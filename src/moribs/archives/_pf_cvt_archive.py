"""Contains the PFCVTArchive class."""

import logging
import numpy as np

from ribs._utils import (
    check_batch_shape,
    check_solution_batch_dim,
    check_finite,
)
from ribs.archives import CVTArchive, ArrayStore
from ribs.archives._archive_stats import ArchiveStats

from ._nondominatedarchive import NonDominatedList
from ._pf_utils import (
    compute_crowding_distances,
    batch_entry_pf,
    compute_moqd_score,
    compute_best_index,
)


logger = logging.getLogger(__name__)


class PFCVTArchive(CVTArchive):
    """Similar to CVTArchive, except each archive cell contains a Pareto Front,
    instead of a single solution.

    Args:
        objective_dim (int): The dimensionality of the (multi)objective.
        max_pf_size (int): The maximum number of solutions each Pareto Front can contain.
            If None, no maximum Pareto Front size is enforced.
        hvi_cutoff_threshold (int): The cutoff value below which hypervolume improvement will be
            set to 0. Does not affect the dominated case in UHVI (FIXME: refer to paper).
            If None, HVI is not clipped.
        reference_point (np.ndarray): A 1D array with length equal to ``objective_dim``.
            The reference point used when calculating hypervolumes.
        bias_sampling (bool): Whether to uniformly or biasedly sample elites from the archive.
            See sample_elites() for more details.


        Other params are the same as their counterparts in CVTArchive. The current
        version of PFCVTArchive does not implement annealing, and assumes the objective
        values it receives are already scaled to [0,100].
    """

    def __init__(
        self,
        *,
        solution_dim,
        objective_dim,
        reference_point,
        cells,
        ranges,
        bias_sampling,
        init_discount,
        alpha,
        max_pf_size=None,
        hvi_cutoff_threshold=None,
        seed=None,
        samples=100_000,
        custom_centroids=None
    ):
        CVTArchive.__init__(
            self,
            solution_dim=solution_dim,
            cells=cells,
            ranges=ranges,
            seed=seed,
            samples=samples,
            custom_centroids=custom_centroids
        )

        """Compared to CVTArchive, ArrayStore object replaces the fields
        of "solution", "objective", and "measures" with "pf" and "hypervolume"
        
        The former stores the Pareto Front objects within all archive cells.
        The latter stores the hypervolumes of all Pareto Fronts (FIXME: Add an interface?)

        Individual solutions, and their (multi)objective values and measures, can now
        be accessed from within the Pareto Front objects.
            - This is because we need to account for the scenario when we do not limit
            the maximum Pareto Front size.
        """
        self._store = ArrayStore(
            field_desc={"pf": ((), object), "hypervolume": ((), np.float64)},
            capacity=self._cells,
        )

        self._objective_dim = objective_dim
        self._reference_point = reference_point
        self._max_pf_size = max_pf_size
        self._hvi_cutoff_threshold = hvi_cutoff_threshold
        self._bias_sampling = bias_sampling
        # Initialize all Pareto Fronts to be empty.
        for i in range(self._cells):
            self._store._fields["pf"][i] = NonDominatedList(
                init_discount=init_discount, alpha=alpha, maxlen=max_pf_size, reference_point=reference_point, seed=seed
            )

    @property
    def objective_dim(self):
        return self._objective_dim

    @property
    def reference_point(self):
        return self._reference_point

    @property
    def max_pf_size(self):
        return self._max_pf_size

    @property
    def hvi_cutoff_threshold(self):
        return self._hvi_cutoff_threshold

    @property
    def bias_sampling(self):
        return self._bias_sampling

    def add_single(self, solution, objective, measures, **fields):
        """Inserts a single solution into the archive. Currently
        simply a wrapper for add().

        Args:
            solution (np.ndarray): a 1D array with ``solution_dim`` entries.
            objective (np.ndarray): a 1D array with ``objective_dim`` entries.
            measures (np.ndarray): a 1D array with ``measure_dim`` entries.

        Returns:
            See add().
        """
        add_info = self.add(
            np.expand_dims(solution, axis=0),
            np.expand_dims(objective, axis=0),
            np.expand_dims(measures, axis=0),
            **fields,
        )

        return add_info

    def add(self, solution, objective, measures, **fields):
        """Inserts the given solutions, and their corresponding objectives into
        the archive cells they are assigned according to their measures.

        Args:
            solution (np.ndarray): a 2D array containing
            ``batch_size * solution_dim`` entries.
            objective (np.ndarray): a 2D array containing
            ``batch_size * objective_dim`` entries.
            measures (np.ndarray): a 2D array containing
            ``batch_size * measure_dim`` entries.

        Returns:
            dict: Information describing the result of the add operation. The
            dict contains the following keys:

            - ``"status"`` (:class:`numpy.ndarray` of :class:`int`): An array of
              integers that represent the "status" obtained when attempting to
              insert each solution in the batch. Each item has the following
              possible values:

              - ``0``: The solution's UHVI value, after clipped by ``hvi_cutoff_threshold``,
                is less than 0, i.e. the solution is strictly dominated by the
                Pareto Front within its assigned archive cell. This corresponds
                to the "not added" case from CVTArchive.add().
              - ``1``: The solution's UHVI value, after clipped by hvi_cutoff_threshold,
                is larger than or equals 0, i.e. the solution weakly dominates
                the Pareto Front within its assigned archive cell. Moreover, the
                current Pareto Front within the archive cell cannot be empty.
                This corresponds to the "improve existing cell" case from
                CVTArchive.add().
              - ``2``: The current Pareto Front within the archive cell is empty.
                This corresponds to the "new cell" case from CVTArchive.add().

              To convert statuses to a more semantic format, cast all statuses
              to :class:`AddStatus` e.g. with ``[AddStatus(s) for s in
              add_info["status"]]``.

            - ``"value"`` (:class:`numpy.ndarray` of :attr:`dtype`): An array
              with UHVI values w.r.t. Pareto Fronts within the assigned archive
              cells for each solution in the batch.

              - ``0`` (not added): The UHVI value is the negated distance from
                a solution to the Pareto Front.
              - ``1`` (improve existing cell): The UHVI value is the hypervolume
                improvement made by the solution to the Pareto Front.
              - ``2`` (new cell): Same as "improve existing cell".
        """
        batch_size = solution.shape[0]
        check_batch_shape(solution, "solution", self.solution_dim, "solution_dim", "")
        check_solution_batch_dim(solution, "solution", batch_size, extra_msg="")

        check_batch_shape(
            objective, "objective", self.objective_dim, "objective_dim", ""
        )
        check_solution_batch_dim(objective, "objective", batch_size, extra_msg="")
        check_finite(objective, "objective")

        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim", "")
        check_solution_batch_dim(measures, "measures", batch_size, extra_msg="")
        check_finite(measures, "measures")

        data = {
            "solution": solution,
            "objective": objective,
            "measures": measures,
            **fields,
        }

        add_info = self._store.add(
            self.index_of(data["measures"]),
            data,
            {
                "hvi_cutoff_threshold": self.hvi_cutoff_threshold,
                # The ArchiveBase class maintains "_objective_sum" when calculating
                # sum, so we use self._objective_sum here to stay compatible.
                "hypervolume_sum": self._objective_sum,
            },
            [batch_entry_pf, compute_moqd_score, compute_best_index],
        )

        hypervolume_sum = add_info.pop("hypervolume_sum")
        # This is the best_index among new data
        best_index = add_info.pop("best_index")
        if not np.all(add_info["status"] == 0):
            self._stats_update(hypervolume_sum, best_index)

        return add_info

    def sample_elites(self, n):
        """This function has two modes with ``self._bias_sampling``

        When ``self._bias_sampling=False``, n solutions are uniformly sampled among
        occupied archive cells.

        When ``self._bias_sampling=True``, n solutions are sampled by first uniformly
        sampling n occupied archive cells (with replacement), and then within each cell,
        sampling a single solution with probability proportional to its crowding distance
        w.r.t. the PF within its archive cell.

        Args:
            n (int): Number of elites to sample.
        Returns:
            dict: Holds a batch of elites randomly selected from the archive.
        Raises:
            IndexError: The archive is empty.
        """
        if self.empty:
            raise IndexError("No elements in archive.")

        random_indices = self._rng.integers(len(self._store), size=n)
        selected_indices = self._store.occupied_list[random_indices]
        _, selected_cells = self._store.retrieve(selected_indices)

        solutions = []
        for pf in selected_cells["pf"]:
            # When PF only has 2 solutions, both their crowding distances will be inf.
            #   In this case, simply sample uniformly.
            if self._bias_sampling and len(pf) >= 3:
                crowding_distances = compute_crowding_distances(
                    np.array(pf.objectives), boundary_inf=False
                )
                probs = crowding_distances / np.sum(crowding_distances)
            else:
                probs = np.ones(len(pf)) / len(pf)

            sol = self._rng.choice(pf.solutions, p=probs)
            solutions.append(sol)

        return {"solution": np.array(solutions)}

    def _stats_update(self, new_objective_sum, new_best_index):
        """Changes ``new_best_elite["objective"]`` to
        ``new_best_elite["hypervolume"]``.
        """
        self._objective_sum = new_objective_sum

        if new_best_index is None:
            return

        new_qd_score = (
            self._objective_sum - self.dtype(len(self)) * self._qd_score_offset
        )

        _, new_best_elite = self._store.retrieve([new_best_index])

        if (
            self._stats.obj_max is None
            or new_best_elite["hypervolume"] > self._stats.obj_max
        ):
            # Convert batched values to single values.
            new_best_elite = {k: v[0] for k, v in new_best_elite.items()}

            new_obj_max = new_best_elite["hypervolume"]
            self._best_elite = new_best_elite
        else:
            new_obj_max = self._stats.obj_max

        self._stats = ArchiveStats(
            num_elites=len(self),
            coverage=self.dtype(len(self) / self.cells),
            qd_score=new_qd_score,
            norm_qd_score=self.dtype(new_qd_score / self.cells),
            obj_max=new_obj_max,
            obj_mean=self._objective_sum / self.dtype(len(self)),
        )
