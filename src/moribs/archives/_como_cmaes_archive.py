import numpy as np
import logging
from ribs._utils import (
    check_batch_shape,
    check_solution_batch_dim,
    check_finite,
)
from ribs.archives._add_status import AddStatus
from src.moribs.archives import PFCVTArchive
from ._pf_utils import (
    batch_entry_pf,
    compute_moqd_score,
    compute_best_index,
    compute_total_numvisits
)
# from ._nondominatedarchive import NonDominatedList
from ._nda_fast import BiobjectiveNondominatedSortedList


logger = logging.getLogger(__name__)


class COMOCMAESArchive(PFCVTArchive):
    """Adapts COMO-CMA-ES to MOQD by pairing a Pareto Front which returns UHVI
    indicator value on add() with a PFCVTArchive as passive QD archive.

    COMO-CMA-ES paper: <https://arxiv.org/abs/1904.08823>

    A few differences compared to vanilla COMO-CMA-ES:
        - Pareto Front is represented with ``pop_size`` solutions instead of a number
        of CMA-ES instances and their mean values (incumbent solutions).
        - Each CMA-ES instance is free to optimize any solution on the Pareto Front.
        There is no longer a one-to-one correspondence between CMA-ES instance and solution.
        - Since any CMA-ES instance can optimize any solution, it is no longer necessary
        to randomize scheduler query order to avoid bias.
    """

    def __init__(
        self,
        *,
        solution_dim,
        objective_dim,
        reference_point,
        cells,
        ranges,
        pop_size,
        seed=None,
        samples=100_000,
    ):

        PFCVTArchive.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=objective_dim,
            reference_point=reference_point,
            cells=cells,
            ranges=ranges,
            bias_sampling=False,
            init_discount=1,
            alpha=1,
            max_pf_size=None,
            hvi_cutoff_threshold=None,
            seed=seed,
            samples=samples,
        )

        self._pop_size = pop_size

        # self.comocmaes = NonDominatedList(
        #     maxlen=self.pop_size, reference_point=self.reference_point, seed=seed
        # )
        self.comocmaes = BiobjectiveNondominatedSortedList(
            init_discount=1, alpha=1, maxlen=self.pop_size, reference_point=self.reference_point, seed=seed
        )

    # @property
    # def num_adds(self):
    #     """Number of times add() function has been called on this archive
    #     should equal ``itr * batch_size``
    #     """
    #     return self._num_adds

    @property
    def pop_size(self):
        return self._pop_size

    def qd_update(self, **fields):
        """Re-add solutions, objectives, and measures stored in the COMO-CMA-ES population to the
        passive archive.
        The passive archive is updated by first emptying and then re-adding NSGA2Population.population.

        FIXME: Maybe more efficient to identify changed individuals and modify those only?

        Returns:
            Doesn't return add_info since COMO-CMA-ES uses UHVI values calculated w.r.t. the entire PF.
        """
        self.clear()
        data = {
            "solution": np.array(self.comocmaes.solutions),
            "objective": np.array(self.comocmaes.objectives),
            "measures": np.array(self.comocmaes.measures),
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
                "total_numvisits": self.total_numvisits
            },
            [batch_entry_pf, compute_moqd_score, compute_best_index, compute_total_numvisits],
        )

        # Updates passive archive QD metrics.
        hypervolume_sum = add_info.pop("hypervolume_sum")
        best_index = add_info.pop("best_index")
        total_numvisits = add_info.pop("total_numvisits")
        if not np.all(add_info["status"] == 0):
            self._stats_update(hypervolume_sum, best_index, total_numvisits)

    def add_single(self, solution, objective, measures, **fields):
        raise NotImplementedError("Please use batch add() for COMO-CMA-ES.")

    def add(self, solution, objective, measures, **fields):
        """Inserts the new solutions and objectives into the population represented by
        NonDominatedList.

        NOTE: Call ``qd_update()`` manually before accessing passive archive.

        Args:
            solution (np.ndarray): a 2D array containing
            ``batch_size * solution_dim`` entries.
            objective (np.ndarray): a 2D array containing
            ``batch_size * objective_dim`` entries.
            measures (np.ndarray): a 2D array containing
            ``batch_size * measure_dim`` entries.

        Returns:
            add_info (dict): Contains ``state`` and ``value`` for the batch solutions w.r.t.
                to the Pareto Front population.
        """
        batch_size = solution.shape[0]
        check_batch_shape(solution, "solution", self.solution_dim, "solution_dim", "")
        check_solution_batch_dim(solution, "solution", batch_size, extra_msg="")

        check_batch_shape(
            objective, "objective", self.objective_dim, "objective_dim", ""
        )
        check_solution_batch_dim(objective, "objective", batch_size, extra_msg="")

        check_batch_shape(measures, "measures", self.measure_dim, "measure_dim", "")
        check_solution_batch_dim(measures, "measures", batch_size, extra_msg="")
        check_finite(measures, "measures")

        # add_info for the batch solutions are calculated w.r.t. to the Pareto Front population.
        add_info = {
            "status": np.zeros(batch_size, dtype=np.int32),
            "value": np.zeros(batch_size, dtype=np.float64),
        }
        for i, (sol, objs, meas) in enumerate(zip(solution, objective, measures)):
            value, status = self.comocmaes.hypervolume_improvement(objs, uhvi=True)

            # Since NonDominatedList doesn't have measures, there can no longer be AddStatus.NEW
            if status == AddStatus.NEW:
                status = AddStatus.IMPROVE_EXISTING

            if (
                (not self.hvi_cutoff_threshold is None)
                and (status == AddStatus.IMPROVE_EXISTING)
                and (value < self.hvi_cutoff_threshold)
            ):
                value, status = 0, AddStatus.NOT_ADDED

            add_info["value"][i], add_info["status"][i] = value, status
        
        can_insert = (add_info["status"] != AddStatus.NOT_ADDED)

        logger.info(
            f"Among {batch_size} generated solutions, {sum(can_insert)} are non-dominated."
        )

        actually_inserted = np.full(np.sum(can_insert), True)
        # Add batch solututions to the Pareto Front population.
        for i, (sol, objs, meas) in enumerate(zip(solution[can_insert], objective[can_insert], measures[can_insert])):
            added_at = self.comocmaes.add(sol, objs, meas)
            if added_at is None:
                actually_inserted[i] = False

        logger.info(
            f"{sum(actually_inserted)} are dominated by others within batch."
        )

        return add_info

    @property
    def empty(self):
        """Since passive archive is not always in sync with the main pop,
        checks whether the main PF is empty."""
        return len(self.comocmaes) == 0

    def sample_elites(self, n):
        """Since passive archive is not always in sync with the main pop,
        samples an elite from the main pop instead of passive archive
        """
        if self.empty:
            raise IndexError("No elements in archive.")

        return {"solution": self._rng.choice(self.comocmaes.solutions, size=n)}
