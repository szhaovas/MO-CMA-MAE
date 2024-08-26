import numpy as np
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
)
# from ._nondominatedarchive import NonDominatedList
from ._nda_fast import BiobjectiveNondominatedSortedList


class MOCMAMEArchive(PFCVTArchive):
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
        bias_sampling,
        init_discount,
        alpha,
        # qd_update_freq,
        pop_size=None,
        max_pf_size=None,
        hvi_cutoff_threshold=None,
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
            bias_sampling=bias_sampling,
            max_pf_size=max_pf_size,
            init_discount=1,
            alpha=1,
            hvi_cutoff_threshold=hvi_cutoff_threshold,
            seed=seed,
            samples=samples,
        )

        # self._num_adds = 0
        # self._qd_update_freq = qd_update_freq

        if pop_size is None and max_pf_size is None:
            raise ValueError(
                "max_pf_size must be provided to calculate pop_size if"
                "pop_size is not provided."
            )
        elif pop_size is not None:
            self._pop_size = pop_size
        else:
            self._pop_size = cells * max_pf_size

        self.main = PFCVTArchive(
            solution_dim=solution_dim,
            objective_dim=objective_dim,
            reference_point=reference_point,
            cells=cells,
            ranges=ranges,
            bias_sampling=bias_sampling,
            max_pf_size=max_pf_size,
            init_discount=init_discount,
            alpha=alpha,
            hvi_cutoff_threshold=hvi_cutoff_threshold,
            seed=seed,
            samples=samples,
            custom_centroids=np.copy(self.centroids)
        )

    def qd_update(self, **fields):
        """No need for qd_update since passive and main archives are always in sync.
        """
        pass

    def add_single(self, solution, objective, measures, **fields):
        raise NotImplementedError("Please use batch add() for COMO-CMA-ES.")

    def add(self, solution, objective, measures, **fields):
        super().add(solution, objective, measures, **fields)
        return self.main.add(solution, objective, measures, **fields)

    @property
    def empty(self):
        """Since passive archive is not always in sync with the main pop,
        checks whether the main PF is empty."""
        return self.main.empty

    def sample_elites(self, n):
        """Since passive archive is not always in sync with the main pop,
        samples an elite from the main pop instead of passive archive
        """
        return self.main.sample_elites(n)

    # def _stats_reset(self):
    #     super()._stats_reset()
    #     self._num_adds = 0
