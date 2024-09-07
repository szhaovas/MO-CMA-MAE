import numpy as np
from src.moribs.archives import PFCVTArchive


class MOQDArchive(PFCVTArchive):
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
    ):

        PFCVTArchive.__init__(
            self,
            solution_dim=solution_dim,
            objective_dim=objective_dim,
            reference_point=reference_point,
            cells=cells,
            ranges=ranges,
            bias_sampling=bias_sampling,
            max_pf_size=None,
            init_discount=1,
            alpha=1,
            hvi_cutoff_threshold=hvi_cutoff_threshold,
            seed=seed,
            samples=samples,
        )

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

    def add(self, solution, objective, measures, **fields):
        super().add(solution, objective, measures, **fields)
        return self.main.add(solution, objective, measures, **fields)

    @property
    def empty(self):
        return self.main.empty

    def sample_elites(self, n):
        return self.main.sample_elites(n)
