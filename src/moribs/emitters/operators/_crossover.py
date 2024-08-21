"""Crossover Operator"""

import numpy as np

from ribs.emitters.operators._operator_base import OperatorBase


class CrossoverOperator(OperatorBase):
    """Implements crossover operator which produces a new population by mixing
    the parameters of two parents.

    Args:
        crossover_proportion (float): The proportion of parameters that should
            be mixed.
        lower_bounds (array-like): Upper bounds of the solution space. Passed in
            by emitter
        upper_bounds (array-like): Upper bounds of the solution space. Passed in
            by emitter
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    """

    def __init__(
        self, crossover_proportion, lower_bounds, upper_bounds, seed=None
    ):
        if crossover_proportion > 1:
            raise ValueError(
                "Invalid crossover_proportion; must not exceed 1,"
                f"received {crossover_proportion}."
            )

        if not np.all(upper_bounds >= lower_bounds):
            raise ValueError(
                "Invalid lower/upper bounds; lower bounds cannot be larger than upper bounds,"
                f"received lower bounds {lower_bounds},"
                f"received upper bounds {upper_bounds},"
            )

        self._crossover_proportion = crossover_proportion
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        self._rng = np.random.default_rng(seed)

    @property
    def parent_type(self):
        """int: Parent Type to be used by selector."""
        return 2

    def ask(self, parents):
        """Returns a new population by mixing ``crossover_proportion`` of
        two parents

        Example:
            crossover_proportion: 0.75
            parent_1:  [0,1,2,3]
                          | | |
            parent_2:  [4,5,6,7]
            offspring: [0,5,6,7]

        Args:
            parents (array-like): Array of shape (2, batch_size, solution_dim),
                containing pairs of parent solutions.

        Returns:
            numpy.ndarray: ``(batch_size, solution_dim)`` array that contains
            ``batch_size`` new solutions.
        """
        parents = np.asarray(parents)
        batch_size = parents.shape[1]
        solution_dim = parents.shape[2]

        x1 = parents[0].copy()
        x2 = parents[1]
        crossover_num = int(self._crossover_proportion * solution_dim)
        # Each individual within batch samples its own crossover indices
        for i in range(batch_size):
            # Crossover indices within each individual cannot repeat.
            crossover_indices = self._rng.choice(
                solution_dim, size=crossover_num, replace=False
            )
            x1[i][crossover_indices] = x2[i][crossover_indices]

        return np.clip(x1, self._lower_bounds, self._upper_bounds)
