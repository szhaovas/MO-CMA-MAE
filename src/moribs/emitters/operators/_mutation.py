"""Mutation Operator"""

import numpy as np

from ribs.emitters.operators._operator_base import OperatorBase


class MutationOperator(OperatorBase):
    """Implements mutation operator which produces a new population by mutating a
    portion of the parent. Modified from the mutation_operators.py from QDax.

    Please see
    <https://github.com/adaptive-intelligent-robotics/QDax/blob/main/qdax/core/emitters/mutation_operators.py>

    Args:
        mutation_proportion (float): The proportion of parameters that should
            be mutated.
        eta (float): The inverse of the power of the mutation applied.
        lower_bounds (array-like): Upper bounds of the solution space. Passed in
            by emitter
        upper_bounds (array-like): Upper bounds of the solution space. Passed in
            by emitter
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    """

    def __init__(
        self, mutation_proportion, eta, lower_bounds, upper_bounds, seed=None
    ):
        if mutation_proportion > 1:
            raise ValueError(
                "Invalid mutation_proportion; must not exceed 1,"
                f"received {mutation_proportion}."
            )

        if not np.all(upper_bounds >= lower_bounds):
            raise ValueError(
                "Invalid lower/upper bounds; lower bounds cannot be larger than upper bounds,"
                f"received lower bounds {lower_bounds},"
                f"received upper bounds {upper_bounds},"
            )

        self._mutation_proportion = mutation_proportion
        self._eta = eta
        self._lower_bounds = lower_bounds
        self._upper_bounds = upper_bounds

        self._rng = np.random.default_rng(seed)

    @property
    def parent_type(self):
        """int: Parent Type to be used by selector."""
        return 1

    def _scale_to_range(self, sol, flip=False):
        if not flip:
            return (sol - self._lower_bounds) / (
                self._upper_bounds - self._lower_bounds
            )
        else:
            return (self._upper_bounds - sol) / (
                self._upper_bounds - self._lower_bounds
            )

    def ask(self, parents):
        """Returns a new population by mutating ``mutation_proportion`` of
        the parent.

        Args:
            parents (array-like): (batch_size, solution_dim) Array of shape
                (batch_size, solution_dim) containing solutions to be mutated.

        Returns:
            numpy.ndarray: ``(batch_size, solution_dim)`` array that contains
            ``batch_size`` mutated solutions.
        """
        parents = np.asarray(parents)
        batch_size, solution_dim = parents.shape

        x1 = parents.copy()
        mutation_num = int(self._mutation_proportion * solution_dim)
        # Each individual within batch samples its own mutation indices
        for i in range(batch_size):
            # Mutation indices within each individual cannot repeat.
            mutation_indices = self._rng.choice(
                solution_dim, size=mutation_num, replace=False
            )

            delta_1 = self._scale_to_range(x1[i][mutation_indices])
            delta_2 = self._scale_to_range(x1[i][mutation_indices], flip=True)
            mutpow = 1.0 / (1.0 + self._eta)

            rand = self._rng.uniform(size=mutation_num)
            value1 = 2.0 * rand + (
                np.power(delta_1, 1.0 + self._eta) * (1.0 - 2.0 * rand)
            )
            value2 = 2.0 * (1 - rand) + 2.0 * (
                np.power(delta_2, 1.0 + self._eta) * (rand - 0.5)
            )
            value1 = np.power(value1, mutpow) - 1.0
            value2 = 1.0 - np.power(value2, mutpow)

            delta_q = np.where(rand < 0.5, value1, value2)

            x1[i][mutation_indices] += delta_q * (
                self._upper_bounds - self._lower_bounds
            )

        return np.clip(x1, self._lower_bounds, self._upper_bounds)
