"""Provides the MutationEmitter."""

import numpy as np

from ribs._utils import check_batch_shape
from ribs.emitters._emitter_base import EmitterBase
from src.moribs.emitters.operators import MutationOperator


class MutationEmitter(EmitterBase):
    """Emits solutions by doing polynomial mutation on existing archive solutions.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        mutation_proportion (float): The proportion of parameters that should
            be mutated.
        eta (float): The inverse of the power of the mutation applied.
        initial_solutions (array-like): An (n, solution_dim) array of solutions
            to be used when the archive is empty. If this argument is None, then
            solutions will be sampled from a Gaussian distribution centered at
            ``x0`` with standard deviation ``sigma``.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: The archive is empty at the time when ask() is called.
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(
        self,
        archive,
        *,
        mutation_proportion,
        eta=1,
        initial_solutions=None,
        bounds=None,
        batch_size=64,
        seed=None
    ):
        self._batch_size = batch_size

        self._mutation_proportion = mutation_proportion
        self._eta = eta

        self._initial_solutions = None
        if initial_solutions is not None:
            self._initial_solutions = np.asarray(initial_solutions)
            check_batch_shape(
                self._initial_solutions,
                "initial_solutions",
                archive.solution_dim,
                "archive.solution_dim",
            )

        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        self._operator = MutationEmitter(
            mutation_proportion=self._mutation_proportion,
            eta=self._eta,
            lower_bounds=self._lower_bounds,
            upper_bounds=self._upper_bounds,
            seed=seed,
        )

    @property
    def initial_solutions(self):
        """numpy.ndarray: The initial solutions which are returned when the
        archive is empty."""
        return self._initial_solutions

    @property
    def mutation_proportion(self):
        """float: The proportion of parameters that should be mutated."""
        return self._mutation_proportion

    @property
    def eta(self):
        """float: The inverse of the power of the mutation applied."""
        return self._eta

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        """Generates ``batch_size`` solutions.

        If the archive is empty and ``self._initial_solutions`` is set, we
        return ``self._initial_solutions``.
        Otherwise, solutions are generated by first randomly sampling ``self._batch_size``
        parents from the archive, and then calling MutationOperator on sampled parents.

        Returns:
            If the archive is not empty, ``(batch_size, solution_dim)`` array
            -- contains ``batch_size`` new solutions to evaluate. If the
            archive is empty, we return ``self._initial_solutions``, which
            might not have ``batch_size`` solutions.
        """
        if self.archive.empty:
            if self._initial_solutions is not None:
                return np.clip(
                    self._initial_solutions,
                    self.lower_bounds,
                    self.upper_bounds,
                )
            raise ValueError(
                "The archive cannot be empty when ask() is called unless initial_solutions is set."
            )
        else:
            parents = self.archive.sample_elites(self._batch_size)["solution"]

        return self._operator.ask(parents=parents)
