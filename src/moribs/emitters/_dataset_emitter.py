import numpy as np
import pickle as pkl
from pathlib import Path
from ribs.emitters._emitter_base import EmitterBase


class DatasetEmitter(EmitterBase):
    """This emitter emits solutions from a pre-constructed dataset, such that it
    emits the same solutions regardless of the state of the archive.

    Args:
        archive (ribs.archives.ArchiveBase): An archive to use when creating and
            inserting solutions. For instance, this can be
            :class:`ribs.archives.GridArchive`.
        bounds (None or array-like): Bounds of the solution space. Solutions are
            clipped to these bounds. Pass None to indicate there are no bounds.
            Alternatively, pass an array-like to specify the bounds for each
            dim. Each element in this array-like can be None to indicate no
            bound, or a tuple of ``(lower_bound, upper_bound)``, where
            ``lower_bound`` or ``upper_bound`` may be None to indicate no bound.
        itrs (int): Number of iterations the experiment will run. Defines the size
            of the pre-constructed dataset together with batch_size.
        batch_size (int): Number of solutions to return in :meth:`ask`.
        seed (int): Value to seed the random number generator. Set to None to
            avoid a fixed seed.
    Raises:
        ValueError: There is an error in the bounds configuration.
    """

    def __init__(self, archive, *, x0, itrs, bounds=None, batch_size=64, seed=None):
        self._rng = np.random.default_rng(seed)
        self._itrs = itrs
        self._batch_size = batch_size

        EmitterBase.__init__(
            self,
            archive,
            solution_dim=archive.solution_dim,
            bounds=bounds,
        )

        dataset_filename = Path.cwd() / f"dataset_{itrs}:{batch_size}.txt"
        if dataset_filename.is_file():
            # If another compatible dataset is already initialized by another emitter,
            #   copy that dataset instead of generating its own.
            self.dataset = np.loadtxt(dataset_filename)
        else:
            bounds_arr = np.array(bounds)
            self.dataset = self._rng.uniform(
                bounds_arr[:, 0],
                bounds_arr[:, 1],
                size=((itrs + 1) * batch_size, self.archive.solution_dim),
            )

            np.savetxt(dataset_filename, self.dataset)

        self.itr_counter = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["dataset"]
        return state

    @property
    def itrs(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._itrs

    @property
    def batch_size(self):
        """int: Number of solutions to return in :meth:`ask`."""
        return self._batch_size

    def ask(self):
        data = self.dataset[
            self.itr_counter
            * self.batch_size : (self.itr_counter + 1)
            * self.batch_size,
            :,
        ]
        self.itr_counter += 1
        return data
