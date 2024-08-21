import warnings
import numpy as np
import ribs.schedulers


class Scheduler(ribs.schedulers.Scheduler):

    def _validate_tell_data(self, data):
        """Preprocesses data passed into tell methods."""
        for name, arr in data.items():
            data[name] = np.asarray(arr)
            self._check_length(name, arr)

        # Convenient to have solutions be part of data, so that everything is
        # just one dict.
        data["solution"] = self._cur_solutions

        return data

    def _add_to_archives(self, data, success_mask):
        """Adds solutions to both the regular archive and the result archive. Returns
        both add_info and masked_add_info. add_info basically has invalid info at
        indices where success_mask is False, but the array length will be maintained,
        allowing proper passing to the emitters."""

        archive_empty_before = self.archive.empty
        if self._result_archive is not None:
            # Check self._result_archive here since self.result_archive is a
            # property that always provides a proper archive.
            result_archive_empty_before = self.result_archive.empty

        # Add solutions to the archive. Assuming success_mask is always passed since the
        # tell methods of this scheduler do that. Add functions in the archive class
        # will only see the masked data.
        if self._add_mode == "batch":
            masked_data = {name: arr[success_mask] for name, arr in data.items()}
            masked_add_info = self.archive.add(**masked_data)

            # Add solutions to result_archive.
            if self._result_archive is not None:
                self._result_archive.add(**masked_data)
        elif self._add_mode == "single":
            raise NotImplementedError("Please use batch add().")

        # Warn the user if nothing was inserted into the archives.
        if archive_empty_before and self.archive.empty:
            warnings.warn(self.EMPTY_WARNING.format(name="archive"))
        if self._result_archive is not None:
            if result_archive_empty_before and self.result_archive.empty:
                warnings.warn(self.EMPTY_WARNING.format(name="result_archive"))

        # Create add_info with invalid info but the same array length as the original
        # solutions.
        add_info = {}
        for name, arr in masked_add_info.items():
            full_shape = [len(self._cur_solutions)] + list(arr.shape[1:])
            add_info[name] = np.empty(full_shape, dtype=arr.dtype)
            add_info[name][success_mask] = arr

        return add_info, masked_add_info

    def tell(self, objective, measures, success_mask=None, **fields):
        """Returns info for solutions from :meth:`ask`.

        .. note:: The objective and measures arrays must be in the same order as
            the solutions created by :meth:`ask_dqd`; i.e. ``objective[i]`` and
            ``measures[i]`` should be the objective and measures for
            ``solution[i]``.

        Args:
            objective ((batch_size,) array): Each entry of this array contains
                the objective function evaluation of a solution.
            measures ((batch_size, measures_dm) array): Each row of this array
                contains a solution's coordinates in measure space.
            success_mask ((batch_size,) array): Array of bool indicating which
                solutions succeeded. Solutions that failed will not be passed
                back to the emitter in tell(), and their data is ignored.
            fields (keyword arguments): Additional data for each solution. Each
                argument should be an array with batch_size as the first
                dimension.
        Raises:
            RuntimeError: This method is called without first calling
                :meth:`ask`.
            ValueError: One of the inputs has the wrong shape.
        """
        if self._last_called != "ask":
            raise RuntimeError("tell() was called without calling ask().")
        self._last_called = "tell"

        success_mask = (
            np.asarray(success_mask, dtype=bool)
            if success_mask is not None
            else np.ones_like(objective, dtype=bool)
        )
        self._check_length("success_mask", success_mask)

        data = self._validate_tell_data(
            {
                "objective": objective,
                "measures": measures,
                **fields,
            }
        )

        add_info, _ = self._add_to_archives(data, success_mask)

        # Keep track of pos because emitters may have different batch sizes.
        pos = 0
        for emitter, n in zip(self._emitters, self._num_emitted):
            end = pos + n
            mask = success_mask[pos:end]

            emitter.tell(
                **{name: arr[pos:end][mask] for name, arr in data.items()},
                add_info={name: arr[pos:end][mask] for name, arr in add_info.items()},
            )
            pos = end
