import numbers
import logging
import numpy as np
import ribs.emitters


logger = logging.getLogger(__name__)


class EvolutionStrategyEmitter(ribs.emitters.EvolutionStrategyEmitter):
    def _check_restart(self, num_parents, measures=None):
        """Emitter-side checks for restarting the optimizer.

        The optimizer also has its own checks.

        Args:
            num_parents (int): The number of solution to propagate to the next
                generation from the solutions generated by CMA-ES.

        Raises:
          ValueError: If :attr:`restart_rule` is invalid.
        """
        if isinstance(self._restart_rule, numbers.Integral):
            return self._itrs % self._restart_rule == 0
        if self._restart_rule == "no_improvement":
            return num_parents == 0
        if self._restart_rule == "basic":
            return False
        if self._restart_rule == "cycle":
            if measures is None:
                return False
            occupied, data = self.archive.retrieve(measures)
            if np.any(occupied):
                avg_numvisits = self.archive.total_numvisits / len(self.archive)
                return (np.max(data["numvisits"]) / avg_numvisits) > 10
            return False
        raise ValueError(f"Invalid restart_rule {self._restart_rule}")

    def tell(self, solution, objective, measures, add_info, **fields):
        """Same as in vanilla EvolutionStrategyEmitter, except skips the
        dimension checks, since now objective is no longer 1d.

        Dimension checks are done at the archive level instead.
        """
        data = {
            "solution": np.asarray(solution),
            "objective": np.asarray(objective),
            "measures": np.asarray(measures),
        }

        for k, v in add_info.items():
            add_info[k] = np.asarray(v)

        # Increase iteration counter.
        self._itrs += 1

        # Count number of new solutions.
        new_sols = add_info["status"].astype(bool).sum()

        # Sort the solutions using ranker.
        indices, ranking_values = self._ranker.rank(self, self.archive, data, add_info)

        # Select the number of parents.
        num_parents = (
            new_sols if self._selection_rule == "filter" else self._batch_size // 2
        )

        # Update Evolution Strategy.
        self._opt.tell(indices, ranking_values, num_parents)

        # Check for reset.
        if self._opt.check_stop(ranking_values[indices]) or self._check_restart(
            new_sols, measures
        ):
            new_x0 = self.archive.sample_elites(1)["solution"][0]
            self._opt.reset(new_x0)
            self._ranker.reset(self, self.archive)
            self._restarts += 1
            logging.info(
                f"Restart triggered! \n Ranker type: {type(self._ranker)} \n new_x0: \n {new_x0} \n numof_restarts = {self._restarts}".replace(
                    "\n", "\n\t"
                )
            )
