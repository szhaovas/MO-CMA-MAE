import numpy as np
from typing import List


class RastriginManager:
    def __init__(self, solution_dim: int, shift: List[float], A: float):
        self.solution_dim = solution_dim
        self.shift = shift
        self.A = A

        # Best and worst objs used to scale all evaluations to [0, 100]
        # FIXME: Higher objective dimensions?
        # Calculate the worst obj value within parameter range with
        # grid search
        x_grid = np.linspace(-10.24, 10.24, int(1e6))

        obj_grid = np.array([np.square(x_grid - s) - self.A * np.cos(
            2 * np.pi * (x_grid - self.shift[1])
        ) for s in self.shift])

        self._best_obj = np.zeros(len(shift))
        self._worst_obj = (
            np.max(obj_grid, axis=1)
            * self.solution_dim
        )

    def evaluate(self, sols: np.ndarray):
        if not sols.shape[1] == self.solution_dim:
            raise ValueError(
                f"Expects sols to have shape (,{self.solution_dim}), actually gets shape {sols.shape}"
            )

        displacement = np.vstack(
            ([[sols - s for s in self.shift]])
        ).T
        sum_terms = np.square(displacement) - self.A * np.cos(
            2 * np.pi * displacement
        )
        raw_obj = self.A * self.solution_dim + np.sum(sum_terms, axis=0)

        objs = (
            (raw_obj - self._worst_obj)
            / (self._best_obj - self._worst_obj)
            * 100
        )

        clipped = sols.copy()
        clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
        clipped[clip_indices] = 5.12 / clipped[clip_indices]
        measures = np.concatenate(
            (
                np.sum(
                    clipped[:, : self.solution_dim // 2], axis=1, keepdims=True
                ),
                np.sum(
                    clipped[:, self.solution_dim // 2 :], axis=1, keepdims=True
                ),
            ),
            axis=1,
        )

        return objs, measures
