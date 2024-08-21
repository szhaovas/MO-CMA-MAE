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

        obj1_grid = np.square(x_grid - self.shift[0]) - self.A * np.cos(
            2 * np.pi * (x_grid - self.shift[0])
        )
        obj2_grid = np.square(x_grid - self.shift[1]) - self.A * np.cos(
            2 * np.pi * (x_grid - self.shift[1])
        )

        self._best_obj = np.array([0, 0])
        self._worst_obj = (
            np.max(np.vstack((obj1_grid, obj2_grid)), axis=1)
            * self.solution_dim
        )

    def evaluate(self, sols: np.ndarray):
        if not sols.shape[1] == self.solution_dim:
            raise ValueError(
                f"Expects sols to have shape (,{self.solution_dim}), actually gets shape {sols.shape}"
            )

        displacement = np.vstack(
            ([sols - self.shift[0]], [sols - self.shift[1]])
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

        # Calculate BCs.
        # FIXME: update clip to match MOME implementation?
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
