import numpy as np
from typing import List


class SphereManager:
    def __init__(self, solution_dim: int, shift: List[float]):
        self.solution_dim = solution_dim
        self.shift = shift

        # Best and worst objs used to scale all evaluations to [0, 100]
        # FIXME: Higher objective dimensions?
        self._best_obj = np.array([0, 0])
        # Compute worst_obj with x in [-10.24, 10.24]
        self._worst_obj = np.array(
            [
                max(
                    (10.24 - self.shift[0]) ** 2 * self.solution_dim,
                    (-10.24 - self.shift[0]) ** 2 * self.solution_dim,
                ),
                max(
                    (10.24 - self.shift[1]) ** 2 * self.solution_dim,
                    (-10.24 - self.shift[1]) ** 2 * self.solution_dim,
                ),
            ]
        )

    def evaluate(self, sols: np.ndarray):
        if not sols.shape[1] == self.solution_dim:
            raise ValueError(
                f"Expects sols to have shape (,{self.solution_dim}), actually gets shape {sols.shape}"
            )

        displacement = np.vstack(([sols - self.shift[0]], [sols - self.shift[1]])).T
        raw_obj = np.sum(np.square(displacement), axis=0)
        objs = (raw_obj - self._worst_obj) / (self._best_obj - self._worst_obj) * 100

        # FIXME: update clip to match MOME implementation?
        clipped = sols.copy()
        clip_indices = np.where(np.logical_or(clipped > 5.12, clipped < -5.12))
        clipped[clip_indices] = 5.12 / clipped[clip_indices]
        measures = np.concatenate(
            (
                np.sum(clipped[:, : self.solution_dim // 2], axis=1, keepdims=True),
                np.sum(clipped[:, self.solution_dim // 2 :], axis=1, keepdims=True),
            ),
            axis=1,
        )

        # return objs, measures
        return objs, measures
