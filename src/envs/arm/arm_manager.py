import numpy as np
from typing import Tuple
import matplotlib
import matplotlib.pyplot as plt


class ArmManager:
    def __init__(self, solution_dim: int) -> None:
        self.solution_dim = solution_dim

        if self.solution_dim < 3:
            raise ValueError(
                f"arm domain needs at least 3 arm links, received {self.solution_dim}"
            )

        self._link_lengths = np.ones(solution_dim)

    def evaluate(self, sols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not sols.shape[1] == self.solution_dim:
            raise ValueError(
                f"Expects sols to have shape (,{self.solution_dim}), actually gets shape {sols.shape}"
            )

        half_index = sols.shape[1] // 2

        obj1 = -np.var(sols[:, :half_index], axis=1)
        obj2 = -np.var(sols[:, half_index:], axis=1)
        objs = np.column_stack((obj1, obj2))

        # Remap the objective from [-1, 0] to [0, 100]
        objs = (objs + 6.58) / 6.58 * 100.0

        # theta_1, theta_1 + theta_2, ...
        cum_theta = np.cumsum(sols, axis=1)
        # l_1 * cos(theta_1), l_2 * cos(theta_1 + theta_2), ...
        x_pos = self._link_lengths[None] * np.cos(cum_theta)
        # l_1 * sin(theta_1), l_2 * sin(theta_1 + theta_2), ...
        y_pos = self._link_lengths[None] * np.sin(cum_theta)

        measures = np.concatenate(
            (
                np.sum(x_pos, axis=1, keepdims=True),
                np.sum(y_pos, axis=1, keepdims=True),
            ),
            axis=1,
        )

        objs = np.where((objs > 100) | (objs < 0), np.nan, objs)

        return objs, measures
