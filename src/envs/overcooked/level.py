"""Provides OvercookedLevel."""
import numpy as np

INT_TO_OBJ_TYPES = "12XSPOD "
OBJ_TYPES_TO_INT = {obj: i for i, obj in enumerate(INT_TO_OBJ_TYPES)}
NUM_OBJS = len(INT_TO_OBJ_TYPES)


class OvercookedLevel:
    """Utilities for handling Overcooked levels.

    Mostly useful when evaluating the levels individually, as pyribs and the GAN
    typically handle levels in batch.

    Args:
        data: 2D numpy array of integers where the integers represent the object
            types. The array must be of shape (lvl_height, lvl_width).
    """

    def __init__(self, data):
        self.data = data

    @property
    def lvl_height(self):
        return self.data.shape[0]

    @property
    def lvl_width(self):
        return self.data.shape[1]

    @staticmethod
    def str_to_number(lvl_str):
        """Converts list of strings (each string is a row) to numpy array."""
        np_lvl = np.zeros((len(lvl_str), len(lvl_str[0])))
        for x, row in enumerate(lvl_str):
            row = row.strip()
            for y, tile in enumerate(row):
                np_lvl[x, y] = OBJ_TYPES_TO_INT[tile]
        return np_lvl

    def to_str_grid(self):
        """Converts level to grid of characters."""
        return [[INT_TO_OBJ_TYPES[x] for x in row] for row in self.data]

    def to_str(self):
        """Converts level to single string."""
        return "\n".join(["".join(row) for row in self.to_str_grid()])
