"""Generator for Overcooked."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


class OvercookedGenerator(nn.Module):
    """Generator DCGAN.

    Args:
        i_size: size of input image
        nz: size of latent z vector
        nc: total number of objects in the environment
        ngf: number of output channels of initial conv2d layer
        n_extra_layers: number of extra layers with out_channels to be ngf to
            add
        lvl_height: Height of final Overcooked level.
        lvl_width: Width of final Overcooked level.
        model_file: Name of the file with saved weights. This is assumed to be
            in `src/overcooked/emulation_model/data`, but you can pass a tuple
            of (model_file, False) to indicate that we should not take the
            model_file path relative to the data directory.

    Note:
        input is a latent vector of size nz
    """

    def __init__(
        self,
        i_size: int,
        nz: int,
        nc: int,
        ngf: int,
        n_extra_layers: int,
        lvl_height: int,
        lvl_width: int,
        model_file: str,
    ):
        super().__init__()
        self.lvl_height = lvl_height
        self.lvl_width = lvl_width

        if isinstance(model_file, tuple) and not model_file[1]:
            # Get full path from first entry of tuple.
            self.model_file = model_file[0]
        else:
            # Set model file relative to the data directory.
            self.model_file = Path(__file__).parent / "data" / model_file

        self.nz = nz
        assert i_size % 16 == 0, "i_size has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != i_size:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module(
            f"initial:{nz}-{cngf}:convt",
            nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False),
        )
        main.add_module(f"initial:{cngf}:batchnorm", nn.BatchNorm2d(cngf))
        main.add_module(f"initial:{cngf}:relu", nn.ReLU(True))

        csize, cndf = 4, cngf  # pylint: disable = unused-variable
        while csize < i_size // 2:
            main.add_module(
                f"pyramid:{cngf}-{cngf // 2}:convt",
                nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False),
            )
            main.add_module(f"pyramid:{cngf // 2}:batchnorm", nn.BatchNorm2d(cngf // 2))
            main.add_module(f"pyramid:{cngf // 2}:relu", nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module(
                f"extra-layers-{t}:{cngf}:conv",
                nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False),
            )
            main.add_module(f"extra-layers-{t}:{cngf}:batchnorm", nn.BatchNorm2d(cngf))
            main.add_module(f"extra-layers-{t}:{cngf}:relu", nn.ReLU(True))

        main.add_module(
            f"final:{cngf}-{nc}:convt",
            nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False),
        )
        main.add_module(f"final:{nc}:tanh", nn.Tanh())
        self.main = main

    def forward(self, latent):
        """Runs the generator on the batch of latent vectors.

        Args:
            latent: (n, nz) array of latent vectors.
        Returns:
            (n, nc, i_size, i_size) array of uncropped level images.
        """
        # Reshaping needed since the layers are all convolutions.
        return self.main(latent.reshape(*latent.shape, 1, 1))

    def levels_from_latent(self, latent: np.ndarray) -> np.ndarray:
        """Generates Overcooked levels from latent vectors.

        Args:
            latent: (n, nz) array of latent vectors.
        Returns:
            Each level is a lvl_height x lvl_width array of integers where each
            integer is the type of object, so the output of this method is an
            array of levels of shape (n, lvl_height, lvl_width).
        """
        # Handle no_grad here since we expect everything to be numpy arrays.
        with torch.no_grad():
            latent = torch.as_tensor(
                np.asarray(latent),
                dtype=torch.float,
                device="cpu",
            )
            lvls = self(latent)
            cropped_lvls = lvls[:, :, : self.lvl_height, : self.lvl_width]
            # Convert from one-hot encoding to ints.
            int_lvls = torch.argmax(cropped_lvls, dim=1)
            return int_lvls.cpu().detach().numpy()

    def softmax_levels_from_latent(
        self, latent: torch.Tensor, grad: bool = True
    ) -> torch.Tensor:
        """Generates softmax levels from latent tensor.

        Args:
            latent: (n, nz) tensor of latent vectors.
            grad: True if gradient should be calculated
        Returns:
            Each level is a nc x lvl_height x lvl_width array where each element
            is the probability of the corresponding type of object, so the
            output of this method is an array of levels of shape (n, nc,
            lvl_height, lvl_width).
        """
        if grad:
            lvls = self(latent)
            cropped_lvls = lvls[:, :, : self.lvl_height, : self.lvl_width]
            return nn.functional.softmax(cropped_lvls, dim=1)
        else:
            with torch.no_grad():
                lvls = self(latent)
                cropped_lvls = lvls[:, :, : self.lvl_height, : self.lvl_width]
                return nn.functional.softmax(cropped_lvls, dim=1)

    def load_from_saved_weights(self):
        self.load_state_dict(
            torch.load(
                self.model_file,
                map_location="cpu",
            ),
        )
        return self
