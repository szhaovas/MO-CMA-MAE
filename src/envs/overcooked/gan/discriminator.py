"""Discriminator for Overcooked.

Note we do not use the discriminator during our experiments -- it is only used
when training the generator.
"""
from torch import nn


class OvercookedDiscriminator(nn.Module):
    """Discriminator DCGAN.

    Args:
        isize: size of input image
        nz: size of latent z vector
        nc: total number of objects in the environment
        ndf: number of output channels of initial conv2d layer
        n_extra_layers: number of extra layers with out_channels to be ndf to
            add

    Note:
        input to the GAN is nc x isize x isize
        output from the GAN is the likelihood of the image being real
    """

    def __init__(self, isize, nz, nc, ndf, n_extra_layers=0):
        super().__init__()
        self.nz = nz
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module(f"initial:conv:{nc}-{ndf}",
                        nn.Conv2d(nc, ndf, 4, 2, 1, bias=False))
        main.add_module(f"initial:relu:{ndf}", nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = isize / 2, ndf

        # add extra layers with out_channels set to ndf
        for t in range(n_extra_layers):
            main.add_module(f"extra-layers-{t}:{cndf}:conv",
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module(f"extra-layers-{t}:{cndf}:batchnorm",
                            nn.BatchNorm2d(cndf))
            main.add_module(f"extra-layers-{t}:{cndf}:relu",
                            nn.LeakyReLU(0.2, inplace=True))

        # add more conv2d layers with exponentially more out_channels
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module(f"pyramid:{in_feat}-{out_feat}:conv",
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module(f"pyramid:{out_feat}:batchnorm",
                            nn.BatchNorm2d(out_feat))
            main.add_module(f"pyramid:{out_feat}:relu",
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module(f"final:{cndf}-{1}:conv",
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))

        # sigmoid to keep output in range [0, 1]
        main.add_module('final:sigmoid', nn.Sigmoid())
        self.main = main

    def forward(self, inputs):
        return self.main(inputs)
