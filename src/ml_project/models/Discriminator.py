from src.project_config import *
from torch import nn

class Discriminator_config(project_config):
    def __init__(self,
                 in_channels=1,
                 layers=5,
                 batch_norm=True,
                 dropout=0.25,
                 output_style='patchGAN',
                 dimensionality='3D',
                 **kwargs):
        super(Discriminator_config, self).__init__(
            config_type='model_PTDiscriminator')
        self.in_channels = in_channels
        self.layers = layers
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_style = output_style
        self.dimensionality = dimensionality

class Discriminator(nn.Module):
    def __init__(self, config = Discriminator_config()):
        super(Discriminator, self).__init__()
        self.config = config

        def _layer(channels_in, channels_out, norm=False, dropout=0.0):
            _ = [nn.Conv3d(in_channels=channels_in,
                           out_channels=channels_out,
                           kernel_size=4,
                           stride=2,
                           padding=1)]
            if norm:
                _.append(nn.BatchNorm3d(channels_out))
            _.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout>0:
                _.append(nn.Dropout(dropout))
            return _

        layers = _layer(channels_in=self.config.in_channels,
                        channels_out=32,
                        norm=False,
                        dropout=self.config.dropout)
        for _ in range(1, self.config.layers,1):
            layers.extend(_layer(channels_in=32*2**(_-1),
                                 channels_out=32*2**_,
                                 norm=self.config.batch_norm,
                                 dropout=self.config.dropout))

        self.discriminator = nn.Sequential(
            *layers
        )
        self.final = nn.Sequential(
            nn.Conv3d(in_channels=32*2**(self.config.layers-1),
                      out_channels=32*2**(self.config.layers-1),
                      kernel_size=1),
            # nn.Sigmoid()
        )

    def forward(self,x):
        x = self.discriminator(x)
        return self.final(x)