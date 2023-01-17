import torch.nn.functional

from src.project_config import project_config
import numpy as np
from torch import nn
from src.ml_project.models.UNet_fcns import *

class UNet_config(project_config):
    def __init__(self,
                 model_name='UNet',
                 in_channels=1,
                 out_channels=1,
                 dimensionality='3D',
                 depth=4,
                 base_filters=64,
                 inplane_layers=2,
                 skip_connections=False,
                 batch_norm=True,
                 activation='relu',
                 strided_encoder=False,
                 strided_decoder=False,
                 dropout=0.0,
                 inplane_kernel=None,
                 downsample_kernel=(2,2,2),
                 upsample_kernel=(2,2,2),
                 conv_bias=False,
                 **kwargs
                 ):
        super(UNet_config, self).__init__('model_UNet')
        assert in_channels>0, "Input cannot have 0 channels"
        assert out_channels>0, "Output cannot have 0 channels."
        assert depth>1, "UNet depth cannot be zero"
        assert base_filters>0, "base_filters cannot be 0."
        if not strided_decoder or not strided_decoder:
            if skip_connections:
               print('ML Message: UNet config Warning: Using MaxPooling or '
                     'UpSampling will not use skip connections in this '
                     'implementation. ')
        assert not (strided_decoder==False and
                    strided_encoder==False and
                    skip_connections==True)
        assert dimensionality!='3D' "The 2D implementation was redacted. " \
                               "Needs to be fixed. "
        if inplane_layers==0:
            assert strided_encoder==True and strided_decoder==True
        self.model_name = model_name
        if self.model_name=='model_VNet':
            base_filters = 16
            depth = 4
            inplane_layers = [1,2,3,3,3,3,3,2,1]
            strided_decoder = True
            strided_encoder = True
            skip_connections = True
            activation = 'preLU'
            inplane_kernel = (5,5,5) if None else inplane_kernel
            upsample_kernel = (2,2,2)
            downsample_kernel = (2,2,2)
            dimensionality = '3D'

        # Global model parameters
        self.in_channels= in_channels
        self.out_channels = out_channels
        self.dimensionality = dimensionality
        self.base_filters = base_filters
        self.skip_connections = skip_connections
        self.depth = depth
        self.activation = activation
        self.strided_encoder = strided_encoder
        self.strided_decoder = strided_decoder
        self.inplane_kernel = inplane_kernel if inplane_kernel is not None \
            else (3,3,3)
        self.downsample_kernel = downsample_kernel
        self.upsample_kernel = upsample_kernel
        self.conv_bias = conv_bias

        # Level specific parameters
        if type(inplane_layers)==list and len(inplane_layers) == int(2*depth+1):
            if any([lyr==0 for lyr in inplane_layers]):
                assert strided_encoder==True and \
                       strided_decoder==True, "To use inplane_layers=0, " \
                                             "both encoder and decoder must " \
                                              "be strided. " \
                                             "'strided_encoder==True and " \
                                             "strided_decoder==True'"
            self.inplane_layers = inplane_layers
        elif type(inplane_layers)==int and inplane_layers>=0:
            if inplane_layers==0:
                assert strided_encoder==True and \
                       strided_decoder==True and \
                       skip_connections==False, "To use inplane_layers=0, " \
                                                "both encoder and decoder " \
                                                "must be strided and " \
                                                "skip_connections turned off. " \
                                                "'strided_encoder==True and " \
                                                "strided_decoder==True and " \
                                                "skip_connections==False'"
            self.inplane_layers = [inplane_layers for _ in range(int(2*depth
                                                                     +1))]
        else:
            raise Exception('Batch Norm option is not acceptable. ' + str(
                batch_norm))

        if type(batch_norm)==list and len(batch_norm) == int(2*depth+1):
            self.batch_norm_scheme = batch_norm
        elif type(batch_norm)==bool or batch_norm=='Batch' or \
                batch_norm=='Instance':
            self.batch_norm_scheme = [batch_norm
                                      for _ in range(int(2*depth +1))]
        else:
            raise Exception('Batch Norm option is not acceptable. ' + str(
                batch_norm))

        if type(dropout)==list and len(dropout) == int(2*depth+1):
            self.dropout_scheme = dropout
        elif type(dropout)==float and dropout>=0.0 and dropout<=1.0:
            self.dropout_scheme = [dropout for _ in range(2*depth-1)]
        else:
            raise Exception('Dropout option is not acceptable. ' + str(
                batch_norm))

class UNet(nn.Module):
    def __init__(self, config = UNet_config()):
        super(UNet, self).__init__()
        self.config = config
        self.encoder = UNetEncoder(config)
        self.decoder = UNetDecoder(config)

    def forward(self,x):
        input_shape = x.shape[2:]
        assert np.min(input_shape)/2**(self.config.depth-1)>=1, \
            "Input size must not disappear during downsampling. np.min(" \
            "input_shape)/2**(self.config.depth-1)<1"
        levels = self.encoder(x, return_levels=True)

        return self.decoder(levels)

class UNetEncoder(nn.Module):
    def __init__(self, config=UNet_config()):
        super(UNetEncoder, self).__init__()
        self.config = config
        factor = 2 if not self.config.strided_decoder else 1
        if config.dimensionality=='3D':
            input_func = UNet_base
            down_sample_func = UNet_DownFunction
        elif config.dimensionality=='2D':
            input_func = UNet_base2d
            down_sample_func = UNet_DownFunction2d
        else:
            raise ValueError('Dimensionality of ' + str(
                config.dimensionality) + ' is not an option. Choose 2D or 3D.')

        self.layers = []
        if self.config.inplane_layers[0]>0:
            self.inc = input_func(self.config.in_channels,
                                 self.config.base_filters,
                                  skip=config.skip_connections,
                                 normalize=self.config.batch_norm_scheme[0],
                                 activation=self.config.activation,
                                 layers=self.config.inplane_layers[0],
                                 dropout=self.config.dropout_scheme[0],
                                 kernel=self.config.inplane_kernel,
                                  bias=self.config.conv_bias)
        else:
            self.inc = None
            self.layers.append(
                down_sample_func(
                    in_ch=self.config.in_channels,
                    out_ch=self.config.base_filters,
                    actvtn=config.activation,
                    norm=config.batch_norm_scheme[0],
                    layers=config.inplane_layers[0],
                    strided=config.strided_encoder,
                    dropout=config.dropout_scheme[0],
                    inplane_kernel=config.inplane_kernel,
                    down_kernel=config.downsample_kernel,
                    skip=config.skip_connections,
                    bias=self.config.conv_bias
                )
            )

        for lr in range(config.depth):
            _in = self.config.base_filters*2**lr
            _out = self.config.base_filters*2**(lr+1) \
                if lr < self.config.depth-1 else self.config.base_filters \
                                                 *2 ** (lr+1) // factor
            self.layers.append(
                down_sample_func(
                    in_ch=_in,
                    out_ch=_out,
                    actvtn=config.activation,
                    norm=config.batch_norm_scheme[lr+1],
                    layers=config.inplane_layers[lr+1],
                    strided=config.strided_encoder,
                    dropout=config.dropout_scheme[lr+1],
                    inplane_kernel=config.inplane_kernel,
                    down_kernel=config.downsample_kernel,
                    skip=config.skip_connections,
                    bias=self.config.conv_bias
                )
            )
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, return_levels=False):
        if self.inc:
            x = self.inc(x)
        output_lvls = [x]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            output_lvls.append(x)
        if return_levels:
            return output_lvls
        else:
            return x

class UNetDecoder(nn.Module):
    def __init__(self, config=UNet_config()):
        super(UNetDecoder, self).__init__()
        self.config = config
        factor = 2 if not self.config.strided_decoder else 1
        if config.dimensionality=='3D':
            if self.config.inplane_layers[-1]>0:
                self.outc = nn.Conv3d(config.base_filters,
                                      config.out_channels,
                                      (1,1,1),
                                      padding='same',
                                      bias=True)
            else:
                self.outc = nn.Sequential(
                    nn.Upsample(scale_factor=2,
                                mode='trilinear',
                                align_corners=True),
                    nn.Conv3d(2 * config.base_filters,
                              config.out_channels,
                              kernel_size=4,
                              padding='same')
                )
            up_sample_func = UNet_UpFunction
        elif config.dimensionality=='2D':
            if self.config.inplane_layers[-1]>0:
                self.outc = nn.Conv2d(config.base_filters,
                                      config.out_channels,
                                      (1,1),
                                      padding='same',
                                      bias=True)
            else:
                self.outc = nn.Sequential(
                    nn.Upsample(scale_factor=2,
                                mode='bilinear',
                                align_corners=True),
                    nn.Conv2d(2 * config.base_filters,
                              config.out_channels,
                              kernel_size=4,
                              padding='same')
                )
            up_sample_func = UNet_UpFunction2d
        else:
            raise ValueError('Dimensionality of ' + str(
                config.dimensionality) + ' is not an option. Choose 2D or 3D.')
        self.layers = []
        for lr in range(self.config.depth, 0, -1):
            _in = self.config.base_filters*2**(lr)
            _out = self.config.base_filters*2**(lr-1) // factor \
                if lr-1!=0 else self.config.base_filters
            if lr!=self.config.depth and config.inplane_layers[-lr]==0:
                _in *=2
            self.layers.append(
                up_sample_func(
                    in_ch=_in,
                    out_ch=_out,
                    actvtn=config.activation,
                    norm=config.batch_norm_scheme[-lr],
                    layers=config.inplane_layers[-lr],
                    strided=config.strided_decoder,
                    dropout=config.dropout_scheme[-lr],
                    inplane_kernel=config.inplane_kernel,
                    up_kernel=config.upsample_kernel,
                    skip=config.skip_connections,
                    bias=self.config.conv_bias
                )
            )
        self.layers = nn.ModuleList(self.layers)


    def forward(self, lvls):
        if self.config.inplane_layers[-1]>0:
            for i in range(1, len(lvls)):
                if i==1:
                    u = self.layers[i-1](lvls[-i], lvls[-i-1])
                else:
                    u = self.layers[i-1](u, lvls[-i-1])
            return self.outc(u)
        else:
            for i in range(1, len(lvls)-1):
                if i==1:
                    u = self.layers[i-1](lvls[-i], lvls[-i-1])
                else:
                    u = self.layers[i-1](u, lvls[-i-1])
            return self.outc(u)
