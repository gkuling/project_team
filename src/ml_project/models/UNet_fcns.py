import torch.nn as nn
import torch
import numpy as np

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride, padding, bias,
                 normalize, activation):
        super(UNetDown, self).__init__()
        layers = [nn.Conv3d(in_size, out_size, kernel, stride, padding,
                            bias=bias)]
        if normalize==True or normalize=='Batch':
            layers.append(nn.BatchNorm3d(out_size))
        elif normalize=='Instance':
            layers.append(nn.InstanceNorm3d(out_size))
        elif type(normalize)==dict and normalize['type']=='Batch':
            layers.append(nn.BatchNorm3d(out_size,**normalize['kwargs']))
        elif type(normalize)==dict and normalize['type']=='Instance':
            layers.append(nn.InstanceNorm3d(out_size,**normalize['kwargs']))

        if activation=='relu':
            layers.append(
                nn.ReLU(inplace=True)
            )
        elif activation=='leakyrelu':
            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif activation == 'preLU':
            layers.append(
                nn.PReLU(out_size)
            )
        else:
            raise ValueError(activation + ' is not recognized.')
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride, padding, bias,
                 normalize, activation):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose3d(in_size, out_size, kernel, stride, padding,
                               bias=bias)]
        if normalize==True or normalize=='Batch':
            layers.append(nn.BatchNorm3d(out_size))
        elif normalize=='Instance':
            layers.append(nn.InstanceNorm3d(out_size))
        elif type(normalize)==dict and normalize['type']=='Batch':
            layers.append(nn.BatchNorm3d(out_size,**normalize['kwargs']))
        elif type(normalize)==dict and normalize['type']=='Instance':
            layers.append(nn.InstanceNorm3d(out_size,**normalize['kwargs']))

        if activation=='relu':
            layers.append(
                nn.ReLU(inplace=True)
            )
        elif activation=='leakyrelu':
            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif activation == 'preLU':
            layers.append(
                nn.PReLU(out_size)
            )
        else:
            raise ValueError(activation + ' is not recognized.')
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return  x

class UNet_base(nn.Module):
    def __init__(self, in_size, out_size, mid_size=None,
                 skip=False, normalize=True, bias=False,
                 activation='relu', layers=1, dropout=0,
                 kernel=(3,3,3)):
        super(UNet_base, self).__init__()
        self.skip_connection = skip
        modules = []
        if not mid_size:
            mid_size = out_size

        for lyr in range(layers):
            if lyr == 0:
                _in, _out = in_size, mid_size
            elif lyr==layers-1:
                _in, _out = mid_size, out_size
            else:
                _in, _out = mid_size, mid_size
            modules.append(nn.Conv3d(_in,_out, kernel, (1,1,1),
                                     padding='same',
                                     bias=bias))

            if normalize==True or normalize=='Batch':
                modules.append(nn.BatchNorm3d(_out))
            elif normalize=='Instance':
                modules.append(nn.InstanceNorm3d(_out))
            elif type(normalize)==dict and normalize['type']=='Batch':
                modules.append(nn.BatchNorm3d(_out,**normalize['kwargs']))
            elif type(normalize)==dict and normalize['type']=='Instance':
                modules.append(nn.InstanceNorm3d(_out,**normalize['kwargs']))
            if lyr!=layers-1:
                if activation=='relu':
                    modules.append(
                        nn.ReLU(inplace=True)
                    )
                elif activation=='leakyrelu':
                    modules.append(
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                elif activation == 'preLU':
                    modules.append(
                        nn.PReLU(_out)
                    )
                else:
                    raise ValueError(activation + ' is not recognized.')

        if len(modules)!=0:
            self.ConvBlocks = nn.Sequential(*modules)
        else:
            self.ConvBlocks = lambda x: x


        if dropout>0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout= None


        if activation=='relu':
            self.final = nn.ReLU(inplace=True)
        elif activation=='leakyrelu':
            self.final = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'preLU':
            self.final = nn.PReLU(out_size)
        else:
            raise ValueError(activation + ' is not recognized.')

    def forward(self, x, add_in=None):
        out = self.ConvBlocks(x)
        if self.skip_connection:
            if x.shape[1]!=out.shape[1] and (out.shape[1]/x.shape[1]).is_integer():
                x = torch.cat([x]*int(out.shape[1]/x.shape[1]),axis=1)
            if add_in is not None:
                out = self.final(out + add_in)
            else:
                out = self.final(out + x)

        if self.dropout is not None:
            out = self.dropout(
                self.final(out)
            )
        return out

class UNet_DownFunction(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 actvtn,
                 norm,
                 layers,
                 strided,
                 dropout,
                 down_kernel,
                 inplane_kernel,
                 skip,
                 bias):
        super(UNet_DownFunction, self).__init__()
        down_padding = tuple([np.ceil(0.5*k-1).astype(int) for k in
                              down_kernel])
        if strided:
            self.down = UNetDown(in_ch,
                                 out_ch,
                                 down_kernel,
                                 (2,2,2),
                                 down_padding,
                                 bias=bias,
                                 normalize=norm,
                                 activation=actvtn)
            self.base_d = UNet_base(out_ch,
                                    out_ch,
                                    normalize=norm,
                                    activation=actvtn,
                                    layers=layers,
                                    dropout=dropout,
                                    kernel=inplane_kernel,
                                    skip=skip,
                                    bias=bias)
        else:
            self.down = nn.MaxPool3d(kernel_size=down_kernel,
                                     stride=2,
                                     padding=down_padding)

            self.base_d = UNet_base(in_ch,
                                    out_ch,
                                    normalize=norm,
                                    activation=actvtn,
                                    layers=layers,
                                    dropout=dropout,
                                    kernel=inplane_kernel,
                                    bias=bias)
    def forward(self, x):
        return self.base_d(self.down(x))

class UNet_UpFunction(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 actvtn,
                 norm,
                 layers,
                 dropout,
                 strided,
                 up_kernel,
                 inplane_kernel,
                 skip,
                 bias):
        super(UNet_UpFunction, self).__init__()
        self.layers = layers
        if strided:
            up_padding = tuple([np.ceil(0.5*k-1).astype(int) for k in
                                up_kernel])
            self.up = UNetUp(in_ch,
                             out_ch,
                             up_kernel,
                             (2,2,2),
                             up_padding,
                             bias=bias,
                             normalize=norm,
                             activation=actvtn)
            self.base_d = UNet_base(in_ch,
                                    out_ch,
                                    normalize=norm,
                                    activation=actvtn,
                                    layers=layers,
                                    dropout=dropout,
                                    kernel=inplane_kernel,
                                    skip=skip,
                                    bias=bias)
        else:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='trilinear',
                                  align_corners=True)
            self.base_d = UNet_base(in_ch,
                                    out_ch,
                                    mid_size=in_ch/2,
                                    normalize=norm,
                                    activation=actvtn,
                                    layers=layers,
                                    dropout=dropout,
                                    kernel=inplane_kernel,
                                    bias=bias)

    def forward(self, x, skip_input):
        x = self.up(x)
        if self.layers>0:
            if self.base_d.skip_connection:
                cated = torch.cat((x, skip_input), 1)
                return self.base_d(cated, x)
            else:
                x = torch.cat((x, skip_input), 1)
                return self.base_d(x)
        else:
            return torch.cat((self.base_d(x), skip_input), 1)

class UNetDown2d(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride, padding, normalize,
                 activation):
        super(UNetDown2d, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel, stride, padding,
                            bias=False)]
        if normalize==True or normalize=='Batch':
            layers.append(nn.BatchNorm2d(out_size))
        elif normalize=='Instance':
            layers.append(nn.InstanceNorm2d(out_size))
        elif type(normalize)==dict and normalize['type']=='Batch':
            layers.append(nn.BatchNorm2d(out_size,**normalize['kwargs']))
        elif type(normalize)==dict and normalize['type']=='Instance':
            layers.append(nn.InstanceNorm2d(out_size,**normalize['kwargs']))

        if activation=='relu':
            layers.append(
                nn.ReLU(inplace=True)
            )
        elif activation=='leakyrelu':
            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif activation == 'preLU':
            layers.append(
                nn.PReLU(out_size)
            )
        else:
            raise ValueError(activation + ' is not recognized.')
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp2d(nn.Module):
    def __init__(self, in_size, out_size, kernel, stride, padding,
                 normalize, activation):
        super(UNetUp2d, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, kernel, stride, padding,
                               bias=False)]
        if normalize==True or normalize=='Batch':
            layers.append(nn.BatchNorm2d(out_size))
        elif normalize=='Instance':
            layers.append(nn.InstanceNorm2d(out_size))
        elif type(normalize)==dict and normalize['type']=='Batch':
            layers.append(nn.BatchNorm2d(out_size,**normalize['kwargs']))
        elif type(normalize)==dict and normalize['type']=='Instance':
            layers.append(nn.InstanceNorm2d(out_size,**normalize['kwargs']))

        if activation=='relu':
            layers.append(
                nn.ReLU(inplace=True)
            )
        elif activation=='leakyrelu':
            layers.append(
                nn.LeakyReLU(0.2, inplace=True)
            )
        elif activation == 'preLU':
            layers.append(
                nn.PReLU(out_size)
            )
        else:
            raise ValueError(activation + ' is not recognized.')
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        return  x

class UNet_base2d(nn.Module):
    def __init__(self, in_size, out_size, mid_size=None,
                 skip=False, normalize=True,
                 activation='relu', layers=1, dropout=0,
                 kernel=(3,3,3)):
        super(UNet_base2d, self).__init__()
        self.skip_connection = skip
        modules = []
        if not mid_size:
            mid_size = out_size

        for lyr in range(layers):
            if lyr == 0:
                modules.append(nn.Conv2d(in_size,mid_size, kernel, (1,1),
                                         padding='same',
                                         bias=normalize==False))
            elif lyr==layers-1:
                modules.append(nn.Conv2d(mid_size,out_size, kernel, (1,1),
                                         padding='same',
                                         bias=normalize==False))
            else:
                modules.append(nn.Conv2d(mid_size,mid_size, kernel, (1,1),
                                         padding='same',
                                         bias=normalize==False))

            if normalize or normalize=='Batch':
                modules.append(nn.BatchNorm2d(out_size))
            elif normalize=='Instance':
                modules.append(nn.InstanceNorm2d(out_size))
            if lyr!=layers-1:
                if activation=='relu':
                    modules.append(
                        nn.ReLU(inplace=True)
                    )
                elif activation=='leakyrelu':
                    modules.append(
                        nn.LeakyReLU(0.2, inplace=True)
                    )
                elif activation == 'preLU':
                    modules.append(
                        nn.PReLU(out_size)
                    )
                else:
                    raise ValueError(activation + ' is not recognized.')

        if len(modules)!=0:
            self.ConvBlocks = nn.Sequential(*modules)
        else:
            self.ConvBlocks = lambda x: x

    def forward(self, x):
        if self.skip_connection:
            return self.ConvBlocks(x) + x
        else:
            return self.ConvBlocks(x)

class UNet_DownFunction2d(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 actvtn,
                 norm,
                 layers,
                 strided,
                 dropout,
                 down_kernel,
                 inplane_kernel,
                 skip):
        super(UNet_DownFunction2d, self).__init__()
        down_padding = tuple([np.ceil(0.5*k-1).astype(int) for k in
                              down_kernel])
        if strided:
            self.down = UNetDown2d(in_ch,
                                   out_ch,
                                   down_kernel,
                                   (2,2),
                                   down_padding,
                                   normalize=norm,
                                   activation=actvtn)
            self.base_d = UNet_base2d(out_ch,
                                      out_ch,
                                      normalize=norm,
                                      activation=actvtn,
                                      layers=layers,
                                      dropout=dropout,
                                      kernel=inplane_kernel,
                                      skip=skip)
        else:
            self.down = nn.MaxPool2d(kernel_size=down_kernel,
                                     stride=2,
                                     padding=down_padding)

            self.base_d = UNet_base2d(in_ch,
                                      out_ch,
                                      normalize=norm,
                                      activation=actvtn,
                                      layers=layers,
                                      dropout=dropout,
                                      kernel=inplane_kernel)
    def forward(self, x):
        return self.base_d(self.down(x))

class UNet_UpFunction2d(nn.Module):
    def __init__(self,
                 in_ch,
                 out_ch,
                 actvtn,
                 norm,
                 layers,
                 dropout,
                 strided,
                 up_kernel,
                 inplane_kernel,
                 skip):
        super(UNet_UpFunction2d, self).__init__()
        self.layers = layers
        if strided:
            up_padding = tuple([np.ceil(0.5*k-1).astype(int) for k in
                                up_kernel])
            self.up = UNetUp2d(in_ch,
                               out_ch,
                               up_kernel,
                               (2,2),
                               up_padding,
                               normalize=norm,
                               activation=actvtn)
            self.base_d = UNet_base2d(in_ch,
                                      out_ch,
                                      normalize=norm,
                                      activation=actvtn,
                                      layers=layers,
                                      dropout=dropout,
                                      kernel=inplane_kernel,
                                      skip=skip)
        else:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)

            self.base_d = UNet_base2d(in_ch,
                                      out_ch,
                                      mid_size=in_ch/2,
                                      normalize=norm,
                                      activation=actvtn,
                                      layers=layers,
                                      dropout=dropout,
                                      kernel=inplane_kernel)

    def forward(self, x, skip_input):
        x = self.up(x)
        if self.layers>0:
            x = torch.cat((x, skip_input), 1)
            return self.base_d(x)
        else:
            return torch.cat((self.base_d(x), skip_input), 1)
