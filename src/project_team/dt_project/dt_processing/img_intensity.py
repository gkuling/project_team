from skimage import exposure
import numpy as np
from . import _TensorProcessing

class MnStdNormalize_Numpy(_TensorProcessing):
    '''
    intensity normalization based on mean and standard deviation
    '''
    def __init__(self, norm=[(0,1)], percentiles=(1,99), field_oi='X'):
        super(MnStdNormalize_Numpy, self).__init__()
        self.field_oi = field_oi
        self.norm = norm
        self.percentiles = percentiles

    def __call__(self, ipt):
        img = ipt[self.field_oi]
        if img.shape[0]!=len(self.norm):
            raise Exception('The amount of normalization factors does not '
                            'equal the number of input channels. Please '
                            'provide more than ' + str(len(self.norm)) +
                            'normalization parameter(s). ')
        output = np.zeros(img.shape)
        for i, nrm in enumerate(self.norm):
            if nrm is None:
                output[i] = img[i]
                continue
            # if the image values are all the same, do nothing to the channel
            if img[i].max()==img[i].min():
                clipped = img[i]
            else:
                if self.percentiles is not None:
                    upper = np.percentile(img[i][img[i]!=0], self.percentiles[1])

                    lower = np.percentile(img[i][img[i]!=0], self.percentiles[0])
                else:
                    upper=img[i].max()
                    lower = img[i].min()

                assert upper>lower
                clipped = np.clip(img[i], a_min=lower, a_max=upper)

                clipped = (clipped - lower)
                clipped = (clipped / clipped.max())

            norm_img = (clipped-nrm[0])/nrm[1]

            output[i] = norm_img

        ipt[self.field_oi] = output
        return ipt

class MxMnNormalize_Numpy(_TensorProcessing):
    '''
    value normalization based on min max
    '''
    def __init__(self, mxmn=[(0.,1.)], percentiles=[(1,99)], field_oi='X'):
        super(MxMnNormalize_Numpy, self).__init__()
        self.field_oi = field_oi
        self.norm = mxmn
        self.percentiles = percentiles

    def __call__(self, ipt):
        img = ipt[self.field_oi]
        if type(ipt[self.field_oi])==list:
            img = np.array(img)
        if img.shape[0]!=len(self.norm):
            raise Exception('The amount of normalization factors does not '
                            'equal the number of input channels. Please '
                            'provide more than ' + str(len(self.norm)) +
                            'normalization parameter(s). ')
        output = np.zeros(img.shape)
        for i, nrm in enumerate(self.norm):
            if img.max==1.0 and img.min()==-1.0:
                print('')
            try:
                upper = np.percentile(img[i][img[i]!=0], self.percentiles[i][1])
            except:
                upper = 0.0
            try:
                lower = np.percentile(img[i][img[i]!=0], self.percentiles[i][0])
            except:
                lower = 0.0
            clipped = np.clip(img[i], lower, upper)

            clipped -= lower
            clipped /= clipped.max()

            output[i] = (self.norm[i][1]-self.norm[i][0]) * clipped + \
                        self.norm[i][0]
        if type(ipt[self.field_oi])==list:
            ipt[self.field_oi] = [output[i] for i in range(output.shape[0])]
        else:
            ipt[self.field_oi] = output
        return ipt

class Clip_Numpy(_TensorProcessing):
    '''
    clip a numpy values between a given min and max
    '''
    def __init__(self, max=1.0, min=0.0, field_oi='X'):
        super(Clip_Numpy, self).__init__()
        self.max = max
        self.min = min
        self.field_oi = field_oi

    def __call__(self, ipt):
        img = ipt[self.field_oi]

        img = [np.clip(i, self.min, self.max) for i in img]

        ipt[self.field_oi] = img

        return ipt

class Histogram_Equalization_Numpy(_TensorProcessing):
    '''
    perform histogram normalization on the given numpy array
    '''
    def __init__(self, bins=1000, mask_mxmn=(0.1,0.99), field_oi='X'):
        super(Histogram_Equalization_Numpy, self).__init__()
        self.bins=bins
        self.mask_mxmn = mask_mxmn
        self.field_oi = field_oi

    def __call__(self, ipt):
        ipt[self.field_oi] = [exposure.equalize_adapthist(np_ar,
                                                          nbins=self.bins,
                                                          clip_limit=0.03
                                                          )
                              for np_ar in ipt[self.field_oi]]
        return ipt
