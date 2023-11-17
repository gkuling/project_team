import torchvision.transforms
from skimage import exposure
import numpy as np
from . import _TensorProcessing
import SimpleITK as sitk

class MnStdNormalize_Numpy(_TensorProcessing):
    '''
    intensity normalization based on mean and standard deviation
    '''
    def __init__(self, norm=[(0,1)], field_oi='X'):
        super(MnStdNormalize_Numpy, self).__init__()
        self.field_oi = field_oi
        self.norm = norm

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

            norm_img = (img[i]-nrm[0])/nrm[1]

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
    def __init__(self, max_min=[(0., 1.)], field_oi='X'):
        super(Clip_Numpy, self).__init__()
        self.max_min = max_min
        self.field_oi = field_oi

    def __call__(self, ipt):
        img = ipt[self.field_oi]

        img = [np.clip(a=i, a_min=mn[0], a_max=mn[1])
               for i, mn in zip(img, self.max_min)]

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

class SITK_N4BiasFieldCorrection(_TensorProcessing):
    '''
    This will run the SITK N4 bias field correction
    '''
    def __init__(self,
                 field_oi ='X',
                 shrink_factor=2,
                 mask_image=None,
                 num_iterations=None,
                 num_OffFittingLevels=None):
        super(SITK_N4BiasFieldCorrection, self).__init__()
        self.field_oi = field_oi
        self.shrink_factor = shrink_factor
        self.mask_image = mask_image
        self.num_iterations = num_iterations
        self.num_OffFittingLevels = num_OffFittingLevels

    def __call__(self, ipt):
        temp_res = []
        for inputImage in ipt[self.field_oi]:
            if self.mask_image:
                if type(ipt[self.mask_image]) and all([type(im)==sitk.Image for im
                                                       in ipt[
                                                           self.mask_image]]):
                    maskImage = ipt[self.mask_image][0]
                else:
                    maskImage = sitk.ReadImage(ipt[self.mask_image],
                                           sitk.sitkUInt8)
            else:
                maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 200)

            if maskImage.GetPixelID()!=sitk.sitkUInt8:
                maskImage = sitk.Cast(maskImage, sitk.sitkUInt8)

            if self.shrink_factor and all([sz%self.shrink_factor for sz in
                                           inputImage.GetSize()]):
                shrinkFactor = int(self.shrink_factor)
                if shrinkFactor>1:
                    image = sitk.Shrink(inputImage,
                                        [shrinkFactor] *
                                        inputImage.GetDimension())
                    maskImage = sitk.Shrink(maskImage,
                                            [shrinkFactor] *
                                            inputImage.GetDimension())
                else:
                    image = inputImage
            else:
                image = inputImage
            corrector = sitk.N4BiasFieldCorrectionImageFilter()

            numberFittingLevels = 4

            if self.num_OffFittingLevels:
                numberFittingLevels = int(self.num_OffFittingLevels)

            if self.num_iterations:
                corrector.SetMaximumNumberOfIterations(
                    [int(self.num_iterations)] * numberFittingLevels
                )

            corrected_image = corrector.Execute(image, maskImage)


            log_bias_field = corrector.GetLogBiasFieldAsImage(inputImage)

            corrected_image_full_resolution = inputImage / \
                                              sitk.Exp( log_bias_field )
            temp_res.append(corrected_image_full_resolution)
        ipt[self.field_oi] = temp_res
        return ipt