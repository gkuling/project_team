import SimpleITK as sitk
from skimage import exposure
from .augmentation_utils import *
from . import _TensorProcessing

class Normalize_nii(_TensorProcessing):
    def __init__(self, min=0., max=1., field_oi='X'):
        super(Normalize_nii, self).__init__()
        self.field_oi = field_oi
        self.filter = sitk.RescaleIntensityImageFilter()
        self.filter.SetOutputMinimum(min)
        self.filter.SetOutputMaximum(max)

    def __call__(self, ipt):

        ipt[self.field_oi] = [self.filter.Execute(img) for img in ipt[self.field_oi]]
        return ipt

class MnStdNormalize_Numpy(_TensorProcessing):
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
                upper = np.percentile(img[i][img[i]!=0], self.percentiles[1])

                lower = np.percentile(img[i][img[i]!=0], self.percentiles[0])
                assert upper>lower
                clipped = np.clip(img[i], a_min=lower, a_max=upper)

                clipped -= lower
                if clipped.max()==0:
                    print('')
                clipped /= clipped.max()

            norm_img = (clipped-nrm[0])/nrm[1]

            output[i] = norm_img

        ipt[self.field_oi] = output
        return ipt

class MxMnNormalize_Numpy(_TensorProcessing):
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
    def __init__(self, bins=1000, mask_mxmn=(0.1,0.99), field_oi='X'):
        super(Histogram_Equalization_Numpy, self).__init__()
        self.bins=bins
        self.mask_mxmn = mask_mxmn
        self.field_oi = field_oi

    def __call__(self, ipt):
        ipt[self.field_oi] = [exposure.equalize_adapthist(np_ar,
                                                          nbins=self.bins,
                                                          clip_limit=0.03
                                                          # mask=np.logical_and(
                                                          #     np_ar>self.mask_mxmn[0],
                                                          #     np_ar<self.mask_mxmn[1])
                                                          )
                              for np_ar in ipt[self.field_oi]]
        return ipt

class SITK_N4BiasFieldCorrection(_TensorProcessing):
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