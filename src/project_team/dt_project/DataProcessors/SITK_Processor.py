from ._Processor import _Processor, DT_config
from ..dt_processing.img_shape import *
import SimpleITK as sitk
from torchvision import transforms
import pandas as pd
from ..datasets import SITK_Dataset

class SITK_Processor_config(DT_config):
    def __init__(self,
                 spacing=(1.0, 1.0, 1.0),
                 silo_dtype=sitk.sitkFloat32,
                 numpy_shape=(64,64,64),
                 pre_load=True,
                 x_centering=False,
                 y_centering=False,
                 z_centering=False,
                 resample_shape=False,
                 **kwargs
                 ):
        '''
        :param spacing: spacing of the data
        :param silo_dtype: the data type to save data as in the files silo
        :param numpy_shape: the shape of the input data
        :param pre_load: bool. Indicate if preloading will be done
        :param x_centering: None. bool. or int. This is an indicator to
            centre x dimension in the padding function. None: no padding.
            bool. if True then it will find the centre of values. int. it
            will shift the dimensional centre by the given value.
        :param y_centering: same as x_centering but in y dimension.
        :param z_centering: same as x_centering but in z dimension.
        :param resample_shape: bool. indicator to resample by shape
        '''
        super(SITK_Processor_config, self).__init__(pre_load)
        self.spacing=spacing
        self.silo_dtype=silo_dtype
        self.numpy_shape=numpy_shape
        self.x_centering=x_centering
        self.y_centering=y_centering
        self.z_centering=z_centering
        self.resample_shape=resample_shape

class SITK_Processor(_Processor):
    '''
    a SITK image processor parent class
    '''
    def __init__(self, sitk_processor_config=SITK_Processor_config()):
        '''
        :param sitk_processor_config: an SITK image processor config
        '''
        super(SITK_Processor, self).__init__(sitk_processor_config)
        self.config = sitk_processor_config

        pre_transforms = [
            OpenSITK_file()]
        if self.config.resample_shape:
            pre_transforms.append(
                Resample_SITK_shape(
                    shape=self.config.numpy_shape,
                    output_dtype=self.config.silo_dtype
                )
            )
        else:
            pre_transforms.append(
                Resample_SITK_spacing(
                    spacing=self.config.spacing,
                    output_dtype=self.config.silo_dtype
                )
            )

        pre_transforms.extend([
            SITKToNumpy(),
            Pad_to_Size_numpy(shape=self.config.numpy_shape)
        ])
        self.pre_transforms = transforms.Compose(pre_transforms)

    def get_dataset(self, data, name, transforms):
        '''
        function that sets the dataset atribute for the given name
        :param data: pandas dataframe to be loaded as a dataset
        :param name: the name of the dataset
        :param transforms: the pretransforms to be given to the dataset
        '''
        assert type(data) == pd.DataFrame
        # GCK: This is the point that I pictured having a different dataset
        # for slice based images (for radiology data) or patch based dataset
        # (for pathology and 2D data)
        setattr(self, name, SITK_Dataset(
            data,
            preload_data=self.config.pre_load,
            preload_transforms=transforms
        ))
