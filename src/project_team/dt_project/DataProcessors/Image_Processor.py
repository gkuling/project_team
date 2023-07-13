import pandas as pd
from torchvision import transforms

from ..datasets import Images_Dataset
from ._Processor import _Processor, DT_config
from ..dt_processing import *

class Image_Processor_config(DT_config):
    '''
    Configuration for the Image processor
    '''
    def __init__(self,
                 silo_dtype='numpy.float32',
                 numpy_shape=(28,28),
                 pad_shape=None,
                 pre_load=True,
                 one_hot_encode=False,
                 max_classes=1,
                 **kwargs
                 ):
        '''
        :param silo_dtype: data type used to store data in the files_silo
        :param numpy_shape: shape of the numpy arrays stored in the files_silo
        :param pad_shape: the amount of zero padding added to the nump arrays
        :param pre_load: booleean indicating whether to load all examples
        into memory before comencing or load data on the fly
        :param one_hot_encode: boolean on whether to one hot encode the y label
        :param max_classes: the amount of labeled classes for the task
        '''
        super(Image_Processor_config, self).__init__(pre_load)
        self.silo_dtype=silo_dtype
        self.numpy_shape=numpy_shape
        self.pad_shape = pad_shape
        self.one_hot_encode = one_hot_encode
        self.max_classes = max_classes

class Image_Processor(_Processor):
    '''
    a image processor parent class
    '''
    def __init__(self, image_processor_config=Image_Processor_config()):
        '''
        :param image_processor_config: an images processor config
        '''
        super(Image_Processor, self).__init__(image_processor_config)

        # the pre_transforms for a standard MNIST classification experiment.
        # This can be changed in child classes
        pre_transforms = [
            OpenImage_file(),
            Resample_Image_shape(new_size=self.config.numpy_shape,
                         output_dtype=self.config.silo_dtype),
            ImageToNumpy()
        ]
        if self.config.pad_shape is tuple and all([x is int for x in
                                                   self.config.pad_shape]):
            pre_transforms.append(
                Pad_to_Size_numpy(shape=self.config.pad_shape)
            )
        if self.config.one_hot_encode:
            pre_transforms.append(OneHotEncode(
                max_class=self.config.max_classes,
                field_oi='y'
            ))
        self.pre_transforms = transforms.Compose(pre_transforms)

    def get_dataset(self, data, name, transforms):
        '''
        function that sets the dataset atribute for the given name
        :param data: pandas dataframe to be loaded as a dataset
        :param name: the name of the dataset
        :param transforms: the pretransforms to be given to the dataset
        '''
        assert type(data)==pd.DataFrame
        # GCK: This is the point that I pctured having a different dataset
        # for slice based images (for radiology data) or pathc based dataset
        # (for pathology and 2D data)
        setattr(self, name, Images_Dataset(
            data,
            preload_data=self.config.pre_load,
            preload_transforms=transforms,
            filter_out_zero_X=self.config.filter_out_zero_X
        ))
