import pandas as pd
from torchvision import transforms
import SimpleITK as sitk

from src.dt_project.DT_config import DT_config
from src.dt_project.datasets import SITK_Dataset, SITK_Dataset_Patchwise, \
    SITK_Dataset_Slicewise
from src.dt_project.dt_processing import *

class Image_Processor_config(DT_config):

    def __init__(self,
                 silo_dtype='numpy.float32',
                 numpy_shape=(28,28),
                 pre_load=True,
                 **kwargs
                 ):
        super(Image_Processor_config, self).__init__(pre_load)
        self.silo_dtype=silo_dtype
        self.numpy_shape=numpy_shape
