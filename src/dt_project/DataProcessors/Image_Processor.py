import pandas as pd
from torchvision import transforms

from src.dt_project.DT_config import DT_config
from src.dt_project.datasets import Images_Dataset
from src.dt_project.dt_processing import *

class Image_Processor_config(DT_config):

    def __init__(self,
                 silo_dtype='numpy.float32',
                 numpy_shape=(28,28),
                 pad_shape=None,
                 pre_load=True,
                 one_hot_encode=False,
                 max_classes=1,
                 **kwargs
                 ):
        super(Image_Processor_config, self).__init__(pre_load)
        self.silo_dtype=silo_dtype
        self.numpy_shape=numpy_shape
        self.pad_shape = pad_shape
        self.one_hot_encode = one_hot_encode
        self.max_classes = max_classes

class Image_Processor(object):
    def __init__(self, image_processor_config=Image_Processor_config()):
        self.config = image_processor_config

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

    def set_pretransforms(self, pre_transforms_list):
        self.pre_transforms = transforms.Compose(pre_transforms_list)

    def add_pretransforms(self, more_transforms_list):
        self.pre_transforms.transforms.extend(more_transforms_list)

    def set_training_data(self, train_data_csv_location):
        print('DT message: Setting up the training dataset. ')
        if type(train_data_csv_location)==pd.DataFrame:
            pass
        else:
            if not os.path.exists(train_data_csv_location + '/train_set.csv'):
                raise FileExistsError(train_data_csv_location + ' does not '
                                                                'contain a '
                                                                'train_set.csv ')
            train_data_csv_location = pd.read_csv(train_data_csv_location
                                                  + '/train_set.csv')
        # GCK: This is the point that I pctured having a different dataset
        # for slice based images (for radiology data) or pathc based dataset
        # (for pathology and 2D data)
        self.tr_dset = Images_Dataset(
            train_data_csv_location,
            preload_data=self.config.pre_load,
            preload_transforms=self.pre_transforms
        )

        if self.config.pre_load:
            self.tr_dset.perform_preload()

    def set_validation_data(self,validation_data_csv_location):
        print('DT message: Setting up the validation dataset. ')
        if type(validation_data_csv_location)==pd.DataFrame:
            pass
        else:
            if not os.path.exists(validation_data_csv_location + '/val_set.csv'):
                raise FileExistsError(validation_data_csv_location + ' does not '
                                                                     'contain a '
                                                                     'val_set.csv ')
            validation_data_csv_location = pd.read_csv(validation_data_csv_location
                                                       + '/val_set.csv')


        self.vl_dset = Images_Dataset(
            validation_data_csv_location,
            preload_data=self.config.pre_load,
            preload_transforms=self.pre_transforms)
        if self.config.pre_load:
            self.vl_dset.perform_preload()

    def set_inference_data(self, inference_data_csv_location,
                           pre_process_y=False):
        print('DT message: Setting up the inference dataset. ')
        if type(inference_data_csv_location)==pd.DataFrame:
            pass
        else:
            if not os.path.exists(inference_data_csv_location + '/inf_set.csv'):
                raise FileExistsError(inference_data_csv_location + ' does not '
                                                                     'contain a '
                                                                     'inf_set.csv ')
            inference_data_csv_location = pd.read_csv(inference_data_csv_location
                                                       + '/inf_set.csv')
        if not pre_process_y:
            pre_transforms = [tr for tr in self.pre_transforms.transforms
                              if tr.field_oi == 'X']
        else:
            pre_transforms = [tr for tr in self.pre_transforms.transforms]

        pre_transforms = transforms.Compose(pre_transforms)
        self.if_dset = Images_Dataset(inference_data_csv_location,
                                    preload_data=self.config.pre_load,
                                    preload_transforms=pre_transforms)
        if self.config.pre_load:
            self.if_dset.perform_preload()

    def set_dataset_filter(self,dataset_lambda_condition):
        if hasattr(self,'tr_dset'):
            self.tr_dset.set_filter(dataset_lambda_condition)

        if hasattr(self, 'vl_dset'):
            self.vl_dset.set_filter(dataset_lambda_condition)

        if hasattr(self, 'if_dset'):
            self.if_dset.set_filter(dataset_lambda_condition)

    def transfer_dataset(self, input_data_list,
                         dset_to_save, pred_y_rename=None):
        print('DT Message: SITK_Processor moving data.')
        if dset_to_save=='if_dset':
            self.__setattr__(dset_to_save,
                             Images_Dataset(
                                 preload_transforms=
                                 transforms.Compose(
                                     [tr for tr in self.pre_transforms.transforms
                                      if tr.field_oi!='y']
                                 )))
        else:
            self.__setattr__(dset_to_save,
                             Images_Dataset(preload_transforms=self.pre_transforms))

        input_data_list = [{k if k != 'X_location'
                            else 'X':v for k, v in d.items() }
                           for d in input_data_list]
        input_data_list = [{k if k != 'y_location'
                            else 'y':v for k, v in d.items() }
                           for d in input_data_list]

        self.__getattribute__(dset_to_save).transfer_list(input_data_list)

        print('DT Message: SITK_Processor finished moving Inference Results.')
