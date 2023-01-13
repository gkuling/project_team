import pandas as pd
from torchvision import transforms
import SimpleITK as sitk

from src.dt_project.datasets import SITK_Dataset, SITK_Dataset_Patchwise, \
    SITK_Dataset_Slicewise
from src.dt_project.dt_processing import *

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
                 mask_by=None,
                 **kwargs
                 ):
        super(SITK_Processor_config, self).__init__(pre_load)
        self.spacing=spacing
        self.silo_dtype=silo_dtype
        self.numpy_shape=numpy_shape
        self.x_centering=x_centering
        self.y_centering=y_centering
        self.z_centering=z_centering
        self.resample_shape=resample_shape
        self.mask_by = mask_by

class SITK_Processor(object):
    def __init__(self, sitk_processor_config=SITK_Processor_config()):
        self.config = sitk_processor_config

        pre_transforms = [
            OpenSITK_file(),
            Resample_nii_spacing(spacing=self.config.spacing,
                         output_dtype=self.config.silo_dtype),
            SITKToNumpy(),
            Pad_to_Size_numpy(shape=self.config.numpy_shape)
        ]
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


        if self.config.patch_based:
            self.tr_dset = SITK_Dataset_Patchwise(
                train_data_csv_location,
                preload_data=self.config.pre_load,
                preload_transforms=self.pre_transforms,
                patch_size=self.config.patch_size,
                overlapped_patches=self.config.overlap
            )
        elif self.config.slice_based:
            self.tr_dset = SITK_Dataset_Slicewise(
                train_data_csv_location,
                preload_data=self.config.pre_load,
                preload_transforms=self.pre_transforms,
                axis_oi=self.config.slice_axis
            )
        else:
            self.tr_dset = SITK_Dataset(
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
        if self.config.patch_based:
            self.vl_dset = SITK_Dataset_Patchwise(
                validation_data_csv_location,
                preload_data=self.config.pre_load,
                preload_transforms=self.pre_transforms,
                patch_size=self.config.patch_size,
                overlapped_patches=self.config.overlap
            )
        elif self.config.slice_based:
            self.vl_dset = SITK_Dataset_Slicewise(
                validation_data_csv_location,
                preload_data=self.config.pre_load,
                preload_transforms=self.pre_transforms,
                axis_oi=self.config.slice_axis
            )
        else:
            self.vl_dset = SITK_Dataset(
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
        self.if_dset = SITK_Dataset(inference_data_csv_location,
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
                             SITK_Dataset(
                                 preload_transforms=
                                 transforms.Compose(
                                     [tr for tr in self.pre_transforms.transforms
                                      if tr.field_oi!='y']
                                 )))
        else:
            self.__setattr__(dset_to_save,
                             SITK_Dataset(preload_transforms=self.pre_transforms))

        input_data_list = [{k if k != 'X_location'
                            else 'X':v for k, v in d.items() }
                           for d in input_data_list]
        input_data_list = [{k if k != 'y_location'
                            else 'y':v for k, v in d.items() }
                           for d in input_data_list]

        self.__getattribute__(dset_to_save).transfer_list(input_data_list)

        print('DT Message: SITK_Processor finished moving Inference Results.')

class SITK_Processor_Segmentation(SITK_Processor):
    def __init__(self, sitk_processor_config=SITK_Processor_config()):
        super(SITK_Processor_Segmentation, self).__init__(sitk_processor_config)
        if self.config.resample_shape:
            self.set_pretransforms(
                [OpenSITK_file(),
                 OpenSITK_file(field_oi='y'),
                 Resample_nii_shape(shape=self.config.numpy_shape,
                              output_dtype=self.config.silo_dtype),
                 Resample_nii_shape(shape=self.config.numpy_shape,
                              field_oi='y',
                              output_dtype=self.config.silo_dtype),
                 SITKToNumpy(),
                 SITKToNumpy(field_oi='y')]
            )
        else:
            self.set_pretransforms(
                [OpenSITK_file(),
                 OpenSITK_file(field_oi='y'),
                 Resample_nii_spacing(spacing=self.config.spacing,
                              output_dtype=self.config.silo_dtype),
                 Resample_nii_spacing(spacing=self.config.spacing,
                              field_oi='y',
                              output_dtype=self.config.silo_dtype),
                 SITKToNumpy(),
                 SITKToNumpy(field_oi='y'),
                 Pad_to_Size_numpy(shape=self.config.numpy_shape,
                                   img_centering=(self.config.x_centering,
                                                  self.config.y_centering,
                                                  self.config.z_centering)),
                 Pad_to_Size_numpy(shape=self.config.numpy_shape,
                                   field_oi='y')]
            )


    def post_process_inference_results(self, list_of_results):
        print('DT Message: Beginning Postprocessing of model prediction '
              'results. ')
        self.inference_results = []
        post_process = [tr for tr in self.pre_transforms.transforms
                        if tr.field_oi == 'X']

        post_process = [tr.get_reciprical(field_oi='pred_y') for tr in
                        post_process if
                        tr.get_reciprical() is not None]

        post_process = transforms.Compose(post_process[::-1])

        for ex in tqdm(list_of_results):
            if post_process:
                ex = post_process(ex)
            self.inference_results.append(ex)