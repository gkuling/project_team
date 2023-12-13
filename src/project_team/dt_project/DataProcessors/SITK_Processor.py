from ._Processor import _Processor, DT_config
from ..dt_processing.img_shape import *
import SimpleITK as sitk
from torchvision import transforms
import pandas as pd
from ..datasets import SITK_Dataset
import inspect
from tqdm import tqdm

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
        super(SITK_Processor_config, self).__init__(pre_load, **kwargs)
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
            preload_transforms=transforms,
            filter_out_zero_X=self.config.filter_out_zero_X,
            debug_pretransform=self.config.debug_pretransform
        ))

class SITK_Processor_Segmentation(SITK_Processor):
    '''
    a data processor to handle SITK data for segmentation
    '''
    def __init__(self, sitk_processor_config=SITK_Processor_config()):
        '''
        constructor
        :param sitk_processor_config
        '''
        super(SITK_Processor_Segmentation, self).__init__(sitk_processor_config)
        # dupicalte the transforms for y as well for training
        self.duplicate_transforms_for('y')

    def duplicate_transforms_for(self, new_field_oi):
        '''
        this will duplicate all the pre_transforms changing the field_oi to
        the new name
        :param new_field_oi: new desired field_oi
        '''
        # goup the transforms together
        tr_groups = {}
        for item in self.pre_transforms.transforms:
            attribute = str(item.__class__)
            if attribute in tr_groups:
                tr_groups[attribute].append(item)
            else:
                tr_groups[attribute] = [item]

        # for each individual type of transform, give it a new field_oi and
        # put it into the sequence
        for tr in tr_groups:
            try:
                # easiest is to do a deepcopy
                new_tr = deepcopy(tr_groups[tr][0])
                new_tr.field_oi = new_field_oi
                tr_groups[tr].append(new_tr)
            except:
                # sometimes we can't deepcopy all attributes so we make a
                # brand new version of that object
                def get_input_args(obj):
                    signature = inspect.signature(obj)
                    return [param.name for param in
                            signature.parameters.values()]

                args = get_input_args(tr_groups[tr][0].__init__)

                tr_groups[tr].append(
                    tr_groups[tr][0].__class__(
                        **{k: v if k != 'field_oi' else new_field_oi
                           for k, v, in tr_groups[tr][0].__dict__.items()
                           if k in args}
                    )
                )
        self.pre_transforms = transforms.Compose(sum([value for key, value in
                                                      tr_groups.items()], []))

    def post_process_inference_results(self, list_of_results):
        '''
        run post processing on the list_of_results
        :param list_of_results: results from the practitioner
        :return:
        '''
        print('DT Message: Beginning Postprocessing of model prediction '
              'results. ')
        self.inference_results = []

        # get the transforms of X
        post_process = [tr for tr in self.pre_transforms.transforms
                        if tr.field_oi == 'X']
        # retrieve the reciprical of these transforms
        post_process = [tr.get_reciprical(field_oi='pred_y') for tr in
                        post_process if
                        tr.get_reciprical() is not None]

        post_process = transforms.Compose(post_process[::-1])

        for ex in tqdm(list_of_results):
            if post_process:
                ex = post_process(ex)
            self.inference_results.append(ex)

    def transfer_dataset(self, input_data_list, dset_to_save):
        '''
        transfer in a a dataset from another processor
        :param input_data_list: data to be brought in
        :param dset_to_save: the name of the data set. This should be
        tr_dset, vl_dset, or if_dset.
        '''
        print('DT Message: SITK_Processor moving data.')
        if dset_to_save=='if_dset':
            self.__setattr__(dset_to_save,
                             SITK_Dataset(
                                 preload_transforms=
                                 transforms.Compose(
                                     [tr for tr in self.pre_transforms.transforms
                                      if tr.field_oi!='y']
                                 ),
                                 debug_pretransform=self.config.debug_pretransform
                            ))
        else:
            self.__setattr__(dset_to_save,
                             SITK_Dataset(preload_transforms=self.pre_transforms,
                                 debug_pretransform=self.config.debug_pretransform))

        input_data_list = [{k if k != 'X_location'
                            else 'X':v for k, v in d.items() }
                           for d in input_data_list]
        input_data_list = [{k if k != 'y_location'
                            else 'y':v for k, v in d.items() }
                           for d in input_data_list]

        self.__getattribute__(dset_to_save).transfer_list(input_data_list)

        print('DT Message: SITK_Processor finished moving Inference Results.')

class nnUNetSITK_Processor(SITK_Processor_Segmentation):
    def __init__(self,
                 save_folder,
                 sitk_processor_config=SITK_Processor_config()
                 ):
        super(nnUNetSITK_Processor, self).__init__(sitk_processor_config)
        self.save_folder = save_folder
        transforms = [
            OpenSITK_file(field_oi='X'),
            nnUNet_SaveSITK_file(field_oi='X',
                          save_folder=os.path.join(
                              save_folder,
                              'nnunetv2_raw',
                              'Dataset001_Current',
                              'imagesTr'
                          )
                          ),
            OpenSITK_file(field_oi='y'),
            nnUNet_SaveSITK_file(field_oi='y',
                          save_folder=os.path.join(
                              save_folder,
                              'nnunetv2_raw',
                              'Dataset001_Current',
                              'labelsTr'
                          )
                          ),
        ]
        self.set_pretransforms(transforms)

    def set_dataset(self, csv_location, set_name, pre_process_y=False):
        '''
        set a dataset of a specific name, has to custom inorder to build the
        proper folders for nnUNetV2
        :param csv_location: csv location of the data or a dataframe
        :param set_name: the name of the set to be loaded
        :param pre_process_y: boolean on whther to preprocess y label or not
        '''
        assert type(set_name) == str
        if set_name == 'if_dset':
            pr_name = 'inference'
        elif set_name=='vl_dset':
            pr_name='validation'
        elif set_name=='tr_dset':
            pr_name='training'
        else:
            pr_name = set_name

        print('DT message: Setting up the {} dataset. '.format(pr_name))

        if type(csv_location) == pd.DataFrame:
            pass
        else:
            if not os.path.exists(
                    os.path.join(csv_location, '{}.csv'.format(set_name))
            ):
                raise FileExistsError(
                    csv_location + ' does not contain a {}.csv '.format(
                        set_name))
            csv_location = pd.read_csv(
                os.path.join(csv_location, '{}.csv'.format(set_name)),
                na_filter=False
            )
        if set_name=='if_dset':
            # if loading an inference set, the preprocessing may not need to
            # be done for the y label, if there is no y label for these
            # examples.
            # split the transforms if there is no need to pre_process_y
            if not pre_process_y:
                pre_transforms = [
                    OpenSITK_file(field_oi='X'),
                    nnUNet_SaveSITK_file(field_oi='X',
                                  save_folder=os.path.join(
                                      self.save_folder,
                                      'nnunetv2_raw',
                                      'Dataset001_Current',
                                      'imagesTs'
                                  )
                                  )
                ]
            else:
                pre_transforms = [
                    OpenSITK_file(field_oi='X'),
                    nnUNet_SaveSITK_file(field_oi='X',
                                  save_folder=os.path.join(
                                      self.save_folder,
                                      'nnunetv2_raw',
                                      'Dataset001_Current',
                                      'imagesTs'
                                  )
                                  ),
                    OpenSITK_file(field_oi='y'),
                    nnUNet_SaveSITK_file(field_oi='y',
                                  save_folder=os.path.join(
                                      self.save_folder,
                                      'nnunetv2_raw',
                                      'Dataset001_Current',
                                      'labelsTs'
                                  )
                                  ),
                ]

            pre_transforms = transforms.Compose(pre_transforms)
        else:
            pre_transforms = self.pre_transforms

        # build processor_dataset
        self.get_dataset(csv_location, set_name, pre_transforms)
        # remove fingerprinting for this one processor because nnunet already
        # does it
        delattr(getattr(self, set_name), 'dataset_fingerprint')
        # pre_load the data
        if self.config.pre_load:
            getattr(getattr(self, set_name), 'perform_preload').__call__()
