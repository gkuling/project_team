from torchvision import transforms
import pandas as pd
import os

from project_team.project_config import project_config

class DT_config(project_config):
    def __init__(self,
                 pre_load=True,
                 **kwargs
                 ):
        '''
        initiator of a dataprocessor config
        :param pre_load: boolean to indicate whether data should be preloaded
        into memory or loaded on the fly
        :param kwargs:
        '''
        super(DT_config, self).__init__('DT')
        self.pre_load = pre_load

class _Processor(object):
    '''
    a processor parent class
    '''

    def __init__(self, config=DT_config()):
        self.config = config

    def get_dataset(self, data, name, transforms):
        '''
        :param data: pandas dataframe to be loaded as a dataset
        :param name: the name of the dataset
        :param transforms: the pretransforms to be given to the dataset
        '''
        raise NotImplementedError()

    def set_pretransforms(self, pre_transforms_list):
        '''
        set the pre_transforms
        :param pre_transforms_list: list of transforms
        '''
        self.pre_transforms = transforms.Compose(pre_transforms_list)

    def add_pretransforms(self, more_transforms_list):
        '''
        add more transforms to the pre_transforms
        :param more_transforms_list: list of more transforms to add on to the end
        :return:
        '''
        self.pre_transforms.transforms.extend(more_transforms_list)

    def set_training_data(self, train_data_csv_location):
        '''
        set the training data of the processor and build a dataset
        :param train_data_csv_location: csv file or pandas dataframe of data used for training
        '''
        self.set_dataset(train_data_csv_location, 'tr_dset', False)

    def set_validation_data(self,validation_data_csv_location):
        '''
        set the validation data of the processor and build a dataset
        :param validation_data_csv_location: csv file or pandas dataframe
        of data used for validation
        '''
        self.set_dataset(validation_data_csv_location, 'vl_dset', False)

    def set_inference_data(self, inference_data_csv_location,
                           pre_process_y=False):
        '''
        set the test data of the processor and build a dataset
        :param inference_data_csv_location: csv file or pandas dataframe of data used for inference
        :param pre_process_y: boolean. default: False. choice to preprocess the y variable
            for possible validation in the transform space
        '''
        self.set_dataset(inference_data_csv_location, 'if_dset', pre_process_y)

    def set_dataset(self, csv_location, set_name, pre_process_y=False):
        '''
        set a dataset of a specific name
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
                os.path.join(csv_location, '{}.csv'.format(set_name))

            )
        if set_name=='if_dset':
            # if loading an inference set, the preprocessing may not need to
            # be done for the y label, if there is no y label for these
            # examples.
            # split the transforms if there is no need to pre_process_y
            if not pre_process_y:
                pre_transforms = [tr for tr in self.pre_transforms.transforms
                                  if tr.field_oi == 'X']
            else:
                pre_transforms = [tr for tr in self.pre_transforms.transforms]

            pre_transforms = transforms.Compose(pre_transforms)
        else:
            pre_transforms = self.pre_transforms

        # build processor_dataset
        self.get_dataset(csv_location, set_name, pre_transforms)
        # pre_load the data
        if self.config.pre_load:
            getattr(getattr(self, set_name), 'perform_preload').__call__()

    def set_dataset_filter(self,dataset_lambda_condition):
        '''
        The option to set a filter on the dataset
        :param dataset_lambda_condition: the filtering function of the dataset
        :return:
        '''
        if hasattr(self,'tr_dset'):
            self.tr_dset.set_filter(dataset_lambda_condition)

        if hasattr(self, 'vl_dset'):
            self.vl_dset.set_filter(dataset_lambda_condition)

        if hasattr(self, 'if_dset'):
            self.if_dset.set_filter(dataset_lambda_condition)
    # GCK: Abandoned function that could be revived
    # For a project I wanted to transfer data from one dataset to another
    # with different pretransforms for two different ML algorithms, so I would
    # have to transfer the  results to the original data space, and then
    # transfer to the second algorithms  data space
    #
    # def transfer_dataset(self, input_data_list,
    #                      dset_to_save, pred_y_rename=None):
    #     print('DT Message: SITK_Processor moving data.')
    #     if dset_to_save=='if_dset':
    #         self.__setattr__(dset_to_save,
    #                          Images_Dataset(
    #                              preload_transforms=
    #                              transforms.Compose(
    #                                  [tr for tr in self.pre_transforms.transforms
    #                                   if tr.field_oi!='y']
    #                              )))
    #     else:
    #         self.__setattr__(dset_to_save,
    #                          Images_Dataset(preload_transforms=self.pre_transforms))
    #
    #     input_data_list = [{k if k != 'X_location'
    #                         else 'X':v for k, v in d.items() }
    #                        for d in input_data_list]
    #     input_data_list = [{k if k != 'y_location'
    #                         else 'y':v for k, v in d.items() }
    #                        for d in input_data_list]
    #
    #     self.__getattribute__(dset_to_save).transfer_list(input_data_list)
    #
    #     print('DT Message: SITK_Processor finished moving Inference Results.')