from project_team.io_project.IO_config import io_config
import os
import pandas as pd
from ._Statistical_Project import _Statistical_Project, LabelNotFoundError

class io_traindeploy_config(io_config):
    '''
    Configuration for a train test split experiment
    '''
    def __init__(self, **kwargs):
        super(io_traindeploy_config, self).__init__(**kwargs)

class _TrainDeploy(_Statistical_Project):
    '''
    Train Deployment Statistical Project
    Functionality:
    - straight train: test_size must be 0, and validation size must be zero
    - train test: have a portion of test data
    - train validation and test: have a portion of testing data and validation
        data
    - train validation: have a portion of validation but not test
    '''
    def __init__(self, io_config_input=None):
        if io_config_input is None:
            io_config_input = io_traindeploy_config()
        # validate before super() creates any directories on disk
        if not isinstance(io_config_input, io_traindeploy_config):
            raise TypeError(
                '_TrainDeploy requires an io_traindeploy_config, got ' +
                type(io_config_input).__name__ + '. Wrap your parameters '
                'in io_traindeploy_config(...).')
        super(_TrainDeploy, self).__init__(io_config_input)

    def prepare_for_experiment(self):
        '''
        Preliminary organization tasks before training begins.
        1. Load data sets, rename columns for X and y so they are consistent
            in downstream tasks
        2. save the data used in the experiment folder for records
        '''
        print('IO Message: Setting up data for training')
        self.config.save_pretrained(self.root)

        # load dataset, rename data and group examples
        data_file, session_list = self.load_rename_group_data()

        if self.config.val_data_csv_location and \
                os.path.exists(self.config.val_data_csv_location):
            # process validation dataset if a dataframe is given
            vl_data_df = pd.read_csv(self.config.val_data_csv_location, na_filter=False)

            vl_data_df = self.remap_X(vl_data_df)
            vl_data_df = self.remap_y(vl_data_df)

            vl_data_df.to_csv(os.path.join(self.root, 'vl_dset.csv'),
                              index=False)
            tr_data_df = data_file
            tr_data_df.to_csv(os.path.join(self.root, 'tr_dset.csv'),
                              index=False)
        else:
            # split and process the data given the proportions
            if self.config.stratify_by:
                # split the data with a stratification characteristic
                strat = self.stratify_data(data_file, session_list)
                train_list, val_list, test_list = self.stratified_data_split(
                    session_list, strat)
            else:
                # split the data with out a stratification characteristic
                train_list, val_list, test_list = self.data_split(
                    session_list)
            # group data together and save the datasets in the experiment
            # folder
            # training
            tr_data_df = data_file[
                data_file[self.config.group_data_by].isin(train_list)
            ]
            tr_data_df.to_csv(os.path.join(self.root, 'tr_dset.csv'),
                              index=False)

            # validation
            if val_list:
                vl_data_df = data_file[
                    data_file[self.config.group_data_by].isin(val_list)
                ]
                vl_data_df.to_csv(os.path.join(self.root, 'vl_dset.csv'),
                                  index=False)

            # inference
            if test_list:
                ts_data_df = data_file[
                    data_file[self.config.group_data_by].isin(test_list)
                ]
                ts_data_df.to_csv(os.path.join(self.root, 'if_dset.csv'),
                                  index=False)

    def prepare_for_inference(self, data_file=None):
        '''
        Preliminary organization tasks to perform before running inference on data
        :param data_file: optional. default: None. Can be a csv location or a pandas dataframe
        :return:
        '''
        ### Case for running inference:
        # 1. test_size>0.0 => this would be done when prepare_for_experiment
        # is ran
        if self.config.test_size>0.0 and os.path.exists(
                os.path.join(self.root,'if_dset.csv')):
            pass
        else:
            # 2. data_file given to the manager.
            if isinstance(data_file, str) and \
                    os.path.exists(data_file) and \
                    data_file.endswith('.csv') and \
                    self.config.inf_data_csv_location is None and \
                    self.config.test_size==0.0:
                data_set = pd.read_csv(data_file, na_filter=False)
            # 3. datafile is a dataframe
            elif isinstance(data_file, pd.DataFrame):
                data_set = data_file
            # 4. inf_data_csv_location is not None
            elif self.config.inf_data_csv_location is not None and \
                    isinstance(self.config.inf_data_csv_location, str) and \
                    os.path.exists(self.config.inf_data_csv_location) and \
                    self.config.inf_data_csv_location.endswith('.csv') and \
                    self.config.test_size==0.0:
                data_set = pd.read_csv(self.config.inf_data_csv_location, na_filter=False)

            else:
                raise Exception(
                    "The four criteria for setting inference data is not "
                    "being met. 'data_file' must be a file location, "
                    "'data_file' is a pandas Dataframe, config "
                    "'inf_data_csv_location' must not be None, or the config "
                    "'test_size' must be >0.0"
                )

            # required to rename X because we are running inference
            data_set = self.remap_X(data_set)
            # warn if y cannot be remapped because we may not know y for
            # the set; any other failure still raises
            try:
                data_set = self.remap_y(data_set)
            except LabelNotFoundError:
                print('IO Message: WARNING: Inference data does not '
                      'contain y labels. It cannot be used for performance '
                      'evaluation.')

            data_set.to_csv(os.path.join(self.root, 'if_dset.csv'), index=False)
        print('IO Message: Inference data is set up. ')