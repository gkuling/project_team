from project_team.io_project.IO_config import io_config
import os
import pandas as pd
from ._Statistical_Project import _Statistical_Project

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
    def __init__(self, io_config_input=io_traindeploy_config()):
        super(_TrainDeploy, self).__init__(io_config_input)
        assert type(io_config_input)==io_traindeploy_config

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

        ### COME BACK FOR THIS SECTION
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
                try:
                    strat = self.stratify_data(data_file, session_list)
                except Exception as e:
                    if type(e)==IndexError:
                        raise IndexError('Using row index to group_data_by '
                                         'requires that the '
                                         'index of the dataframe be 0 to n. '
                                         'Use df.reset_index() to avoid this '
                                         'IndexError. ')
                    else:
                        raise e
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
            if type(data_file)==str and \
                    os.path.exists(data_file) and \
                    data_file.endswith('.csv') and \
                    self.config.inf_data_csv_location is None and \
                    self.config.test_size==0.0:
                data_set = pd.read_csv(data_file, na_filter=False)
            # 3. datafile is a dataframe
            elif type(data_file)==pd.DataFrame:
                data_set = data_file
            # 4. inf_data_csv_location is not None
            elif self.config.inf_data_csv_location is not None and \
                    type(self.config.inf_data_csv_location)==str and \
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

            # required to rename X becausewe are running inference
            data_set = self.remap_X(data_set)
            # throw warning if y can not be remapped because we may not know
            # y for the set

            try:
                data_set = self.remap_y(data_set)
            except Exception as e:
                if str(e).startswith(' The y label is not in the '
                                     'data_csv_location'):
                    print('IO Message: WARNING: Inference data does not '
                          'contain y labels. It cannot be used for performance '
                          'evaluation.')

            data_set.to_csv(os.path.join(self.root, 'if_dset.csv'), index=False)
        print('IO Message: Inference data is set up. ')

    ### CONSIDER DELETING BELOW AS IT WAS USED FOR A SEGENTATION KFOLD
    ### EXPERIMENT.
    ### I DO NOT BELIEVE IT IS PRACTICAL FOR PROJECT TEAM AND THIS PROJECT
    # def finished_inf_validation(self, results_df):
    #     summary = {nm:[] for nm in results_df.columns}
    #     if len(np.unique(results_df['seg_map'].to_list()).tolist())>1:
    #         raise NotImplementedError('Having an output of more than one '
    #                                   'segmentation map is not implemented. '
    #                                   'Changes to this class must be made. ')
    #     summary['Subject'].extend(['Average', 'Std.Dev.'])
    #     summary['seg_map'].extend(
    #         [np.unique(results_df['seg_map'].to_list()).tolist()
    #          for _ in range(2)]
    #     )
    #
    #     for key in summary.keys():
    #         if key!='Subject' and key!='seg_map':
    #             try:
    #                 met_values = np.array(
    #                     results_df[key].apply(
    #                         lambda x: eval(x) if type(x)==str
    #                         else x).to_list()
    #                 )
    #                 summary[key].append(met_values.mean(axis=0))
    #                 summary[key].append(met_values.std(axis=0))
    #             except:
    #                 summary[key].extend(['',''])
    #
    #     summary = pd.DataFrame(summary)
    #
    #     output_results = pd.concat(
    #         [results_df, summary],
    #         ignore_index=True
    #     )
    #     output_results.to_csv(os.path.join(self.root, 'Full_TrainTest_TestResults.csv'),
    #                           index=False)