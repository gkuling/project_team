from src.io_project.IO_config import io_config
import os
import pandas as pd
import numpy as np
from ._Statistical_Project import _Statistical_Project

class io_traindeploy_config(io_config):
    def __init__(self, **kwargs):
        super(io_traindeploy_config, self).__init__(**kwargs)

class _TrainDeploy(_Statistical_Project):
    def __init__(self, io_config_input=io_traindeploy_config()):
        super(_TrainDeploy, self).__init__(io_config_input)
        assert type(io_config_input)==io_traindeploy_config

    def prepare_for_experiment(self):
        print('IO Message: Setting up data for training')
        self.config.save_pretrained(self.root)
        data_file = pd.read_csv(self.config.data_csv_location)

        data_file = self.remap_X(data_file)
        data_file = self.remap_y(data_file)

        if self.config.group_data_by in data_file.columns:
            pass
        else:
            self.config.group_data_by = 'index_column'
            data_file[self.config.group_data_by] = data_file.index
        session_list = list(set(
            data_file[self.config.group_data_by].values.tolist()
        ))


        if self.config.val_data_csv_location and os.path.exists(self.config.val_data_csv_location):
            vl_data_df = pd.read_csv(self.config.val_data_csv_location)

            vl_data_df = self.remap_X(vl_data_df)
            vl_data_df = self.remap_y(vl_data_df)

            vl_data_df.to_csv(self.root + '/val_set.csv', index=False)
            tr_data_df = data_file
            tr_data_df.to_csv(self.root + '/train_set.csv', index=False)
        else:

            if self.config.stratify_by:
                tmp_strtfy_by = self.config.stratify_by
                if tmp_strtfy_by==self.config.y:
                    tmp_strtfy_by = 'y'
                assert type(tmp_strtfy_by)==str
                assert tmp_strtfy_by in data_file.columns
                strat = data_file.iloc[
                    [getattr(data_file,self.config.group_data_by).eq(x).idxmax()
                     for x in session_list]
                ][tmp_strtfy_by].to_list()
                train_list, val_list, test_list = self.stratified_data_split(
                    session_list, strat)
            else:
                train_list, val_list, test_list = self.data_split(
                    session_list)

            tr_data_df = data_file[
                data_file[self.config.group_data_by].isin(train_list)
            ]
            tr_data_df.to_csv(self.root +'/train_set.csv', index=False)
            if val_list:
                vl_data_df = data_file[
                    data_file[self.config.group_data_by].isin(val_list)
                ]
                vl_data_df.to_csv(self.root + '/val_set.csv', index=False)
            if test_list:
                ts_data_df = data_file[
                    data_file[self.config.group_data_by].isin(test_list)
                ]
                ts_data_df.to_csv(self.root + '/inf_set.csv', index=False)

    def prepare_for_inference(self, data_file=None):
        ### Case for running inference:
        # 1. test_size>0.0 => this would be done when prepare_for_experiment
        # is ran
        if self.config.test_size>0.0 and os.path.exists(self.root +
                                                        '/inf_set.csv'):
            pass
        else:
            # 2. data_file given to the manager.
            if type(data_file)==str and \
                    os.path.exists(data_file) and \
                    data_file.endswith('.csv') and \
                    self.config.inf_data_csv_location is None and \
                    self.config.test_size==0.0:
                data_set = pd.read_csv(data_file)
            # 3. datafile is a dataframe
            elif type(data_file)==pd.DataFrame:
                data_set = data_file
            # 4. inf_data_csv_location is not None
            elif self.config.inf_data_csv_location is not None and \
                    type(self.config.inf_data_csv_location)==str and \
                    os.path.exists(self.config.inf_data_csv_location) and \
                    self.config.inf_data_csv_location.endswith('.csv') and \
                    self.config.test_size==0.0:
                data_set = pd.read_csv(self.config.inf_data_csv_location)

            else:
                raise Exception("The four criteria for setting inference data "
                                "is "
                                "not being met. 'data_file' must be a file "
                                "location, 'data_file' is a pandas Dataframe, "
                                "config 'inf_data_csv_location' must not be None, "
                                "or the config 'test_size' must be >0.0")



            data_set = self.remap_X(data_set)

            try:
                data_set = self.remap_y(data_set)
            except Exception as e:
                if str(e).startswith(' The y label is not in the '
                                     'data_csv_location'):
                    print('IO Message: WARNING: Inference data does not '
                          'contain y labels. It cannot be used for performance '
                          'evaluation.')

            data_set.to_csv(self.root + '/inf_set.csv', index=False)
        print('IO Message: Inference data is set up. ')

    def finished_inf_validation(self, results_df):
        summary = {nm:[] for nm in results_df.columns}
        if len(np.unique(results_df['seg_map'].to_list()).tolist())>1:
            raise NotImplementedError('Having an output of more than one '
                                      'segmentation map is not implemented. '
                                      'Changes to this class must be made. ')
        summary['Subject'].extend(['Average', 'Std.Dev.'])
        summary['seg_map'].extend(
            [np.unique(results_df['seg_map'].to_list()).tolist()
             for _ in range(2)]
        )

        for key in summary.keys():
            if key!='Subject' and key!='seg_map':
                try:
                    met_values = np.array(
                        results_df[key].apply(
                            lambda x: eval(x) if type(x)==str
                            else x).to_list()
                    )
                    summary[key].append(met_values.mean(axis=0))
                    summary[key].append(met_values.std(axis=0))
                except:
                    summary[key].extend(['',''])

        summary = pd.DataFrame(summary)

        output_results = pd.concat(
            [results_df, summary],
            ignore_index=True
        )
        output_results.to_csv(self.root + '/Full_TrainTest_TestResults.csv',
                              index=False)