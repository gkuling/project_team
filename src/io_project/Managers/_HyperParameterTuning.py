from src.io_project.IO_config import io_config
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import scipy
import itertools as it
import re
from datetime import datetime as dt_tool
from ._Statistical_Project import _Statistical_Project

class io_hptuning_config(io_config):
    def __init__(self, data_csv_location, technique, training_portion,
                 criterion= 'ACC',
                 penultimate=None,
                 iterations=None,
                 **kwargs):
        super(io_hptuning_config, self).__init__(data_csv_location, **kwargs)
        technique_possibilities = [
            'GridSearch', 'RandomSearch'
        ]
        assert technique in technique_possibilities, "Tuning Technique must be in " + str(technique_possibilities)
        self.iteration=0
        self.technique = technique
        self.training_portion = training_portion
        self.criterion = criterion
        self.penultimate = penultimate
        self.iterations = iterations

class _HyperParameterTuning(_Statistical_Project):
    def __init__(self, io_config_input):
        super(_HyperParameterTuning, self).__init__(io_config_input)
        assert type(io_config_input)==io_hptuning_config

        self.config = io_config_input
        self.root = io_config_input.project_folder + '/' + io_config_input.experiment_name
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.original_root = self.root
        self.timer = dt_tool.now()

    def prepare_data_for_experiment(self):
        data_file = pd.read_csv(self.config.data_csv_location)
        data_file = self.remap_X(data_file)
        data_file = self.remap_y(data_file)

        data_file, session_list = self.get_session_list(data_file)

        if self.config.stratify_by:
            if self.config.training_portion<1.0:
                try:
                    strat = self.stratify_data(data_file, session_list)
                except Exception as e:
                    if type(e) == IndexError:
                        raise IndexError('Using row index to group_data_by '
                                         'requires that the '
                                         'index of the dataframe be 0 to n. '
                                         'Use df.reset_index() to avoid this '
                                         'IndexError. ')
                    else:
                        raise e
                _, session_list = train_test_split(
                    session_list,
                    stratify=strat,
                    test_size=self.config.training_portion,
                    random_state=self.config.r_seed
                )
            try:
                strat = self.stratify_data(data_file, session_list)
            except Exception as e:
                if type(e) == IndexError:
                    raise IndexError('Using row index to group_data_by '
                                     'requires that the '
                                     'index of the dataframe be 0 to n. '
                                     'Use df.reset_index() to avoid this '
                                     'IndexError. ')
                else:
                    raise e
            train_list, val_list, test_list = self.stratified_data_split(
                session_list, strat
            )
        else:
            if self.config.training_portion<1.0:
                _, session_list = train_test_split(
                    session_list,
                    test_size=self.config.training_portion,
                    random_state=self.config.r_seed
                )
            train_list, val_list, test_list = self.data_split(
                session_list)

        tr_data_df = data_file[
            data_file[self.config.group_data_by].isin(train_list)
        ]
        tr_data_df.to_csv(self.root +'/tr_dset.csv', index=False)
        if val_list:
            vl_data_df = data_file[
                data_file[self.config.group_data_by].isin(val_list)
            ]
            vl_data_df.to_csv(self.root + '/vl_dset.csv', index=False)
        if test_list:
            inf_set_df = data_file[
                data_file[self.config.group_data_by].isin(test_list)
            ]
            inf_set_df.to_csv(self.root + '/if_dset.csv', index=False)

    def prepare_for_experiment(self,
                               parameter_domains
                               ):
        print('IO Message: Setting up data for hyper-parameter tuning')
        self.config.save_pretrained(self.root)
        self.prepare_data_for_experiment()

        allNames = sorted(parameter_domains)
        combinations = it.product(*(parameter_domains[Name] for Name in allNames))
        self.parameter_configurations =[dict(tuple(zip(allNames, f))) for f
                                        in list(combinations)]
        self.parameter_performances = []
        print('IO Message: Ready for experiment. There are ' + str(len(
            self.parameter_configurations)) + ' parameter configurations to '
                                              'be explored. ')
        if self.config.technique=='Gridsearch':
            pass
        elif self.config.technique=='RandomSearch':
            np.random.shuffle(
                self.parameter_configurations
            )
            self.parameter_configurations = self.parameter_configurations[
                                            :self.config.iterations]


    def get_gridpoint_args(self):
        gridpoint_args = self.parameter_configurations[self.config.iteration]
        self.config.iteration += 1

        self.root = self.original_root + '/grid_point' + str(
            self.config.iteration)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        return gridpoint_args

    def record_performance(self):

        hptuning_test_res = []
        for path, subdirs, files in os.walk(self.original_root):
            for name in files:
                if name.endswith('test_result_evaluation.csv'):
                    hptuning_test_res.append(os.path.join(path, name))
        self.parameter_performances = []
        for fl in hptuning_test_res:
            try:
                perf = self.evaluate_performance(
                    pd.read_csv(fl)
                )
            except:
                perf = 'FAIL'
            grid_pt_fldr = re.search('grid_point\d+', fl)
            ind = int(''.join([chr for chr in
                               fl[grid_pt_fldr.start():grid_pt_fldr.end()]
                               if chr.isdigit()]))
            parameters = self.parameter_configurations[ind-1]
            self.parameter_performances.append(
                [perf,
                 parameters,
                 fl]
            )
        res = pd.DataFrame(self.parameter_performances,
                           columns=['Performance(' + self.config.criterion +
                                    ')',
                                    'Parameters',
                                    'File'])
        res = pd.concat([res.drop(['Parameters'], axis=1),
                         res['Parameters'].apply(pd.Series)], axis=1)
        res.to_csv(self.original_root + '/Experimental_Results.csv', index=False)

        delta_time = dt_tool.now() - self.timer
        self.timer = dt_tool.now()
        print('This past experimental parameter configuration took ' + str(
            delta_time) + ' long')

    def evaluate_performance(self, df):
        try:
            eval_on = df[self.config.criterion].apply(eval).to_list()
        except:
            eval_on = df[self.config.criterion].to_list()
        if self.config.penultimate=='mean':
            return np.mean(np.mean(eval_on,
                                    axis=0))
        elif self.config.penultimate=='harmonic_mean':
            return scipy.stats.hmean(
                np.mean(eval_on, axis=0)
            )
        elif callable(self.config.penultimate):
            return self.config.penultimate(eval_on)
        elif self.config.penultimate is None:
            return df[self.config.criterion].item()
        else:
            raise NotImplementedError(
                str(self.config.penultimate) + ' is not implmented. '
            )