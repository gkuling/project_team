from project_team.io_project.IO_config import io_config
import ast
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
    '''
    Configuration for a hyperparameter tuning experiment
    '''
    def __init__(self, data_csv_location=None, technique='GridSearch',
                 training_portion=1.0, criterion= 'ACC', iterations=1,
                 penultimate=None, iteration=0,
                 **kwargs):
        '''
        :param data_csv_location: data set csv file or dataframe containing
        the data
        :param technique: technique for performing HPTuning, currently only
        have GridSearch and RandomSearch
        :param training_portion: the portion of the input data to use for the
        search. Highly recommend a small portion to lower computation time
        :param criterion: The criteria that you use to determine success
        :param iterations: for the RandomSearch you decide how many
        parameters to search over
        :param iteration: runtime counter of grid points already dispatched.
        It is a named parameter (not just an attribute) so a reloaded config
        resumes the search where it stopped instead of restarting from 0
        :param kwargs:
        '''
        super(io_hptuning_config, self).__init__(
            data_csv_location=data_csv_location, **kwargs)
        technique_possibilities = [
            'GridSearch', 'RandomSearch'
        ]
        if technique not in technique_possibilities:
            raise ValueError(
                'technique must be one of ' + str(technique_possibilities) +
                ', got ' + repr(technique))
        self.iteration = iteration
        self.technique = technique
        self.training_portion = training_portion
        self.criterion = criterion
        self.iterations = iterations
        self.penultimate = penultimate

class _HyperParameterTuning(_Statistical_Project):
    '''
    HP Tuning Statistical Project
    Functionality:
    - search hyperparamters to find the best model
    '''
    def __init__(self, io_config_input):
        # validate before super() creates any directories on disk
        if not isinstance(io_config_input, io_hptuning_config):
            raise TypeError(
                '_HyperParameterTuning requires an io_hptuning_config, '
                'got ' + type(io_config_input).__name__ + '. Wrap your '
                'parameters in io_hptuning_config(...).')
        super(_HyperParameterTuning, self).__init__(io_config_input)
        self.original_root = self.root
        self.timer = None

    def prepare_data_for_experiment(self):
        '''
        Preliminary organization tasks before training begins.
        1. Load data sets, rename columns for X and y so they are consistent
            in downstream tasks
        2. save the data used in the experiment folder for records
        '''

        # load dataset, rename data and group examples
        data_file, session_list = self.load_rename_group_data()

        if self.config.stratify_by:
            # split the data with a stratification characteristic
            if self.config.training_portion<1.0:
                # if training with less than the full dataset
                strat = self.stratify_data(data_file, session_list)
                # take the portion desired
                _, session_list = train_test_split(
                    session_list,
                    stratify=strat,
                    test_size=self.config.training_portion,
                    random_state=self.config.r_seed
                )
            # split the data with a stratification characteristic
            strat = self.stratify_data(data_file, session_list)
            train_list, val_list, test_list = self.stratified_data_split(
                session_list, strat
            )
        else:
            if self.config.training_portion<1.0:
                # if training with less than the full dataset
                _, session_list = train_test_split(
                    session_list,
                    test_size=self.config.training_portion,
                    random_state=self.config.r_seed
                )
            # split the data with out a stratification characteristic
            train_list, val_list, test_list = self.data_split(
                session_list)
        # group data together and save the datasets in the experiment
        # folder
        # training
        tr_data_df = data_file[
            data_file[self.config.group_data_by].isin(train_list)
        ]
        tr_data_df.to_csv(os.path.join(self.root, 'tr_dset.csv'), index=False)

        # validation
        if val_list:
            vl_data_df = data_file[
                data_file[self.config.group_data_by].isin(val_list)
            ]
            vl_data_df.to_csv(os.path.join(self.root, 'vl_dset.csv'), index=False)

        # inference
        if test_list:
            inf_set_df = data_file[
                data_file[self.config.group_data_by].isin(test_list)
            ]
            inf_set_df.to_csv(os.path.join(self.root, 'if_dset.csv'), index=False)

    def prepare_for_experiment(self,
                               parameter_domains
                               ):
        '''
        Preliminary organization tasks before training begins.
        :param parameter_domains: distionary containing the parameters to be
        tuned as keys and their domains as values
        '''
        print('IO Message: Setting up data for hyper-parameter tuning')
        # save self before beginning
        self.config.save_pretrained(self.root)
        # organize the data to be used in the HP tuning
        self.prepare_data_for_experiment()

        # organize the hyperparameters that will be searched over
        allNames = sorted(parameter_domains)
        combinations = it.product(*(parameter_domains[Name] for Name in allNames))
        self.parameter_configurations =[dict(tuple(zip(allNames, f))) for f
                                        in list(combinations)]
        self.parameter_performances = []
        print('IO Message: Ready for experiment. There are ' + str(len(
            self.parameter_configurations)) + ' parameter configurations to '
                                              'be explored. ')

        # change parameter combinations based on the search technique
        if self.config.technique=='GridSearch':
            # use all combos
            pass
        elif self.config.technique=='RandomSearch':
            # use only a limited amount of randomly chosen parameters
            np.random.shuffle(
                self.parameter_configurations
            )
            self.parameter_configurations = self.parameter_configurations[
                                            :self.config.iterations]
        else:
            raise NotImplementedError(
                str(self.config.technique) + ' is not an implemented '
                'technique.')


    def get_gridpoint_args(self):
        '''
        get the next grid point to be evaluated
        :return: return args to be evaluated for the next grid point
        '''
        # get the new point
        gridpoint_args = self.parameter_configurations[self.config.iteration]

        # keep track of iterations, and re-save the config so an interrupted
        # search can resume from the saved counter instead of grid point 0
        self.config.iteration += 1
        self.config.save_pretrained(self.original_root)

        # change the root for the new point
        self.root = os.path.join(self.original_root,
                                 'grid_point' + str(self.config.iteration))
        if not os.path.exists(self.root):
            os.makedirs(self.root)

        # start timing the evaluation
        self.timer = dt_tool.now()

        return gridpoint_args

    def record_performance(self):
        '''
        record the performance of the current grid point results
        '''

        # load all previous gridpoint results, this will add the newest one
        # onto the end of the results csv
        hptuning_test_res = []
        for path, subdirs, files in os.walk(self.original_root):
            for name in files:
                if name.endswith('test_result_evaluation.csv'):
                    hptuning_test_res.append(os.path.join(path, name))
        self.parameter_performances = []
        for fl in hptuning_test_res:
            try:
                perf = self.evaluate_performance(
                    pd.read_csv(fl, na_filter=False)
                )
            except Exception as e:
                print('IO Message: WARNING: could not evaluate ' + fl +
                      ': ' + type(e).__name__ + ': ' + str(e))
                perf = 'FAIL'
            grid_pt_fldr = re.search(r'grid_point\d+', fl)
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
        res.to_csv(
            os.path.join(self.original_root, 'Experimental_Results.csv'),
            index=False
        )

        # print the results and the time it took to evaluate the point so you
        # can guess how much longer the code will run for
        delta_time = dt_tool.now() - self.timer
        self.timer = dt_tool.now()
        print('This past experimental parameter configuration took ' + str(
            delta_time) + ' long')

    def evaluate_performance(self, df):
        '''
        evaluation of the results given the penultimate function. For basic
        classification and regression this can just be a column of the
        evaluator. But for something like segmentation where you calculate a
        non binary performance metric, you can take the mean of harmonic mean of
        DSC of Jaccard coefficient on all the testing examples.
        :param df: dataframe of results from an evaluator
        :return: the performance amount
        '''
        try:
            eval_on = df[self.config.criterion].apply(
                ast.literal_eval).to_list()
        except (ValueError, SyntaxError, TypeError):
            # the criterion column holds plain scalars, not stringified lists
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