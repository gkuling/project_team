import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from project_team.project_config import project_config

class OrdinalCor_Practitioner_config(project_config):
    def __init__(self,
                 exogenous,
                 endogenous,
                 ordinal_label,
                 kendall_tau=True,
                 spearman_r=True,
                 graph_name=None,
                 histogram=True,
                 boxplot=True,
                 **kwargs):
        '''
        configuration for the ordinal correlation practitioner
        :param exogenous: the exogenous variable (ordinal variable)
        :param endogenous: the endogenous variable (ordinal or continuous
            variable)
        :param ordinal_label: the ordinal label for the exogenous variable
        :param kendall_tau: bool. indicator to calculate kendall tau
        :param spearman_r: bool. indicator to calculate spearman r
        :param graph_name: str. the name to save graphs with
        :param histogram: bool. indicator to generate a histogram
        :param boxplot: bool. indicator to generate a boxplot
        :param kwargs:
        '''
        super(OrdinalCor_Practitioner_config, self).__init__(
            'ML_CorrelationEvalPractitioner'
        )
        # statistical parameters
        self.exog = exogenous
        self.endog = endogenous
        self.cat_names = ordinal_label
        self.kndll_tau = kendall_tau
        self.sprmn_r = spearman_r

        # plotting parameters
        self.graph_name = graph_name
        self.histogram = histogram
        self.boxplot = boxplot

class Ordinal_Correlation_Practitioner():
    def __init__(self,
                 config,
                 dt_processor,
                 io_manager):
        '''
        constructor for the ordinal correlation practitioner
        :param config: confiuration file
        :param dt_processor: the data processor for the project
        :param io_manager: the inputoutput manager for the project
        '''
        self.config = config

        self.dt_processor = dt_processor
        self.io_manager = io_manager

    def FiveNumSum(self, dt):
        '''
        Runs a 5 number summary on the data
        :param dt: input data
        :return: list of a 5 number summary
        '''
        max = np.max(dt)
        min = np.min(dt)
        quartiles = np.percentile(dt, [25, 50, 75])
        return [min] + list(quartiles) + [max]

    def plot_name(self, type, dtype):
        '''
        generate the name of the plot
        :param type: the type of plot being saved
        :param dtype: the datatype to save the plot as
        :return: name of the plot that will be saved
        '''
        _name = [] if self.config.graph_name==None else [self.config.graph_name]
        _name.extend([type, self.config.exog])
        return '.'.join(['_'.join(_name), dtype])

    def histogram_of_x(self, x):
        '''
        will create a histogram of the data in x
        :param x: input data
        :return: saves a jpg and a pdf of the histogram
        '''
        # Histogram of values
        hist_plot_data = [sample[1] for sample in x]
        plt.hist(hist_plot_data, bins=100)

        plt.ylabel('Frequency')
        plt.title(self.config.endog)
        plt.savefig(
            os.path.join(self.io_manager.root,
                         self.plot_name('hist', 'pdf'))
        )
        plt.savefig(
            os.path.join(self.io_manager.root,
                         self.plot_name('hist', 'jpg'))
        )
        plt.clf()
        plt.close()

    def boxplot_of_x(self, x):
        '''
        will create a boxplot of the data in x
        :param x: input data
        :return: saves a jpg and a pdf of the boxplot
        '''
        ### BoxPlotWork
        data = [pt for pt in x]
        box_plot_data = [
            [sample[1] for sample in data if self.config.cat_names[sample[0]]
             == cat]
            for cat in self.config.cat_names]
        plt.boxplot(box_plot_data, vert=True, patch_artist=True,
                    whis=1.5, labels=self.config.cat_names)
        max_val = np.max([x[1] for x in data])
        if max_val < 0.55:
            plt.ylim([-0.05, 0.55])
        elif max_val < 1.05:
            plt.ylim([-0.05, 1.05])

        plt.ylabel(self.config.endog)
        plt.title(self.config.exog)
        plt.savefig(
            os.path.join(self.io_manager.root,
                         self.plot_name('boxplot', 'pdf'))
        )
        plt.savefig(
            os.path.join(self.io_manager.root,
                         self.plot_name('boxplot', 'jpg'))
        )
        plt.clf()
        plt.close()

    def calculate_kendall_tau(self, dt, res_dict):
        '''
        run kendall tau statisical analysis on dt
        :param dt: the dataset to run analysis on. List of tuples.
        :param res_dict: the dictionary to save the results. Must have a
        'Coefficient', 'P_value', and '95%_CI' keys
        :return: res_dict with the saved results
        '''
        # need to ensure the results keys are in the given dictionary
        assert all([ky in res_dict.keys() for ky in
                    ['Coefficient', 'P_value', '95%_CI']])

        # run the statistical analysis
        test_result = kendalltau(x=[x[0] for x in dt],
                                 y=[x[1] for x in dt])
        res_dict['Coefficient'].append(test_result[0])
        res_dict['P_value'].append(test_result[1])

        # calculate the 95% confidence interval
        r = 0.5 * np.log((1 + test_result[0]) / (1 - test_result[0]))
        num = len(dt)
        stderr = 1.0 / np.sqrt(num - 3)
        delta = 1.96 * stderr
        lower_z = r - delta
        upper_z = r + delta
        lower = (np.exp(2 * lower_z) - 1) / (np.exp(2 * lower_z) + 1)
        upper = (np.exp(2 * upper_z) - 1) / (np.exp(2 * upper_z) + 1)
        res_dict['95%_CI'].append([lower, upper])

        return res_dict

    def calculate_spearman_r(self, dt, res_dict):
        '''
        run spearman r statisical analysis on dt
        :param dt: the dataset to run analysis on. List of tuples.
        :param res_dict: the dictionary to save the results. Must have a
        'Coefficient', 'P_value', and '95%_CI' keys
        :return: res_dict with the saved results
        '''
        # need to ensure the results keys are in the given dictionary
        assert all([ky in res_dict.keys() for ky in
                    ['Coefficient', 'P_value', '95%_CI']])

        # run the statistical analysis
        test_result = spearmanr(a=[x[0] for x in dt],
                                b=[x[1] for x in dt])
        res_dict['Coefficient'].append(test_result[0])
        res_dict['P_value'].append(test_result[1])

        # calculate the 95% confidence interval
        r = test_result[0]
        num = len(dt)
        stderr = 1.0 / np.sqrt(num - 3)
        delta = 1.96 * stderr
        lower = np.tanh(np.arctanh(r) - delta)
        upper = np.tanh(np.arctanh(r) + delta)
        res_dict['95%_CI'].append([lower, upper])

        return res_dict

    def evaluate(self, input_data_set=None):
        '''
        evaluation function
        :param input_data_set: pd.DataFrame. the data set to run analysis on
        '''
        if input_data_set is None:
            dataset = pd.DataFrame(
                [self.dt_processor.if_dset.__getitem__(i) for i
                 in range(len(self.dt_processor.if_dset))]
            )
        else:
            dataset = input_data_set

        if 'y' in dataset.columns:
            dataset = dataset.rename(columns={'y':self.config.endog})
        if 'X' in dataset.columns:
            dataset = dataset.rename(columns={'X':self.config.exog})
        dataset = dataset[[self.config.endog, self.config.exog]].dropna()
        data = [tuple(x) for x in
                dataset[[self.config.exog, self.config.endog]].values]
        data = [(self.config.cat_names.index(x[0]), x[1]) for x in data]

        if self.config.histogram:
            self.histogram_of_x(data)

        if self.config.boxplot:
            self.boxplot_of_x(data)

        seperated_data = [
            [sample[1]
             for sample in data
             if self.config.cat_names[sample[0]] == cat]
            for cat in self.config.cat_names
        ]
        ### Spearman and Kendall Tests
        save_data = {
            'Test': [],
            'Sample Sizes': [len(x) for x in seperated_data],
            'DataSet Mean': [np.average([row[1] for row in data])],
            'DataSet STD': [np.std([row[1] for row in data])],
            'DataSet 5-NumSum': [self.FiveNumSum([row[1] for row in data])],
            'Sample Means': [[np.average(x) for x in seperated_data]],
            'Sample Std.Dev.': [[np.std(x) for x in seperated_data]],
            'Coefficient': [],
            'P_value': [],
            '95%_CI': []
        }
        measures = []
        if self.config.kndll_tau:
            measures.append('Kendall Tau')
        if self.config.sprmn_r:
            measures.append('Spearman Correlation')
        if len(measures)>0:
            for measure in measures:
                save_data['Test'].append(measure)
                if measure=='Kendall Tau':
                    save_data = self.calculate_kendall_tau(data, save_data)
                if measure=='Spearman Correlation':
                    save_data = self.calculate_spearman_r(data, save_data)
            save_data = pd.DataFrame(
                dict([(k, pd.Series(v)) for k, v in save_data.items()])
            )
            save_data = save_data.fillna('')
            save_data.to_csv(
                os.path.join(self.io_manager.root,
                             self.plot_name('Stats', 'csv')),
                index=False
            )
        else:
            print('ML Practitioner: No statistics calculated. '
                  'config must have kendall_tau=True or spearman_r=True for '
                  'statistics to be calculated. ')
