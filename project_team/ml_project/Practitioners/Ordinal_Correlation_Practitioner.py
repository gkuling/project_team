import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

class Ordinal_Correlation_Practitioner():
    def __init__(self,
                 categorical,
                 continuous,
                 ordinal_label,
                 dt_processor,
                 io_manager):
        self.category = categorical
        self.continuous = continuous
        self.cat_names = ordinal_label
        self.dt_processor = dt_processor
        self.io_manager = io_manager

    def FiveNumSum(self, dt):
        max = np.max(dt)
        min = np.min(dt)
        quartiles = np.percentile(dt, [25, 50, 75])
        return [min] + list(quartiles) + [max]

    def evaluate(self, input_data_set=None):
        if input_data_set is None:
            dataset = pd.DataFrame(
                [self.dt_processor.if_dset.__getitem__(i) for i
                 in range(len(self.dt_processor.if_dset))]
            )
        else:
            dataset = input_data_set

        if 'y' in dataset.columns:
            dataset = dataset.rename(columns={'y':self.category})
        if 'X' in dataset.columns:
            dataset = dataset.rename(columns={'X':self.continuous})
        dataset = dataset[[self.continuous, self.category]].dropna()
        data = [tuple(x) for x in
                dataset[[self.category, self.continuous]].values]
        data = [(self.cat_names.index(x[0]), x[1]) for x in data]

        # Histogram of values
        hist_plot_data = [sample[1] for sample in data]
        plt.hist(hist_plot_data, bins=100)

        plt.ylabel('Frequency')
        plt.title(self.continuous)
        plt.savefig(self.io_manager.root + '/quanDICOM_hist_' + self.category +
                    '.pdf')
        plt.savefig(self.io_manager.root + '/quanDICOM_hist_' + self.category +
                    '.jpg')
        plt.clf()
        plt.close()
        ### BoxPlotWork
        data = [pt for pt in data]
        box_plot_data = [[sample[1] for sample in data if self.cat_names[sample[0]]==cat]
                         for cat in self.cat_names]
        plt.boxplot(box_plot_data, vert=True, patch_artist=True,
                    whis=1.5, labels=self.cat_names)
        max_val = np.max([x[1] for x in data])
        if max_val<0.55:
            plt.ylim([-0.05, 0.55])
        elif max_val<1.05:
            plt.ylim([-0.05, 1.05])

        plt.ylabel(self.continuous)
        plt.title(self.category)
        plt.savefig(self.io_manager.root + '/quanDICOM_boxplot_' + self.category +
                    '.pdf')
        plt.savefig(self.io_manager.root + '/quanDICOM_boxplot_' + self.category + '.jpg')
        plt.clf()
        plt.close()

        ### Spearman and Kendall Tests

        save_data ={
            'Test': ['Kendall Tau', 'Spearman Correlation'],
            'Sample Sizes': [[len(x) for x in box_plot_data],
                             [len(x) for x in box_plot_data]],
            'DataSet Mean': [np.average([row[1] for row in data]),
                             np.average([row[1] for row in data])],
            'DataSet STD': [np.std([row[1] for row in data]),
                            np.std([row[1] for row in data])],
            'DataSet 5-NumSum': [self.FiveNumSum([row[1] for row in data]),
                                 self.FiveNumSum([row[1] for row in data])],
            # 'ClassThresholds': [cat_thrs, cat_thrs],
            'Sample Means':[
                [np.average(x) for x in box_plot_data],
                [np.average(x) for x in box_plot_data]
            ],
            'Sample Std.Dev.':[
                [np.std(x) for x in box_plot_data],
                [np.std(x) for x in box_plot_data]
            ],
            'Coefficient': [
                kendalltau(x = [x[0] for x in data], y = [x[1] for x in data])[0],
                spearmanr(a = [x[0] for x in data], b = [x[1] for x in data])[0]
            ],
            'P_value': [
                kendalltau(x = [x[0] for x in data], y = [x[1] for x in data])[1],
                spearmanr(a = [x[0] for x in data], b = [x[1] for x in data])[1]
            ]
        }
        test = kendalltau(x = [x[0] for x in data], y = [x[1] for x in data])
        r = 0.5* np.log((1+test[0])/(1-test[0]))
        num = len(data)
        stderr = 1.0 / np.sqrt(num - 3)
        delta = 1.96 * stderr
        lower_z = r - delta
        upper_z = r + delta
        lower = (np.exp(2*lower_z)-1)/(np.exp(2*lower_z)+1)
        upper = (np.exp(2*upper_z)-1)/(np.exp(2*upper_z)+1)
        save_data['95%_CI'] = [[lower,upper]]

        test = spearmanr(a = [x[0] for x in data], b = [x[1] for x in data])
        r = test[0]
        num = len(data)
        stderr = 1.0 / np.sqrt(num - 3)
        delta = 1.96 * stderr
        lower = np.tanh(np.arctanh(r) - delta)
        upper = np.tanh(np.arctanh(r) + delta)
        save_data['95%_CI'].append([lower,upper])
        save_data = pd.DataFrame(save_data)
        save_data.to_csv(
            self.io_manager.root + '/quanDICOM_Stats_' + self.category + '.csv', index=False
        )