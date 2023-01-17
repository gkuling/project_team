import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os
import numpy as np

class OddsRatio_Practitioner():
    def __init__(self,
                 exposure,
                 outcome,
                 threshold,
                 dt_processor,
                 io_manager,
                 print_graphs=False,
                 duration=None):
        self.exposure = exposure
        self.outcome = outcome
        self.threshold = threshold
        self.dt_processor = dt_processor
        self.io_manager = io_manager
        self.print_graphs = print_graphs
        self.duration = duration

    def evaluate(self, dataframe=None):
        if type(dataframe)==pd.DataFrame:
            dataset = dataframe
        else:
            dataset = pd.DataFrame(
                [self.dt_processor.if_dset.__getitem__(i) for i
                 in range(len(self.dt_processor.if_dset))]
            )
        if 'y' in dataset.columns:
            dataset = dataset.rename(columns={'y':self.outcome})
        if 'X' in dataset.columns:
            dataset = dataset.rename(columns={'X':self.exposure})

        if self.print_graphs:
            # Box Plot positive and negative groups
            fig, ax = plt.subplots()
            outcomes = list(set(dataset[self.outcome].to_list()))
            outcomes.sort()
            # build a box plot
            ax.boxplot(
                [dataset[
                     dataset[self.outcome]==outcome
                     ][self.exposure].to_list()
                 for outcome in outcomes]
            )
            # title and axis labels
            ax.set_title(self.outcome + ' vs. ' + self.exposure + ' boxplot')
            ax.set_xlabel(self.outcome)
            ax.set_ylabel(self.exposure)
            xticklabels=outcomes
            ax.set_xticklabels(xticklabels)
            # add horizontal grid lines
            ax.yaxis.grid(True)
            plt.savefig(os.path.join(self.io_manager.root,
                                     self.exposure+'_'+self.outcome+
                                     '_boxplot.png'))
            plt.savefig(os.path.join(self.io_manager.root,
                                     self.exposure+'_'+self.outcome+
                                     '_boxplot.pdf'))

        input_cat = dataset[[self.exposure,self.outcome]].copy()

        if self.threshold=='percentiles':
            thrs = [np.percentile(input_cat[self.exposure].to_list(),_)
                   for _ in range(10,100,10)]
        elif callable(self.threshold):
            thrs = self.threshold(input_cat[self.exposure].to_list())
        else:
            thrs = self.threshold

        for thr in thrs:
            input_cat[self.exposure + '_thr'] = input_cat[self.exposure].apply(
                lambda x: x > thr
            )
            or_test = {'Metric': [],
                       'OddsRatio': [],
                       'OddsRatioCI': [],
                       'OddsRatioPval': []
                       }
            or_test['Metric'].append(self.exposure + '_Thr' + str(thr))
            if all(input_cat[self.exposure + '_thr'].values):
                or_test['OddsRatio'].append('NaN')
                or_test['OddsRatioCI'].append(('NaN','NaN'))
                or_test['OddsRatioPval'].append(1.0)
            elif all([not v for v in input_cat[self.exposure + '_thr'].values]):
                or_test['OddsRatio'].append(0.0)
                or_test['OddsRatioCI'].append((0.0,0.0))
                or_test['OddsRatioPval'].append(1.0)
            else:
                table = sm.stats.Table.from_data(
                    input_cat[
                        [self.exposure + '_thr', self.outcome]
                    ].loc[
                        (input_cat[self.outcome]=='Negative')|
                        (input_cat[self.outcome]=='Positive')
                        ]
                )
                or_rslt = sm.stats.Table2x2(table.table)

                or_test['OddsRatio'].append(or_rslt.oddsratio)
                or_test['OddsRatioCI'].append(or_rslt.oddsratio_confint())
                or_test['OddsRatioPval'].append(or_rslt.oddsratio_pvalue())


            if hasattr(self, 'or_test'):
                self.or_test = pd.concat([self.or_test,
                                          pd.DataFrame(or_test)])
            else:
                self.or_test = pd.DataFrame(or_test)
