import os
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

class LogRegORAnalysis_Practitioner():
    def __init__(self,
                 exposure,
                 outcome,
                 dt_processor,
                 io_manager,
                 intercept=True):
        self.exposure = exposure
        self.outcome = outcome
        self.dt_processor = dt_processor
        self.io_manager = io_manager
        self.intercept = intercept

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

        input_cat = dataset[[self.exposure,self.outcome]].copy(deep=True)
        input_cat = input_cat.reset_index()
        outcome_cats = list(set(input_cat[self.outcome]))
        outcome_cats.sort()
        if outcome_cats==['Negative', 'Positive']:
            input_cat[self.outcome] = input_cat[self.outcome].apply(
                lambda vl: 1.0 if vl=='Positive' else 0.0
            )
        if self.intercept:
            X = input_cat[[self.exposure]]
            # X['intercept'] = pd.Series([1.0 for _ in range(
            #                         input_cat[self.exposure].values.shape[0]
            #                     )])
            X = X.assign(intercept=1.0)
        else:
            X = input_cat[[self.exposure]]
        try:
            lr_mdl = sm.Logit(endog=input_cat[[self.outcome]],
                          exog=X)
        except Exception as e:
            if 'ValueError: Pandas data cast to numpy dtype of object. Check ' \
               'input data with np.asarray(data).' in str(e) and \
                    outcome_cats!=['Negative', 'Positive']:
                raise ValueError("Your endog variable is not binary as a string "
                                 "from ['Negative', 'Positive'] or [0,1]")
            else:
                raise e
        lr_mdl = lr_mdl.fit()

        self.logreg_or_test = pd.DataFrame(
            {'Metric': [self.exposure],
             'OddsRatio': [np.exp(lr_mdl.params[self.exposure])],
             'OddsRatioCI': [tuple(np.exp(lr_mdl.conf_int().loc[self.exposure].values))],
             'OddsRatioPval': [lr_mdl.pvalues[self.exposure]],
             'LogRegSummary': [str(lr_mdl.summary())]
             }
        )
