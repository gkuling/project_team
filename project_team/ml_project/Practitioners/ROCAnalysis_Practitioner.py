import os
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

class ROCAnalysis_Practitioner():
    def __init__(self,
                 exposure,
                 outcome,
                 dt_processor,
                 io_manager):
        self.exposure = exposure
        self.outcome = outcome
        self.dt_processor = dt_processor
        self.io_manager = io_manager

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

        input_cat = dataset[[self.exposure,self.outcome]].copy()
        outcome_cats = list(set(input_cat[self.outcome]))
        outcome_cats.sort()
        if outcome_cats==['Negative', 'Positive']:
            input_cat[self.outcome] = input_cat[self.outcome].apply(
                lambda vl: 1.0 if vl=='Positive' else 0.0
            )
        fpr, tpr, _ = roc_curve(input_cat[self.outcome].values,
                                input_cat[self.exposure].values)
        roc_auc = auc(fpr, tpr)

        self.roc_test = pd.DataFrame(
            {'Metric': [self.exposure],
             'AUC': [roc_auc]}
        )

        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % roc_auc,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.io_manager.root,
                                 self.exposure+'_'+self.outcome+
                                 '_ROCAUCCurve.png'))
        plt.savefig(os.path.join(self.io_manager.root,
                                 self.exposure+'_'+self.outcome+
                                 '_ROCAUCCurve.pdf'))
        plt.clf()
