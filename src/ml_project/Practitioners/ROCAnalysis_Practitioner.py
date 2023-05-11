import os
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

class ROCAnalysis_Practitioner():
    def __init__(self,
                 prediction,
                 groundtruth,
                 io_manager):
        '''
        a quick statstical practitioner that will run ROC analysis on the
        input data
        :param prediction: the output of the model
        :param groundtruth: the ground truth binary labels
        :param io_manager:
        '''
        self.prediction = prediction
        self.groundtruth = groundtruth
        self.io_manager = io_manager

    def evaluate(self, dataset):
        '''
        evaluate the ROC analysis on the given dataset
        :param dataset: a dataframe of the results
        '''
        # rename columns if X and y are in the dataset
        if 'y' in dataset.columns:
            dataset = dataset.rename(columns={'y':self.groundtruth})
        if 'X' in dataset.columns:
            dataset = dataset.rename(columns={'X':self.prediction})

        # double chek label sets
        input_cat = dataset[[self.prediction,self.groundtruth]].copy()
        groundtruth_cats = list(set(input_cat[self.groundtruth]))
        groundtruth_cats.sort()
        if groundtruth_cats==['Negative', 'Positive']:
            input_cat[self.groundtruth] = input_cat[self.groundtruth].apply(
                lambda vl: 1.0 if vl=='Positive' else 0.0
            )

        # run ROC analysis from sklearn
        fpr, tpr, _ = roc_curve(input_cat[self.groundtruth].values,
                                input_cat[self.prediction].values)
        roc_auc = auc(fpr, tpr)

        # save results
        self.roc_test = pd.DataFrame(
            {'Metric': [self.prediction],
             'AUC': [roc_auc]}
        )

        # plot the ROC curve
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
                                 self.prediction+'_'+self.groundtruth+
                                 '_ROCAUCCurve.png'))
        plt.savefig(os.path.join(self.io_manager.root,
                                 self.prediction+'_'+self.groundtruth+
                                 '_ROCAUCCurve.pdf'))
        plt.clf()
