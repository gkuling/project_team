import os
from sklearn.metrics import roc_curve, auc
import pandas as pd
import matplotlib.pyplot as plt

class ROCAnalysis_Practitioner():
    def __init__(self,
                 prediction,
                 groundtruth,
                 io_manager,
                 positive_label=None):
        '''
        a quick statistical practitioner that will run ROC analysis on the
        input data
        :param prediction: the output of the model
        :param groundtruth: the ground truth binary labels
        :param io_manager: manager whose root the plots and stats save into
        :param positive_label: the ground-truth value counted as positive.
            default: None, which auto-detects a ['Negative', 'Positive']
            encoding and otherwise passes the labels through unconverted
        '''
        self.prediction = prediction
        self.groundtruth = groundtruth
        self.io_manager = io_manager
        self.positive_label = positive_label

    def evaluate(self, dataset):
        '''
        evaluate the ROC analysis on the given dataset
        :param dataset: a dataframe of the results
        '''
        # rename columns if X and y are in the dataset
        # NOTE: renaming 'X' (the model input) to the prediction column
        # looks suspect — flagged for review; kept for compatibility
        if 'y' in dataset.columns:
            dataset = dataset.rename(columns={'y':self.groundtruth})
        if 'X' in dataset.columns:
            dataset = dataset.rename(columns={'X':self.prediction})

        # double check label sets
        input_cat = dataset[[self.prediction,self.groundtruth]].copy()
        groundtruth_cats = list(set(input_cat[self.groundtruth]))
        groundtruth_cats.sort()
        pos_label = self.positive_label
        if pos_label is None and groundtruth_cats==['Negative', 'Positive']:
            pos_label = 'Positive'
        if pos_label is not None:
            input_cat[self.groundtruth] = (
                input_cat[self.groundtruth] == pos_label).astype(float)

        # run ROC analysis from sklearn
        fpr, tpr, _ = roc_curve(input_cat[self.groundtruth].values,
                                input_cat[self.prediction].values)
        roc_auc = auc(fpr, tpr)

        # save results
        self.roc_test = pd.DataFrame(
            {'Metric': [self.prediction],
             'AUC': [roc_auc]}
        )
        self.roc_test.to_csv(
            os.path.join(self.io_manager.root,
                         self.prediction + '_' + self.groundtruth +
                         '_ROCAUCStats.csv'),
            index=False)

        # plot the ROC curve
        fig = plt.figure()
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
        plt.title("ROC curve: " + self.prediction + " vs " +
                  self.groundtruth)
        plt.legend(loc="lower right")

        plt.savefig(os.path.join(self.io_manager.root,
                                 self.prediction+'_'+self.groundtruth+
                                 '_ROCAUCCurve.png'))
        plt.savefig(os.path.join(self.io_manager.root,
                                 self.prediction+'_'+self.groundtruth+
                                 '_ROCAUCCurve.pdf'))
        plt.close(fig)
