import os.path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from project_team.project_config import project_config


class ClassificationEval_Practitioner_config(project_config):
    def __init__(self,
                 classes,
                 ground_truth='y',
                 model_prediction='pred_y',
                 F1=True,
                 sensitivity=True,
                 specificity=True,
                 accuracy=True,
                 save_folder=None,
                 **kwargs
                 ):
        '''
        Configuration file for the classification evaluation practitioner
        :param classes: ground truth classes. Should be a list of int or fields
        :param ground_truth: the column name for the ground truth
        :param model_prediction: the column name for the model prediction
        :param F1: include F1 measure in the evaluation
        :param sensitivity: include the sensitivity in the evaluation
        :param specificity: include the specificity in the evaluation
        :param accuracy: include the accuracy in the evaluation
        '''
        kwargs.setdefault('config_type', 'ML_ClassificationEvalPractitioner')
        super(ClassificationEval_Practitioner_config, self).__init__(**kwargs)

        if not isinstance(classes, list):
            raise TypeError('classes must be a list of the ground truth '
                            'class labels, got ' + repr(type(classes)))
        self.classes = classes
        self.ground_truth = ground_truth
        self.model_prediction = model_prediction

        self.F1 = F1
        self.sensitivity=sensitivity
        self.specificity=specificity
        self.accuracy=accuracy

class ClassificationEval_Practitioner():
    def __init__(self, config, pred_preprocess=None, gt_preprocess=None):
        '''
        constructor for the classification evaluator
        :param config: practitioner config
        :param pred_preprocess: any transforms needed for the model prediction
        :param gt_preprocess: any transforms needed for the ground truth
        '''
        self.config = config
        self.pred_preprocess = pred_preprocess
        self.gt_preprocess = gt_preprocess
        self.metric_options = ['F1', 'Sens.', 'Spec.', 'Acc.']

    def setup_metrics_to_eval(self):
        '''
        set up the dirctionary to save results of evaluation
        '''
        self.eval_results = {}
        if self.config.F1:
            self.eval_results['F1_Overall'] = []
        if self.config.sensitivity:
            self.eval_results['Sens._Overall'] = []
        if self.config.specificity:
            self.eval_results['Spec._Overall'] = []
        if self.config.accuracy:
            self.eval_results['Acc._Overall'] = []
        if len(self.config.classes)>1:
            for _ in self.config.classes:
                if self.config.F1:
                    self.eval_results['F1_' + str(_)] = []
                if self.config.sensitivity:
                    self.eval_results['Sens._' + str(_)] = []
                if self.config.specificity:
                    self.eval_results['Spec._' + str(_)] = []
                if self.config.accuracy:
                    self.eval_results['Acc._' + str(_)] = []

    def evaluate(self, data):
        '''
        Run the evaluation of the data. The results are saved as an atribute
        named "eval_results".
        :param data: input of the results
        '''
        print('ML Message: Beginning Evaluation of classification results.')
        self.setup_metrics_to_eval()
        if isinstance(data, pd.DataFrame):
            pass
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        elif os.path.exists(data) and data.endswith('.csv'):
            data = pd.read_csv(data, na_filter=False)
        else:
            raise TypeError('The data given to the classification evaluator '
                            'is not a DataFrame, a list of results, or a '
                            'path to a csv file.')
        if self.config.ground_truth!='y':
            data['y'] = data[self.config.ground_truth].values.tolist()
        if self.config.model_prediction!='pred_y':
            data['pred_y'] = data[self.config.model_prediction].values.tolist()
        if self.pred_preprocess is not None:
            data['pred_y'] = data['pred_y'].apply(self.pred_preprocess)
        if self.gt_preprocess is not None:
            data['y'] = data['y'].apply(self.gt_preprocess)

        for met in set([m.split('_')[0] for m in self.eval_results.keys()]):
            multiclass_res = self.evaluate_metric(
                met,
                data['pred_y'].values[:, None],
                data['y'].values[:, None]
            )
            if len(self.config.classes) > 1:
                for i, lbl in enumerate(self.config.classes):
                    self.eval_results[
                        met + '_' + str(lbl)
                    ].append(multiclass_res[i])
            if met == 'Acc.':
                self.eval_results[
                    met + '_Overall'
                    ].append(accuracy_score(data[['y']], data[['pred_y']]))
            else:
                self.eval_results[
                    met + '_Overall'
                    ].append(np.mean(multiclass_res))

        self.eval_results = pd.DataFrame(self.eval_results)
        print('ML Message: Finished Evaluation of classification maps.')

    def evaluate_metric(self, met, p, g):
        '''
        calculation of the given metric based on the name prediction (p) and
        groundtruth (g)
        :param met: name of metric
        :param p: prediction
        :param g: groundtruth
        :return: result
        '''
        # one binary map per configured class — built from config.classes
        # directly, so offset, non-contiguous, or string labels all align
        # with the per-class result columns
        individual_label_maps = [(g==cls, p==cls)
                                 for cls in self.config.classes]
        if met=='F1':
            return [(2*(g_p*p_p).sum() + 1e-8)/(g_p.sum() + p_p.sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        elif met=='Sens.':
            return [((g_p*p_p).sum() + 1e-8)/(g_p.sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        elif met=='Spec.':
            # specificity = TN / (TN + FP). Note: results computed before
            # this fix were actually precision (TP / (TP + FP))
            return [(((1-g_p)*(1-p_p)).sum() + 1e-8)/((1-g_p).sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        elif met=='Acc.':
            return [((g_p*p_p).sum() +
                     ((1-g_p)*(1-p_p)).sum() + 1e-8)/
                    (g_p.sum() + (1-g_p).sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        else:
            raise ValueError(met + ' is not an implemented metric. ')
