import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from src.project_config import project_config, is_Primitive

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
        :param save_folder: the folder to save the results
        '''
        super(ClassificationEval_Practitioner_config, self).__init__(
            'ML_ClassificationEvalPractitioner')

        assert (type(classes)==list)
        self.classes = classes
        self.ground_truth = ground_truth
        self.model_prediction = model_prediction

        self.F1 = F1
        self.sensitivity=sensitivity
        self.specificity=specificity
        self.accuracy=accuracy
        self.save_folder = save_folder

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
        if type(data)==pd.DataFrame:
            pass
        elif type(data)==list:
            data = pd.DataFrame(data)
        elif os.path.exists(data) and data.endswith('.csv'):
            data = pd.read_csv(data)
        else:
            raise Exception('The data given to the segmentation evaluator is '
                            'not a list of results or a csv file. ')
        if self.config.ground_truth!='y':
            data['y'] = data[self.config.ground_truth].values.tolist()
        if self.config.model_prediction!='pred_y':
            data['pred_y'] = data[self.config.model_prediction].values.tolist()

        for met in set([m.split('_')[0] for m in self.eval_results.keys()]):
            multiclass_res = self.evaluate_metric(
                met,
                data['pred_y'].values[:, None],
                data['y'].values[:, None]
            )
            for lbl in self.config.classes:
                self.eval_results[
                    met + '_' + str(lbl)
                ].append(multiclass_res[self.config.classes.index(lbl)])
            if met == 'Acc.':
                self.eval_results[
                    met + '_Overall'
                    ].append(accuracy_score(data[['y']], data[['pred_y']]))
            else:
                self.eval_results[
                    met + '_Overall'
                    ].append(np.mean(multiclass_res))

        self.eval_results = pd.DataFrame(self.eval_results)
        print('ML Message: Finished Evaluation of segmentation maps.')

    def evaluate_metric(self, met, p, g):
        individual_label_maps = [(g==float(u),p==float(u)) for u in
                                 range(int(np.unique(g).max())+1)]
        if met=='DSC' or met=='F1':
            return [(2*(g_p*p_p).sum() + 1e-8)/(g_p.sum() + p_p.sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        elif met=='GDSC':
            w = np.array([1/(g_p.sum()**2 + 1e-8) for g_p,p_p in \
                    individual_label_maps])
            intersection = np.array([(g_p*p_p).sum() + 1e-8
                            for g_p,p_p in individual_label_maps])
            denominator = np.array([(g_p.sum() + p_p.sum() + 1e-8)
                            for g_p,p_p in individual_label_maps])
            return [(2*(w*intersection)).sum()/(w*denominator).sum()]
        elif met=='Sens.':
            return [((g_p*p_p).sum() + 1e-8)/(g_p.sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        elif met=='Spec.':
            return [((g_p*p_p).sum() + 1e-8)/(p_p.sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        elif met=='Acc.':
            return [((g_p*p_p).sum() +
                     ((1-g_p)*(1-p_p)).sum() + 1e-8)/
                    (g_p.sum() + (1-g_p).sum() + 1e-8)
                    for g_p,p_p in individual_label_maps]
        elif met=='IOU':
                return [((g_p*p_p).sum() + 1e-8)/(g_p.sum() + p_p.sum() - (
                        g_p*p_p).sum() + 1e-8)
                        for g_p,p_p in individual_label_maps]
        else:
            raise ValueError(met + ' is not an implemented metric. ')
