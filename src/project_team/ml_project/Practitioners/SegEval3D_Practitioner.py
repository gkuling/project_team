import os.path

import numpy as np
import pandas as pd
from tqdm import tqdm
from project_team.project_config import project_config, is_Primitive
import matplotlib.pyplot as plt
import SimpleITK as sitk

def quick_visualize(input_data, folder, score):
    '''
    visualization of a 3d scan for a quick save
    :param input_data: example information
    :param folder: folder to save the image
    :param score: The DSC score to put in the figure
    '''
    fig, ax = plt.subplots(3,3,figsize=(15,15))
    img = sitk.GetArrayFromImage(sitk.ReadImage(input_data['X_location']))
    pred = input_data['pred_y'][0]
    gt = input_data['y'][0]

    mids = [int(sh/2) for sh in gt.shape]
    mids[0] = int(mids[0]/2)

    ax[0,0].imshow(img[mids[0]], cmap='gray', aspect='auto')
    ax[0,1].imshow(img[:, mids[1]],cmap='gray', aspect='auto')
    ax[0,2].imshow(img[:,:,mids[2]], cmap='gray', aspect='auto')
    ax[1,0].imshow(gt[mids[0]], cmap='gray', aspect='auto')
    ax[1,1].imshow(gt[:, mids[1]],cmap='gray', aspect='auto')
    ax[1,2].imshow(gt[:,:,mids[2]], cmap='gray', aspect='auto')
    ax[2,0].imshow(pred[mids[0]], cmap='gray', aspect='auto')
    ax[2,1].imshow(pred[:, mids[1]],cmap='gray', aspect='auto')
    ax[2,2].imshow(pred[:,:,mids[2]], cmap='gray', aspect='auto')
    fig.suptitle(str(score), fontsize=28)
    name = folder + '/' + str(input_data['AccNum']) + '_' + input_data[
        'Acquisition'] + '.pdf'
    plt.savefig(name)
    plt.close(fig)

class SegEval_Practitioner_config(project_config):
    def __init__(self,
                 subject_field,
                 ground_truth='y',
                 model_prediction='pred_y',
                 dice=True,
                 gen_dice=False,
                 sensitivity=False,
                 specificity=False,
                 accuracy=False,
                 iou=False,
                 visualize=False,
                 save_folder=None,
                 **kwargs
                 ):
        '''
        :param subject_field: identifier for individual subjects to be
        evaluated
        :param ground_truth: identifier of the groundtruth
        :param model_prediction: identifier of the prediction

        The rest are metrics to evaluate
        :param dice: dice similairty coefficient
        :param gen_dice: generalized didce similarity coefficient
        :param sensitivity:
        :param specificity:
        :param accuracy:
        :param iou: intersect over union
        :param visualize: bool. to save a visualization of the 3d data
        :param save_folder: folder to save the results
        '''
        super(SegEval_Practitioner_config, self).__init__(
            'ML_SegEvalPractitioner')
        self.subject_field = subject_field
        self.ground_truth = ground_truth
        self.model_prediction = model_prediction
        self.dice = dice
        self.gen_dice = gen_dice
        self.sensitivity=sensitivity
        self.specificity=specificity
        self.accuracy=accuracy
        self.iou=iou
        self.visualize = visualize
        self.save_folder = save_folder

class SegEval3D_Practitioner():
    '''
    a segmentation evaluator for 3d data
    '''
    def __init__(self, config, pred_preprocess=None, gt_preprocess=None):
        '''
        constructor
        :param config: practitioner specific config
        :param pred_preprocess: preprocess of the GT
        :param gt_preprocess: preprocessing of the Prediction
        '''
        self.config = config
        self.pred_preprocess = pred_preprocess
        self.gt_preprocess = gt_preprocess
        self.metric_options = ['DSC', 'GDSC', 'Sens.', 'Spec.', 'Acc.', 'IOU']

    def setup_metrics_to_eval(self):
        '''
        get the results dictionary initialized
        '''
        self.eval_results = {'Subject':[],
                             'seg_map':[]}
        if self.config.dice:
            self.eval_results['DSC'] = []
        if self.config.gen_dice:
            self.eval_results['GDSC'] = []
        if self.config.sensitivity:
            self.eval_results['Sens.'] = []
        if self.config.specificity:
            self.eval_results['Spec.'] = []
        if self.config.accuracy:
            self.eval_results['Acc.'] = []
        if self.config.iou:
            self.eval_results['IOU'] = []

    def evaluate(self, data):
        '''
        evaluation of the metrics on the given data
        :param data: dataframe or a location of a dataframe to evaluate the
        results.
        '''
        # organize results and data for evaluation
        print('ML Message: Beginning Evaluation of segmentation maps.')
        self.setup_metrics_to_eval()
        if type(data)==list:
            data = pd.DataFrame(data)
        elif os.path.exists(data) and data.endswith('.csv'):
            data = pd.read_csv(data)
        else:
            raise Exception('The data given to the segmentation evaluator is '
                            'not a list of results or a csv file. ')
        data['y'] = data[self.config.ground_truth].values.tolist()
        data['pred_y'] = data[self.config.model_prediction].values.tolist()
        data = data.to_dict('records')
        for key in data[0].keys():
            if key not in self.eval_results.keys() and \
                    is_Primitive(data[0][key]) and key != 'y' and key != \
                    'pred_y':
                self.eval_results[key] = []

        # run the evaluations
        for tst_sub in tqdm(data, desc='Running Evaluation: '):
            if self.gt_preprocess:
                for pr in self.gt_preprocess:
                    tst_sub = pr(tst_sub)
            if self.pred_preprocess:
                for pr in self.pred_preprocess:
                    tst_sub = pr(tst_sub)
            preds = tst_sub['pred_y']
            gts = tst_sub['y']

            for i in range(len(preds)):
                pred = preds[i]
                gt = gts[i]
                assert type(pred)==np.ndarray
                assert type(gt)==np.ndarray
                assert pred.shape==gt.shape
                for key in self.eval_results:
                    if key in self.metric_options:
                        self.eval_results[key].append(self.evaluate_metric(
                            key, pred, gt))
                self.eval_results['Subject'].append(tst_sub[
                    self.config.subject_field])
                self.eval_results['seg_map'].append(str(i))
                for key in tst_sub.keys():
                    if key in self.eval_results.keys() and \
                            is_Primitive(tst_sub[key]):
                        self.eval_results[key].append(tst_sub[key])

            if self.config.visualize:
                quick_visualize(tst_sub, self.config.save_folder,
                                self.eval_results['DSC'][-1])
        self.eval_results = {k:v for k,v in self.eval_results.items() if len(
            v)!=0}
        self.eval_results = pd.DataFrame(self.eval_results)
        print('ML Message: Finished Evaluation of segmentation maps.')

    def evaluate_metric(self, met, p, g):
        '''
        function to evaluate the given metric
        :param met: metric to be calculated
        :param p: prediction
        :param g: groundtruth
        :return: the desired metric
        '''
        individual_label_maps = [(g==float(u),p==float(u)) for u in
                                 range(int(np.unique(g).max())+1)]
        if met=='DSC':
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



