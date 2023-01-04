import os

import pandas as pd
import SimpleITK as sitk

from .Clustering_Manager import Clustering_Manager

class MedSeg_Cl_Manager(Clustering_Manager):
    def __init__(self, io_config_input):
        super(MedSeg_Cl_Manager, self).__init__(io_config_input)
        pass

    def save_inference_results(self, list_of_results, reduce_add=None,
                               save_folder=None):

        if reduce_add:
            reduction_consider = [ex[reduce_add] for ex in list_of_results]
            reduction_standard = list(set(reduction_consider))
            list_to_save = []
            for ex in reduction_standard:
                scans_oi = [samp for samp in list_of_results if samp[
                    reduce_add]==ex]
                sum_scans = [s['pred_y'][0] for s in scans_oi]
                res = sitk.Add(*sum_scans)
                scans_oi[0]['pred_y'] = res
                list_to_save.append(scans_oi[0])
        else:
            list_to_save = list_of_results

        for ex in list_to_save:

            name = ex['X_location'].split('.')[0] + '_' + \
                   self.exp_type.config.experiment_name + '.nii.gz'

            if type(ex['pred_y'])==list and len(ex['pred_y'])==1:
                sitk.WriteImage(ex['pred_y'][0], name)
            elif type(ex['pred_y'])==list:
                raise NotImplementedError('Save mutiple outputs has not been '
                                          'implemented yet. ')
            else:
                sitk.WriteImage(ex['pred_y'], name)
            ex[self.exp_type.config.experiment_name + '_pred_y'] = name
        inf_set_data = pd.read_csv(self.root + '/inf_set.csv')
        if reduce_add:
            inf_set_data[self.exp_type.config.experiment_name + '_pred_y'] = \
                inf_set_data['X'].apply(
                    lambda x: [ex for ex in list_to_save if ex[reduce_add]==x][0][
                        self.exp_type.config.experiment_name + '_pred_y'
                        ]
                )
        else:
            inf_set_data[self.exp_type.config.experiment_name + '_pred_y'] = \
                inf_set_data['X'].apply(
                    lambda x: [ex for ex in list_to_save
                               if ex['X_location']==x][0][
                        self.exp_type.config.experiment_name + '_pred_y'
                        ]
                )
        inf_set_data.to_csv(self.root + '/inf_set.csv', index=False)
