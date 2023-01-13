import os

from src.project_config import is_Primitive
import pandas as pd
import SimpleITK as sitk

from .Pytorch_Manager import Pytorch_Manager

class MedSeg_PT_Manager(Pytorch_Manager):
    def __init__(self, io_config_input):
        super(MedSeg_PT_Manager, self).__init__(io_config_input)

    def save_inference_results(self, list_to_save, save_folder=None):
        field = 'pred_y'
        inf_set_data = []
        for ex in list_to_save:
            for i, result in enumerate(ex['pred_y']):
                name = ex['X_location'].split('.')[0] + '_' + \
                       self.exp_type.config.experiment_name + '_' + \
                       self.config.y[i] + '.nii.gz'
                if save_folder:
                    name = name.replace(os.path.dirname(name),save_folder)

                sitk.WriteImage(result, name)
                ex[self.exp_type.config.experiment_name + '_' + \
                   self.config.y[i]] = name
        list_to_save = [{k:v for k,v in d.items() if is_Primitive(v)}
                        for d in list_to_save]
        inf_set_data = pd.DataFrame(list_to_save)
        inf_set_data.to_csv(self.root + '/inf_set.csv', index=False)
