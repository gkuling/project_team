import argparse
import pandas as pd

from project_team import dt_project
from project_team.ml_project import SegEval3D_Practitioner, \
    SegEval_Practitioner_config
import numpy as np

import project_team as proteam
import os
from project_team.dt_project.DataProcessors.SITK_Processor import nnUNetSITK_Processor
from project_team.ml_project.Practitioners.nnUNet_Practitioner import (
    nnUNet_Practitioner, nnUNet_Practitioner_config)

r_seed = 20230824
parser = argparse.ArgumentParser(description='nnUNet Segmentation Example')
parser.add_argument('--working_dir',type=str,
                    default='/amartel_data/Grey/pro_team_examples',
                    help='The current directory to save models, and configs '
                         'of the experiment')
opt = parser.parse_args()

dataset_file = os.path.join(opt.working_dir,
                                   'decathalon',
                                   'segmentation_dataset.csv')

import shutil
if os.path.exists(os.path.join(opt.working_dir, 'nnUNet_TrainTestSplit',
                           'nnunetv2_raw', 'Dataset001_Current',
                           'imagesTs')):
    shutil.rmtree(os.path.join(opt.working_dir, 'nnUNet_TrainTestSplit',
                               'nnunetv2_raw', 'Dataset001_Current',
                               'imagesTs'))
if os.path.exists(os.path.join(opt.working_dir, 'nnUNet_TrainTestSplit',
                               'nnunetv2_raw', 'Dataset001_Current',
                               'imagesTs')):
    shutil.rmtree(os.path.join(opt.working_dir, 'nnUNet_TrainTestSplit',
                               'nnunetv2_raw', 'Dataset001_Current',
                               'imagesTs'))
if os.path.exists(os.path.join(opt.working_dir, 'nnUNet_TrainTestSplit',
                             'nnunetv2_raw', 'Dataset001_Current',
                            'imagesTs_predlowres')):
    shutil.rmtree(os.path.join(opt.working_dir, 'nnUNet_TrainTestSplit',
                               'nnunetv2_raw', 'Dataset001_Current',
                               'imagesTs_predlowres'))
# Prepare data if not already saved and set up
if not os.path.exists(dataset_file):

    dataset = pd.DataFrame(
        {
            'img_data': [os.path.join(opt.working_dir, 'decathalon',
                                      'Task02_Heart', 'imagesTr',fl) for fl
                         in os.listdir(os.path.join(opt.working_dir,
                                                    'decathalon',
                                                    'Task02_Heart',
                                                    'imagesTr'))],
            'label': [os.path.join(opt.working_dir, 'decathalon',
                                   'Task02_Heart', 'labelsTr',fl) for fl in
                      os.listdir(os.path.join(opt.working_dir, 'decathalon',
                                              'Task02_Heart', 'labelsTr'))]
        }
    )
    dataset.to_csv(dataset_file)
    print('Catalogued and saved data set file to ' + opt.working_dir +
          '/data')

# Prepare Manager
io_args = {
    'data_csv_location': dataset_file,
    'inf_data_csv_location': None,
    'val_data_csv_location': None,
    'experiment_name':'nnUNet_TrainTestSplit',
    'project_folder':opt.working_dir,
    'X':'img_data',
    'X_dtype':'SITKImage',
    'y':'label',
    'y_dtype':'SITKImage',
    'test_size': 0.1,
    'validation_size': 0.0,
    'r_seed': r_seed
}

io_project_cnfg = proteam.io_project.io_traindeploy_config(**io_args)

manager = proteam.io_project.Pytorch_Manager(
    io_config_input=io_project_cnfg
)

# Prepare Processor
dt_project_cnfg = proteam.dt_project.SITK_Processor_config(
    **{
    'filter_out_zero_X': False
}
)

processor = nnUNetSITK_Processor(
    sitk_processor_config=dt_project_cnfg,
    save_folder=manager.root
)

# Prepare Practitioner
ml_project_cnfg = proteam.ml_project.nnUNet_Practitioner_config(
    **{
        'n_epochs': 1,
        'labels': {
            "background":0,
            "left atrium":1
        },
        'channel_names': {
            "0": "MRI"
        },
        'nnunet_cofiguration': '3d_fullres'
    }
)

practitioner = nnUNet_Practitioner(
    io_manager=manager,
    dt_processor=processor,
    practitioner_config=ml_project_cnfg,

)

# Perform Training
manager.prepare_for_experiment()


processor.set_training_data(manager.root)

practitioner.train_model()

# Perform Inference

processor.set_inference_data(manager.root)

practitioner.run_inference()

eval_args = {
    'subject_field': 'Unnamed: 0',
    'ground_truth': 'y',
    'model_prediction':'pred_y',
    'dice':True,
    'sensitivity':False,
    'specificity':False,
    'accuracy':False,
    'iou':False,
    'visualize':False,
    'gen_dice': True
}

eval_cfg = SegEval_Practitioner_config(**eval_args)

eval_cfg.save_folder = manager.root
evaluator = SegEval3D_Practitioner(
    eval_cfg,
    pred_preprocess=[dt_project.SITKToNumpy(field_oi='pred_y')],
    gt_preprocess=[dt_project.OpenSITK_file(field_oi='y'),
                   dt_project.SITKToNumpy(field_oi='y')]
)

evaluator.evaluate(processor.inference_results)

print('Model results are:')
print(np.array([vl for vl in evaluator.eval_results['DSC']]).mean(axis=0))
print('End of seg_firstattempt.py')
