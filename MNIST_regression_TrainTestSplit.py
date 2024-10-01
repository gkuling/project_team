'''
Copyright (c) 2023, Martel Lab, Sunnybrook Research Institute

Description: Example code of how to use the project_team to train a model on
regression. This example is performed on the MNIST dataset. This will
perform a Train-Test Split experiment

Input: a working_dir (working directory) to perform the experiment in
Output: in the working directory there will be configs for the manager,
processor, practitioner, and model. Checkpoint saves will be intheir own
folder. Datsets will be saved in individual csv files. Final model weights as a
pth file.
'''

import argparse

import pandas as pd
from scipy.stats import spearmanr
from torchvision import datasets
from sklearn.metrics import accuracy_score

import project_team as proteam
import os
from default_arguements import dt_args, ml_args, mdl_args

r_seed = 20230117
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--working_dir',type=str,
                    default='/amartel_data/Grey/pro_team_examples',
                    help='The current directory to save models, and configs '
                         'of the experiment')
parser.add_argument('--start_from_checkpoint', action='store_true',
                    help='Choice to load the model from checkpoint folder')
opt = parser.parse_args()

# Prepare data if not already saved and set up
if not os.path.exists(os.path.join(opt.working_dir, 'data','dataset_info.csv')):
    if not os.path.exists(os.path.join(opt.working_dir, 'data')):
        os.mkdir(os.path.join(opt.working_dir, 'data'))

    dataset1 = datasets.MNIST('../data', train=True, download=True)

    dataset2 = datasets.MNIST('../data', train=False)
    all_data = []
    cnt = 0
    for ex in dataset1:
        save_local = os.path.join(opt.working_dir, 'data',
                                  'img_' + str(cnt) + '.png')
        ex[0].save(save_local)
        all_data.append(
            {'img_data': save_local,
             'label': ex[1]}
        )
        cnt += 1
    for ex in dataset2:
        save_local = os.path.join(opt.working_dir, 'data',
                                  'img_' + str(cnt) + '.png')
        ex[0].save(save_local)
        all_data.append(
            {'img_data': save_local,
             'label': ex[1]}
        )
        cnt += 1
    all_data = pd.DataFrame(all_data)
    all_data.to_csv(os.path.join(opt.working_dir, 'data','dataset_info.csv'))
    print('Saved all images and data set file to ' + opt.working_dir + '/data')

# Prepare Manager
io_args = {
    'data_csv_location':os.path.join(opt.working_dir, 'data',
                                     'dataset_info.csv'),
    'inf_data_csv_location': None,
    'val_data_csv_location': None,
    'experiment_name':'MNIST_Reg_TrainTestSplit',
    'project_folder':opt.working_dir,
    'X':'img_data',
    'X_dtype':'PIL png',
    'y':'label',
    'y_dtype':'discrete',
    'y_domain': [_ for _ in range(10)],
    'group_data_by':None,
    'test_size': 0.1,
    'validation_size': 0.1,
    'stratify_by': 'label',
    'r_seed': r_seed
}

ml_args['loss_type'] = 'MSE'

io_project_cnfg = proteam.io_project.io_traindeploy_config(**io_args)

manager = proteam.io_project.Pytorch_Manager(
    io_config_input=io_project_cnfg
)

# Prepare Processor
dt_project_cnfg = proteam.dt_project.Image_Processor_config(**dt_args)

processor = proteam.dt_project.Image_Processor(
    image_processor_config=dt_project_cnfg
)

# Prepare model
mdl = proteam.models.MNIST_CNN(
    proteam.models.MNIST_CNN_config(**mdl_args)
)
mdl = proteam.models.PTRegressionModel(
    encoder=mdl,
    config=proteam.models.PTRegression_config(
        encoder='custom',
        regresser_input=10,
        regressor_output=1,
        flatten_assist=False,
        output_style='continuous'
    )
)


# Prepare Practitioner
ml_project_cnfg = proteam.ml_project.PTRegression_Practitioner_config(
    **ml_args)

practitioner = proteam.ml_project.PTRegression_Practitioner(
    model=mdl,
    io_manager=manager,
    data_processor=processor,
    trainer_config=ml_project_cnfg
)

# Perform Training
manager.prepare_for_experiment()

processor.set_training_data(manager.root)
processor.set_validation_data(manager.root)

if opt.start_from_checkpoint:
    manager.from_model_checkpoint(practitioner)

practitioner.train_model()

# Perform Inference
manager.prepare_for_inference()

processor.set_inference_data(manager.root)

practitioner.run_inference()

test_results = processor.inference_results

# Evaluate Inference Results
print('Model Accuracy: ' +
      str(spearmanr(test_results['y'], test_results['pred_y'])))

print('End of MNIST_regression_TrainTestSplit.py')

