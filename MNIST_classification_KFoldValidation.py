import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from sklearn.metrics import accuracy_score

import ml_project
import src as proteam
from src.project_config import project_config
import os
from copy import deepcopy

r_seed = 2023322
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--working_dir',type=str,
                    default='/amartel_data/Grey/pro_team_examples',
                    help='The current directory to save models, and configs '
                         'of the experiment')
opt = parser.parse_args()

class Net_config(project_config):
    def __init__(self, 
                 kernel = 3,
                 hidden_layer_parameters=128,
                 numpy_shape=(28,28),
                 **kwargs):
        super(Net_config, self).__init__('Net')
        self.kernel = kernel
        self.hidden_layer_parameters = hidden_layer_parameters
        self.output_style = 'softmax'
        assert len(numpy_shape)==2
        assert numpy_shape[0]==numpy_shape[1]
        self.input_shape = numpy_shape

class Net(nn.Module):
    def __init__(self, config = Net_config()):
        super(Net, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, 32, self.config.kernel, 1)
        self.conv2 = nn.Conv2d(32, 64, self.config.kernel, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(int(64*(self.config.input_shape[0]/2-2*int((self.config.kernel-1)/2))**2),
                             self.config.hidden_layer_parameters)
        self.fc2 = nn.Linear(self.config.hidden_layer_parameters, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        output = self.fc2(x)
        return output

# Prepare data if not already saved and set up
if not os.path.exists(opt.working_dir + '/data/dataset_info.csv'):
    if not os.path.exists(opt.working_dir + '/data'):
        os.mkdir(opt.working_dir + '/data')

    dataset1 = datasets.MNIST('../data', train=True, download=True)

    dataset2 = datasets.MNIST('../data', train=False)
    all_data = []
    cnt = 0
    for ex in dataset1:
        ex[0].save(opt.working_dir + '/data/img_' + str(cnt) + '.png')
        all_data.append(
            {'img_data': opt.working_dir + '/data/img_' + str(cnt) + '.png',
             'label': ex[1]}
        )
        cnt += 1
    for ex in dataset2:
        ex[0].save(opt.working_dir + '/data/img_' + str(cnt) + '.png')
        all_data.append(
            {'img_data': opt.working_dir + '/data/img_' + str(cnt) + '.png',
             'label': ex[1]}
        )
        cnt += 1
    all_data = pd.DataFrame(all_data)
    all_data.to_csv(opt.working_dir + '/data/dataset_info.csv')
    print('Saved all images and data set file to ' + opt.working_dir + '/data')

# Prepare Manager
# Here use all the same parameters as the Train/Test Split example
from default_arguements import dt_args, ml_args, mdl_args

io_args = {
    'data_csv_location':opt.working_dir + '/data/dataset_info.csv',
    'inf_data_csv_location': None,
    'val_data_csv_location': None,
    'experiment_name':'MNIST_KFoldValidation',
    'project_folder':opt.working_dir,
    'X':'img_data',
    'X_dtype':'PIL png',
    'y':'label',
    'y_dtype':'discrete',
    'y_domain': [_ for _ in range(10)],
    'group_data_by':None,
    'test_size': 0.0,
    'validation_size': 0.1,
    'stratify_by': 'label',
    'r_seed': r_seed
}

io_args['k_folds'] = 4



io_project_cnfg = proteam.io_project.io_kfold_config(**io_args)

manager = proteam.io_project.Pytorch_Manager(
    io_config_input=io_project_cnfg
)

manager.prepare_for_experiment()

# Prepare the Data Processor
dt_project_cnfg = proteam.dt_project.Image_Processor_config(**dt_args)

processor = proteam.dt_project.Image_Processor(
    image_processor_config=dt_project_cnfg
)

# Prepare model
mdl = Net(Net_config(**mdl_args))

# Perform KFold Validation
starting_fold = 0
for k in range(starting_fold, manager.config.k_folds):
    manager.set_fold(k)

    mdl = deepcopy(mdl)
    # Prepare Practitioner
    ml_project_cnfg = proteam.ml_project.PTClassification_Practitioner_config(
        **ml_args)
    practitioner = proteam.ml_project.PTClassification_Practitioner(
        model=mdl,
        io_manager=manager,
        data_processor=processor,
        trainer_config=ml_project_cnfg
    )
    # load data here because we have new sets each fold
    processor.set_training_data(manager.root)
    processor.set_validation_data(manager.root)
    processor.set_inference_data(manager.root)
    practitioner.train_model()
    practitioner.run_inference()
    evaluator = ml_project.ClassificationEval_Practitioner(
        config=ml_project.ClassificationEval_Practitioner_config(
            classes=io_project_cnfg.y_domain,
            save_folder=manager.root
        )
    )
    evaluator.evaluate(processor.inference_results)
    manager.save_dataframe(evaluator.eval_results, 'test_result_evaluation')
    print(' Finished running fold number: ' + str(
        k))
    print('-'*120)


manager.finished_kfold_validation()

print('End of MNIST_Classification_KFoldValidation.py')

