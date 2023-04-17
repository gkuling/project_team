import argparse

import pandas as pd
from torchvision import datasets
from sklearn.metrics import accuracy_score

import src as proteam
import os
from default_arguements import dt_args, ml_args, mdl_args

r_seed = 20230117
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--working_dir',type=str,
                    default='/amartel_data/Grey/pro_team_examples',
                    help='The current directory to save models, and configs '
                         'of the experiment')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 14)')
opt = parser.parse_args()

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
io_args = {
    'data_csv_location':opt.working_dir + '/data/dataset_info.csv',
    'inf_data_csv_location': None,
    'val_data_csv_location': None,
    'experiment_name':'MNIST_TrainTestSplit',
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
# Prepare model
mdl = proteam.models.MNIST_CNN(
    proteam.models.MNIST_CNN_config(**mdl_args)
)

# Prepare Practitioner
ml_project_cnfg = proteam.ml_project.PTClassification_Practitioner_config(
    **ml_args)

practitioner = proteam.ml_project.PTClassification_Practitioner(
    model=mdl,
    io_manager=manager,
    data_processor=processor,
    trainer_config=ml_project_cnfg
)

# Perform Training
manager.prepare_for_experiment()

processor.set_training_data(manager.root)
processor.set_validation_data(manager.root)

practitioner.train_model()

# Perform Inference
manager.prepare_for_inference()

processor.set_inference_data(manager.root)

practitioner.run_inference()

test_results = processor.inference_results

# Evaluate Inference Results
print('Model Accuracy: ' +
      str(accuracy_score(test_results['y'], test_results['pred_y'])))

print('End of MNIST_Classification.py')

