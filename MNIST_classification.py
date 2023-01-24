import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import src as proteam
from src.project_config import project_config
import os

r_seed = 20230117
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--working_dir',type=str,
                    default='/amartel_data/Grey/pro_team_examples'
                            '/TrainTestSplit',
                    help='The current directory to save models, and configs '
                         'of the experiment')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=14, metavar='N',
                    help='number of epochs to train (default: 14)')
parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                    help='learning rate (default: 1.0)')
parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                    help='Learning rate step gamma (default: 0.7)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--no-mps', action='store_true', default=False,
                    help='disables macOS GPU training')
parser.add_argument('--dry-run', action='store_true', default=False,
                    help='quickly check a single pass')
parser.add_argument('--seed', type=int, default=20220113, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')
opt = parser.parse_args()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.config = project_config('Net')
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
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
dt_args={
    'silo_dtype': 'np.uint8',
    'numpy_shape': (28,28),
    'pad_shape':(28,28),
    'pre_load':True,
    'one_hot_encode': False,
    'max_classes': 10
}

dt_project_cnfg = proteam.dt_project.Image_Processor_config(**dt_args)

processor = proteam.dt_project.Image_Processor(
    image_processor_config=dt_project_cnfg
)

# Prepare model
mdl = Net()

# Prepare Practitioner

ml_args = {
    'batch_size':opt.batch_size,
    'n_epochs':opt.epochs,
    'n_steps':None,
    'warmup':0.0,
    'lr_decay':'steplr',
    'lr_decay_stepsize': 1,
    'lr_decay_gamma': 0.1,
    'lr_decay_step_timing': 'epoch',
    'n_saves':10,
    'validation_criteria':'min',
    'optimizer':'adadelta',
    'lr':opt.lr,
    'grad_clip':None,
    'loss_type':'NLL',
    'affine_aug':False,
    'add_Gnoise':False,
    'gaussian_std':1.0,
    'normalization_percentiles':None,
    'normalization_channels':[(0.1307,0.3081)],
    'n_workers':0,
    'visualize_val':False,
    'data_parallel':False
}

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


### Pytorch example
def train(opt, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % opt.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if opt.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

use_cuda = not opt.no_cuda and torch.cuda.is_available()
use_mps = not opt.no_mps and torch.backends.mps.is_available()

if use_cuda:
    device = torch.device("cuda")
elif use_mps:
    device = torch.device("mps")
else:
    device = torch.device("cpu")

train_kwargs = {'batch_size': opt.batch_size}
test_kwargs = {'batch_size': opt.test_batch_size}
if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                   'pin_memory': True,
                   'shuffle': True}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])
train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = Net().to(device)
optimizer = optim.Adadelta(model.parameters(), lr=opt.lr)

scheduler = StepLR(optimizer, step_size=1, gamma=opt.gamma)
for epoch in range(1, opt.epochs + 1):
    train(opt, model, device, train_loader, optimizer, epoch)
    test(model, device, test_loader)
    scheduler.step()

if opt.save_model:
    torch.save(model.state_dict(), "mnist_cnn.pt")

