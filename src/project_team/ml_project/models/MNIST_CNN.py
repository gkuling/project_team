import torch
import torch.nn as nn
import torch.nn.functional as F
from project_team.project_config import project_config

class MNIST_CNN_config(project_config):
    # derived state written by __init__, restored via kwargs on reload
    _derived_config_keys = frozenset({'output_style'})

    def __init__(self,
                 kernel = 3,
                 hidden_layer_parameters=128,
                 numpy_shape=(28,28),
                 **kwargs):
        '''
        Default CNN to be used with MNIST examples, and a good example of how
        to build a model in this project framework
        :param kernel: kernel size of cnn layers
        :param hidden_layer_parameters: number of hidden layer parameters
        between flatten and output
        :param numpy_shape: shape of image input
        '''
        kwargs.setdefault('config_type', 'MNIST_CNN')
        super(MNIST_CNN_config, self).__init__(**kwargs)
        self.kernel = kernel
        self.hidden_layer_parameters = hidden_layer_parameters
        self.output_style = 'softmax'
        if len(numpy_shape) != 2 or numpy_shape[0] != numpy_shape[1]:
            raise ValueError('numpy_shape must be a square 2D shape, got ' +
                             str(numpy_shape))
        # stored under the parameter's own name so save/load round-trips
        self.numpy_shape = numpy_shape

    @property
    def input_shape(self):
        '''Deprecated alias of numpy_shape, kept so old saved configs and
        old code keep working.'''
        return self.numpy_shape

    @input_shape.setter
    def input_shape(self, value):
        self.numpy_shape = value

class MNIST_CNN(nn.Module):
    def __init__(self, config = MNIST_CNN_config()):
        '''
        MNIST CNN example model
        :param config:
        '''
        super(MNIST_CNN, self).__init__()
        self.config = config
        self.conv1 = nn.Conv2d(1, 32, self.config.kernel, 1)
        self.conv2 = nn.Conv2d(32, 64, self.config.kernel, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(int(64*(self.config.numpy_shape[0]/2-2*int((self.config.kernel-1)/2))**2),
                             self.config.hidden_layer_parameters)
        self.fc2 = nn.Linear(self.config.hidden_layer_parameters, 10)

    def forward(self, x):
        '''
        runs a forward pass on x
        :param x: x is an input tensor
        :return: output logits of the model
        '''
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