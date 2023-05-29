import torch
import torch.nn as nn
import torch.nn.functional as F
from project_team import project_config

class MNIST_CNN_config(project_config):
    def __init__(self,
                 kernel = 3,
                 hidden_layer_parameters=128,
                 numpy_shape=(28,28),
                 **kwargs):
        '''
        efaut CNN to be used with MNIST examples, and a good example of how
        to build a model in this project framework
        :param kernel: kernel size of cnn layers
        :param hidden_layer_parameters: number of hidden layer parameters
        between flatten and output
        :param numpy_shape: shape of image input
        '''
        super(MNIST_CNN_config, self).__init__('MNIST_CNN')
        self.kernel = kernel
        self.hidden_layer_parameters = hidden_layer_parameters
        self.output_style = 'softmax'
        assert len(numpy_shape)==2
        assert numpy_shape[0]==numpy_shape[1]
        self.input_shape = numpy_shape

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
        self.fc1 = nn.Linear(int(64*(self.config.input_shape[0]/2-2*int((self.config.kernel-1)/2))**2),
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