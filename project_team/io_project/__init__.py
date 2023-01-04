import random

import numpy

# Base experimental Frameworks
from .Managers._TrainDeploy import *
from .Managers._Kfold import *
from .Managers._HyperParameterTuning import *

# Base IO Manager Class
from .Managers.IO_Manager import *

# Base Specialized Package IO Managers
from .Managers.Pytorch_Manager import *
from .Managers.Clustering_Manager import *

# Base Task Specialized Package IO Managers
from .Managers.MedSeg_PT_Manager import *
from .Managers.MedSeg_Cl_Manager import *

# specified dicom identifier functions. But these are only accustom to
# Sunnybrook so needs to be moved to custom functions before publication


def set_random_seed(input_seed):
    numpy.random.seed(input_seed)
    random.seed(input_seed)
    torch.manual_seed(input_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(input_seed)
        if torch.cuda.device_count()>1:
            torch.cuda.manual_seed_all(input_seed)

