from ._TrainDeploy import io_traindeploy_config, _TrainDeploy
from ._Kfold import io_kfold_config, _Kfold
from ._HyperParameterTuning import io_hptuning_config, _HyperParameterTuning
import pandas as pd
import os
import numpy as np

class IO_Manager(object):
    def __init__(self,
                 io_config_input):
        if type(io_config_input)==io_traindeploy_config:
            self.exp_type = _TrainDeploy(io_config_input)
        elif type(io_config_input)==io_kfold_config:
            self.exp_type = _Kfold(io_config_input)
        elif type(io_config_input)==io_hptuning_config:
            self.exp_type = _HyperParameterTuning(io_config_input)

    def __getattr__(self, item):
        return getattr(self.exp_type, item)

    def save_dataframe(self, df, name):
        df.to_csv(os.path.join(self.root, name + '.csv'), index=False)