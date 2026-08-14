from ._TrainDeploy import io_traindeploy_config, _TrainDeploy
from ._Kfold import io_kfold_config, _Kfold
from ._HyperParameterTuning import io_hptuning_config, _HyperParameterTuning
import os

class IO_Manager(object):
    '''
    Parent class for input output manager
    '''
    def __init__(self,
                 io_config_input):
        '''
        load specific functionality based on the type of project being used
        :param io_config_input: an io_config
        '''
        if isinstance(io_config_input, io_traindeploy_config):
            self.exp_type = _TrainDeploy(io_config_input)
        elif isinstance(io_config_input, io_kfold_config):
            self.exp_type = _Kfold(io_config_input)
        elif isinstance(io_config_input, io_hptuning_config):
            self.exp_type = _HyperParameterTuning(io_config_input)
        else:
            raise TypeError(
                'IO_Manager received an unrecognized config type: ' +
                type(io_config_input).__name__ + '. Expected one of '
                'io_traindeploy_config, io_kfold_config, or '
                'io_hptuning_config.')

    def __getattr__(self, item):
        '''
        IO_Manager holds exactly one "experiment type" object (_TrainDeploy /
        _Kfold / _HyperParameterTuning) chosen by the config's class, stored
        in self.exp_type. This __getattr__ forwards any attribute lookup that
        IO_Manager itself doesn't have onto self.exp_type, so callers can use
        manager.method() without knowing which experiment type they got.
        :param item: attribute to be applied
        :return: the attribute desired
        '''
        if item == 'exp_type':
            # guard against infinite recursion when construction failed
            # before exp_type was assigned
            raise AttributeError(
                "IO_Manager has no 'exp_type' — construction must have "
                "failed before it was set.")
        return getattr(self.exp_type, item)

    def save_dataframe(self, df, name):
        '''
        save the dataframe as a csv file
        :param df: the dataframe
        :param name: the name for the csv file
        '''
        df.to_csv(os.path.join(self.root, name + '.csv'), index=False)
