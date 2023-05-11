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
        load specific functinality based on the type of project being used
        :param io_config_input: an io_config
        '''
        if type(io_config_input)==io_traindeploy_config:
            self.exp_type = _TrainDeploy(io_config_input)
        elif type(io_config_input)==io_kfold_config:
            self.exp_type = _Kfold(io_config_input)
        elif type(io_config_input)==io_hptuning_config:
            self.exp_type = _HyperParameterTuning(io_config_input)

    def __getattr__(self, item):
        '''
        because of the 3 way inheritance, the get attribute function had to
        be changed
        :param item: attribute to be applied
        :return: the attribute desired
        '''
        return getattr(self.exp_type, item)

    def save_dataframe(self, df, name):
        '''
        save the dataframe as a csv file
        :param df: the dataframe
        :param name: the name for the csv file
        '''
        df.to_csv(os.path.join(self.root, name + '.csv'), index=False)