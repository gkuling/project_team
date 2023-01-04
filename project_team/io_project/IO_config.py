from project_team.project_config import project_config
import os
from datetime import datetime as dt

"""
Description: This is an example of a io_config for a machien learning 
project. It will hold all neccessary parameters on what to laod for the 
experiment, what to save, and where the save space is. 

example: 

io_config_test = io_config(
    data_csv_location='/jaylabs/amartel_data2/Grey/ObjectPlay'
                      '/data.csv',
    project_folder='/jaylabs/amartel_data2/Grey/ObjectPlay'
)
io_config_test.save_pretrained()

new_config = io_config.from_pretrained('/jaylabs/amartel_data2/Grey/ObjectPlay')
"""

class io_config(project_config):
    def __init__(self,
                 data_csv_location=None,
                 inf_data_csv_location=None,
                 val_data_csv_location=None,
                 experiment_name='Experiment_on_' + dt.now().__str__().replace(' ','_').replace(':','-').split('.')[0],
                 project_folder=os.getcwd(),
                 X=None,
                 X_dtype=None,
                 y=None,
                 y_dtype=None,
                 y_domain=None,
                 test_size=0.0,
                 validation_size=0.0,
                 group_data_by=None,
                 stratify_by=None,
                 r_seed=0,
                 **kwargs):
        super(io_config, self).__init__('IO')

        self.data_csv_location = data_csv_location
        self.inf_data_csv_location = inf_data_csv_location
        self.val_data_csv_location = val_data_csv_location
        self.experiment_name = experiment_name
        self.project_folder = project_folder

        # data information
        self.X = X
        self.X_dtype = X_dtype
        self.y = y
        self.y_dtype = y_dtype
        self.y_domain = y_domain

        # validation parameters
        self.test_size = test_size
        self.validation_size = validation_size
        self.group_data_by = group_data_by
        self.stratify_by = stratify_by
        self.r_seed = r_seed


