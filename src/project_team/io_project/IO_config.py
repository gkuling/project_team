from project_team import project_config
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

datetime_string = dt.now(

).__str__().replace(' ','_').replace(':','-').split('.')[0]

class io_config(project_config):
    def __init__(self,
                 data_csv_location=None,
                 inf_data_csv_location=None,
                 val_data_csv_location=None,
                 experiment_name= 'Experiment_on_' + datetime_string,
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
        '''
        configuration for input and output
        :param data_csv_location: location of a csv file used in the
        experiment, or a dataframe
        :param inf_data_csv_location: location of a csv file used in the
        experiment, or a dataframe
        :param val_data_csv_location: location of a csv file used in the
        experiment, or a dataframe
        :param experiment_name: given name for the experiment. default:
        "Experiment_on_[datetime]"
        :param project_folder: path to the location filees will be saved for
        the experiment. default: os.getcwd()
        :param X: column name in data csv that is the X variable
        :param X_dtype: datatype of the X variable. I would like to build a
        datatype dictionary for this
        :param y: column name in data csv that is the y variable
        :param y_dtype: datatype of the y variable. I would like to build a
        datatype dictionary for this
        :param y_domain: the domain of the label. This could be a list of the
        labels that the output index belong to. Typiclaly a list for
        classification. a mathematical domain for regression. and
        segmentation map names for the channels
        :param test_size: portion of data_csv to be used for inference. must
        be <1.0 and >0.0
        :param validation_size: portion of data_csv to be used for validation.
        must be <1.0 and >0.0
        :param group_data_by: a column value that the data can be grouped by
        for the data splitting. For example: patient number given you have
        multiple images from one patient. You don't want those images split
        between training and testing. So you can set group_data_by to the
        patient identifier column. default: None which will use the line
        index of the data_csv dataframe
        :param stratify_by: a column value to stratify the data splits by.
        Typically this can be the y label. default: None which will have
        managers not perform any stratification.
        :param r_seed: random state
        '''
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


