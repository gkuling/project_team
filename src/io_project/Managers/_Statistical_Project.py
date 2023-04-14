import os
import pandas as pd
from src.io_project.IO_config import io_config
from sklearn.model_selection import train_test_split

class _Statistical_Project():
    def __init__(self,
                 config):
        self.config = config

        self.config = config
        self.root = config.project_folder + '/' + config.experiment_name
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def set_root(self, lcl):
        self.root = lcl
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def remap_X(self, df):
        mapper = {}
        if type(self.config.X)==list:
            try:
                df['X'] = df[self.config.X].values.tolist()
            except Exception as e:
                if 'are in the [columns]' in str(e):
                    raise Exception(str(e) + ' data_set columns are: ' +
                                    str(df.columns))
                else:
                    raise e
        elif type(self.config.X)==str and self.config.X in df.columns:
            mapper[self.config.X] = 'X'
        else:
            raise Exception(' The X label is not in the data_csv_location '
                            'file. ' + str(self.config.data_csv_location))
        df = df.rename(columns=mapper)
        return df

    def remap_y(self, df):
        mapper = {}
        if type(self.config.y)==list:
            df['y'] = df[self.config.y].values.tolist()
        elif type(self.config.y)==str and self.config.y in df.columns:
            mapper[self.config.y] = 'y'
        else:
            raise Exception(' The y label is not in the data_csv_location '
                            'file. ' + str(self.config.data_csv_location))
        df = df.rename(columns=mapper)
        return df

    def stratified_data_split(self, list_examples, stratification):
        val_list = None
        test_list = None
        if self.config.test_size>0.0:
            ls, test_list = train_test_split(
                list(zip(list_examples,stratification)),
                stratify=stratification,
                test_size=self.config.test_size,
                random_state=self.config.r_seed
            )
            list_examples, stratification = (list(t) for t in list(zip(*ls)))
            test_list, ts_strat  = (list(t) for t in list(zip(*test_list)))
        if self.config.validation_size>0.0:
            train_list, val_list = train_test_split(
                list_examples,
                stratify=stratification,
                test_size=self.config.validation_size,
                random_state=self.config.r_seed
            )
        else:
            train_list = list_examples
        return train_list, val_list, test_list

    def data_split(self, list_examples):
        val_list = None
        test_list = None
        if self.config.test_size>0.0:
            list_examples, test_list = train_test_split(
                list_examples,
                test_size=self.config.test_size,
                random_state=self.config.r_seed
            )
        if self.config.validation_size>0.0:
            train_list, val_list = train_test_split(
                list_examples,
                test_size=self.config.validation_size,
                random_state=self.config.r_seed
            )
        else:
            train_list = list_examples
        return train_list, val_list, test_list
