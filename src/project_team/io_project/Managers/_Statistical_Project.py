import os
import pandas as pd
from sklearn.model_selection import train_test_split

class _Statistical_Project():
    '''
    Parent class of a statistical project.
    Used to collect shared function used by the following types of projects:
    - train (val) test split/ deployment
    - k fold validation
    - hyper parameter grid seraching
    '''
    def __init__(self,
                 config):
        '''
        :param config: an io_config
        '''
        self.config = config

        self.config = config
        self.root = os.path.join(config.project_folder, config.experiment_name)
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def set_root(self, lcl):
        '''
        change the root directory of the experiment
        :param lcl: directory where to save the experiment
        :return:
        '''
        self.root = lcl
        if not os.path.exists(self.root):
            os.makedirs(self.root)

    def remap_X(self, df):
        '''
        the function will change the column name of the X variable in the dataframe to 'X' so it is generic and can be
        used consistently in project team members
        :param df: dataframe you wish to change the X column to 'X'
        :return:
        '''
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
        '''
        the function will change the column name of the y variable in the dataframe to 'y' so it is generic and can be
        used consistently in project team members
        :param df: dataframe you wish to change the X column to 'y'
        :return:
        '''
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
        '''
        stratified split of data for training, val, and test, given the portions that are declared in the config.
        :param list_examples: list of example labels from the 'group_data_by' config setting
        :param stratification: a list of coresponding values that stratification is based on
        :return: train_list, val_list, and test_list of the group_data_by characteristic
        '''
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
        '''
        split of data for training, val, and test, given the portions that are declared in the config.
        :param list_examples: list of example labels from the 'group_data_by' config setting
        :return: train_list, val_list, and test_list of the group_data_by characteristic
        '''
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

    def get_session_list(self, data):
        '''
        the function will find the individual items that are grouped in the
        data given
        :param data: dataset
        :return: data, and a list of individual unique identifiers based on
        group_data_by
        '''
        if self.config.group_data_by in data.columns:
            pass
        else:
            self.config.group_data_by = 'index_column'
            data[self.config.group_data_by] = data.index
        return data, list(set(
            data[self.config.group_data_by].values.tolist()
        ))

    def stratify_data(self, data, sessions):
        '''
        the function will determine the stratification quality the data has
        given the sessions and stratify_by
        :param data: dataset
        :param sessions: list of individual unqiue identifiers based on
        group_data_by
        :return: a list of the stratification quality based on stratify_by
        '''
        tmp_strtfy_by = self.config.stratify_by
        if tmp_strtfy_by == self.config.y:
            tmp_strtfy_by = 'y'
        assert type(tmp_strtfy_by) == str, \
            "Stratify by value must be string."
        assert tmp_strtfy_by in data.columns, \
            "Stratify by value must be a column in your dataset."
        return data.iloc[
            [getattr(data, self.config.group_data_by).eq(x).idxmax()
             for x in sessions]
        ][tmp_strtfy_by].to_list()

    def load_rename_group_data(self):
        '''
        a function to load the data csv file, rename x and y and acquire the
        amount of examples in the dataframe
        :return: dt_fl: data_file, sssn_lst: session_list
        '''
        # load the dataframe
        if type(self.config.data_csv_location) == pd.DataFrame:
            dt_fl = self.config.data_csv_location
        elif os.path.exists(self.config.data_csv_location):
            dt_fl = pd.read_csv(self.config.data_csv_location, na_filter=False)
        else:
            raise Exception('The data_csv_location given is not a pandas '
                            'dataframe or a file that exists. ')

        # designate X and y
        dt_fl = self.remap_X(dt_fl)
        dt_fl = self.remap_y(dt_fl)

        # determine how to group data by and fine the grouped data examples
        dt_fl, sssn_lst = self.get_session_list(dt_fl)
        return dt_fl, sssn_lst