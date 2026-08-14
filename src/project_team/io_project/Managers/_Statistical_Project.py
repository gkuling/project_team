import os
import pandas as pd
from sklearn.model_selection import train_test_split


class LabelNotFoundError(KeyError):
    '''
    Raised when a configured X or y column is missing from the data.
    Caught by managers that can tolerate the miss (for example inference
    data without ground-truth labels).
    '''


class _Statistical_Project():
    '''
    Parent class of a statistical project.
    Used to collect shared function used by the following types of projects:
    - train (val) test split/ deployment
    - k fold validation
    - hyper parameter grid searching
    '''
    def __init__(self,
                 config):
        '''
        :param config: an io_config
        '''
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

    def _remap_column(self, df, col_spec, target_name):
        '''
        rename (or assemble, for a list of columns) the configured column
        into the canonical target name used by all project team members
        :param df: dataframe to remap
        :param col_spec: the configured column name (str) or list of names
        :param target_name: the canonical name, 'X' or 'y'
        :return: the remapped dataframe
        '''
        if isinstance(col_spec, list):
            missing = [c for c in col_spec if c not in df.columns]
            if missing:
                raise LabelNotFoundError(
                    'The ' + target_name + ' columns ' + str(missing) +
                    ' were not found in the data. Available columns: ' +
                    str(list(df.columns)))
            df[target_name] = df[col_spec].values.tolist()
            return df
        if isinstance(col_spec, str) and col_spec in df.columns:
            return df.rename(columns={col_spec: target_name})
        raise LabelNotFoundError(
            'The configured ' + target_name + ' column ' + repr(col_spec) +
            ' was not found in the data. Available columns: ' +
            str(list(df.columns)))

    def remap_X(self, df):
        '''
        the function will change the column name of the X variable in the dataframe to 'X' so it is generic and can be
        used consistently in project team members
        :param df: dataframe you wish to change the X column to 'X'
        :return: the remapped dataframe
        '''
        return self._remap_column(df, self.config.X, 'X')

    def remap_y(self, df):
        '''
        the function will change the column name of the y variable in the dataframe to 'y' so it is generic and can be
        used consistently in project team members
        :param df: dataframe you wish to change the y column to 'y'
        :return: the remapped dataframe
        '''
        return self._remap_column(df, self.config.y, 'y')

    def stratified_data_split(self, list_examples, stratification):
        '''
        stratified split of data for training, val, and test, given the portions that are declared in the config.
        :param list_examples: list of example labels from the 'group_data_by' config setting
        :param stratification: a list of corresponding values that stratification is based on
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
        :param sessions: list of individual unique identifiers based on
        group_data_by
        :return: a list of the stratification quality based on stratify_by
        '''
        tmp_strtfy_by = self.config.stratify_by
        if tmp_strtfy_by == self.config.y:
            tmp_strtfy_by = 'y'
        if not isinstance(tmp_strtfy_by, str):
            raise TypeError('stratify_by must be a string column name, '
                            'got ' + repr(tmp_strtfy_by))
        if tmp_strtfy_by not in data.columns:
            raise ValueError('stratify_by column ' + repr(tmp_strtfy_by) +
                             ' is not in the data. Available columns: ' +
                             str(list(data.columns)))
        try:
            return data.iloc[
                [getattr(data, self.config.group_data_by).eq(x).idxmax()
                 for x in sessions]
            ][tmp_strtfy_by].to_list()
        except IndexError as e:
            raise IndexError(
                'Using row index to group_data_by requires that the index '
                'of the dataframe be 0 to n. Use df.reset_index() to avoid '
                'this IndexError. ') from e

    def load_rename_group_data(self):
        '''
        a function to load the data csv file, rename x and y and acquire the
        amount of examples in the dataframe
        :return: data_file, session_list
        '''
        # load the dataframe
        if isinstance(self.config.data_csv_location, pd.DataFrame):
            data_file = self.config.data_csv_location
        elif os.path.exists(self.config.data_csv_location):
            data_file = pd.read_csv(self.config.data_csv_location,
                                    na_filter=False)
        else:
            raise Exception('The data_csv_location given is not a pandas '
                            'dataframe or a file that exists. ')

        # designate X and y
        data_file = self.remap_X(data_file)
        data_file = self.remap_y(data_file)

        # determine how to group data by and find the grouped data examples
        data_file, session_list = self.get_session_list(data_file)
        return data_file, session_list