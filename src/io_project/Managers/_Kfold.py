from src.io_project.IO_config import io_config
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from ._Statistical_Project import _Statistical_Project
import os

class io_kfold_config(io_config):
    def __init__(self, k_folds, data_csv_location, **kwargs):
        super(io_kfold_config, self).__init__(data_csv_location, **kwargs)
        self.k_folds = k_folds

class _Kfold(_Statistical_Project):
    def __init__(self, io_config_input):
        super(_Kfold, self).__init__(io_config_input)
        assert type(io_config_input)==io_kfold_config
        self.config = io_config_input
        self.root = io_config_input.project_folder + '/' + io_config_input.experiment_name
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        self.original_root = self.root

    def prepare_for_experiment(self):
        print('IO Message: Setting up data for kfold experiment')
        self.config.save_pretrained(self.root)
        data_file = pd.read_csv(self.config.data_csv_location)
        data_file = self.remap_X(data_file)
        data_file = self.remap_y(data_file)

        data_file, session_list = self.get_session_list(data_file)

        self.folds = []
        if self.config.stratify_by:
            strat = self.stratify_data(data_file, session_list)
            if self.config.validation_size>0.0:
                kfold_analyzer = StratifiedKFold(n_splits=self.config.k_folds,
                                                 shuffle=True,
                                                 random_state=self.config.r_seed)
                for train_index, test_index in kfold_analyzer.split(
                        session_list,
                        strat
                ):
                    train_index, val_index = train_test_split(
                        train_index,
                        stratify=[strat[ind] for ind in train_index],
                        test_size=self.config.validation_size,
                        random_state=self.config.r_seed
                    )
                    self.folds.append(
                        {'train': [session_list[ind] for ind in train_index],
                         'val': [session_list[ind] for ind in val_index],
                         'test': [session_list[ind] for ind in test_index]}
                    )
            else:
                kfold_analyzer = StratifiedKFold(n_splits=self.config.k_folds,
                                                 shuffle=True,
                                                 random_state=self.config.r_seed)
                for train_index, test_index in kfold_analyzer.split(
                        session_list,
                        strat
                ):
                    self.folds.append(
                        {'train': [session_list[ind] for ind in train_index],
                         'test': [session_list[ind] for ind in test_index]}
                    )
        else:
            if self.config.validation_size>0.0:
                kfold_analyzer = KFold(n_splits=self.config.k_folds,
                                       shuffle=True,
                                       random_state=self.config.r_seed)
                for train_index, test_index in kfold_analyzer.split(session_list):
                    train_index, val_index = train_test_split(
                        train_index,
                        test_size=self.config.validation_size,
                        random_state=self.config.r_seed
                    )
                    self.folds.append(
                        {'train': [session_list[ind] for ind in train_index],
                         'val': [session_list[ind] for ind in val_index],
                         'test': [session_list[ind] for ind in test_index]}
                    )

            else:
                kfold_analyzer = KFold(n_splits=self.config.k_folds,
                                       shuffle=True,
                                       random_state=self.config.r_seed)
                for train_index, test_index in kfold_analyzer.split(session_list):
                    self.folds.append(
                        {'train': [session_list[ind] for ind in train_index],
                         'test': [session_list[ind] for ind in test_index]}
                    )

    def set_fold(self, k_fold_step):
        print('IO Message: Setting up data for kfold ' + str(
            k_fold_step) + ' experiment')
        self.root = self.original_root + '/Fold_' + str(k_fold_step)
        if not os.path.exists(self.root):
            os.makedirs(self.root)
        data_file = pd.read_csv(self.config.data_csv_location)

        data_file = self.remap_X(data_file)
        data_file = self.remap_y(data_file)

        data_file, session_list = self.get_session_list(data_file)

        fold = self.folds[k_fold_step]
        train_list = fold['train']
        if 'val' in fold.keys():
            val_list = fold['val']
        else:
            val_list = None
        test_list = fold['test']

        tr_data_df = data_file[
            data_file[self.config.group_data_by].isin(train_list)

        ]
        tr_data_df.to_csv(self.root +'/tr_dset.csv', index=False)

        ts_data_df = data_file[
            data_file[self.config.group_data_by].isin(test_list)
        ]
        ts_data_df.to_csv(self.root +'/if_dset.csv', index=False)

        if val_list:
            vl_data_df = data_file[
                data_file[self.config.group_data_by].isin(val_list)
            ]
            vl_data_df.to_csv(self.root + '/vl_dset.csv', index=False)

    def finished_kfold_validation(self):
        self.root = self.original_root

        k_fold_test_res = []
        for path, subdirs, files in os.walk(self.root):
            for name in files:
                if name.endswith('test_result_evaluation.csv'):
                    k_fold_test_res.append(os.path.join(path, name))
        k_fold_test_res = pd.concat(
            [pd.read_csv(fl) for fl in k_fold_test_res]
        )

        # Want to have different types of summaries based on the output,
        # if it is a segmentation or classification/regression task
        if 'seg_map' in k_fold_test_res.columns and \
                'Subject' in k_fold_test_res.columns:
            summary = {nm:[] for nm in k_fold_test_res.columns}
            if len(np.unique(k_fold_test_res['seg_map'].to_list()).tolist())>1:
                raise NotImplementedError('Having an output of more than one '
                                          'segmentation map is not implemented. '
                                          'Changes to this class must be made. ')
            summary['seg_map'].extend(
                [np.unique(k_fold_test_res['seg_map'].to_list()).tolist()
                 for _ in range(2)]
            )
            summary['Subject'].extend(['Average', 'Std.Dev.'])

            for key in summary.keys():
                if key!='Subject' and key!='seg_map':
                    try:
                        met_values = np.array(
                            k_fold_test_res[key].apply(lambda x: eval(x)).to_list()
                        )
                        summary[key].append(met_values.mean(axis=0))
                        summary[key].append(met_values.std(axis=0))
                    except:
                        summary[key].extend(['',''])
            summary = pd.DataFrame(summary)

            output_results = pd.concat(
                [k_fold_test_res, summary],
                ignore_index=True
            )
        else:
            k_fold_test_res.reset_index(drop=True, inplace=True)
            k_fold_test_res.loc['mean'] = k_fold_test_res.mean()
            k_fold_test_res.loc['std'] = k_fold_test_res.std()
            k_fold_test_res.reset_index(drop=False, inplace=True)
            k_fold_test_res.rename(columns={'index':'Fold'}, inplace=True)
            output_results = k_fold_test_res


        output_results.to_csv(self.root + '/Full_KFold_TestResults.csv', index=False)

    def check_folds_finished(self):
        folds_finished = [int(folder.split('_')[1])
                          for folder in os.listdir(self.root)
                          if 'Fold_' in folder and
                          os.path.exists(
                              os.path.join(self.root,
                                           folder)
                              +'/test_result_evaluation.csv'
                          )]
        if len(folds_finished)==0:
            return 0
        else:
            max_fold = np.max(folds_finished)
            return max_fold + 1
