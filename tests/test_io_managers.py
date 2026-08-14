'''
Tests for the IO manager layer: config dispatch, data splitting, kfold and
hyperparameter-tuning orchestration, and the fail-loudly fixes.
'''
import os

import pandas as pd
import pytest
import torch

from project_team.io_project import (
    IO_Manager, Pytorch_Manager, io_traindeploy_config, io_kfold_config,
    io_hptuning_config)
from project_team.io_project.IO_config import io_config
from project_team.io_project.Managers._Statistical_Project import (
    _Statistical_Project, LabelNotFoundError)
from project_team.io_project.Managers._TrainDeploy import _TrainDeploy


def make_manager(cls, tmp_path, df, **kw):
    cfg = cls(data_csv_location=df, project_folder=str(tmp_path),
              experiment_name='exp', X='feat', y='target',
              group_data_by='group', **kw)
    return Pytorch_Manager(cfg)


def test_dispatch_to_experiment_types(tmp_path, tiny_df):
    pairs = [(io_traindeploy_config, '_TrainDeploy'),
             (io_kfold_config, '_Kfold'),
             (io_hptuning_config, '_HyperParameterTuning')]
    for cls, exp_name in pairs:
        mgr = make_manager(cls, tmp_path / exp_name, tiny_df)
        assert type(mgr.exp_type).__name__ == exp_name
        assert mgr.root == os.path.join(str(tmp_path / exp_name), 'exp')
        assert os.path.isdir(mgr.root)


def test_io_manager_rejects_wrong_config_type(tmp_path):
    # regression 2.1/2.2: an unrecognized config used to leave exp_type
    # unset and any attribute access hit a RecursionError
    with pytest.raises(TypeError, match='io_traindeploy_config'):
        IO_Manager(io_config(project_folder=str(tmp_path)))
    with pytest.raises(TypeError):
        IO_Manager('not a config at all')


def test_traindeploy_rejects_wrong_config(tmp_path):
    with pytest.raises(TypeError, match='io_traindeploy_config'):
        _TrainDeploy(io_config(project_folder=str(tmp_path)))


def test_traindeploy_default_config_is_fresh():
    # regression 2.10: the default config used to be a single shared
    # instance evaluated at class-definition time
    import time
    a = _TrainDeploy()
    time.sleep(1.1)
    b = _TrainDeploy()
    assert a.config is not b.config
    assert a.config.experiment_name != b.config.experiment_name


def test_save_dataframe(tmp_path, tiny_df):
    mgr = make_manager(io_traindeploy_config, tmp_path, tiny_df)
    mgr.save_dataframe(tiny_df, 'my_results')
    assert (tmp_path / 'exp' / 'my_results.csv').is_file()


def test_prepare_for_experiment_splits(tmp_path, tiny_df):
    mgr = make_manager(io_traindeploy_config, tmp_path, tiny_df,
                       test_size=0.2, validation_size=0.2)
    mgr.prepare_for_experiment()
    tr = pd.read_csv(os.path.join(mgr.root, 'tr_dset.csv'))
    vl = pd.read_csv(os.path.join(mgr.root, 'vl_dset.csv'))
    ts = pd.read_csv(os.path.join(mgr.root, 'if_dset.csv'))
    groups = [set(d['group']) for d in (tr, vl, ts)]
    # disjoint splits whose union covers everything
    assert groups[0] & groups[1] == set()
    assert groups[0] & groups[2] == set()
    assert groups[1] & groups[2] == set()
    assert groups[0] | groups[1] | groups[2] == set(tiny_df['group'])
    # columns were remapped to the canonical names
    assert 'X' in tr.columns and 'y' in tr.columns


def test_prepare_for_experiment_straight_train(tmp_path, tiny_df):
    mgr = make_manager(io_traindeploy_config, tmp_path, tiny_df)
    mgr.prepare_for_experiment()
    assert os.path.isfile(os.path.join(mgr.root, 'tr_dset.csv'))
    assert not os.path.isfile(os.path.join(mgr.root, 'vl_dset.csv'))
    assert not os.path.isfile(os.path.join(mgr.root, 'if_dset.csv'))


def test_stratified_split_proportions(tmp_path, tiny_df):
    mgr = make_manager(io_traindeploy_config, tmp_path, tiny_df,
                       test_size=0.3, stratify_by='strat')
    mgr.prepare_for_experiment()
    ts = pd.read_csv(os.path.join(mgr.root, 'if_dset.csv'))
    # 30 rows, 3 balanced strat classes, test_size 0.3 -> 9 rows, 3 each
    assert sorted(ts['strat'].value_counts().to_list()) == [3, 3, 3]


def test_remap_missing_column_lists_available_columns(tmp_path, tiny_df):
    # regression 2.6/7.6: the error used to repeat the file path instead of
    # naming the missing column and what is actually available
    mgr = make_manager(io_traindeploy_config, tmp_path, tiny_df)
    with pytest.raises(LabelNotFoundError, match='feat'):
        mgr.exp_type._remap_column(tiny_df.drop(columns=['feat']),
                                   'feat', 'X')
    with pytest.raises(LabelNotFoundError, match='target'):
        mgr.exp_type._remap_column(
            tiny_df.drop(columns=['target']), ['target', 'feat'], 'y')


def test_prepare_for_inference_reraises_unexpected_errors(tmp_path, tiny_df):
    # regression 2.11: the except block used to swallow every exception
    mgr = make_manager(io_traindeploy_config, tmp_path, tiny_df)
    # missing y column -> warning, not crash; if_dset still written
    mgr.prepare_for_inference(tiny_df.drop(columns=['target']))
    assert os.path.isfile(os.path.join(mgr.root, 'if_dset.csv'))
    # missing X column -> a real error must propagate
    os.remove(os.path.join(mgr.root, 'if_dset.csv'))
    with pytest.raises(LabelNotFoundError):
        mgr.prepare_for_inference(tiny_df.drop(columns=['feat']))


def test_stratify_data_nonzero_index_message(tmp_path, tiny_df):
    # regression 2.3: the friendly IndexError message now lives in
    # stratify_data itself
    shifted = tiny_df.copy()
    shifted.index = shifted.index + 5
    cfg = io_traindeploy_config(
        data_csv_location=shifted, project_folder=str(tmp_path),
        experiment_name='exp', X='feat', y='target', stratify_by='strat')
    mgr = Pytorch_Manager(cfg)
    data, sessions = mgr.get_session_list(shifted)
    with pytest.raises(IndexError, match='reset_index'):
        mgr.stratify_data(data, sessions)


def test_kfold_lifecycle(tmp_path, tiny_df):
    mgr = make_manager(io_kfold_config, tmp_path, tiny_df, k_folds=3)
    mgr.prepare_for_experiment()
    assert len(mgr.folds) == 3
    all_test = []
    for fold in mgr.folds:
        all_test.extend(fold['test'])
    # every session appears in exactly one fold's test list
    assert sorted(all_test) == sorted(set(tiny_df['group']))

    mgr.set_fold(0)
    assert mgr.root.endswith('Fold_0')
    assert os.path.isfile(os.path.join(mgr.root, 'tr_dset.csv'))
    assert os.path.isfile(os.path.join(mgr.root, 'if_dset.csv'))

    # plant results and aggregate
    for k in range(3):
        mgr.set_fold(k)
        pd.DataFrame({'Acc.': [0.5 + 0.1 * k]}).to_csv(
            os.path.join(mgr.root, 'test_result_evaluation.csv'),
            index=False)
    assert mgr.check_folds_finished() == 3
    mgr.finished_kfold_validation()
    full = pd.read_csv(os.path.join(mgr.original_root,
                                    'Full_KFold_TestResults.csv'))
    assert 'mean' in full['Fold'].astype(str).to_list()


def test_hptuning_grid_and_resume(tmp_path, tiny_df):
    mgr = make_manager(io_hptuning_config, tmp_path, tiny_df,
                       criterion='Acc.', validation_size=0.2)
    mgr.prepare_for_experiment({'lr': [0.1, 0.2], 'kernel': [3, 5]})
    assert len(mgr.parameter_configurations) == 4

    args1 = mgr.get_gridpoint_args()
    assert set(args1) == {'lr', 'kernel'}
    assert mgr.config.iteration == 1
    assert mgr.root.endswith('grid_point1')

    # regression NEW-4b: the advancing counter is re-saved, so a reloaded
    # config resumes instead of restarting from 0
    reloaded = io_hptuning_config.from_pretrained(mgr.original_root)
    assert reloaded.iteration == 1


def test_hptuning_random_search_truncates(tmp_path, tiny_df):
    mgr = make_manager(io_hptuning_config, tmp_path, tiny_df,
                       technique='RandomSearch', iterations=2,
                       criterion='Acc.')
    mgr.prepare_for_experiment({'lr': [0.1, 0.2], 'kernel': [3, 5]})
    assert len(mgr.parameter_configurations) == 2


def test_hptuning_record_and_evaluate(tmp_path, tiny_df):
    mgr = make_manager(io_hptuning_config, tmp_path, tiny_df,
                       criterion='Acc.')
    mgr.prepare_for_experiment({'lr': [0.1, 0.2]})
    for expected in (0.7, 0.9):
        mgr.get_gridpoint_args()
        pd.DataFrame({'Acc.': [expected]}).to_csv(
            os.path.join(mgr.root, 'test_result_evaluation.csv'),
            index=False)
        mgr.record_performance()
    res = pd.read_csv(os.path.join(mgr.original_root,
                                   'Experimental_Results.csv'))
    assert len(res) == 2
    assert set(res['Performance(Acc.)']) == {0.7, 0.9}


def test_evaluate_performance_list_valued_metric(tmp_path, tiny_df):
    # regression 2.7: eval() on csv cells replaced with ast.literal_eval
    mgr = make_manager(io_hptuning_config, tmp_path, tiny_df,
                       criterion='DSC', penultimate='mean')
    df = pd.DataFrame({'DSC': ['[0.5, 0.7]']})
    assert mgr.evaluate_performance(df) == pytest.approx(0.6)
    # scalar column falls back cleanly
    mgr.config.penultimate = None
    assert mgr.evaluate_performance(pd.DataFrame({'DSC': [0.25]})) == 0.25


def test_strip_module_prefix_nested_keys():
    # regression 2.5: split('module.')[1] returned the wrong fragment for
    # nested keys like module.module_list.0.weight
    state = {'module.encoder.module_list.0.weight': torch.zeros(1),
             'plain.weight': torch.ones(1)}
    stripped = Pytorch_Manager._strip_module_prefix(state)
    assert set(stripped) == {'encoder.module_list.0.weight', 'plain.weight'}


def test_model_save_and_reload_roundtrip(tmp_path, tiny_df):
    from project_team.ml_project.models.MNIST_CNN import (
        MNIST_CNN, MNIST_CNN_config)
    mgr = make_manager(io_traindeploy_config, tmp_path, tiny_df)
    model = MNIST_CNN(MNIST_CNN_config(numpy_shape=(8, 8),
                                       hidden_layer_parameters=8))

    class FakePractitioner:
        pass

    class FakeProcessor:
        pass

    prac = FakePractitioner()
    prac.model = model
    from project_team.ml_project.Practitioners.PT_Practitioner import \
        PTPractitioner_config
    prac.config = PTPractitioner_config(n_steps=10, affine_aug=False)
    prac.config.trained_steps = 7
    prac.data_processor = FakeProcessor()
    from project_team.dt_project.DataProcessors.Image_Processor import \
        Image_Processor_config
    prac.data_processor.config = Image_Processor_config(numpy_shape=(8, 8))

    mgr.set_final_model(model.state_dict())
    mgr.model_save_pretrained(prac)
    assert mgr.check_if_model_trained()

    # reload into a fresh practitioner and confirm state + config restore
    prac2 = FakePractitioner()
    prac2.model = MNIST_CNN(MNIST_CNN_config(numpy_shape=(8, 8),
                                             hidden_layer_parameters=8))
    prac2.config = PTPractitioner_config(affine_aug=False)
    prac2.data_processor = FakeProcessor()
    prac2.data_processor.config = Image_Processor_config()
    mgr.model_from_pretrained(prac2)
    for k, v in prac2.model.state_dict().items():
        assert torch.equal(v, model.state_dict()[k])
    assert prac2.config.trained_steps == 7  # resume works now
