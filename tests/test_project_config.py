'''
Tests for the config layer: save/load round-trips, the kwargs policy, and
the checkpoint-state fixes.
'''
import json
import os
import warnings

import pytest

from project_team.project_config import (
    project_config, is_primitive, is_Primitive, CONFIG_FILE_SUFFIX)
from project_team.io_project.IO_config import io_config
from project_team.io_project.Managers._TrainDeploy import \
    io_traindeploy_config
from project_team.io_project.Managers._Kfold import io_kfold_config
from project_team.io_project.Managers._HyperParameterTuning import \
    io_hptuning_config
from project_team.dt_project.DataProcessors.Image_Processor import \
    Image_Processor_config
from project_team.dt_project.DataProcessors.Text_Processor import \
    Text_Processor_config
from project_team.ml_project.Practitioners.PT_Practitioner import \
    PTPractitioner_config
from project_team.ml_project.Practitioners.PTClassification_Practitioner \
    import PTClassification_Practitioner_config
from project_team.ml_project.Practitioners.ClassificationEval_Practitioner \
    import ClassificationEval_Practitioner_config
from project_team.ml_project.models.MNIST_CNN import MNIST_CNN_config
from project_team.ml_project.models.PTRegressionModel import \
    PTRegression_config


def test_save_pretrained_writes_classname_json(tmp_path):
    cfg = io_config(X='col_a', y='col_b', project_folder=str(tmp_path))
    cfg.save_pretrained(str(tmp_path))
    expected = tmp_path / ('io_config' + CONFIG_FILE_SUFFIX)
    assert expected.is_file()
    saved = json.loads(expected.read_text())
    assert saved['X'] == 'col_a'
    assert saved['config_type'] == 'IO'


def test_config_roundtrip_preserves_every_field(tmp_path):
    # regression: from_pretrained used to look for config.json (never
    # written) and silently return an all-defaults config
    cfg = PTClassification_Practitioner_config(
        batch_size=7, n_steps=99, lr=0.123, loss_type='CE',
        affine_aug=False, add_Gnoise=False)
    cfg.save_pretrained(str(tmp_path))
    reloaded = PTClassification_Practitioner_config.from_pretrained(
        str(tmp_path))
    assert reloaded.batch_size == 7
    assert reloaded.n_steps == 99
    assert reloaded.lr == 0.123
    assert reloaded.loss_type == 'CE'
    assert reloaded.affine_aug is False


def test_subclass_config_type_is_preserved(tmp_path):
    # regression NEW-1: the child's config_type used to be silently
    # replaced by the parent's 'ML_PTPractitioner'
    cfg = PTClassification_Practitioner_config()
    assert cfg.config_type == 'ML_PTClassificationPractitioner'
    cfg.save_pretrained(str(tmp_path))
    reloaded = PTClassification_Practitioner_config.from_pretrained(
        str(tmp_path))
    assert reloaded.config_type == 'ML_PTClassificationPractitioner'


def test_save_pretrained_works_for_configs_with_required_params(tmp_path):
    # regression NEW-2: use_diff=True constructed self.__class__() and
    # crashed for any config with a required parameter
    cfg = ClassificationEval_Practitioner_config(classes=[0, 1, 2])
    cfg.save_pretrained(str(tmp_path))
    reloaded = ClassificationEval_Practitioner_config.from_pretrained(
        str(tmp_path))
    assert reloaded.classes == [0, 1, 2]


def test_two_configs_same_directory_no_collision(tmp_path):
    io_cfg = io_config(X='a')
    img_cfg = Image_Processor_config()
    io_cfg.save_pretrained(str(tmp_path))
    img_cfg.save_pretrained(str(tmp_path))
    assert (tmp_path / 'io_config.json').is_file()
    assert (tmp_path / 'Image_Processor_config.json').is_file()


def test_io_config_roundtrip_with_extra_kwargs(tmp_path):
    # regression 1.4: io_config used to drop **kwargs entirely, and naive
    # forwarding collided on the config_type key
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        cfg = io_config(X='a', extra_field=17)
    assert cfg.extra_field == 17
    cfg.save_pretrained(str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        reloaded = io_config.from_pretrained(str(tmp_path))
    assert reloaded.extra_field == 17
    assert reloaded.config_type == 'IO'


def test_unrecognized_kwarg_warns_with_suggestion():
    with pytest.warns(UserWarning, match='regressor_input'):
        PTRegression_config(regresser_input=10)


def test_dt_config_forwards_kwargs(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        cfg = Image_Processor_config(custom_note='hello')
    assert cfg.custom_note == 'hello'
    cfg.save_pretrained(str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        reloaded = Image_Processor_config.from_pretrained(str(tmp_path))
    assert reloaded.custom_note == 'hello'


def test_checkpoint_resume_restores_trained_steps(tmp_path):
    # regression NEW-4: state attrs used to be reset after super().__init__
    # clobbered anything restored from json
    cfg = PTPractitioner_config(n_steps=100, affine_aug=False)
    cfg.trained_steps = 42
    cfg.best_vl_loss = 0.5
    cfg.best_vl_step = 40
    cfg.save_pretrained(str(tmp_path))
    reloaded = PTPractitioner_config.from_pretrained(str(tmp_path))
    assert reloaded.trained_steps == 42
    assert reloaded.best_vl_loss == 0.5
    assert reloaded.best_vl_step == 40


def test_best_vl_loss_default_is_json_safe(tmp_path):
    # regression NEW-4: np.inf serialized as bare Infinity (invalid json)
    cfg = PTPractitioner_config(affine_aug=False)
    assert cfg.best_vl_loss is None
    cfg.save_pretrained(str(tmp_path))
    # strict json parse must succeed
    text = (tmp_path / 'PTPractitioner_config.json').read_text()
    json.loads(text, parse_constant=lambda c: pytest.fail(
        'non-standard json constant ' + c + ' in saved config'))


def test_hptuning_iteration_roundtrip(tmp_path):
    cfg = io_hptuning_config(iteration=3)
    cfg.save_pretrained(str(tmp_path))
    reloaded = io_hptuning_config.from_pretrained(str(tmp_path))
    assert reloaded.iteration == 3
    assert reloaded.iterations == 1  # unrelated search budget untouched


def test_mnist_cnn_config_roundtrip(tmp_path):
    # regression NEW-3: numpy_shape was stored as input_shape, so a custom
    # shape silently reverted to the default after a save/load
    cfg = MNIST_CNN_config(numpy_shape=(32, 32), kernel=5)
    cfg.save_pretrained(str(tmp_path))
    reloaded = MNIST_CNN_config.from_pretrained(str(tmp_path))
    assert tuple(reloaded.numpy_shape) == (32, 32)
    assert tuple(reloaded.input_shape) == (32, 32)  # deprecated alias
    assert reloaded.kernel == 5


def test_tuples_reload_as_lists(tmp_path):
    # pinned as documented behavior: json has no tuple type
    cfg = Image_Processor_config(numpy_shape=(16, 16))
    cfg.save_pretrained(str(tmp_path))
    reloaded = Image_Processor_config.from_pretrained(str(tmp_path))
    assert reloaded.numpy_shape == [16, 16]


def test_two_configs_same_session_get_distinct_experiment_names():
    # regression 1.6: the timestamp default was baked in at import time
    import time
    a = io_config()
    time.sleep(1.1)
    b = io_config()
    assert a.experiment_name != b.experiment_name


def test_test_size_validation():
    with pytest.raises(ValueError, match='test_size'):
        io_config(test_size=1.5)
    with pytest.raises(ValueError, match='validation_size'):
        io_config(validation_size=-0.1)
    io_config(test_size=0.0)  # 0.0 is the documented skip sentinel


def test_config_type_must_be_string():
    with pytest.raises(TypeError, match='config_type'):
        project_config(config_type=None)


def test_vl_interval_never_zero():
    # regression 5.2: n_steps=1, n_saves=10 used to produce vl_interval=0
    # and a ZeroDivisionError two steps into training
    cfg = PTPractitioner_config(n_steps=1, n_saves=10, affine_aug=False)
    assert cfg.vl_interval >= 1
    default_cfg = PTPractitioner_config(affine_aug=False)
    assert default_cfg.n_steps is None  # regression 7.9


def test_text_processor_config_requires_model():
    with pytest.raises(ValueError, match='model'):
        Text_Processor_config('BertTokenizerFast')


def test_is_primitive():
    assert is_primitive(1)
    assert is_primitive('a')
    assert is_primitive(None)
    assert is_primitive([1, 'b', None])
    assert is_primitive({'k': [1, 2]})
    assert is_primitive((1, 2))
    import numpy as np
    assert not is_primitive(np.zeros(3))
    assert not is_primitive(object())


def test_is_Primitive_alias_warns():
    with pytest.warns(DeprecationWarning):
        assert is_Primitive(1)


def test_traindeploy_and_kfold_config_roundtrip(tmp_path):
    for cls, kw in ((io_traindeploy_config, dict(X='a', test_size=0.2)),
                    (io_kfold_config, dict(k_folds=3, X='a'))):
        folder = tmp_path / cls.__name__
        cfg = cls(**kw)
        cfg.save_pretrained(str(folder))
        reloaded = cls.from_pretrained(str(folder))
        for key, value in kw.items():
            assert getattr(reloaded, key) == value, (cls.__name__, key)
