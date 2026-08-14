'''
End-to-end CPU smoke tests: config -> Pytorch_Manager -> Image_Processor ->
PTClassification_Practitioner on a tiny synthetic dataset, plus the named
training-schedule regressions. No MNIST, no downloads.
'''
import os

import numpy as np
import pytest
import torch

from project_team.io_project import Pytorch_Manager, io_traindeploy_config
from project_team.dt_project.DataProcessors.Image_Processor import (
    Image_Processor, Image_Processor_config)
from project_team.ml_project.Practitioners.PTClassification_Practitioner \
    import (PTClassification_Practitioner,
            PTClassification_Practitioner_config)
from project_team.ml_project.Practitioners.PTRegression_Practitioner import \
    PTRegression_Practitioner
from project_team.ml_project.models.MNIST_CNN import (
    MNIST_CNN, MNIST_CNN_config)
from project_team.ml_project.models.PTRegressionModel import (
    PTRegressionModel, PTRegression_config)
from project_team.dt_project.dt_processing import ToTensor


def build_pipeline(tmp_path, tiny_image_dataset, **ml_kw):
    io_cfg = io_traindeploy_config(
        data_csv_location=tiny_image_dataset,
        project_folder=str(tmp_path), experiment_name='exp',
        X='img_data', y='label', y_domain=[0, 1, 2],
        test_size=0.25, validation_size=0.25, X_dtype='Image')
    manager = Pytorch_Manager(io_cfg)
    manager.prepare_for_experiment()

    processor = Image_Processor(Image_Processor_config(
        numpy_shape=(8, 8), silo_dtype='np.uint8'))
    processor.set_training_data(manager.root)
    processor.set_validation_data(manager.root)

    model = MNIST_CNN(MNIST_CNN_config(numpy_shape=(8, 8),
                                       hidden_layer_parameters=8))
    defaults = dict(batch_size=4, n_epochs=1, n_steps=6, n_saves=2,
                    affine_aug=False, add_Gnoise=False, loss_type='CE',
                    normalization_percentiles=[(0, 255)],
                    normalization_channels=[(0.5, 0.5)])
    defaults.update(ml_kw)
    ml_cfg = PTClassification_Practitioner_config(**defaults)
    practitioner = PTClassification_Practitioner(
        model=model, io_manager=manager, data_processor=processor,
        trainer_config=ml_cfg)
    return manager, processor, practitioner


def test_full_train_save_infer(tmp_path, tiny_image_dataset):
    manager, processor, practitioner = build_pipeline(
        tmp_path, tiny_image_dataset)
    practitioner.train_model()

    # artifacts on disk
    assert os.path.isfile(os.path.join(manager.root, 'final_model.pth'))
    for cfg_file in ('PTClassification_Practitioner_config.json',
                     'Image_Processor_config.json',
                     'MNIST_CNN_config.json'):
        assert os.path.isfile(os.path.join(manager.root, cfg_file)), cfg_file
    assert os.path.isdir(os.path.join(manager.root, 'checkpoint'))
    assert manager.check_if_model_trained()
    assert practitioner.config.trained_steps == 6

    # inference end to end
    manager.prepare_for_inference()
    processor.set_inference_data(manager.root)
    practitioner.run_inference()
    results = processor.inference_results
    assert 'pred_y' in results.columns
    assert set(results['pred_y']).issubset({0, 1, 2})


def test_checkpoint_resume_continues_steps(tmp_path, tiny_image_dataset):
    # regression NEW-4 end-to-end: a resumed run must continue from the
    # saved step count instead of restarting at 0
    manager, processor, practitioner = build_pipeline(
        tmp_path, tiny_image_dataset)
    practitioner.train_model()
    assert practitioner.config.trained_steps == 6

    model2 = MNIST_CNN(MNIST_CNN_config(numpy_shape=(8, 8),
                                        hidden_layer_parameters=8))
    practitioner2 = PTClassification_Practitioner(
        model=model2, io_manager=manager, data_processor=processor,
        trainer_config=PTClassification_Practitioner_config(
            affine_aug=False))
    manager.from_model_checkpoint(practitioner2)
    # the checkpoint folder holds the last mid-training snapshot (the
    # vl_interval boundary at step 4 with this schedule), not the final
    # post-training state — resuming continues from there, not from 0
    assert practitioner2.config.trained_steps == 4


def test_steplr_sets_scheduler(tmp_path, tiny_image_dataset):
    # regression 5.3: the scheduler was assigned to a local variable and
    # every 'steplr' run silently trained with a constant learning rate
    manager, processor, practitioner = build_pipeline(
        tmp_path, tiny_image_dataset,
        lr=0.1, lr_decay='steplr', lr_decay_stepsize=1, lr_decay_gamma=0.1,
        lr_decay_step_timing='epoch', n_epochs=2, n_steps=8)
    practitioner.train_model()
    assert hasattr(practitioner, 'scheduler')
    assert practitioner.optmzr.param_groups[0]['lr'] < 0.1


def test_subclass_transforms_present_without_auto(tmp_path,
                                                  tiny_image_dataset):
    # regression 5.1: with explicit (non-'auto') normalization params the
    # child's y-transforms used to be silently dropped
    _, _, practitioner = build_pipeline(tmp_path, tiny_image_dataset)
    y_transforms = [t for t in practitioner.standard_transforms
                    if getattr(t, 'field_oi', None) == 'y']
    assert any(isinstance(t, ToTensor) for t in y_transforms)


def test_warmup_int_sets_warmup_steps(tmp_path, tiny_image_dataset):
    # regression 5.4: the int path read warmup_steps before assignment
    _, _, practitioner = build_pipeline(
        tmp_path, tiny_image_dataset, warmup=3, n_steps=10, n_epochs=None)
    practitioner.setup_steps(2)
    assert practitioner.config.warmup_steps == 3
    _, _, p2 = build_pipeline(tmp_path / 'b', tiny_image_dataset,
                              warmup=0.5, n_steps=10, n_epochs=None)
    p2.setup_steps(2)
    assert p2.config.warmup_steps == 5
    _, _, p3 = build_pipeline(tmp_path / 'c', tiny_image_dataset,
                              warmup=None, n_steps=10, n_epochs=None)
    p3.setup_steps(2)
    assert p3.config.warmup_steps == 0


def test_short_run_does_not_zerodivide(tmp_path, tiny_image_dataset):
    # regression 5.2/7.9: n_steps < n_saves used to yield vl_interval=0 and
    # a ZeroDivisionError on the second step
    _, _, practitioner = build_pipeline(
        tmp_path, tiny_image_dataset, n_steps=3, n_saves=10, n_epochs=1)
    practitioner.train_model()
    assert practitioner.config.trained_steps == 3


def test_auto_fingerprint_normalization(tmp_path, tiny_image_dataset):
    _, _, practitioner = build_pipeline(
        tmp_path, tiny_image_dataset,
        normalization_percentiles='auto_min_max',
        normalization_channels='auto')
    practitioner.train_model()
    # 'auto' resolved into concrete numbers from the dataset fingerprint
    assert not isinstance(practitioner.config.normalization_percentiles, str)
    assert not isinstance(practitioner.config.normalization_channels, str)


def test_setup_loss_functions_does_not_move_model(tmp_path,
                                                  tiny_image_dataset):
    # regression 5.11 (CPU-only assertion: the method must not blow up and
    # must set the loss function without touching the model attribute)
    _, _, practitioner = build_pipeline(tmp_path, tiny_image_dataset)
    model_before = practitioner.model
    practitioner.setup_loss_functions()
    assert practitioner.model is model_before
    assert isinstance(practitioner.loss_function, torch.nn.CrossEntropyLoss)


def test_regression_model_default_flatten_encoder():
    # regression 6.1: the constructor assert made the documented default
    # encoder='flatten' unconstructable
    model = PTRegressionModel(PTRegression_config(regressor_input=64))
    out = model(torch.zeros((2, 1, 8, 8)))
    assert out.shape == (2, 1)
    with pytest.raises(ValueError, match='custom'):
        PTRegressionModel(PTRegression_config(encoder='custom'))


def test_mnist_cnn_fc1_matches_forward_for_even_kernel():
    # regression 6.2: the fc1 size formula was wrong for even kernels
    for kernel in (3, 4, 5):
        model = MNIST_CNN(MNIST_CNN_config(numpy_shape=(28, 28),
                                           kernel=kernel,
                                           hidden_layer_parameters=8))
        out = model(torch.zeros((2, 1, 28, 28)))
        assert out.shape == (2, 10), 'kernel=' + str(kernel)


def test_regression_unknown_output_style_raises(tmp_path,
                                                tiny_image_dataset):
    # regression 5.9b: regression inference silently passed the raw array
    # through for an unrecognized output_style
    io_cfg = io_traindeploy_config(
        data_csv_location=tiny_image_dataset,
        project_folder=str(tmp_path), experiment_name='exp',
        X='img_data', y='label', test_size=0.25, X_dtype='Image')
    manager = Pytorch_Manager(io_cfg)
    manager.prepare_for_experiment()
    processor = Image_Processor(Image_Processor_config(
        numpy_shape=(8, 8), silo_dtype='np.uint8'))
    class WeirdStyleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(64, 1)
            self.config = type('C', (), {'output_style': 'not_a_style'})()

        def forward(self, x):
            return self.lin(torch.flatten(x, 1))

    practitioner = PTRegression_Practitioner(
        model=WeirdStyleModel(), io_manager=manager,
        data_processor=processor)
    manager.prepare_for_inference()
    processor.set_inference_data(manager.root)
    with pytest.raises(ValueError, match='output_style'):
        practitioner.run_inference()
