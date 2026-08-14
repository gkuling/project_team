'''
Tests for the dataset silo mechanics, preload error handling, dataset
fingerprint, and the processor classes.
'''
import numpy as np
import pandas as pd
import pytest
from torchvision import transforms as pt_transforms

from project_team.dt_project.datasets import (
    Project_Team_Dataset, Images_Dataset, Dataset_Fingerprint)
from project_team.dt_project.DataProcessors.Image_Processor import (
    Image_Processor, Image_Processor_config)
from project_team.dt_project.dt_processing import (
    OpenImage_file, Resample_Image_shape, ImageToNumpy, OneHotEncode,
    Pad_to_Size_numpy)


def build_processor(**cfg_kw):
    cfg = Image_Processor_config(numpy_shape=(8, 8), **cfg_kw)
    return Image_Processor(cfg)


def test_default_image_processor_constructs():
    # regression 3.3: the default silo_dtype='numpy.float32' used to raise
    # a NameError inside eval()
    proc = Image_Processor()
    assert proc.pre_transforms is not None


def test_preload_and_on_the_fly_parity(tiny_image_dataset):
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    trs = pt_transforms.Compose([OpenImage_file(),
                                 Resample_Image_shape(new_size=(8, 8),
                                                      output_dtype='np.uint8'),
                                 ImageToNumpy()])
    pre = Images_Dataset(df, preload_transforms=trs, preload_data=True)
    pre.perform_preload()
    fly = Images_Dataset(df, preload_transforms=trs, preload_data=False)
    np.testing.assert_array_equal(pre[0]['X'][0], fly[0]['X'][0])
    assert len(pre) == len(fly) == len(df)


def test_silo_mechanics(tiny_image_dataset):
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    trs = pt_transforms.Compose([OpenImage_file(),
                                 Resample_Image_shape(new_size=(8, 8),
                                                      output_dtype='np.uint8'),
                                 ImageToNumpy()])
    dset = Images_Dataset(df, preload_transforms=trs, preload_data=True)
    dset.perform_preload()
    # heavy objects live in the silo; rows hold sentinels
    assert len(dset.files_silo) > 0
    raw_row = dset.dfiles[0]
    assert isinstance(raw_row['X'], str) and raw_row['X'].startswith(
        'save_name_')
    assert raw_row['_silo_fields'] == ['X']
    # __getitem__ substitutes them back and strips the bookkeeping key
    example = dset[0]
    assert isinstance(example['X'], list)
    assert '_silo_fields' not in example


def test_silo_sentinel_not_confused_by_data(tiny_image_dataset):
    # regression 3.7: a genuine text field whose value looks like a
    # sentinel used to be silently replaced with silo contents
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    df['note'] = 'save_name_0'
    trs = pt_transforms.Compose([OpenImage_file(),
                                 Resample_Image_shape(new_size=(8, 8),
                                                      output_dtype='np.uint8'),
                                 ImageToNumpy()])
    dset = Images_Dataset(df, preload_transforms=trs, preload_data=True)
    dset.perform_preload()
    assert dset[0]['note'] == 'save_name_0'


def test_preload_all_failed_raises(tiny_image_dataset):
    # regression 3.1: a dataset where every example fails used to become
    # silently empty
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    df['X'] = '/nonexistent/path.png'

    trs = pt_transforms.Compose([OpenImage_file()])
    dset = Images_Dataset(df, preload_transforms=trs, preload_data=True)
    with pytest.raises(RuntimeError, match='debug_pretransform'):
        dset.perform_preload()
    # and with debug_pretransform=True the original exception surfaces
    dset_dbg = Images_Dataset(df, preload_transforms=trs, preload_data=True,
                              debug_pretransform=True)
    with pytest.raises(ValueError):
        dset_dbg.perform_preload()


def test_preload_partial_failure_keeps_the_rest(tiny_image_dataset, capsys):
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    df.loc[0, 'X'] = '/nonexistent/path.png'
    trs = pt_transforms.Compose([OpenImage_file(),
                                 Resample_Image_shape(new_size=(8, 8),
                                                      output_dtype='np.uint8'),
                                 ImageToNumpy()])
    dset = Images_Dataset(df, preload_transforms=trs, preload_data=True)
    dset.perform_preload()
    assert len(dset) == len(df) - 1
    # regression 7.1: the warning now says WHY the example failed
    assert 'Reason' in capsys.readouterr().out


def test_one_hot_preload_keeps_dataset(tiny_image_dataset):
    # regression CONFLICT-4: one_hot_encode=True with pandas int64 labels
    # used to silently drop every example
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    proc = build_processor(one_hot_encode=True, max_classes=3,
                           silo_dtype='np.uint8')
    proc.set_training_data(df)
    assert len(proc.tr_dset) == len(df)
    assert proc.tr_dset[0]['y'] in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])


def test_pad_shape_applied_after_roundtrip(tmp_path, tiny_image_dataset):
    # regression 3.2 + P1c: 'is tuple' was always False, and a reloaded
    # config carries a list rather than a tuple
    cfg = Image_Processor_config(numpy_shape=(8, 8), pad_shape=(12, 12),
                                 silo_dtype='np.uint8')
    cfg.save_pretrained(str(tmp_path))
    reloaded = Image_Processor_config.from_pretrained(str(tmp_path))
    proc = Image_Processor(reloaded)
    assert any(isinstance(t, Pad_to_Size_numpy)
               for t in proc.pre_transforms.transforms)
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    proc.set_training_data(df)
    assert proc.tr_dset[0]['X'][0].shape == (12, 12)


def test_inference_strips_y_transforms(tiny_image_dataset):
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    proc = build_processor(one_hot_encode=True, max_classes=3,
                           silo_dtype='np.uint8')
    proc.set_inference_data(df, pre_process_y=False)
    used = proc.if_dset.preload_transforms.transforms
    assert all(t.field_oi == 'X' for t in used)
    # y stays a raw label because OneHotEncode(field_oi='y') was stripped
    assert isinstance(proc.if_dset[0]['y'], (int, np.integer))


def test_set_dataset_missing_csv(tmp_path):
    # regression 3.6: raised FileExistsError for a MISSING file
    proc = build_processor(silo_dtype='np.uint8')
    with pytest.raises(FileNotFoundError):
        proc.set_training_data(str(tmp_path))


def test_dataset_filter(tiny_image_dataset):
    df = tiny_image_dataset.rename(columns={'img_data': 'X', 'label': 'y'})
    trs = pt_transforms.Compose([OpenImage_file(),
                                 Resample_Image_shape(new_size=(8, 8),
                                                      output_dtype='np.uint8'),
                                 ImageToNumpy()])
    dset = Images_Dataset(df, preload_transforms=trs, preload_data=True)
    dset.perform_preload()
    dset.set_filter(lambda ex: ex['y'] == 0)
    assert len(dset) == len(df) // 3
    dset.clear_filter()
    assert len(dset) == len(df)


def test_fingerprint_single_array_and_channels():
    fp = Dataset_Fingerprint()
    fp.update('X', [np.ones((4, 4)), np.zeros((4, 4))])
    means = fp.get_mean_std('X')
    assert means[0][0] == pytest.approx(1.0)
    assert means[1][0] == pytest.approx(0.0)
    mins = fp.get_min_max('X')
    assert mins[0] == (1.0, 1.0)
    pct = fp.get_percentiles('X', 0.5, 99.5)
    assert len(pct) == 2
    # regression 3.10: a bare ndarray used to be unreachable in the
    # non-iterable branch; a dict must not be treated as a batch
    assert fp.is_iterable(np.zeros((2, 2)))
    assert not fp.is_iterable({'mean': 1.0})
    assert not fp.is_iterable('save_name_0')


def test_fingerprint_deprecated_alias_warns():
    fp = Dataset_Fingerprint()
    with pytest.warns(DeprecationWarning):
        fp.isititerable([1])
