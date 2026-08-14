'''
Unit tests for dt_processing transforms: dict-in/dict-out contracts and the
correctness fixes (pad axis/fill, one-hot alignment, ndarray handling).
'''
import numpy as np
import pytest
import torch
from PIL import Image

from project_team.dt_project.dt_processing import (
    OpenImage_file, Resample_Image_shape, Cast_numpy, ToNumpy, ImageToNumpy,
    NumpyToImage, Numpy_resize, Pad_to_Size_numpy, Reverse_Pad_to_Size_numpy,
    Add_Channel, ToTensor, OneHotEncode, OneHotEncode_Seg,
    MnStdNormalize_Numpy, MxMnNormalize_Numpy, Clip_Numpy,
    AffineAugmentation, Translate_3DNumpy, Rotate_3DNumpy, Scale_3DNumpy,
    AddGaussianNoise, AddGaussainNoise)
from project_team.dt_project.dt_processing.img_shape import resolve_dtype
from project_team.dt_project.dt_processing.functional import \
    make_all_tensors_same_size


def test_open_image_from_path(tiny_image_dataset):
    path = tiny_image_dataset['img_data'].iloc[0]
    out = OpenImage_file()({'X': path})
    assert isinstance(out['X'], list)
    assert isinstance(out['X'][0], Image.Image)
    assert out['X_location'] == [path]


def test_open_image_accepts_ndarray():
    # regression 4.1: type(img)!=np.array compared against a function and
    # was always True, so ndarrays were passed to Image.open
    arr = np.zeros((4, 4), dtype=np.uint8)
    out = OpenImage_file()({'X': [arr]})
    assert out['X'][0] is arr
    assert out['X_location'] == ['Image Given Unknown Location']


def test_open_image_bad_cell_message():
    # regression 4.2: raw eval() on the cell replaced with literal_eval
    with pytest.raises(ValueError, match='not an existing file path'):
        OpenImage_file()({'X': 'os.system("echo pwned")'})


def test_resolve_dtype_registry():
    # regression 3.3: the default 'numpy.float32' string was a NameError
    assert resolve_dtype('numpy.float32') is np.float32
    assert resolve_dtype('np.float32') is np.float32
    assert resolve_dtype('float32') is np.float32
    assert resolve_dtype(np.uint8) is np.uint8
    with pytest.raises(ValueError, match='float32'):
        resolve_dtype('np.floaty32')


def test_resample_applies_output_dtype(tiny_image_dataset):
    # regression 4.8: output_dtype was parsed but never applied
    path = tiny_image_dataset['img_data'].iloc[0]
    sample = OpenImage_file()({'X': path})
    sample = Resample_Image_shape(new_size=(6, 6),
                                  output_dtype='np.uint8')(sample)
    sample = ImageToNumpy()(sample)
    assert sample['X'][0].shape == (6, 6)
    assert sample['X'][0].dtype == np.uint8
    assert 'resize_meta_data' in sample


def test_image_numpy_roundtrip():
    arr = np.arange(16, dtype=np.uint8).reshape(4, 4)
    to_img = NumpyToImage()({'X': [arr]})
    back = ImageToNumpy()(to_img)
    np.testing.assert_array_equal(back['X'][0], arr)


def test_cast_numpy_variants():
    caster = Cast_numpy(data_type=np.float32)
    assert caster({'X': np.zeros(3, dtype=np.uint8)})['X'].dtype == np.float32
    assert caster({'X': [np.zeros(2, dtype=int)]})['X'][0].dtype == np.float32
    assert caster({'X': 3})['X'].dtype == np.float32
    assert caster({'X': {'a': np.zeros(2)}})['X']['a'].dtype == np.float32


def test_pad_uses_fill_value():
    # regression 4.5: the canvas was filled with the corner voxel, not
    # fill_value
    img = np.full((2, 2), 9.0)
    out = Pad_to_Size_numpy(shape=(4, 4), fill_value=7.0,
                            img_centering=(None, None))({'X': [img]})
    padded = out['X'][0]
    assert padded.shape == (4, 4)
    assert padded[0, 0] == 7.0
    assert padded[1, 1] == 9.0


def test_get_bounds_of_axis_per_axis():
    # regression 4.4: both returns used self.shape[0] regardless of axis
    pad = Pad_to_Size_numpy(shape=(2, 3, 4), img_centering=(True, True, True))
    img = np.zeros((6, 6, 6))
    img[2:4, 2:5, 1:5] = 1.0
    for axis in range(3):
        lo, hi = pad.get_bounds_of_axis(axis, img)
        assert hi - lo == pad.shape[axis]


def test_reverse_pad_2d():
    # regression 4.6: the reverse hard-coded 3 axes and crashed on 2D
    img = np.arange(6.0).reshape(2, 3)
    sample = Pad_to_Size_numpy(shape=(4, 5), fill_value=0.0,
                               img_centering=(None, None))({'X': [img]})
    sample['pred_y'] = sample['X']
    reversed_ = Reverse_Pad_to_Size_numpy()(sample)
    np.testing.assert_array_equal(reversed_['pred_y'][0], img)


def test_pad_reverse_3d_roundtrip():
    img = np.random.rand(3, 4, 5)
    sample = Pad_to_Size_numpy(shape=(6, 6, 6), fill_value=0.0,
                               img_centering=(None, None, None))({'X': [img]})
    sample['pred_y'] = sample['X']
    reversed_ = Reverse_Pad_to_Size_numpy()(sample)
    np.testing.assert_allclose(reversed_['pred_y'][0], img)


def test_onehotencode_accepts_numpy_int():
    # regression 4.7 (half of the silent-empty-dataset pair): pandas labels
    # arrive as np.int64 and used to fail the type()==int assert
    out = OneHotEncode(max_class=3)({'y': np.int64(1)})
    assert out['y'] == [0, 1.0, 0]
    assert out['y_original'] == 1
    with pytest.raises(TypeError, match='integer'):
        OneHotEncode(max_class=3)({'y': 'one'})


def test_onehot_seg_missing_class_channel_alignment():
    # regression 4.3: channels were assigned by enumeration order of the
    # values present, so an absent class shifted every later channel
    seg = np.array([[[0, 2], [2, 0]]])  # class 1 absent
    out = OneHotEncode_Seg(max_class=3)({'y': seg})
    one_hot = out['y']
    np.testing.assert_array_equal(one_hot[0], (seg[0] == 0).astype(float))
    np.testing.assert_array_equal(one_hot[1], np.zeros((2, 2)))
    np.testing.assert_array_equal(one_hot[2], (seg[0] == 2).astype(float))


def test_totensor_and_add_channel():
    out = Add_Channel()({'X': [np.zeros((2, 2))]})
    assert out['X'].shape == (1, 2, 2)
    tensored = ToTensor()({'X': out['X']})
    assert isinstance(tensored['X'], torch.Tensor)
    d = ToTensor()({'X': {'a': np.ones(2)}})
    assert isinstance(d['X']['a'], torch.Tensor)


def test_mnstd_normalize_exact():
    img = np.ones((1, 2, 2)) * 10.0
    out = MnStdNormalize_Numpy(norm=[(4.0, 2.0)])({'X': img})
    np.testing.assert_allclose(out['X'][0], 3.0)
    with pytest.raises(Exception, match='normalization factors'):
        MnStdNormalize_Numpy(norm=[(0, 1), (0, 1)])({'X': img})
    passthrough = MnStdNormalize_Numpy(norm=[None])({'X': img})
    np.testing.assert_allclose(passthrough['X'][0], 10.0)


def test_clip_numpy():
    img = [np.array([[-5.0, 0.5], [2.0, 7.0]])]
    out = Clip_Numpy(max_min=[(0.0, 1.0)])({'X': img})
    assert out['X'][0].min() >= 0.0
    assert out['X'][0].max() <= 1.0


def test_percentile_guard_all_zero_channel():
    # regression 4.11: the bare excepts around np.percentile are narrowed;
    # an all-zero channel still normalizes without raising
    img = [np.zeros((4, 4)), np.random.rand(4, 4) + 1.0]
    out = MxMnNormalize_Numpy(mxmn=[(0., 1.), (0., 1.)],
                              percentiles=[(1, 99), (1, 99)])({'X': img})
    assert len(out['X']) == 2


def test_affine_augmentation_meta_data_reuse():
    img = [np.random.rand(6, 6, 6)]
    aug = AffineAugmentation(shift=1, rot=2, scale=0.05, order=0)
    out = aug({'X': [i.copy() for i in img]})
    assert out['X'][0].shape == (6, 6, 6)
    assert 'augmentation_meta_data' in out
    # reusing recorded meta_data reproduces the same augmentation
    again = aug({'X': [i.copy() for i in img],
                 'augmentation_meta_data': out['augmentation_meta_data']})
    np.testing.assert_allclose(again['X'][0], out['X'][0])


def test_translate_rotate_scale_shape_preserving():
    img = [np.random.rand(5, 5, 5)]
    for tr in (Translate_3DNumpy(shifts=(1, 0, 0)),
               Rotate_3DNumpy(angles=(10, 0, 0)),
               Scale_3DNumpy(scale=1)):
        out = tr({'X': [i.copy() for i in img]})
        assert out['X'][0].shape == (5, 5, 5)


def test_add_gaussian_noise_and_alias():
    img = [np.zeros((4, 4))]
    out = AddGaussianNoise(std=1.0)({'X': [i.copy() for i in img]})
    assert not np.allclose(out['X'][0], 0.0)
    # the old misspelled name stays importable as an alias
    assert AddGaussainNoise is AddGaussianNoise


def test_numpy_resize():
    out = Numpy_resize(output_shape=(3, 3))({'X': [np.ones((6, 6))]})
    assert out['X'][0].shape == (3, 3)


def test_get_reciprocal_and_deprecated_alias():
    pad = Pad_to_Size_numpy(shape=(4, 4))
    assert isinstance(pad.get_reciprocal(), Reverse_Pad_to_Size_numpy)
    with pytest.warns(DeprecationWarning):
        pad.get_reciprical()
    with pytest.raises(NotImplementedError):
        Resample_Image_shape().get_reciprocal()


def test_make_all_tensors_same_size():
    batch = [{'X': torch.tensor([1., 2.]), 'name': 'a'},
             {'X': torch.tensor([3., 4., 5.]), 'name': 'b'}]
    out = make_all_tensors_same_size(batch)
    assert out['X'].shape == (2, 3)
    assert out['X'][0, 2] == 0.0  # padded
    assert out['name'] == ['a', 'b']
