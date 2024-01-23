from PIL import Image
import numpy as np

import torch
import os
from skimage.transform import resize

from . import _TensorProcessing
from copy import deepcopy
import SimpleITK as sitk

class OpenImage_file(_TensorProcessing):
    '''
    Open a PIL Image file
    '''
    def __init__(self, field_oi='X'):
        super(OpenImage_file, self).__init__()
        self.field_oi = field_oi

    def __call__(self, ipt):

        if type(ipt[self.field_oi])==str and os.path.exists(ipt[self.field_oi]):
            ipt[self.field_oi + '_location'] = [ipt[self.field_oi]]
            ipt[self.field_oi] = [Image.open(ipt[self.field_oi])]
        else:
            if type(ipt[self.field_oi])!=list:
                try:
                    ipt[self.field_oi] = eval(ipt[self.field_oi])
                except:
                    raise ValueError('The Sample Location is not a python '
                                     'object. ex. list, dict, etc.')
            ipt[self.field_oi + '_location'] = [img if
                                                type(img)!=np.array else
                                                'Image Given Unknown ' \
                                                'Location'
                                                for img in ipt[self.field_oi]]
            ipt[self.field_oi] = [Image.open(img) if
                                  type(img)!=np.array else
                                  img
                                  for img in ipt[self.field_oi]]

        return ipt

class OpenSITK_file(_TensorProcessing):
    '''
    a tensor process that opens the simple itk file at the location given
    '''
    def __init__(self, field_oi='X'):
        super(OpenSITK_file, self).__init__()
        self.field_oi = field_oi

    def __call__(self, ipt):

        if type(ipt[self.field_oi])==str and os.path.exists(ipt[self.field_oi]):
            # open the file if a location is given and not a list
            ipt[self.field_oi + '_location'] = [ipt[self.field_oi]]
            ipt[self.field_oi] = [sitk.ReadImage(ipt[self.field_oi])]
        else:
            # make sure the input values are a list
            if type(ipt[self.field_oi])!=list:
                try:
                    ipt[self.field_oi] = eval(ipt[self.field_oi])
                except:
                    raise ValueError('The Sample Location is not a python '
                                     'object. ex. list, dict, etc.')
            ipt[self.field_oi + '_location'] = [img if
                                                type(img)!=sitk.Image else
                                                'SITKImage Given Unknown ' \
                                                'Location'
                                                for img in ipt[self.field_oi]]
            ipt[self.field_oi] = [sitk.ReadImage(img) if
                                  type(img)!=sitk.Image else
                                  img
                                  for img in ipt[self.field_oi]]
            # reutrn the SimpleITK Images in a list
        return ipt

class nnUNet_SaveSITK_file(_TensorProcessing):
    def __init__(self,
                 save_folder,
                 field_oi='X',
                 save_message='proteam'):
        super(nnUNet_SaveSITK_file, self).__init__()
        self.field_oi = field_oi
        self.save_folder = save_folder
        self.save_message = save_message

    def __call__(self, sample):
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
        file_num = len(os.listdir(self.save_folder))
        save_name = os.path.join(self.save_folder,
                                     self.save_message +
                                     '_' +
                                     str(file_num).zfill(4) +
                                 ('_0000.nii.gz' if self.field_oi=='X'
                                     else '.nii.gz'))
        sitk.WriteImage(sample[self.field_oi][0],
                        save_name)
        sample[self.field_oi] = save_name
        return sample

class Resample_Image_shape(_TensorProcessing):
    '''
    Resample a PIL Image based on the dimension of the image
    '''
    def __init__(self, mode = Image.BILINEAR, new_size = (32, 32), field_oi='X',
                 output_dtype=np.uint8):
        super(Resample_Image_shape, self).__init__()
        self.field_oi= field_oi
        self.new_size = list(new_size)
        self.mode = mode
        if type(output_dtype)==str:
            try:
                output_dtype = eval(output_dtype)
            except:
                raise Exception('The datatype input is not a type that can be '
                                'put through eval()')
        self.output_dtype = output_dtype

    def get_reciprical(self, **kwargs):
        return NotImplementedError()

    def resample_function(self, image, meta_data=None):
        if not meta_data:
            orig_size = image.size
            new_size = self.new_size
            meta_data = {'orig_size': [int(v) for v in orig_size],
                         'new_size': [int(v) for v in new_size]}

        return image.resize(self.new_size, self.mode), \
               meta_data

    def __call__(self, ipt):
        res = []
        if all([sp is None for sp in self.new_size]):
            return ipt
        if 'resize_meta_data' in ipt.keys():
            res_data = ipt['resize_meta_data']
        else:
            res_data = None
        for img in ipt[self.field_oi]:

            res_img, res_data = self.resample_function(img, res_data)
            res.append(res_img)
        ipt[self.field_oi] = res
        ipt['resize_meta_data'] = res_data
        return ipt

class Reverse_Resample_Image(_TensorProcessing):
    def __init__(self, mode , field_oi='pred_y'):
        super(Reverse_Resample_Image, self).__init__()
        raise NotImplementedError
        # This is the copy from SITK
        # self.field_oi = field_oi
        # self.resampler = sitk.ResampleImageFilter()
        # self.mode = mode

    # def __call__(self, ipt):
    #     original = deepcopy(ipt)
    #     meta_data = ipt['resample_meta_data']
    #     self.resampler.SetInterpolator(self.mode)
    #     self.resampler.SetSize([int(x) for x in meta_data['orig_size']])
    #     self.resampler.SetOutputSpacing(meta_data['orig_spc'])
    #     self.resampler.SetOutputOrigin(meta_data['orig_org'])
    #     self.resampler.SetOutputDirection(meta_data['orig_dir'])
    #     img_res = []
    #     for img in ipt[self.field_oi]:
    #         img = self.resampler.Execute(img)
    #         img_res.append(sitk.Cast(img, sitk.sitkUInt8))
    #     ipt[self.field_oi] = img_res
    #
    #     return ipt

class Resample_SITK_shape(_TensorProcessing):
    '''
    a tensor process that will resample the scan based on the shape of the scan
    '''
    def __init__(self,
                 mode = sitk.sitkLinear,
                 shape = (64,64,64),
                 field_oi='X',
                 output_dtype=sitk.sitkFloat32):
        '''
        :param mode: the resampling mode, must be aa sitk mode like
            sitk.sitkLinear
        :param shape: the output shape desired
        :param field_oi:
        :param output_dtype: the datatype to save the output
        '''
        super(Resample_SITK_shape, self).__init__()
        self.resampler = sitk.ResampleImageFilter()
        self.field_oi= field_oi
        self.shape = shape
        self.mode = mode
        self.output_dtype = output_dtype
        # desired SITK Image direction : (-0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0,
        # -1.0, 0.0)

    def get_reciprical(self, **kwargs):
        return Reverse_Resample_SITK(**kwargs)

    def resample_function(self, image, meta_data=None):
        '''
        this will resample the image input
        :param image: sitk Image
        :param meta_data: the meta data needed for resampling the image
        :return: the image resampled to a new size
        '''

        if not meta_data:
            # collection of important meta data for resampling
            orig_size = np.array(image.GetSize(), dtype=int)
            orig_spacing = image.GetSpacing()
            new_size = np.array([sz if sz is not None
                                 else orig_size[i]
                                 for i, sz in enumerate(self.shape)],dtype=int)
            new_spacing = orig_spacing*(orig_size/new_size)
            new_size = [int(sz) for sz in new_size]

            meta_data = {'orig_size': list(orig_size),
                         'orig_spc': list(orig_spacing),
                         'orig_dir': image.GetDirection(),
                         'orig_org': image.GetOrigin(),
                         'new_size': list(new_size),
                         'new_spc': list(new_spacing)}

        assert all([ky in meta_data.keys() for ky in
                    ['orig_org', 'orig_dir', 'new_size', 'new_spc']])

        # setting up the resampler with the meta data
        self.resampler.SetInterpolator(self.mode)
        self.resampler.SetOutputOrigin(meta_data['orig_org'])
        self.resampler.SetOutputDirection(meta_data['orig_dir'])
        self.resampler.SetSize(meta_data['new_size'])
        self.resampler.SetOutputSpacing(meta_data['new_spc'])

        return self.resampler.Execute(sitk.Cast(image, self.output_dtype)), \
               meta_data

    def __call__(self, ipt):
        '''
        execute process on the input
        :param ipt: dict.
        :return: dict.
        '''
        res = []
        if all([sp is None for sp in self.shape]):
            return ipt
        if 'resample_meta_data' in ipt.keys():
            res_data = ipt['resample_meta_data']
        else:
            res_data = None
        for img in ipt[self.field_oi]:
            res_img, res_data = self.resample_function(img, res_data)
            res.append(res_img)
        ipt[self.field_oi] = res
        ipt['resample_meta_data'] = res_data
        return ipt

class Reverse_Resample_SITK(_TensorProcessing):
    '''
    a tensor process that will reverse the  resample the scan based on the
    shape of the scan or the spacing
    '''
    def __init__(self, mode = sitk.sitkLinear, field_oi='pred_y'):
        '''
        :param mode: the resampling mode, must be aa sitk mode like
            sitk.sitkLinear
        '''
        super(Reverse_Resample_SITK, self).__init__()
        self.field_oi = field_oi
        self.resampler = sitk.ResampleImageFilter()
        self.mode = mode

    def __call__(self, ipt):
        '''
        :param ipt: dict.
        :return: dict.
        '''
        assert 'resample_meta_data' in ipt.keys()
        meta_data = ipt['resample_meta_data']

        assert all([ky in meta_data.keys() for ky in
                    ['orig_org', 'orig_dir', 'orig_spc', 'orig_size']])

        self.resampler.SetInterpolator(self.mode)
        self.resampler.SetSize([int(x) for x in meta_data['orig_size']])
        self.resampler.SetOutputSpacing(meta_data['orig_spc'])
        self.resampler.SetOutputOrigin(meta_data['orig_org'])
        self.resampler.SetOutputDirection(meta_data['orig_dir'])

        img_res = []
        for img in ipt[self.field_oi]:
            img = self.resampler.Execute(img)
            img_res.append(sitk.Cast(img, sitk.sitkUInt8))
        ipt[self.field_oi] = img_res

        return ipt

class Resample_SITK_spacing(_TensorProcessing):
    '''
    a tensor process that will resample the scan based on the spacing
    '''
    def __init__(self,
                 mode = sitk.sitkLinear,
                 spacing = (1.0, 1.0, 1.0),
                 field_oi='X',
                 output_dtype=sitk.sitkFloat32):
        '''
        :param mode: the resampling mode, must be aa sitk mode like
            sitk.sitkLinear
        :param spacing: the desired output spacing
        :param output_dtype: the datatype to save the output
        '''
        super(Resample_SITK_spacing, self).__init__()
        self.resampler = sitk.ResampleImageFilter()
        self.field_oi= field_oi
        self.spacing = list(spacing)
        self.mode = mode
        self.output_dtype = output_dtype

    def get_reciprical(self, **kwargs):
        return Reverse_Resample_SITK(**kwargs)

    def resample_function(self, image, meta_data=None):
        '''
        this will resample the image input
        :param image: sitk Image
        :param meta_data: the meta data needed for resampling the image
        :return: the image resampled to a new size
        '''
        if not meta_data:
            # collection of important meta data for resampling
            output_spacing = [sp if sp is not None
                              else image.GetSpacing()[i]
                              for i, sp in enumerate(self.spacing)]
            orig_size = np.array(image.GetSize(), dtype=int)
            orig_spacing = image.GetSpacing()
            new_size = orig_size * (np.array(orig_spacing) / np.array(output_spacing))
            new_size = np.ceil(new_size).astype(int).tolist()
            meta_data = {'orig_size': [int(v) for v in orig_size],
                         'orig_spc': list(orig_spacing),
                         'orig_dir': image.GetDirection(),
                         'orig_org': image.GetOrigin(),
                         'new_size': [int(v) for v in new_size],
                         'new_spc': list(output_spacing)}

        assert all([ky in meta_data.keys() for ky in
                    ['orig_org', 'orig_dir', 'new_size', 'new_spc']])

        # setting up the resampler with the meta data
        self.resampler.SetInterpolator(self.mode)
        self.resampler.SetOutputOrigin(meta_data['orig_org'])
        self.resampler.SetOutputDirection(meta_data['orig_dir'])
        self.resampler.SetSize(meta_data['new_size'])
        self.resampler.SetOutputSpacing(meta_data['new_spc'])

        return self.resampler.Execute(sitk.Cast(image, self.output_dtype)), \
               meta_data

    def __call__(self, ipt):
        '''
        execute process on the input
        :param ipt: dict.
        :return: dict.
        '''
        res = []
        if all([sp is None for sp in self.spacing]):
            return ipt
        if 'resample_meta_data' in ipt.keys():
            res_data = ipt['resample_meta_data']
        else:
            res_data = None
        for img in ipt[self.field_oi]:
            res_img, res_data = self.resample_function(img, res_data)
            res.append(res_img)
        ipt[self.field_oi] = res
        ipt['resample_meta_data'] = res_data
        return ipt

class Cast_numpy(_TensorProcessing):
    '''
    cast a numpy array with a specific data type
    '''
    def __init__(self, data_type=np.float16, field_oi='X'):
        super(Cast_numpy, self).__init__()
        self.output_dtype = data_type
        self.field_oi = field_oi

    def __call__(self, ipt):
        if type(ipt[self.field_oi])==np.ndarray:
            ipt[self.field_oi] = ipt[self.field_oi].astype(self.output_dtype)
        elif type(ipt[self.field_oi])==list:
            ipt[self.field_oi] = [img.astype(self.output_dtype)
                                  for img in ipt[self.field_oi]]
        elif type(ipt[self.field_oi])==int or type(ipt[self.field_oi])==float:
            ipt[self.field_oi] = np.asarray(
                [ipt[self.field_oi]]
            ).astype(self.output_dtype)
        elif type(ipt[self.field_oi])==dict:
            ipt[self.field_oi] = {
                ky:ipt[self.field_oi][ky].astype(self.output_dtype)
                if isinstance(ipt[self.field_oi][ky], np.ndarray)
                else np.asarray(ipt[self.field_oi][ky], dtype=self.output_dtype)
                for ky in ipt[self.field_oi].keys()
            }
        else:
            raise Exception(str(type(ipt[self.field_oi])) +
                            ' is not a recognized data type for CastNumpy')
        return ipt

class ToNumpy(_TensorProcessing):
    '''
    cast the input to a numpy array with a specific data type
    '''
    def __init__(self, field_oi='X', dtype=np.float32):
        super(ToNumpy, self).__init__()
        self.field_oi = field_oi
        self.dtype = dtype

    def __call__(self, ipt):

        ipt[self.field_oi] = np.asarray(ipt[self.field_oi], dtype=self.dtype)
        return ipt

class ImageToNumpy(_TensorProcessing):
    '''
    convert a PIL Image to a numpy array
    '''
    def __init__(self, field_oi='X'):
        super(ImageToNumpy, self).__init__()
        self.field_oi = field_oi

    def get_reciprical(self, **kwargs):
        return NumpyToImage(**kwargs)

    def __call__(self, ipt):
        ipt[self.field_oi] = [np.array(img) for img in ipt[self.field_oi]]
        return ipt

class NumpyToImage(_TensorProcessing):
    '''
    convert a numpy array to a PIL image
    '''
    def __init__(self, field_oi='X', mode=None):
        super(NumpyToImage, self).__init__()
        self.field_oi = field_oi
        self.mode= mode

    def get_reciprical(self, **kwargs):
        return ImageToNumpy(**kwargs)

    def __call__(self, ipt):
        ipt[self.field_oi] = [Image.fromarray(img, mode=self.mode) for img in
                              ipt[
            self.field_oi]]
        return ipt

class SITKToNumpy(_TensorProcessing):
    '''
    a tensor process to convert sitk Image to numpy array
    '''
    def __init__(self, field_oi='X'):
        super(SITKToNumpy, self).__init__()
        self.field_oi = field_oi

    def get_all_image_data(self, image):
        '''
        returns a dictionary of image meta data
        :param image: sitk.Image
        :return: dict.
        '''
        return {'orig_size': image.GetSize(),
                'orig_spc': image.GetSpacing(),
                'orig_dir': image.GetDirection(),
                'orig_org': image.GetOrigin()}

    def get_reciprical(self, **kwargs):
        return NumpyToSITK(**kwargs)

    def __call__(self, ipt):
        '''
        execute process on the input
        :param ipt: dict.
        :return: dict.
        '''
        ipt[self.field_oi + '_SITKToNumpy_data'] = [self.get_all_image_data(img)
                                                    for img in
                                                    ipt[self.field_oi]]
        ipt[self.field_oi] = [sitk.GetArrayFromImage(img) for img in
                              ipt[self.field_oi]]
        return ipt

class NumpyToSITK(_TensorProcessing):
    '''
    a tensor process to convert numpy array to sitk.Image
    '''
    def __init__(self, field_oi='X'):
        super(NumpyToSITK, self).__init__()
        self.field_oi = field_oi

    def set_all_image_data(self, image, data):
        '''
        sets the image origin, direction and spacing in the data
        :param image: sitk.Image
        :param data: dict.
        :return: sitk.Image
        '''
        image.SetOrigin(data['orig_org'])
        image.SetSpacing(data['orig_spc'])
        image.SetDirection(data['orig_dir'])
        return image

    def get_reciprical(self, **kwargs):
        return SITKToNumpy(**kwargs)

    def __call__(self, ipt):
        '''
        execute process on the input
        :param ipt: dict.
        :return: dict.
        '''
        ipt[self.field_oi] = [sitk.GetImageFromArray(img) for img in ipt[self.field_oi]]
        if self.field_oi + '_SITKToNumpy_data' in ipt.keys():
            ipt[self.field_oi] = [
                self.set_all_image_data(img, data)
                for img, data in zip(ipt[self.field_oi],
                                     ipt[self.field_oi + '_SITKToNumpy_data'])
            ]
        elif self.field_oi=='pred_y':
            if len(ipt['X_SITKToNumpy_data'])==len(ipt[self.field_oi]):
                ipt[self.field_oi] = [
                    self.set_all_image_data(img, data)
                    for img, data in zip(ipt[self.field_oi],
                                         ipt['X_SITKToNumpy_data']
                                         )
                ]
            elif len(ipt['X_SITKToNumpy_data'])==1 or \
                    all(item==ipt['X_SITKToNumpy_data'] for item in ipt[
                        'X_SITKToNumpy_data']):
                ipt[self.field_oi] = [
                    self.set_all_image_data(img, data)
                    for img, data in zip(ipt[self.field_oi],
                                         [ipt['X_SITKToNumpy_data'][0] for _ in
                                          range(
                                             len(ipt[self.field_oi]))]
                                         )
                ]
            else:
                raise Exception('This is a very stupid hack, I need to see '
                                'more use cases where this fails? ')
        return ipt

class Numpy_resize(_TensorProcessing):
    '''
    resample a numpy array to a new size using skimage resize and an order of 0
    '''
    def __init__(self, output_shape, field_oi='X'):
        super(Numpy_resize, self).__init__()
        self.output_shape = output_shape
        self.field_oi = field_oi

    def __call__(self, ipt):
        ipt[self.field_oi] = [resize(
            img,
            output_shape=self.output_shape,
            order=0,
            cval=0,
            mode='constant',
            preserve_range=True
        )
            for img in ipt[self.field_oi]]
        return ipt

class Pad_to_Size_numpy(_TensorProcessing):
    '''
    pad a numpy array to a given size by a given value
    '''
    def __init__(self, shape = (128, 256, 256), img_centering=(None,None,None),
                 centre=None, fill_value=0.0,
                 field_oi='X'):
        super(Pad_to_Size_numpy, self).__init__()
        self.field_oi = field_oi
        self.shape = shape
        self.img_centering = img_centering
        self.centre = centre
        self.value = fill_value

    def get_reciprical(self, **kwargs):
        return Reverse_Pad_to_Size_numpy(**kwargs)

    def get_bounds_of_axis(self, axis, img):
        noise = 0.01*img.max()
        thr_img = img>noise
        mv_ax = np.moveaxis(thr_img, axis, 0)
        mv_ax.reshape(mv_ax.shape[0],-1).sum(-1)
        beginning = mv_ax.reshape(mv_ax.shape[0],-1).sum(-1).argmax()
        if beginning+self.shape[axis]<=img.shape[axis]:
            return (beginning, beginning+self.shape[0])
        else:
            return (0, self.shape[0])

    def __call__(self, ipt):

        scan_shape = ipt[self.field_oi][0].shape
        if 'padding_meta_data' in ipt.keys():
            scan_indices = ipt['padding_meta_data']['scan_indices']
            opt_indices = ipt['padding_meta_data']['opt_indices']
        else:
            scan_indices = [(0, sc_sh) for sc_sh in scan_shape]
            opt_indices = [(0, sh) for sh in self.shape]
            centering = [0 for _ in range(len(scan_shape))]
            if self.centre:
                centering = [c if c is not None else centering[i]
                             for i, c in enumerate(self.centre)]
            for i1 in range(len(scan_indices)):

                if scan_shape[i1] <= self.shape[i1]:
                    shift = (self.shape[i1]-scan_shape[i1])/2
                    opt_indices[i1] = (
                        int(np.ceil(shift)),
                        int(np.ceil(self.shape[i1] - shift))
                    )
                else:
                    if self.img_centering[i1]:
                        scan_indices[i1] = self.get_bounds_of_axis(
                            axis=i1,
                            img=ipt[self.field_oi][0]
                        )
                    else:
                        l = (scan_shape[i1]-self.shape[i1]) / 2 - centering[i1]
                        u = (scan_shape[i1]-self.shape[i1]) / 2 + centering[i1]
                        if u<0:
                            l = scan_shape[i1]-self.shape[i1]
                            u = 0
                        if l<0:
                            l = 0
                            u = scan_shape[i1]-self.shape[i1]

                        scan_indices[i1] = (
                            int(np.ceil(l)),
                            int(np.ceil(scan_shape[i1] - u))
                        )
        meta_data = {'scan_indices': scan_indices,
                     'opt_indices': opt_indices,
                     'orig_size': scan_shape}
        ipt['padding_meta_data'] = meta_data

        temp = ipt[self.field_oi]

        start = [np.ones(self.shape, dtype=t.dtype) * t.flatten()[0] for t in
                 temp]

        for i in range(len(temp)):
            if len(scan_shape) == 3:
                start[i][
                opt_indices[0][0]:opt_indices[0][1],
                opt_indices[1][0]:opt_indices[1][1],
                opt_indices[2][0]:opt_indices[2][1]
                ] = \
                    temp[i][
                    scan_indices[0][0]:scan_indices[0][1],
                    scan_indices[1][0]:scan_indices[1][1],
                    scan_indices[2][0]:scan_indices[2][1]
                    ]
            elif len(scan_shape) == 2:
                start[i][
                opt_indices[0][0]:opt_indices[0][1],
                opt_indices[1][0]:opt_indices[1][1]
                ] = \
                    temp[i][
                    scan_indices[0][0]:scan_indices[0][1],
                    scan_indices[1][0]:scan_indices[1][1]
                    ]
            else:
                raise Exception('Pad_to_Size_numpy does not support numpy '
                                'arrays outside of 2 and 3 dimensionality. ')
        ipt[self.field_oi] = start

        return ipt

class Reverse_Pad_to_Size_numpy(_TensorProcessing):
    '''
    reverse the padding to a specific numpy size
    '''
    def __init__(self, field_oi='pred_y'):
        super(Reverse_Pad_to_Size_numpy, self).__init__()
        self.field_oi = field_oi

    def __call__(self, ipt):
        new_segm = [np.zeros(ipt['padding_meta_data']['orig_size']) for
                    _ in range(len(ipt[self.field_oi]))]
        op_indices = ipt['padding_meta_data']['scan_indices']
        ip_indices = ipt['padding_meta_data']['opt_indices']
        for i in range(len(ipt[self.field_oi])):
            new_segm[i][
            op_indices[0][0]:op_indices[0][1],
            op_indices[1][0]:op_indices[1][1],
            op_indices[2][0]:op_indices[2][1]
            ] = \
                ipt[self.field_oi][i][
                ip_indices[0][0]:ip_indices[0][1],
                ip_indices[1][0]:ip_indices[1][1],
                ip_indices[2][0]:ip_indices[2][1]
                ]
        ipt[self.field_oi] = new_segm
        return ipt

class Add_Channel(_TensorProcessing):
    '''
    convert given field into a full numpy array with the list len as the
    channels
    '''
    def __init__(self, compress_labels=False, field_oi='X'):
        super(Add_Channel, self).__init__()
        self.field_oi = field_oi
        self.compress_labels = compress_labels

    def __call__(self, ipt):
        ipt[self.field_oi] = np.asarray(ipt[self.field_oi])
        return ipt

class ToTensor(_TensorProcessing):
    '''
    cast object to a pytorch Tensor
    '''
    def __init__(self, dtype=torch.float32, field_oi='X'):
        super(ToTensor, self).__init__()
        self.field_oi = field_oi
        self.dtype = dtype

    def __call__(self, ipt):
        if type(ipt[self.field_oi])==dict:
            ipt[self.field_oi] = {
                ky: torch.tensor(ipt[self.field_oi][ky], dtype=self.dtype)
                for ky in ipt[self.field_oi].keys()
            }
        else:
            ipt[self.field_oi] = torch.tensor(np.array(ipt[self.field_oi]),
                                              dtype=self.dtype)
        return ipt

class OneHotEncode(_TensorProcessing):
    '''
    one hot encode the given y label
    '''
    def __init__(self,max_class, field_oi='y'):
        super(OneHotEncode, self).__init__()
        self.number_of_classes = max_class
        self.field_oi = field_oi

    def __call__(self, ipt):
        # Fast wrote this, it maybe completely wrong. Apologies
        assert type(ipt[self.field_oi])==int

        new_y = [0] * self.number_of_classes

        new_y[ipt[self.field_oi]] = 1.0
        ipt[self.field_oi + '_original'] = deepcopy(ipt[self.field_oi])

        ipt[self.field_oi] = new_y

        return ipt

class OneHotEncode_Seg(_TensorProcessing):
    def __init__(self, max_class, field_oi='y'):
        super(OneHotEncode_Seg, self).__init__()
        self.max_class = max_class
        self.field_oi = field_oi

    def __call__(self, ipt):
        assert ipt[self.field_oi].shape[0]==1

        def one_hotEncode(im):
            one_hot = np.zeros((self.max_class, *im.shape))
            for i, unique_value in enumerate(np.unique(im)):
                one_hot[i,...][im == unique_value] = 1
            return one_hot
        if self.max_class>1:
            ipt[self.field_oi] = one_hotEncode(ipt[self.field_oi][0])
        return ipt