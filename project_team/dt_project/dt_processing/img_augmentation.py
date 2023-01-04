import scipy.ndimage.interpolation as sni
from .augmentation_utils import *
from . import _TensorProcessing

class AffineAugmentation(_TensorProcessing):
    def __init__(self, shift=1, rot=1, scale=0.1, dimensionality=3, field_oi='X',
                 order=0, distribution_type='Gaussian'):
        super(AffineAugmentation, self).__init__()
        if type(shift)==int or type(shift)==float:
            shift = tuple([shift for _ in range(dimensionality)])
        if type(shift)==list and len(shift)==dimensionality:
            shift = tuple(shift)
        if type(rot)==int or type(rot)==float:
            rot = tuple([rot for _ in range(dimensionality)])
        if type(rot)==list and len(rot)==dimensionality:
            rot = tuple(rot)
        assert type(shift)==tuple
        assert type(rot)==tuple

        self.shift = shift
        self.rot = rot
        self.scale = scale
        self.field_oi = field_oi
        self.order = order
        self.dist_type = distribution_type

    def calc_from_normal(self, x):
        value = np.random.normal(0, x)
        value = np.clip(value, -2 * x, 2 * x)
        return value

    def build_meta_data(self):
        if self.dist_type=='Uniform':
            return {
                'scaling': 1+np.random.uniform(-self.scale,self.scale),
                'rotation': tuple([np.deg2rad(np.random.uniform(-r,r)) for r in
                                   self.rot]),
                'translation': tuple([np.random.randint(-t,t) for t in self.shift])
            }
        elif self.dist_type=='Gaussian':
            return {
                'scaling': 1+self.calc_from_normal(self.scale),
                'rotation': tuple([np.deg2rad(self.calc_from_normal(r)) for
                                   r in self.rot]),
                'translation': tuple([np.round(self.calc_from_normal(t)) for t
                                      in self.shift])
            }
        else:
            return {
                'scaling': 1+np.random.uniform(-self.scale,self.scale),
                'rotation': tuple([np.deg2rad(np.random.uniform(-r,r)) for r in
                                   self.rot]),
                'translation': tuple([np.random.randint(-t,t) for t in self.shift])
            }

    def __call__(self, ipt):

        if 'augmentation_meta_data' in ipt.keys():
            meta_data = ipt['augmentation_meta_data']
        else:
            meta_data = self.build_meta_data()

        output = {'X': ipt[self.field_oi]}

        output = Scale_3DNumpy(meta_data['scaling'], order=self.order)(output)

        output = Rotate_3DNumpy(meta_data['rotation'], order=self.order)(output)

        output = Translate_3DNumpy(meta_data['translation'],
                                   order=self.order)(output)

        ipt[self.field_oi] = output['X']
        ipt['augmentation_meta_data'] = meta_data
        return ipt

class Translate_3DNumpy(_TensorProcessing):
    def __init__(self, shifts=(0,0,0), order=0, field_oi='X'):
        super(Translate_3DNumpy, self).__init__()
        self.shifts = shifts
        self.order = order
        self.field_oi = field_oi

    def __call__(self, ipt):

        result = []
        for image in ipt[self.field_oi]:
            image = sni.shift(image, self.shifts, order=self.order, cval=image[0,0,0])
            result.append(image)

        ipt[self.field_oi] = result
        return ipt

class Rotate_3DNumpy(_TensorProcessing):
    def __init__(self, angles=(0,0,0), order=0, field_oi='X'):
        super(Rotate_3DNumpy, self).__init__()
        self.angles = angles
        self.order = order
        self.field_oi = field_oi

    def __call__(self, ipt):

        result = []
        for image in ipt[self.field_oi]:
            angle = self.angles[0]
            image = sni.rotate(image, angle, order=self.order, axes=(0, 1),
                               reshape=False, cval=image[0,0,0])
            angle = self.angles[1]
            image = sni.rotate(image, angle, order=self.order, axes=(0, 2),
                               reshape=False, cval=image[0,0,0])
            angle = self.angles[2]
            image = sni.rotate(image, angle, order=self.order, axes=(1, 2),
                               reshape=False, cval=image[0,0,0])
            result.append(image)
        ipt[self.field_oi] = result
        return ipt

class Scale_3DNumpy(_TensorProcessing):
    def __init__(self, scale=1, field_oi='X', order=0):
        super(Scale_3DNumpy, self).__init__()
        self.scale = scale
        self.field_oi = field_oi
        self.order = order

    def __call__(self, ipt):


        result = []
        for img in ipt[self.field_oi]:

            h, w, d = img.shape

            # For multichannel images we don't want to apply the zoom factor to the RGB
            # dimension, so instead we create a tuple of zoom factors, one per array
            # dimension, with 1's for any trailing dimensions after the width and height.
            zoom_tuple = (self.scale,) * img.ndim

            # Zooming out
            if self.scale < 1:

                # Bounding box of the zoomed-out image within the output array
                zh = int(np.round(h * self.scale))
                zw = int(np.round(w * self.scale))
                zd = int(np.round(d * self.scale))
                top = (h - zh) // 2
                left = (w - zw) // 2
                far = (d - zd) // 2

                # Zero-padding
                if img[0,0,0]<0.5:
                    out = np.zeros_like(img)
                else:
                    out = np.ones_like(img)
                out[top:top+zh, left:left+zw, far:far+zd] = \
                    sni.zoom(img, zoom_tuple, order=self.order)

            # Zooming in
            elif self.scale > 1:
                out = sni.zoom(img, zoom_tuple, order=self.order)
                top_start, left_start, far_start = [int(ind) for ind in
                                                    np.round((np.array(
                                                        out.shape)-np.array(img.shape))/2).tolist()]
                out = out[top_start:top_start+h, left_start:left_start+w,
                      far_start:far_start + d]

            # If zoom_factor == 1, just return the input array
            else:
                out = img
            if out.shape!=img.shape:
                out = img
            result.append(out)
        ipt[self.field_oi] = result
        return ipt

class GMM_DA_Augmentation(_TensorProcessing):
    def __init__(self):
        super(GMM_DA_Augmentation, self).__init__()
        pass

    def __call__(self, ipt):
        # res = deepcopy(ipt)
        ipt['X'] = [generate_gmm_image(im,
                                       mask=ipt['y'][0],
                                       n_components=2,
                                       std_means=(50,50),
                                       std_sigma=(50,50),
                                       normalize=False) for im in ipt['X']]
        return ipt

class AddGaussainNoise(_TensorProcessing):
    def __init__(self, std=1, field_oi='X'):
        super(AddGaussainNoise, self).__init__()
        self.std = std
        self.field_oi = field_oi

    def __call__(self, ipt):

        ipt[self.field_oi] = [np_ar + np.random.normal(0.0, self.std, np_ar.shape)
                              for np_ar in ipt[self.field_oi]]
        return ipt
