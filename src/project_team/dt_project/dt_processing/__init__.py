### TensorProcessing Parent class must go before img processing imports.
# Because this object is used in the img processing files
import warnings


class _TensorProcessing(object):
    def __init__(self):
        pass

    def get_reciprocal(self, **kwargs):
        '''
        return a transform that undoes this one, or None when no reverse
        exists. This is an extension point for invertible pipelines —
        nothing in the package calls it automatically yet.
        :param kwargs: forwarded to the reverse transform's constructor
        :return: a transform instance or None
        '''
        return None

    def get_reciprical(self, **kwargs):
        '''Deprecated misspelling of get_reciprocal().'''
        warnings.warn(
            'get_reciprical() is deprecated; use get_reciprocal().',
            DeprecationWarning, stacklevel=2)
        return self.get_reciprocal(**kwargs)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('The call has not been implemented for ' +
                                  str(self.__class__))

### Img Processing Files
from .img_intensity import *
from .img_shape import *
from .img_augmentation import *
from .nlp import *
