### TensorProcessing Parent class must go before img processing imports.
# Because this object is used in the img processing files
class _TensorProcessing(object):
    def __init__(self):
        pass

    def get_reciprical(self, **kwargs):
        return None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError('The call has not been impleneted for ' +
                                  str(self.__class__))

### Img Processing Files
from .img_intensity import *
from .img_shape import *
from .img_augmentation import *
from .nlp import *