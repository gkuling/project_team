from . import _TensorProcessing
import transformers as tk
from copy import deepcopy

__all__ = ['HG_Tokenizer']

class HG_Tokenizer(_TensorProcessing):
    '''
    tokenize the string using a tokenizer built by the hugging face library
    '''
    def __init__(self, tokenizer, model=None, field_oi='X', truncation=True,
                 max_length=512):
        super(HG_Tokenizer, self).__init__()
        if hasattr(tk, tokenizer):
            self.tokenizer = getattr(tk, tokenizer)
        else:
            raise Exception('The given tokenizer is not a tokenizer of '
                            'hugging face. ' + str(tokenizer))

        self.model = model
        self.field_oi = field_oi
        self.tokenizer = self.tokenizer.from_pretrained(self.model)
        if self.tokenizer.model_max_length>max_length:
            self.tokenizer.model_max_length = max_length
        self.truncation = truncation

    def __call__(self, ipt):
        if not isinstance(ipt[self.field_oi], str):
            raise TypeError('A tokenizer must be run on text; field ' +
                            repr(self.field_oi) + ' holds ' +
                            type(ipt[self.field_oi]).__name__)

        ipt[self.field_oi + '_originaltext'] = deepcopy([ipt[self.field_oi]])
        result = self.tokenizer(ipt[self.field_oi],
                                truncation=self.truncation)
        ipt.update(result)
        ipt[self.field_oi] = ipt['input_ids']
        return ipt
