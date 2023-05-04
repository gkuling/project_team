from . import _TensorProcessing
import transformers as tk
from copy import deepcopy
class HG_Tokenizer(_TensorProcessing):
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
        assert type(ipt[self.field_oi])==str, 'A tokenizer must be ran on text'

        ipt[self.field_oi + '_originaltext'] = deepcopy([ipt[self.field_oi]])
        result = self.tokenizer(ipt[self.field_oi],
                                truncation=self.truncation)
        ipt.update(result)
        ipt[self.field_oi] = ipt['input_ids']
        return ipt