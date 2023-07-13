import pandas as pd
from torchvision import transforms

from ..datasets import Text_Dataset
from ._Processor import _Processor, DT_config
from ..dt_processing.nlp import *

class Text_Processor_config(DT_config):
    '''
    Configuration for the Text processor
    '''
    def __init__(self,
                 tokenizer,
                 model=None,
                 pre_load=True,
                 max_length=512,
                 **kwargs
                 ):
        '''
        :param tokenizer: required for converting text data into tokens. Currently only programmed
            to handle transformer tokenizers
        :param model: The type of model tokenizer to load
        :param pre_load: see parent class
        :param kwargs:
        '''
        super(Text_Processor_config, self).__init__(pre_load, **kwargs)
        self.tokenizer = tokenizer
        self.model = model
        self.max_length = max_length

class Text_Processor(_Processor):
    '''
    a text processor parent class
    '''
    def __init__(self, text_processor_config=Text_Processor_config('BertTokenizerFast')):
        '''
        :param text_processor_config: a text processor config
        '''
        super(Text_Processor, self).__init__(text_processor_config)

        # the pre_transforms for a standard BERT experiment.
        # This can be changed in child classes
        self.pre_transforms = transforms.Compose([
            # Huggingface tokenizer transform
            HG_Tokenizer(tokenizer=self.config.tokenizer,
                         model=self.config.model,
                         max_length=self.config.max_length)
        ])

    def get_dataset(self, data, name, transforms):
        '''
        function that sets the dataset atribute for the given name
        :param data: pandas dataframe to be loaded as a dataset
        :param name: the name of the dataset
        :param transforms: the pretransforms to be given to the dataset
        '''
        assert type(data)==pd.DataFrame
        setattr(self, name, Text_Dataset(
            data,
            preload_data=self.config.pre_load,
            preload_transforms=transforms,
            filter_out_zero_X=self.config.filter_out_zero_X
        ))
