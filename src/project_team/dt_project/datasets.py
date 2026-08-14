import numpy as np

from ..project_config import is_primitive
import pandas as pd
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm
from torchvision import transforms as pt_transforms

__all__ = ['Project_Team_Dataset', 'Images_Dataset', 'Text_Dataset',
           'Dataset_Fingerprint']

class Project_Team_Dataset(Dataset):
    '''
    Custom dataset built for the project_team package
    '''
    def __init__(self,
                 data_df=None,
                 preload_transforms=None,
                 transforms=None,
                 preload_data=False,
                 filter_out_zero_X=True,
                 debug_pretransform=False):
        '''
        :param data_df: dataframe of the data to be used
        :param preload_transforms: the transforms desired for once data has
            been preloaded
        :param transforms: the transforms desired for the data once
            __getitem__ has been called by a dataloader.
        :param preload_data: boolean. default: False. Indicator whether to
        load all the data into memory or to open data on the fly.
        :param filter_out_zero_X: boolean. default: True. Drop examples
        whose processed X is all zeros (for example a blank image).
        :param debug_pretransform: boolean. default: False. When True, a
        failing pretransform re-raises its original exception instead of
        dropping the example with a warning.
        '''
        if data_df is None:
            data_df = pd.DataFrame([])
        self.dfiles = data_df.to_dict('records')
        self.preload_transforms = preload_transforms
        self.transforms = transforms
        self.preloaded = preload_data
        self.condition = None
        self.filter_out_zero_X = filter_out_zero_X
        self.debug_pretransform = debug_pretransform

        if preload_data:
            self.files_silo = []
            self.catalogue = {}
    def perform_preload(self):
        '''
        function used to preload all data and store objects in a files_silo
        '''
        print(type(self).__name__ + ' Message: ')
        cnt=0
        dropped = 0
        new_dfiles = []
        for item in tqdm(range(len(self.dfiles)),
                         desc='Preloading Dataset: ',
                         ncols=80):
            try:
                # take example from dfiles
                pre_loaded_ex = deepcopy(self.dfiles[item])

                # don't take any information if it has meta_data
                pre_loaded_ex = {k:v for k,v in pre_loaded_ex.items() if
                                 'meta_data' not in k}

                # apply the preload_transforms
                post_loaded_ex = \
                    self.preload_transforms(
                        deepcopy(pre_loaded_ex)
                    )

                # determine the items that will need to be stored in the
                # files_silo if it is not a Primitive data type or it isn't
                # the same as it was prior to transformation it will be stored.
                items_to_save = [key for key in pre_loaded_ex.keys()
                                 if not is_primitive(post_loaded_ex[key]) or
                                 post_loaded_ex[key]!=pre_loaded_ex[key]]

                # Take a data fingerprint of array-valued fields (the
                # fingerprint only works on numpy arrays)
                if hasattr(self, 'dataset_fingerprint'):
                    for save_key in items_to_save:
                        value = post_loaded_ex[save_key]
                        if isinstance(value, np.ndarray) or (
                                isinstance(value, list) and len(value) > 0
                                and all(isinstance(v, np.ndarray)
                                        for v in value)):
                            self.dataset_fingerprint.update(save_key, value)

                pre_loaded_ex.update({key:value for key, value in post_loaded_ex.items()
                                      if key not in pre_loaded_ex.keys()})
                # check the transformation should be kept
                if not self.filter_out_zero_X or \
                        self.keep_data_type_specific_function(post_loaded_ex):
                    for key in items_to_save:
                        self.catalogue['save_name_' + str(cnt)] = cnt
                        self.files_silo.append(post_loaded_ex[key])
                        pre_loaded_ex[key] = 'save_name_' + str(cnt)
                        cnt += 1
                    # remember exactly which fields hold silo sentinels so
                    # __getitem__ never confuses genuine data for one
                    pre_loaded_ex['_silo_fields'] = list(items_to_save)
                    new_dfiles.append(pre_loaded_ex)
                else:
                    # if all inputs are zero, do not keep in the dataset.
                    dropped += 1
                    print(type(self).__name__ +
                          ' Message: Warning an example has an input of all '
                          'zeros. Example at line '
                          + str(item) + ' of the current dataset. ')
            except Exception as e:
                if self.debug_pretransform:
                    raise e
                dropped += 1
                print(type(self).__name__ +
                      ' Message: Warning an example has failed the '
                      'preloading transforms. Example at line ' +
                      str(item) + ' of the current dataset. Reason: ' +
                      type(e).__name__ + ': ' + str(e))
        if len(self.dfiles) > 0 and len(new_dfiles) == 0:
            raise RuntimeError(
                'Every example (' + str(dropped) + ') failed or was '
                'filtered out during preloading — the resulting dataset is '
                'empty. Re-run with debug_pretransform=True to see the '
                'original exception.')
        self.dfiles = new_dfiles

    def keep_data_type_specific_function(self,x):
        '''
        Depending on the data type the dataset is specialized for this check
        could be different. Majority of it is to check that the input is not
        all zeros.
        :param x: input value
        :return: boolean
        '''
        raise NotImplementedError

    def transfer_list(self, input_data_list):
        '''
        transfer input_data_list into the dataset files_silo
        :param input_data_list: list of data examples to be processed and
        stored.
        '''
        self.preloaded = True
        self.files_silo = []
        self.catalogue = {}
        cnt = 0
        for ex in tqdm(input_data_list,
                       desc='Transferring data from list'):

            ex = {k:v for k,v in ex.items() if 'meta_data' not in k}

            # apply preload_transforms on the example
            if self.preload_transforms:
                ex = self.preload_transforms(ex)
            dfile_ex = {}
            silo_fields = []
            for k,v in ex.items():
                if is_primitive(v):
                    dfile_ex[k] = v
                else:
                    dfile_ex[k] = 'save_name_' + str(cnt)
                    silo_fields.append(k)
                    self.files_silo.append(v)
                    self.catalogue['save_name_' + str(cnt)] = cnt
                    cnt+=1
            dfile_ex['_silo_fields'] = silo_fields
            self.dfiles.append(dfile_ex)

    def set_filter(self, lambda_condition):
        '''
        set the filter of dataset
        :param lambda_condition: function condition. recommend a lambda x:
        x==a type function.
        '''
        self.condition = lambda_condition

    def clear_filter(self):
        '''
        Clear the set filter
        '''
        self.condition = None

    def set_transforms(self, transforms_compose):
        '''
        set the data transforms for __getitem__
        :param transforms_compose: list or pt_rensforms.Compose
        '''
        if isinstance(transforms_compose, list):
            self.transforms = pt_transforms.Compose(transforms_compose)
        else:
            self.transforms = transforms_compose

    def set_preload_transforms(self, transforms_compose):
        '''
        set the data transforms for pre_loading
        :param transforms_compose: list or pt_transforms.Compose
        '''
        if isinstance(transforms_compose, list):
            self.preload_transforms = pt_transforms.Compose(transforms_compose)
        else:
            self.preload_transforms = transforms_compose

    def list_of_examples(self):
        '''
        retrieve a list of examples in the dataset
        :return: list
        '''
        if self.condition:
            return [ex for ex in self.dfiles if self.condition(ex)]
        else:
            return self.dfiles

    def __len__(self):
        '''
        length of the dataset
        :return: int
        '''
        return len(self.list_of_examples())

    def __getitem__(self, item):
        '''
        retrieve a dataset example
        :param item:
        :return:
        '''
        example = deepcopy(self.list_of_examples()[item])

        if self.preloaded:
            # only fields recorded in _silo_fields are sentinel-substituted;
            # genuine data that happens to look like 'save_name_3' is safe
            for key in example.pop('_silo_fields', []):
                example[key] = self.files_silo[self.catalogue[example[key]]]
        else:
            example = self.preload_transforms(example)

        if self.transforms:
            example = self.transforms(example)

        return example

class Images_Dataset(Project_Team_Dataset):
    '''
    A dataset that is designated to handling imaging files
    '''
    def __init__(self, data_df=None, preload_transforms=None,
                 transforms=None, preload_data=False, filter_out_zero_X=True,
                 debug_pretransform=False):
        super(Images_Dataset, self).__init__(data_df, preload_transforms,
                                             transforms, preload_data, filter_out_zero_X,
                                             debug_pretransform)
        self.dataset_fingerprint = Dataset_Fingerprint()

    def keep_data_type_specific_function(self, processed_x):
        '''
        checking if the input images are all zeros
        :param processed_x: processed example
        :return: boolean
        '''
        # Checking that the entire input_data of the model is not 0.0
        return not all([input_data.max() == 0 for input_data in processed_x['X']])

class Text_Dataset(Project_Team_Dataset):
    '''
    A dataset that is designated to handling text files
    '''
    def __init__(self, data_df=None, preload_transforms=None,
                 transforms=None, preload_data=False, filter_out_zero_X=True,
                 debug_pretransform=False):
        super(Text_Dataset, self).__init__(data_df, preload_transforms,
                                        transforms,
                           preload_data, filter_out_zero_X,
                                           debug_pretransform)

    def keep_data_type_specific_function(self, processed_x):
        # Checking that the entire input_data of the model is not 0.0
        return not all([input_data == 0 for input_data in processed_x['X']])

class Dataset_Fingerprint():
    def __init__(self):
        self.fingerprint = {}

    def average_dictionaries(self, dict1, dict2):
        result_dict = {}

        for key in dict1.keys():
            if key in dict2:
                if isinstance(dict1[key], (int, float)):
                    result_dict[key] = (dict1[key] + dict2[key]) / 2
                elif isinstance(dict1[key], list):
                    result_dict[key] = [(x + y) / 2 for x, y in
                                        zip(dict1[key], dict2[key])]
                elif isinstance(dict1[key], tuple):
                    result_dict[key] = tuple(
                        (x + y) / 2 for x, y in zip(dict1[key], dict2[key]))
        return result_dict
    def finger_print(self, item):
        return_result = {}
        if type(item)==np.ndarray:
            return_result['shape'] = item.shape
            return_result['mean'] = item.mean().astype(float)
            return_result['std'] = item.std().astype(float)
            return_result['min'] = item.min().astype(float)
            return_result['max'] = item.max().astype(float)
            return_result['median'] = np.median(item).astype(float)
            return_result['99_5percentile'] = np.percentile(item, 99.5).astype(float)
            return_result['0_5percentile'] = np.percentile(item, 0.5).astype(float)
            return_result['99percentile'] = np.percentile(item, 99).astype(float)
            return_result['1percentile'] = np.percentile(item, 1).astype(float)
            return_result['95percentile'] = np.percentile(item, 95).astype(float)
            return_result['5percentile'] = np.percentile(item, 5).astype(float)
        else:
            raise Exception('Not a numpy array. Fingerprinting only works on '
                            'numpy arrays currently. ')
        return return_result
    def is_iterable(self, itm):
        '''
        whether itm is a sequence of separate items to fingerprint
        individually. Deliberately excludes dict and str, which are iterable
        in python but are not "a list of examples" in this class's sense.
        :param itm: the value to check
        :return: boolean
        '''
        return isinstance(itm, (list, tuple, np.ndarray))

    def isititerable(self, itm):
        '''Deprecated: use is_iterable().'''
        import warnings
        warnings.warn('isititerable() is deprecated; use is_iterable().',
                      DeprecationWarning, stacklevel=2)
        return self.is_iterable(itm)

    def update(self, item_nm, item):
        itrbl = self.is_iterable(item)

        if itrbl:
            subject = [self.finger_print(i) for i in item]
        else:
            subject = self.finger_print(item)

        if item_nm not in self.fingerprint.keys():
            self.fingerprint[item_nm] = subject
        elif itrbl:
            self.fingerprint[item_nm] = [
                self.average_dictionaries(new, current)
                for current, new in zip(self.fingerprint[item_nm],subject)
            ]
        else:
            self.fingerprint[item_nm] = self.average_dictionaries(
                self.fingerprint[item_nm], subject
            )

    def get_percentiles(self, field_oi, min, max):
        itrbl = self.is_iterable(self.fingerprint[field_oi])
        if itrbl:
            return [
                (i[str(min).replace('.','_')+'percentile'],
                 i[str(max).replace('.','_')+'percentile'])
                for i in self.fingerprint[field_oi]
            ]
        else:
            return (
                self.fingerprint[field_oi][str(min).replace('.',
                                                            '_')+'percentile'],
                self.fingerprint[field_oi][str(max).replace('.',
                                                            '_')+'percentile']
            )

    def get_mean_std(self, field_oi):
        itrbl = self.is_iterable(self.fingerprint[field_oi])
        if itrbl:
            return [
                (i['mean'], i['std'])
                for i in self.fingerprint[field_oi]
            ]
        else:
            return (
                self.fingerprint[field_oi]['mean'],
                self.fingerprint[field_oi]['std']
            )

    def get_min_max(self, field_oi):
        itrbl = self.is_iterable(self.fingerprint[field_oi])
        if itrbl:
            return [
                (i['min'], i['max'])
                for i in self.fingerprint[field_oi]
            ]
        else:
            return (
                self.fingerprint[field_oi]['min'],
                self.fingerprint[field_oi]['max']
            )