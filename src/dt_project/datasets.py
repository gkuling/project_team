from src.project_config import is_Primitive
import pandas as pd
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm
from torchvision import transforms as pt_transforms

class Project_Team_Dataset(Dataset):
    '''
    Custom dataset built for the project_team package
    '''
    def __init__(self,
                 data_df=pd.DataFrame([]),
                 preload_transforms=None,
                 transforms=None,
                 preload_data=False):
        '''
        :param data_df: dataframe of the data to be used
        :param preload_transforms: the transforms desired for once data has
            been preloaded
        :param transforms: the transforms desired for the data once
            __getitem__ has been called by a dataloader.
        :param preload_data: boolean. default: False. Indicator whether to
        load all the data into memory or to open data on the fly.
        '''
        self.dfiles = data_df.to_dict('records')
        self.preload_transforms = preload_transforms
        self.transforms = transforms
        self.preloaded = preload_data
        self.condition = None

        if preload_data:
            self.files_silo = []
            self.catalogue = {}
    def perform_preload(self):
        '''
        function used to preload all data and store objects in a files_silo
        '''
        print(self.__str__().split(' ')[0].split('.')[-1] + ' Message: ')
        cnt=0
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
                                 if not is_Primitive(post_loaded_ex[key]) or
                                 post_loaded_ex[key]!=pre_loaded_ex[key]]

                pre_loaded_ex.update({key:value for key, value in post_loaded_ex.items()
                                      if key not in pre_loaded_ex.keys()})
                # check the transformation should be kept
                if self.keep_data_type_specific_function(post_loaded_ex):
                    for key in items_to_save:
                        self.catalogue['save_name_' + str(cnt)] = cnt
                        self.files_silo.append(post_loaded_ex[key])
                        pre_loaded_ex[key] = 'save_name_' + str(cnt)
                        cnt+=1
                    new_dfiles.append(pre_loaded_ex)
                else:
                    # if all inputs are zero, do not keep in the dataset.
                    print(self.__str__().split(' ')[0].split('.')[-1] +
                          ' Message: Warning an example has an input of all '
                          'zeros. Example at line '
                          + str(item) + ' of the current dataset. ')
            except:
                print(self.__str__().split(' ')[0].split('.')[-1] +
                      ' Message: Warning an example has failed the preloading '
                      'transforms. Example at line ' + str(item) +
                      ' of the current dataset. ')
        self.dfiles = new_dfiles

    def keep_data_type_specific_function(self,x):
        '''
        Depending on the data typoe the dataset is specialized for this check
        could be different. Majority of it is to check that the iminput in not
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
                       desc='Transfering data from list'):

            ex = {k:v for k,v in ex.items() if 'meta_data' not in k}

            # apply preload_transforms on the example
            if self.preload_transforms:
                ex = self.preload_transforms(ex)
            dfile_ex = {}
            for k,v in ex.items():
                if is_Primitive(v):
                    dfile_ex[k] = v
                else:
                    dfile_ex[k] = 'save_name_' + str(cnt)
                    self.files_silo.append(v)
                    self.catalogue['save_name_' + str(cnt)] = cnt
                    cnt+=1
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
        if type(transforms_compose)==list:
            self.transforms = pt_transforms.Compose(transforms_compose)
        else:
            self.transforms = transforms_compose

    def set_preload_transforms(self, transforms_compose):
        '''
        set the data transforms for pre_loading
        :param transforms_compose: list or pt_transforms.Compose
        '''
        if type(transforms_compose) == list:
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
            for key in example:
                if str(example[key]) in self.catalogue.keys():
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
    def __init__(self, data_df=pd.DataFrame([]), preload_transforms=None,
                 transforms=None, preload_data=False):
        super(Images_Dataset, self).__init__(data_df, preload_transforms, transforms, preload_data)

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
    def __init__(self, data_df=pd.DataFrame([]), preload_transforms=None,
                 transforms=None, preload_data=False):
        super(Text_Dataset, self).__init__(data_df, preload_transforms,
                                        transforms,
                           preload_data)

    def keep_data_type_specific_function(self, processed_x):
        # Checking that the entire input_data of the model is not 0.0
        return not all([input_data == 0 for input_data in processed_x['X']])
