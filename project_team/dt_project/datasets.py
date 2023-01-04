import pandas as pd
from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm
from .Scan_utils import *
from project_team.project_config import is_Primitive
from torchvision import transforms as pt_transforms
# from .file_silo import File_Silo

class Images_Dataset(Dataset):
    def __init__(self, data_df=pd.DataFrame([]), preload_transforms=None,
                 transforms=None,
                 preload_data=False):
        self.dfiles = data_df.to_dict('records')
        self.preload_transforms = preload_transforms
        self.transforms = transforms
        self.preloaded = preload_data
        self.condition = None

        if preload_data:
            self.files_silo = []
            self.catalogue = {}

    def perform_preload(self):
        print(self.__str__().split(' ')[0].split('.')[-1] + ' Message: ')
        cnt=0
        new_dfiles = []
        for item in tqdm(range(len(self.dfiles)),
                         desc='Preloading Dataset: ',
                         ncols=80):
            try:
                pre_loaded_ex = deepcopy(self.dfiles[item])
                pre_loaded_ex = {k:v for k,v in pre_loaded_ex.items() if
                                 'meta_data' not in k}
                post_loaded_ex = \
                    self.preload_transforms(
                        deepcopy(pre_loaded_ex)
                    )
                items_to_save = [key for key in pre_loaded_ex.keys()
                                 if not is_Primitive(post_loaded_ex[key]) or post_loaded_ex[key]!=pre_loaded_ex[key]]

                pre_loaded_ex.update({key:value for key, value in post_loaded_ex.items()
                                      if key not in pre_loaded_ex.keys()})
                # Checking that the entire input of the model is not 0.0
                if type(post_loaded_ex['X'])==list:
                    if not all([img.max()==0 for \
                        img in post_loaded_ex['X']]):
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
                else:
                    new_dfiles.append(post_loaded_ex)
            except:
                print(self.__str__().split(' ')[0].split('.')[-1] +
                      ' Message: Warning an example has failed the preloading '
                      'transforms. Example at line ' + str(item) +
                      ' of the current dataset. ')
        self.dfiles = new_dfiles

    def transfer_list(self, input_data_list):
        self.preloaded = True
        self.files_silo = []
        self.catalogue = {}
        cnt = 0
        for ex in tqdm(input_data_list,
                       desc='Transfering data from list'):
            ex = {k:v for k,v in ex.items() if 'meta_data' not in k}
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
        self.condition = lambda_condition

    def clear_filter(self):
        self.condition = None

    def set_transforms(self, transforms_compose):
        if type(transforms_compose)==list:
            self.transforms = pt_transforms.Compose(transforms_compose)
        else:
            self.transforms = transforms_compose

    def set_preload_transforms(self, transforms_compose):
        self.preload_transforms = transforms_compose

    def list_of_examples(self):
        if self.condition:
            return [ex for ex in self.dfiles if self.condition(ex)]
        else:
            return self.dfiles

    def __len__(self):
        return len(self.list_of_examples())

    def __getitem__(self, item):

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

class SITK_Dataset(Images_Dataset):
    def __init__(self, data_df=pd.DataFrame([]), preload_transforms=None, transforms=None,
                 preload_data=False):
        super(SITK_Dataset, self).__init__(data_df,
                                           preload_transforms,
                                           transforms,
                                           preload_data)

class SITK_Dataset_Patchwise(Images_Dataset):
    def __init__(self, data_df=pd.DataFrame([]), preload_transforms=None, transforms=None,
                 preload_data=False, patch_size=(16,16,16),
                 overlapped_patches=False,):
        super(SITK_Dataset_Patchwise, self).__init__(data_df,
                                           preload_transforms,
                                           transforms,
                                           preload_data)
        self.pch_sz = patch_size
        self.overlap = overlapped_patches

    def perform_preload(self):
        print('SITK_Dataset_Patchwise Message: ')
        cnt = 0
        updated_dfiles = []
        for item in tqdm(range(len(self.dfiles)),
                         desc='Preloading Dataset: ',
                         ncols=80):
            temp = self.preload_transforms(
                deepcopy(self.dfiles[item])
            )
            items_to_save = [[key, self.dfiles[item][key],temp[key]]
                             for key in temp
                             if key in self.dfiles[item] and
                             self.dfiles[item][key]!=temp[key]]
            for save_item in items_to_save:
                self.files_silo['save' + str(cnt)] = save_item[2]
                self.dfiles[item][save_item[0]] = 'save' + str(cnt)
                cnt+=1
            # self.files_silo.update(items_to_save)
            x = [slice_3Dmatrix(img, self.pch_sz, self.overlap) for img in temp['X']]

            for pt_idx in range(len(x[0])):
                if x[0][pt_idx].mean()+1.0>1e-3:
                    pt_temp = {key: self.dfiles[item][key] for key in self.dfiles[item]}
                    pt_temp['patch_location'] = pt_idx
                    updated_dfiles.append(pt_temp)

        self.dfiles = updated_dfiles


    def __len__(self):
        if self.condition:
            return len([fl for fl in self.dfiles if self.condition(fl)])
        else:
            return len(self.dfiles)

    def __getitem__(self, item):
        if self.condition:
            files = [fl for fl in self.dfiles if self.condition(fl)]
        else:
            files = self.dfiles

        if self.preloaded:
            example = deepcopy(files[item])
            add_to_example = {key:self.files_silo[example[key]] for key in
                              example if example[key] in self.files_silo.keys()}
            example.update(add_to_example)

            x = [slice_3Dmatrix(img, self.pch_sz, self.overlap) for img in example['X']]
            y = [slice_3Dmatrix(img, self.pch_sz, self.overlap) for img in
                 example['y']]

            x = np.array([ch[example['patch_location']] for ch in x])
            y = np.array([ch[example['patch_location']] for ch in y])
            example['X'] = x
            example['y'] = y
        else:
            raise Exception('Dataset without preload functions has not been '
                            'implemented.')

        if self.transforms:
            example = self.transforms(example)

        return example

class SITK_Dataset_Slicewise(Images_Dataset):
    def __init__(self, axis_oi, data_df=pd.DataFrame([]),
                 preload_transforms=None,
                 transforms=None, preload_data=False):
        super(SITK_Dataset_Slicewise, self).__init__(data_df,
                                                     preload_transforms,
                                                     transforms,
                                                     preload_data)
        self.axis_oi = axis_oi

    def perform_preload(self):
        print('SITK_Dataset_Patchwise Message: ')
        cnt = 0
        updated_dfiles = []
        for item in tqdm(range(len(self.dfiles)),
                         desc='Preloading Dataset: ',
                         ncols=80):
            temp = self.preload_transforms(
                deepcopy(self.dfiles[item])
            )
            items_to_save = [[key, self.dfiles[item][key],temp[key]]
                             for key in temp
                             if key in self.dfiles[item] and
                             self.dfiles[item][key]!=temp[key]]
            for save_item in items_to_save:
                self.files_silo['save' + str(cnt)] = save_item[2]
                self.dfiles[item][save_item[0]] = 'save' + str(cnt)
                cnt+=1
            # self.files_silo.update(items_to_save)
            x = [np.split(img, img.shape[self.axis_oi], axis=self.axis_oi) for
                 img in temp['X']]

            for pt_idx in range(len(x[0])):
                if x[0][pt_idx].mean()>1e-3:
                    pt_temp = {key: self.dfiles[item][key] for key in self.dfiles[item]}
                    pt_temp['slice_location'] = pt_idx
                    updated_dfiles.append(pt_temp)

        self.dfiles = updated_dfiles


    def __len__(self):
        if self.condition:
            return len([fl for fl in self.dfiles if self.condition(fl)])
        else:
            return len(self.dfiles)

    def __getitem__(self, item):
        if self.condition:
            files = [fl for fl in self.dfiles if self.condition(fl)]
        else:
            files = self.dfiles

        if self.preloaded:
            example = deepcopy(files[item])
            add_to_example = {key:self.files_silo[example[key]] for key in
                              example if example[key] in self.files_silo.keys()}
            example.update(add_to_example)

            x = [np.split(img, img.shape[self.axis_oi], axis=self.axis_oi)
                 for img in example['X']]
            y = [np.split(img, img.shape[self.axis_oi], axis=self.axis_oi)
                 for img in example['y']]

            x = np.array([ch[example['slice_location']] for ch in x])[:,0,...]
            y = np.array([ch[example['slice_location']] for ch in y])[:,0,...]
            example['X'] = x
            example['y'] = y
        else:
            raise Exception('Dataset without preload functions has not been '
                            'implemented.')

        if self.transforms:
            example = self.transforms(example)

        return example