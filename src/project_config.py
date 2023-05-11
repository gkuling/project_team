'''
base project configuration inspired by the hugging face transformers
configuration classes
'''

import os
import copy
import json
import inspect

CONFIG_NAME = '.json'

def is_Primitive(thing):
    '''
    check if the thing input is a primitive feature
    :param thing:
    :return:
    '''
    primitives = (int, float, str, bool)
    if isinstance(thing, type(None)):
        return True
    elif isinstance(thing, list):
        return all([is_Primitive(v) for v in thing])
    elif isinstance(thing, dict):
        return all(is_Primitive(x) for x in thing.values()) and \
            all(is_Primitive(x) for x in thing.keys())
    elif isinstance(thing, tuple):
        return all([is_Primitive(v) for v in thing])
    else:
        return isinstance(thing, primitives)

class project_config(object):
    '''
    base proejct config. Big contribution is it continually saves cnfigs as
    json dictionaries that are easy to read in notepad and edit manually
    '''
    def __init__(self,
                 config_type,
                 **kwargs
                 ):
        assert type(config_type)==str, "Config type must be a string. "
        self.config_type = config_type

    def save_pretrained(self, save_folder=None):
        if hasattr(self, 'project_folder') and not save_folder:
            save_folder = self.project_folder

        if os.path.isfile(save_folder):
            raise AssertionError(f"Provided path ({save_folder}) "
                                 f"should be a directory, not a file")
        output_config_file = os.path.join(save_folder,
                                          type(self).__name__ + CONFIG_NAME)
        with open(output_config_file, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())

    def to_json_string(self):
        config_dict = self.to_dict()

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        for k, o in output.items():
            if isinstance(o, str) or o is None or o is True or o is False or \
                    isinstance(o, int) or isinstance(o, float) or \
                    isinstance(o, (list, tuple, dict)):
                pass
            else:
                output.update({k:o.__str__()})
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path):

        config_dict = cls.get_config_dict(pretrained_model_name_or_path)

        return cls.from_dict(config_dict)

    @classmethod
    def _dict_from_json_file(cls, json_file):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)

    @classmethod
    def get_config_dict(cls, pretrained_model_name_or_path):
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path,
                                       cls.__name__+'.json')
        elif os.path.isfile(pretrained_model_name_or_path):
            config_file = pretrained_model_name_or_path
        else:
            raise ValueError(pretrained_model_name_or_path +
                             " does not contain a " +
                             cls.__name__ + ".json or is not a proper file.")

        try:
            config_dict = cls._dict_from_json_file(config_file)
        except:
            extra_check = [fl for fl in
                           os.listdir(pretrained_model_name_or_path)
                           if cls.__name__ in fl]
            if len(extra_check)==1:
                config_file = os.path.join(pretrained_model_name_or_path,
                                           extra_check[0])
                config_dict = cls._dict_from_json_file(config_file)
            elif len(extra_check)>1:
                raise TypeError(
                    "The folder " + str(pretrained_model_name_or_path) +
                    " has more than one config json file that matches " +
                    cls.__name__ + ". Double check thhis input folder"
                )
            else:
                raise TypeError("You are not loading a config in .json format. ")

        config_dict.pop(('config_type'))

        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        args_in_init = inspect.getfullargspec(cls).args

        config = cls(**{key:value for key, value in config_dict.items() if
                        key in args_in_init})

        extra_config_values = {key:value for key, value in config_dict.items() if
                               key not in args_in_init}
        if extra_config_values:
            for key, value in extra_config_values.items():
                config.__setattr__(key, value)
        return config
