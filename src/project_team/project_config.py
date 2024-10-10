'''
base project configuration inspired by the hugging face transformers
configuration classes
'''

import os
import copy
import json
import inspect
import warnings

from transformers import PretrainedConfig
from transformers.utils import logging
from transformers.dynamic_module_utils import custom_object_save
logger = logging.get_logger(__name__)
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
from transformers import PretrainedConfig
class project_config(PretrainedConfig):
    '''
    base proejct config. Big contribution is it continually saves cnfigs as
    json dictionaries that are easy to read in notepad and edit manually
    '''
    def __init__(self,
                 config_type,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        assert type(config_type)==str, "Config type must be a string. "
        self.config_type = config_type

    def save_pretrained(self, save_directory, push_to_hub = False, **kwargs):
        """
        Save a configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~PretrainedConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """
        self._set_token_in_kwargs(kwargs)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        non_default_generation_parameters = self._get_non_default_generation_parameters()
        if len(non_default_generation_parameters) > 0:
            # TODO (joao): this should be an exception if the user has modified the loaded config. See #33886
            warnings.warn(
                "Some non-default generation parameters are set in the model config. These should go into either a) "
                "`model.generation_config` (as opposed to `model.config`); OR b) a GenerationConfig file "
                "(https://huggingface.co/docs/transformers/generation_strategies#save-a-custom-decoding-strategy-with-your-model)."
                "This warning will become an exception in the future."
                f"\nNon-default generation parameters: {str(non_default_generation_parameters)}",
                UserWarning,
            )

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        # If we have a custom config, we copy the file defining it in the folder and set the attributes so it can be
        # loaded from the Hub.
        if self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, type(self).__name__ + CONFIG_NAME)

        self.to_json_file(output_config_file, use_diff=True)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    # def save_pretrained(self, save_folder=None):
    #     if hasattr(self, 'project_folder') and not save_folder:
    #         save_folder = self.project_folder
    #
    #     if os.path.isfile(save_folder):
    #         raise AssertionError(f"Provided path ({save_folder}) "
    #                              f"should be a directory, not a file")
    #     output_config_file = os.path.join(save_folder,
    #                                       type(self).__name__ + CONFIG_NAME)
    #     with open(output_config_file, "w", encoding="utf-8") as writer:
    #         writer.write(self.to_json_string())
#
#     def to_json_string(self):
#         config_dict = self.to_dict()
#
#         return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
#
#     def to_dict(self):
#         output = copy.deepcopy(self.__dict__)
#         for k, o in output.items():
#             if isinstance(o, str) or o is None or o is True or o is False or \
#                     isinstance(o, int) or isinstance(o, float) or \
#                     isinstance(o, (list, tuple, dict)):
#                 pass
#             else:
#                 output.update({k:o.__str__()})
#         return output
#
#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path):
#
#         config_dict = cls.get_config_dict(pretrained_model_name_or_path)
#
#         return cls.from_dict(config_dict)
#
#     @classmethod
#     def _dict_from_json_file(cls, json_file):
#         with open(json_file, "r", encoding="utf-8") as reader:
#             text = reader.read()
#         return json.loads(text)
#
#     @classmethod
#     def get_config_dict(cls, pretrained_model_name_or_path):
#         pretrained_model_name_or_path = str(pretrained_model_name_or_path)
#         if os.path.isdir(pretrained_model_name_or_path):
#             config_file = os.path.join(pretrained_model_name_or_path,
#                                        cls.__name__+'.json')
#         elif os.path.isfile(pretrained_model_name_or_path):
#             config_file = pretrained_model_name_or_path
#         else:
#             raise ValueError(pretrained_model_name_or_path +
#                              " does not contain a " +
#                              cls.__name__ + ".json or is not a proper file.")
#
#         try:
#             config_dict = cls._dict_from_json_file(config_file)
#         except:
#             extra_check = [fl for fl in
#                            os.listdir(pretrained_model_name_or_path)
#                            if cls.__name__ in fl]
#             if len(extra_check)==1:
#                 config_file = os.path.join(pretrained_model_name_or_path,
#                                            extra_check[0])
#                 config_dict = cls._dict_from_json_file(config_file)
#             elif len(extra_check)>1:
#                 raise TypeError(
#                     "The folder " + str(pretrained_model_name_or_path) +
#                     " has more than one config json file that matches " +
#                     cls.__name__ + ". Double check thhis input folder"
#                 )
#             else:
#                 raise TypeError("You are not loading a config in .json format. ")
#
#         config_dict.pop(('config_type'))
#
#         return config_dict
#
#     @classmethod
#     def from_dict(cls, config_dict):
#         args_in_init = inspect.getfullargspec(cls).args
#
#         config = cls(**{key:value for key, value in config_dict.items() if
#                         key in args_in_init})
#
#         extra_config_values = {key:value for key, value in config_dict.items() if
#                                key not in args_in_init}
#         if extra_config_values:
#             for key, value in extra_config_values.items():
#                 config.__setattr__(key, value)
#         return config
