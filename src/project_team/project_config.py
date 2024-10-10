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
