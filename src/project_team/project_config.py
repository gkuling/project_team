'''
base project configuration inspired by the hugging face transformers
configuration classes
'''

import difflib
import inspect
import os
import warnings

from transformers import PretrainedConfig
from transformers.utils import logging

try:
    # Private transformers module: present in 4.30-5.x today, but guard the
    # import so a future removal only disables the (unused) auto-class branch.
    from transformers.dynamic_module_utils import custom_object_save
except ImportError:
    custom_object_save = None

logger = logging.get_logger(__name__)

# Every config saves as <ClassName> + CONFIG_FILE_SUFFIX so one experiment
# folder can hold each team member's config side by side. Named to avoid
# confusion with transformers' own CONFIG_NAME, which is 'config.json'.
CONFIG_FILE_SUFFIX = '.json'

# Keys PretrainedConfig itself adds to a saved config. Derived
# programmatically because the set differs between transformers versions;
# underscore-stripped variants cover keys stored privately but saved bare.
_HF_RESERVED_KEYS = frozenset(PretrainedConfig().to_dict()) | frozenset(
    k.lstrip('_') for k in PretrainedConfig().to_dict()) | {
    'torch_dtype', 'dtype', 'attn_implementation'
}


def is_primitive(thing):
    '''
    check if the thing input is a primitive feature that can be stored in a
    json config file
    :param thing: any object
    :return: bool, whether thing is a primitive or a container of primitives
    '''
    primitives = (int, float, str, bool)
    if isinstance(thing, type(None)):
        return True
    elif isinstance(thing, (list, tuple)):
        return all([is_primitive(v) for v in thing])
    elif isinstance(thing, dict):
        return all(is_primitive(x) for x in thing.values()) and \
            all(is_primitive(x) for x in thing.keys())
    else:
        return isinstance(thing, primitives)


def is_Primitive(thing):
    '''Deprecated: use is_primitive().'''
    warnings.warn('is_Primitive() is deprecated; use is_primitive().',
                  DeprecationWarning, stacklevel=2)
    return is_primitive(thing)


class project_config(PretrainedConfig):
    '''
    base project config. Big contribution is it continually saves configs as
    json dictionaries that are easy to read in notepad and edit manually
    '''
    # some subclasses have required parameters, so transformers must never
    # try to build a default instance of them (its repr/diff machinery
    # otherwise calls self.__class__() and crashes)
    has_no_defaults_at_init = True

    # keys a subclass __init__ derives and stores (not parameters) that
    # legitimately round-trip through kwargs on reload — subclasses extend
    # this so those keys don't trigger the unrecognized-kwarg warning
    _derived_config_keys = frozenset()

    def __init__(self,
                 config_type=None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        if not isinstance(config_type, str):
            raise TypeError(
                "config_type must be a string naming this config's kind, "
                "got " + repr(config_type) + ". Subclasses set it with "
                "kwargs.setdefault('config_type', '...') before calling "
                "super().__init__(**kwargs)."
            )
        self.config_type = config_type
        self._warn_unrecognized_kwargs(kwargs)

    def _warn_unrecognized_kwargs(self, kwargs):
        '''
        warn (never raise) about keyword arguments that are not a declared
        parameter of any __init__ in this config's class hierarchy. They are
        still stored as attributes (the huggingface convention, and required
        for save/load round-trips), but a typo'd parameter would otherwise
        silently have no effect.
        :param kwargs: the keyword arguments that reached this constructor
        :return: None
        '''
        recognized = set()
        for klass in type(self).__mro__:
            init = getattr(klass, '__init__', None)
            if init is None:
                continue
            try:
                recognized.update(inspect.signature(init).parameters)
            except (TypeError, ValueError):
                continue
        for key in kwargs:
            if key.startswith('_') or key in _HF_RESERVED_KEYS \
                    or key in recognized \
                    or key in type(self)._derived_config_keys:
                continue
            suggestion = difflib.get_close_matches(
                key, sorted(recognized), n=1)
            did_you_mean = ' Did you mean: ' + suggestion[0] + '?' \
                if suggestion else ''
            warnings.warn(
                type(self).__name__ + ": '" + key + "' is not a recognized "
                "parameter and will only be stored as an attribute." +
                did_you_mean,
                UserWarning, stacklevel=2
            )

    def save_pretrained(self, save_directory, push_to_hub=False, **kwargs):
        """
        Save a configuration object to the directory `save_directory` as
        <ClassName>.json, so that it can be re-loaded using the
        [`~project_config.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved
                (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether to push the config to the Hugging Face hub after
                saving, via the inherited public `push_to_hub` method.
            kwargs (`Dict[str, Any]`, *optional*):
                `repo_id`, `commit_message`, `private`, `token` and
                `create_pr` are forwarded to `push_to_hub`; anything else
                is ignored.
        """
        if os.path.isfile(save_directory):
            raise AssertionError(
                "Provided path (" + str(save_directory) + ") should be a "
                "directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        # If we have a custom config, we copy the file defining it in the
        # folder so it can be loaded from the Hub.
        if custom_object_save is not None and self._auto_class is not None:
            custom_object_save(self, save_directory, config=self)

        # use_diff=False writes every attribute, so the saved file is a
        # complete, hand-editable record. (use_diff=True would construct a
        # default instance internally, which crashes for configs with
        # required parameters.)
        output_config_file = os.path.join(
            save_directory, type(self).__name__ + CONFIG_FILE_SUFFIX)
        self.to_json_file(output_config_file, use_diff=False)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            repo_id = kwargs.pop(
                'repo_id', os.path.basename(os.path.normpath(save_directory)))
            allowed = ('commit_message', 'private', 'token', 'create_pr')
            hub_kwargs = {k: kwargs[k] for k in allowed if k in kwargs}
            return self.push_to_hub(repo_id, **hub_kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load a configuration saved by `save_pretrained`. Looks for
        <ClassName>.json in the given directory — transformers' own loader
        looks for config.json, which this package never writes, and it
        silently returns a default-valued config when that file is missing.
        """
        kwargs.setdefault('_configuration_file',
                          cls.__name__ + CONFIG_FILE_SUFFIX)
        return super().from_pretrained(
            pretrained_model_name_or_path, **kwargs)
