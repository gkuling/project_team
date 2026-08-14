'''Deprecated module name (a misspelling); use default_arguments instead.'''
import warnings

warnings.warn('default_arguements is a deprecated module name; import '
              'default_arguments instead.', DeprecationWarning, stacklevel=2)

from default_arguments import *  # noqa: E402,F401,F403
