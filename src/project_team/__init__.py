
__version__ = "0.0.2"

import io
import contextlib

from . import dt_project
from . import io_project
from . import ml_project
from .ml_project import models


# Not sure if this function is doing anything???
def maketabbed(ml_project):
    def tabbed():
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ml_project
        for line in output.getvalue().splitlines():
            print('\t' + line)
    return tabbed