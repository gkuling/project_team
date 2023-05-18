
__version__ = "1.0.0.dev0"

from . import io_project as io_project, dt_project as dt_project
from . import ml_project as ml_project
from .ml_project import models as models

import io
import contextlib


# Not sure if this function is doing anything???
def maketabbed(ml_project):
    def tabbed():
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ml_project
        for line in output.getvalue().splitlines():
            print('\t' + line)
    return tabbed