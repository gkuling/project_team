
__version__ = "1.0.0.dev0"

import io
import contextlib

import dt_project
import io_project
import ml_project


# Not sure if this function is doing anything???
def maketabbed(ml_project):
    def tabbed():
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ml_project
        for line in output.getvalue().splitlines():
            print('\t' + line)
    return tabbed