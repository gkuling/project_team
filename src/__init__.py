from src import io_project as io_project, dt_project as dt_project
from src import ml_project as ml_project
from src.ml_project import models
import io
import contextlib

def maketabbed(ml_project):
    def tabbed():
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ml_project
        for line in output.getvalue().splitlines():
            print('\t' + line)
    return tabbed