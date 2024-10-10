'''
This si top test out a weird config save bug
'''

from project_team import io_project
import os

io_args = {
    'root': 'data',
    'data': 'mnist',
    'x': 'image',
    'y': 'label',
    'y_dtype': 'discrete',
    'y_domain': [_ for _ in range(10)],
    'group_data_by': None,
    'test_size': 0.0,
    'validation_size': 0.1,
    'stratify_by': 'label',
    'r_seed': 0
}
from transformers import BertConfig

another = BertConfig()
another.save_pretrained(os.getcwd())
io_project_cnfg = io_project.io_traindeploy_config(**io_args)
io_project_cnfg.save_pretrained(os.getcwd())
