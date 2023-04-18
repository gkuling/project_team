from src.project_config import project_config

class DT_config(project_config):
    def __init__(self,
                 pre_load=True,
                 **kwargs
                 ):
        super(DT_config, self).__init__('DT')

        self.pre_load = pre_load
