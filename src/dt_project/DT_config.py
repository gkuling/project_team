from src.project_config import project_config

class DT_config(project_config):
    def __init__(self,
                 pre_load=True,
                 **kwargs
                 ):
        super(DT_config, self).__init__('DT')

        self.pre_load = pre_load
        self.patch_based = False
        self.slice_based = False

    def initialize_patch_based(self,
                               patch_size=(16,16,16),
                               overlap=(8,8,8)):
        self.patch_based = True
        self.patch_size = patch_size
        self.overlap = overlap

    def initialize_slice_based(self,
                               axis=0):
        self.slice_based = True
        self.slice_axis = axis