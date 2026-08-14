import os
import torch

from copy import deepcopy

from .IO_Manager import IO_Manager

class Pytorch_Manager(IO_Manager):
    '''
    an input output manager that is specialized in pytorch models
    '''
    def __init__(self, io_config_input):
        super(Pytorch_Manager, self).__init__(io_config_input)

    @staticmethod
    def _strip_module_prefix(state_dict):
        '''
        remove the 'module.' prefix DataParallel adds to state_dict keys.
        Slicing by prefix length (not str.split) so nested submodules that
        happen to be named 'module' are left intact.
        :param state_dict: a model state dictionary
        :return: the state dictionary with leading 'module.' prefixes removed
        '''
        prefix = 'module.'
        return {(k[len(prefix):] if k.startswith(prefix) else k): v
                for k, v in state_dict.items()}

    def check_if_model_trained(self):
        '''
        check if there is a saved model in the project folder
        :return: bool
        '''
        # the config filenames are <ClassName>.json, so the practitioner and
        # processor configs are matched by suffix — their concrete class
        # names are not known here
        files = os.listdir(self.root)
        return os.path.isfile(os.path.join(self.root, 'final_model.pth')) \
            and any('Practitioner_config.json' in x for x in files) \
            and any('Processor_config.json' in x for x in files)

    def set_final_model(self, pt_model_state_dict):
        '''
        keep a copy of the best model during training
        :param pt_model_state_dict: state dictionary of the best model during
        training
        '''
        self.final_model = deepcopy(pt_model_state_dict)

    def model_save_pretrained(self, practitioner, model_folder=None):
        '''
        save the final trained model in the working directory
        :param practitioner: practitioner that needs to be saved
        :param model_folder: folder to save the files. default is the
        io_manager root
        '''
        model_folder = model_folder or self.root
        torch.save(self.final_model,
                   os.path.join(model_folder, 'final_model.pth'))
        # Save the trainer config file
        practitioner.config.save_pretrained(model_folder)
        practitioner.data_processor.config.save_pretrained(model_folder)
        practitioner.model.config.save_pretrained(model_folder)

    def model_from_pretrained(self, practitioner, model_folder=None):
        '''
        load a pre_trained model in the given model_folder
        :param practitioner: practitioner that is being loaded
        :param model_folder: folder that holds all the saved files. default is
        the manager root
        '''
        model_folder = model_folder or self.root
        state = torch.load(os.path.join(model_folder, 'final_model.pth'),
                           weights_only=True)
        state = self._strip_module_prefix(state)
        practitioner.model.load_state_dict(state)
        self.set_final_model(state)
        # load the trainer config file
        practitioner.config = practitioner.config.from_pretrained(model_folder)
        practitioner.data_processor.config = \
            practitioner.data_processor.config.from_pretrained(model_folder)
        practitioner.model.config = practitioner.model.config.from_pretrained(
            model_folder)

    def save_model_checkpoint(self, practitioner):
        '''
        save the current model and the best model at the given checkpoint
        :param practitioner: practitioner to save checkpoint of
        :return:
        '''
        output_dir_lcl = os.path.join(
            self.root,'checkpoint'
        )
        if not os.path.exists(output_dir_lcl):
            os.makedirs(output_dir_lcl)
        self.model_save_pretrained(practitioner, output_dir_lcl)

        torch.save(practitioner.model.state_dict(),
                   os.path.join(output_dir_lcl, 'current_model.pth'))

    def from_model_checkpoint(self, practitioner):
        '''
        save the current model and the best model at the given checkpoint
        :param practitioner: practitioner to load checkpoint from
        '''
        output_dir_lcl = os.path.join(
            self.root, 'checkpoint'
        )
        if not os.path.exists(output_dir_lcl):
            os.makedirs(output_dir_lcl)
        self.model_from_pretrained(practitioner, output_dir_lcl)

        state = torch.load(os.path.join(output_dir_lcl, 'current_model.pth'),
                           weights_only=True)
        state = self._strip_module_prefix(state)
        practitioner.model.load_state_dict(state)