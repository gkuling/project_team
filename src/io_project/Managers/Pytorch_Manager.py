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

    def check_if_model_trained(self):
        '''
        check if there is a saved model in the project folder
        :return: bool
        '''
        if any([x=='final_model.pth' for x in
                os.listdir(self.root)]) and \
                any(['Practitioner_config.json' in x for x in
                     os.listdir(self.root)]) and \
                any(['Processor_config.json' in x for x in
                     os.listdir(self.root)]):
            return True
        else:
            return False

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
        :param practitioner: practitioenr that needs to be saved
        :param model_folder: folder to save the files. default is the
        io_manager root
        '''
        if model_folder:
            pass
        else:
            model_folder = self.root
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
        :param model_folder: folder tha holds all the saved files. default is
        the manager root
        '''
        if model_folder:
            pass
        else:
            model_folder = self.root
        state = torch.load(os.path.join(model_folder, 'final_model.pth'))
        state = {k.split('module.')[1] if k.startswith('module.') else k: v for
                 k, v in state.items()}
        practitioner.model.load_state_dict(state)
        self.set_final_model(state)
        # load the trianer config file
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

        state = torch.load( os.path.join(output_dir_lcl, 'current_model.pth'))
        state = {k.split('module.')[1] if k.startswith('module.') else k: v for
                 k, v in state.items()}
        practitioner.model.load_state_dict(state)