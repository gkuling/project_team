import os
import torch

from copy import deepcopy

from .IO_Manager import IO_Manager

class Pytorch_Manager(IO_Manager):
    def __init__(self, io_config_input):
        super(Pytorch_Manager, self).__init__(io_config_input)

    def check_if_model_trained(self):
        if any([x=='final_model.pth' for x in
                os.listdir(self.root)]) and \
                any(['Practitioner_config.json' in x for x in
                     os.listdir(self.root)]) and \
                any(['Processor_config.json' in x for x in
                     os.listdir(self.root)]):
            return True
        else:
            return False

    def set_best_model(self, pt_model_state_dict):
        self.best_model = deepcopy(pt_model_state_dict)

    def save_model_checkpoint(self, trainer_config):
        output_dir_lcl = os.path.join(
            self.root,'checkpoint'
        )
        trainer_config = deepcopy(trainer_config)
        trainer_config.trained_setps = trainer_config.best_vl_step
        if not os.path.exists(output_dir_lcl):
            os.makedirs(output_dir_lcl)
        torch.save(self.best_model,
                   output_dir_lcl + '/final_model.pth')
        # Save the trianer config file
        trainer_config.save_pretrained(output_dir_lcl)


    def save_final_model(self, trainer_config, data_processor_config,
                         model_config):
        torch.save(self.best_model,
                   self.root + '/final_model.pth')
        # Save the trianer config file
        trainer_config.save_pretrained(self.root)
        data_processor_config.save_pretrained(self.root)
        model_config.save_pretrained(self.root)