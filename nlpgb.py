import pandas as pd
import torch.nn
from tqdm import tqdm
import gc
import numpy as np
from scipy.special import expit as sigmoid, softmax

from project_team.src.project_config import project_config, is_Primitive
from .PT_Practitioner import PTPractitioner_config, PT_Practitioner
from project_team.src.dt_project.dt_processing import *
from torchvision import transforms

class PTClassification_Practitioner_config(PTPractitioner_config,
                                       project_config):
    def __init__(self,
                 **kwargs):
        '''
        Specific configuration for running pytorch classification
        practitioner
        '''
        super(PTClassification_Practitioner_config, self).__init__(
            config_type ='ML_PTClassificationPractitioner', **kwargs
        )


class PTClassification_Practitioner(PT_Practitioner):
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=PTClassification_Practitioner_config()):
        '''
        constructor of the pytorch classification practitioner. Inherits the
        pytorch practitioner
        :param model: pytorch model
        :param io_manager: manager to be used
        :param data_processor: data processor to be used
        :param trainer_config: the configuration that holds parameters for a
        practitioner
        '''
        super(PTClassification_Practitioner, self).__init__(model=model,
                                                io_manager=io_manager,
                                                data_processor=data_processor,
                                                trainer_config=trainer_config)

        self.practitioner_name = 'PTClassification'
        # Standard transfroms for training would be in ensure all input and
        # out put are tensors
        self.standard_transforms.extend([ToTensor(field_oi='X'),
                                         ToTensor(field_oi='y')])

    def validate_model(self, val_dataloader):
        '''
        function that will run validation of the model
        :param mdl: the currently trained model.
        :param val_dataloader: validation data laoder
        :return: the overall validation loss
        '''
        print('')
        self.model.eval()
        epoch_iterator = tqdm(val_dataloader, desc="  Validation",
                              position=0, leave=True)
        epoch_iterator.set_postfix({'loss': 'Initialized'})
        vl_lss = []
        with torch.no_grad():
            for batch_idx, data in enumerate(epoch_iterator):
                # prepare data
                if torch.cuda.is_available():
                    btch_x = data['X'].cuda()
                    btch_y = data['y'].cuda()
                else:
                    btch_x = data['X']
                    btch_y = data['y']
                # run forward pass and test result
                mdl_pred = self.model(btch_x)
                loss = self.calculate_loss(mdl_pred, btch_y)

                if torch.cuda.is_available():
                    loss = loss.cpu().numpy()[None]
                else:
                    loss = loss.numpy()[None]
                epoch_iterator.set_postfix({'loss': np.round(loss,
                                                             decimals=2).tolist()})
                vl_lss.append(loss)
        # calculate average loss for the validaiton data
        vl_loss = np.array(vl_lss).mean(0)
        print(" ML Message: Validation Loss: " + str(vl_loss))
        return vl_loss[0]

    def run_inference(self, return_output=False):
        '''
        run inference on the iference dataset in the processor
        :return: all prediction results are saved on the data processor
        inference_results
        '''
        # here we only use the standard transforms that affects the input
        trnsfrms = [trsnfrm for trsnfrm in self.standard_transforms if
                    trsnfrm.field_oi=='X']

        self.data_processor.if_dset.set_transforms(transforms.Compose(trnsfrms))

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        epoch_iterator = tqdm(self.data_processor.if_dset, desc="  Inference",
                              position=0, leave=True)
        return_results = []
        for batch_idx, data in enumerate(epoch_iterator):
            # prepare data
            res = data.copy()
            if torch.cuda.is_available():
                btch_x = data['X'][None,...].cuda()
            else:
                btch_x = data['X'][None,...]

            # run inference
            with torch.no_grad():
                pred = self.model(btch_x)

            # save logits if desired
            if return_output:
                if torch.cuda.is_available():
                    res['outputs'] = pred.cpu().numpy()[0].tolist()
                else:
                    res['outputs'] = pred.numpy()[0].tolist()

            # post process the result
            if torch.cuda.is_available():
                res['pred_y'] = pred.cpu().numpy()
            else:
                res['pred_y'] = pred.numpy()

            if hasattr(self.model, 'config') and  hasattr(self.model.config, \
                    'output_style'):
                style = self.model.config.output_style
            elif hasattr(self.model, 'module') and \
                hasattr(self.model.module, 'config') and \
                hasattr(self.model.module.config, 'output_style'):
                style = self.model.module.config.output_style
            else:
                style = 'softmax'


            if style == 'softmax':
                res['pred_y'] = res['pred_y'].argmax()
                if hasattr(self.io_manager.exp_type.config,'y_domain'):
                    res['pred_y'] = self.io_manager.exp_type.config.y_domain[
                        res['pred_y']
                    ]
            elif style == 'sigmoid':
                res['pred_y'] = sigmoid(res['pred_y'])>0.5
                if hasattr(self.io_manager.exp_type.config,'y_domain'):
                    res['pred_y'] = [
                        self.io_manager.exp_type.config.y_domain[_]
                        for _ in range(res['pred_y'].shape[1])
                        if res['pred_y'][_]==1
                    ]
            else:
                raise Exception(' the ouput_style is not recognized. ' + str(
                    style))


            return_results.append(res)

        # save results in the data_processor
        if return_output:
            self.data_processor.inference_results = pd.DataFrame(return_results)
        else:
            self.data_processor.inference_results = pd.DataFrame(
                [
                    {ky:v for ky, v in ex.items() if is_Primitive(v)}
                    for ex in return_results
                ]
            )

        torch.cuda.empty_cache()
        gc.collect()
        print(" ML Message: " + self.practitioner_name +
              " Practitioner has finished Running Inference ")