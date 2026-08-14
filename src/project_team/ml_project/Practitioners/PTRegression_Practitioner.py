import pandas as pd
import torch.nn
from tqdm import tqdm
import gc
from scipy.special import expit as sigmoid, softmax
import numpy as np

from project_team.project_config import project_config, is_primitive
from .PT_Practitioner import PTPractitioner_config, PT_Practitioner
from torchvision import transforms
from project_team.dt_project.dt_processing import ToTensor, Cast_numpy

class PTRegression_Practitioner_config(PTPractitioner_config,
                                       project_config):
    def __init__(self,
                 **kwargs):
        '''
        Specific configuration for running pytorch regression
        practitioner
        '''
        kwargs.setdefault('config_type', 'ML_PTRegressionPractitioner')
        super(PTRegression_Practitioner_config, self).__init__(**kwargs)

class PTRegression_Practitioner(PT_Practitioner):
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=None):
        '''
        constructor of the pytorch regression practitioner. Inherits the
        pytorch practitioner
        :param model: pytorch model
        :param io_manager: manager to be used
        :param data_processor: data processor to be used
        :param trainer_config: the configuration that holds parameters for a
        practitioner
        '''
        if trainer_config is None:
            trainer_config = PTRegression_Practitioner_config()
        super(PTRegression_Practitioner, self).__init__(model=model,
                                                io_manager=io_manager,
                                                data_processor=data_processor,
                                                trainer_config=trainer_config)
        self.practitioner_name = 'PTRegression'

    def get_subclass_standard_transforms(self):
        '''
        standard transforms ensuring the y value ends up a float tensor.
        Declared as a method override (not an attribute set after __init__)
        so the parent picks it up when it builds standard_transforms.
        :return: list of transforms
        '''
        return [
            Cast_numpy(field_oi='y', data_type=np.float32),
            ToTensor(field_oi='y')
        ]

    def validate_model(self, val_dataloader):
        '''
        function that will run validation of the model
        :param val_dataloader: validation data loader
        :return: the overall validation loss
        '''
        self.model.eval()
        epoch_iterator = tqdm(val_dataloader, desc="  Validation",
                              position=0, leave=True)
        epoch_iterator.set_postfix({'loss': 'Initialized'})
        vl_lss = []
        with torch.no_grad():
            for batch_idx, data in enumerate(epoch_iterator):
                if torch.cuda.is_available():
                    btch_x = data['X'].cuda()
                    btch_y = data['y'].cuda()
                else:
                    btch_x = data['X']
                    btch_y = data['y']
                mdl_pred = self.model(btch_x)
                loss = self.calculate_loss(mdl_pred, btch_y)

                if torch.cuda.is_available():
                    loss = loss.cpu().numpy()[None]
                else:
                    loss = loss.numpy()[None]
                epoch_iterator.set_postfix(
                    {'loss': np.round(loss,  decimals=2).tolist()}
                )
                vl_lss.append(loss)
        # calculate average loss for the validaiton data
        vl_loss = np.array(vl_lss).mean(0)
        print(" ML Message: Validation Loss: " + str(vl_loss))
        return vl_loss[0]

    def run_inference(self, return_output=False):
        '''
        run inference on the iference dataset in the processor
        :param: return_output: bool to indicate logits are desired with the
        results
        :return: all prediction results are saved on the data processor
        inference_results
        '''
        # here we only use the standard transforms that affects the input
        trnsfrms = [trsnfrm for trsnfrm in self.standard_transforms if
                    trsnfrm.field_oi == 'X']

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

            if hasattr(self.model, 'config')and  hasattr(self.model.config, \
                    'output_style'):
                style = self.model.config.output_style
            elif hasattr(self.model, 'module') and \
                hasattr(self.model.module, 'config') and \
                hasattr(self.model.module.config, 'output_style'):
                style = self.model.module.config.output_style
            else:
                style = 'continuous'


            if style == 'CORAL':
                res['pred_y'] = (
                                        sigmoid(res['pred_y'])>0.5
                                ).sum() / res['pred_y'].shape[1]
            elif style=='softlabel':
                res['pred_y'] = np.argmax(
                    res['pred_y']
                ) / (res['pred_y'].shape[1] - 1)
            elif style=='continuous':
                res['pred_y'] = res['pred_y'].item()
            elif style=='patchGAN':
                res['pred_y'] = res['pred_y'].mean().item()
            elif style=='binary':
                res['pred_y'] = softmax(res['pred_y'],
                                        axis=1)[:,1].item()
            else:
                raise ValueError('the output_style is not recognized: ' +
                                 str(style))

            return_results.append(res)
        # save results in the data_processor
        if return_output:
            self.data_processor.inference_results = pd.DataFrame(
                return_results)
        else:
            self.data_processor.inference_results = pd.DataFrame(
                [
                    {ky: v for ky, v in ex.items() if is_primitive(v)}
                    for ex in return_results
                ]
            )

        torch.cuda.empty_cache()
        gc.collect()
        print(" ML Message: " + self.practitioner_name +
              " Practitioner has finished Running Inference ")