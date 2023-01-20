import pandas as pd
import torch.nn
from tqdm import tqdm
import gc
import numpy as np
from scipy.special import expit as sigmoid, softmax

from src.project_config import project_config, is_Primitive
from .PT_Practitioner import PTPractitioner_config, PT_Practitioner
from src.dt_project.dt_processing import *
from torchvision import transforms

class PTClassification_Practitioner_config(PTPractitioner_config,
                                       project_config):
    def __init__(self,
                 **kwargs):
        super(PTClassification_Practitioner_config, self).__init__(
            config_type ='ML_PTClassificationPractitioner', **kwargs
        )

class PTClassification_Practitioner(PT_Practitioner):
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=PTClassification_Practitioner_config()):
        super(PTClassification_Practitioner, self).__init__(model=model,
                                                io_manager=io_manager,
                                                data_processor=data_processor,
                                                trainer_config=trainer_config)
        self.practitioner_name = 'PTClassification'

        self.standard_transforms.extend([
            OneHotEncode(),
            ToTensor(field_oi='y')
        ])

    def validate_model(self, mdl, val_dataloader):
        print('')
        mdl.eval()
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
                mdl_pred = mdl(btch_x)
                loss = self.calculate_loss(mdl_pred, btch_y)

                if torch.cuda.is_available():
                    loss = loss.cpu().numpy()[None]
                else:
                    loss = loss.numpy()[None]
                epoch_iterator.set_postfix({'loss': np.round(loss,
                                                             decimals=2).tolist()})
                vl_lss.append(loss)
        vl_loss = np.array(vl_lss).mean(0)
        print(" ML Message: Validation Loss: " + str(vl_loss))
        return vl_loss[0]

    def run_inference(self):
        trnsfrms = [
            Add_Channel(field_oi='X'),
            MnStdNormalize_Numpy(norm=list(self.config.normalization_channels),
                                 percentiles=self.config.normalization_percentiles,
                                 field_oi='X'),
            ToTensor(field_oi='X')
        ]

        self.data_processor.if_dset.set_transforms(transforms.Compose(trnsfrms))

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
        epoch_iterator = tqdm(self.data_processor.if_dset, desc="  Inference",
                              position=0, leave=True)
        return_results = []
        print(hasattr(self.model, 'config'))
        for batch_idx, data in enumerate(epoch_iterator):
            res = data.copy()
            if torch.cuda.is_available():
                btch_x = data['X'][None,...].cuda()
            else:
                btch_x = data['X'][None,...]

            with torch.no_grad():
                pred = self.model(btch_x)

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
            if style=='softlabel':
                res['pred_y'] = np.argmax(
                    res['pred_y']
                ) / (res['pred_y'].shape[1] - 1)
            if style=='continuous':
                res['pred_y'] = res['pred_y'].item()
            if style=='patchGAN':
                res['pred_y'] = res['pred_y'].mean().item()
            if style=='binary':
                res['pred_y'] = softmax(res['pred_y'],
                                        axis=1)[:,1].item()

            return_results.append(res)
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