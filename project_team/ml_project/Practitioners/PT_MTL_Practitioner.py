import pandas as pd
import torch.nn
from tqdm import tqdm
import gc
import numpy as np

from project_team.project_config import project_config, is_Primitive
from .PT_Practitioner import PTPractitioner_config, PT_Practitioner
from project_team.dt_project.dt_processing import *
from torchvision import transforms

class PT_MTL_Practitioner_config(PTPractitioner_config,
                                       project_config):
    def __init__(self,
                 **kwargs):
        super(PT_MTL_Practitioner_config, self).__init__(
            config_type ='ML_PT_MTL_Practitioner', **kwargs
        )

class PT_MTL_Practitioner(PT_Practitioner):
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=PT_MTL_Practitioner_config()):
        super(PT_MTL_Practitioner, self).__init__(
            model=model,
            io_manager=io_manager,
            data_processor=data_processor,
            trainer_config=trainer_config
        )
        self.practitioner_name = 'PT_MTL'
        self.standard_transforms.extend([
            Cast_numpy(field_oi='y', data_type=np.float32),
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
                btch_x, btch_y = self.organize_input_and_output(data)
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
        for batch_idx, data in enumerate(epoch_iterator):
            res = data.copy()
            if torch.cuda.is_available():
                btch_x = data['X'][None,...].cuda()
            else:
                btch_x = data['X'][None,...]

            with torch.no_grad():
                pred = self.model(btch_x)
            for task in self.io_manager.config.y:
                if torch.cuda.is_available():
                    res[task + '_pred_y'] = pred[task].cpu().numpy().item()
                else:
                    res[task +'_pred_y'] = pred[task].numpy().item()

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