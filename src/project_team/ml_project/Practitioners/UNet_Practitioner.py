import gc

import torch.optim
from project_team.ml_project.Practitioners.segmentation_losses import \
    GeneralizedDiceLoss
from sklearn.metrics import multilabel_confusion_matrix
from torchvision import transforms
from tqdm import tqdm
from scipy.stats import hmean
from project_team.project_config import project_config
from .PT_Practitioner import PTPractitioner_config, PT_Practitioner
from skimage.filters import threshold_otsu
import numpy as np
from project_team.dt_project.dt_processing import *

class UNet_Practitioner_config(PTPractitioner_config, project_config):
    def __init__(self,
                 bce_weights=(1.),
                 loss_type='GDSC',
                 gdsc_normalization='softmax',
                 dsc_epsilon=1e-8,
                 output_threshold='argmax',
                 **kwargs):
        '''
        configuration for the UNet practitioner
        :param bce_weights: weights usd for weighted BCE loss
        :param loss_type: loass type. See below for options
        :param gdsc_normalization: activation of output layer
        :param dsc_epsilon: epsilon for DSC. default: 1e-08
        :param output_threshold: choice of output post processing, either
        threshold or argmax
        :param kwargs:
        '''
        super(UNet_Practitioner_config, self).__init__(
            config_type='ML_UNetPractitioner', **kwargs)
        # Training Parameters
        self.bce_weights = bce_weights
        self.loss_type = loss_type
        self.gdsc_normalization = gdsc_normalization
        self.dsc_epsilon = dsc_epsilon

        if type(output_threshold)==float:
            assert output_threshold>=0.0 and output_threshold<=1.0
        elif type(output_threshold)==str:
            assert output_threshold in ['otsu', 'argmax']

        self.output_threshold = output_threshold
        if 'affine_aug' in kwargs.keys():
            self.affine_aug_y = kwargs['affine_aug']

class UNet_Practitioner(PT_Practitioner):
    """"
    This should be a child class of a pytorch practitioner
    """
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=UNet_Practitioner_config()):
        super(UNet_Practitioner, self).__init__(model=model,
                                                io_manager=io_manager,
                                                data_processor=data_processor,
                                                trainer_config=trainer_config)
        self.practitioner_name = 'UNet'
        self.standard_transforms.extend([
            Add_Channel(field_oi='y'),
            OneHotEncode_Seg(max_class=self.model.config.out_channels),
            ToTensor(field_oi='y')
        ])

    def setup_loss_functions(self):
        '''
        segmentation losses are different from other PT tasks
        '''
        self.bce_criterion = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.config.bce_weights)
        )
        self.dsc_criterion = GeneralizedDiceLoss(
            normalization=self.config.gdsc_normalization,
            epsilon=self.config.dsc_epsilon
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.bce_criterion = self.bce_criterion.cuda()
            self.dsc_criterion = self.dsc_criterion.cuda()

    def calculate_loss(self, py, y):
        '''
        loss calculation for segmentation
        :param py: prediction
        :param y: groundtruth
        :return: loss value
        '''
        if type(self.config.loss_type)==str and 'CE' in self.config.loss_type:
            bce = self.bce_criterion(py, torch.argmax(y, dim=1))
        else:
            bce = None

        if type(self.config.loss_type)==str and 'DSC' in self.config.loss_type:
            dsc = self.dsc_criterion(py, y)
        else:
            dsc = None

        if self.config.loss_type=='Balanced CE and GDSC':
            loss = 0.5 * (bce + dsc)
        elif self.config.loss_type=='GDSC':
            loss = dsc
        elif self.config.loss_type=='CE':
            loss = bce
        elif callable(self.config.loss_type):
            loss = self.config.loss_type(py, y)
        else:
            raise ValueError(
                "The " + str(self.config.loss_type) + ' is not an implemented'
                                                      ' loss type. '
            )

        return loss

    def validate_model(self, val_dataloader):
        '''
        run validation on the validation set
        :param val_dataloader: data loader of validation examples
        :return: average validation loss
        '''
        print('')
        self.model.eval()
        epoch_iterator = tqdm(val_dataloader, desc="  Validation",
                              position=0, leave=True)
        epoch_iterator.set_postfix({'loss': 'Initialized'})
        vl_lss = []
        with torch.no_grad():
            for batch_idx, data in enumerate(epoch_iterator):
                if torch.cuda.is_available():
                    btch_x = data['X'].cuda()
                else:
                    btch_x = data['X']
                btch_y = data['y']
                mdl_pred = self.model(btch_x)
                if self.config.gdsc_normalization=='softmax':
                    pred = torch.softmax(mdl_pred, dim =1)
                    if torch.cuda.is_available():
                        pred = pred.detach().cpu()
                    else:
                        pred = pred.detach()
                elif self.config.gdsc_normalization=='sigmoid':
                    pred = torch.sigmoid(mdl_pred)
                    if torch.cuda.is_available():
                        pred = pred.detach().cpu()
                    else:
                        pred = pred.detach()

                if type(self.config.output_threshold)==float:
                    assert self.config.output_threshold<=1.0 and \
                           self.config.output_threshold>=0.0
                    btch_y = np.array(
                        btch_y.numpy()>self.config.output_threshold
                    ).astype(btch_y.numpy().dtype)
                    pred = np.array(
                        pred.numpy()>self.config.output_threshold
                    ).astype(pred.numpy().dtype)
                elif self.config.output_threshold == 'otsu':
                    btch_y = np.array(
                        btch_y.numpy()>threshold_otsu(pred.numpy())
                    ).astype(btch_y.numpy().dtype)
                    pred = np.array(
                        pred.numpy()>threshold_otsu(pred.numpy())
                    ).astype(pred.numpy().dtype)
                elif self.config.output_threshold == 'argmax':
                    btch_y = btch_y.numpy().argmax(axis=1)
                    pred = pred.numpy().argmax(axis=1)
                else:
                    raise Exception(str(self.config.output_threshold) +
                                    "is not an output threshold option")


                mcm = multilabel_confusion_matrix(btch_y.flatten(), pred.flatten())
                lbls = list(np.union1d(btch_y.flatten(), pred.flatten()))
                dsc = [1.0 for _ in range(len(mcm))]

                for ind, cm in enumerate(mcm):
                    tn, fp, fn, tp= cm.ravel()
                    dsc[int(lbls[ind])] = 2 * tp / (2 * tp + fp + fn)
                if len(dsc) != self.model.config.out_channels + 1 \
                        if self.model.config.out_channels==1 else \
                        self.model.config.out_channels:
                    print('ML Message: WARNING, a segmentation GT dosent have enough classes.')
                    continue
                vl_lss.append(dsc)
                epoch_iterator.set_postfix({'loss': [np.round(d, decimals=2)
                                                     for d in dsc]})

        vl_loss = np.array(vl_lss).mean(0)
        print(" ML Message: Validation Loss: " + str(vl_loss))
        vl_loss = hmean(vl_loss)
        return vl_loss

    def run_inference(self):
        '''
        run inference on the iference dataset in the processor
        :return: all prediction results are saved on the data processor
        inference_results
        '''
        # here we only use the standard transforms that affects the input
        trnsfrms = [trsnfrm for trsnfrm in self.standard_transforms if
                    trsnfrm.field_oi == 'X']

        self.data_processor.if_dset.set_transforms(transforms.Compose(trnsfrms))
        # remember that using a data loader here will mess up post processing
        # steps
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
                mdl_pred = self.model(btch_x)

            # post process the result
            if self.config.gdsc_normalization=='softmax':
                pred = torch.softmax(mdl_pred, dim =1)
                if torch.cuda.is_available():
                    pred = pred.detach().cpu()
                else:
                    pred = pred.detach()
            elif self.config.gdsc_normalization=='sigmoid':
                pred = torch.sigmoid(mdl_pred)
                if torch.cuda.is_available():
                    pred = pred.detach().cpu()
                else:
                    pred = pred.detach()
            if type(self.config.output_threshold)==float:
                assert self.config.output_threshold<=1.0 and \
                       self.config.output_threshold>=0.0
                pred = np.array(
                    pred.numpy()>self.config.output_threshold
                ).astype(pred.numpy().dtype)
            elif self.config.output_threshold == 'otsu':
                pred = np.array(
                    pred.numpy()>threshold_otsu(pred.numpy())
                ).astype(pred.numpy().dtype)
            elif self.config.output_threshold == 'argmax':
                pred = pred.numpy().argmax(axis=1)[:,None,...]
            else:
                raise Exception(str(self.config.output_threshold) +
                                  "is not an output threshold option")
            pred = pred[0]
            res['pred_y'] = \
                    [pred[_] for _ in range(pred.shape[0])]

            return_results.append(res)
        self.data_processor.post_process_inference_results(return_results)

        torch.cuda.empty_cache()
        gc.collect()
        print(" ML Message: UNet Practitioner has finished Running Inference ")
