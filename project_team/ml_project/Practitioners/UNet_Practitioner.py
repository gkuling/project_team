import gc

import torch.optim
from project_team.ml_project.segmentation_losses import GeneralizedDiceLoss
from sklearn.metrics import multilabel_confusion_matrix
from project_team.dt_project.dt_processing import *
from torchvision import transforms
from tqdm import tqdm
from scipy.stats import hmean
from project_team.project_config import project_config
from .PT_Practitioner import PTPractitioner_config, PT_Practitioner
from torch.utils.data import DataLoader
from skimage.filters import threshold_otsu
from copy import deepcopy
import numpy as np


class UNet_Practitioner_config(PTPractitioner_config, project_config):
    def __init__(self,
                 bce_weights=(1.),
                 loss_type='GDSC',
                 gdsc_normalization='softmax',
                 balanced_sampler=False,
                 dsc_epsilon=1e-8,
                 output_threshold='argmax',
                 **kwargs):
        super(UNet_Practitioner_config, self).__init__(
            config_type='ML_UNetPractitioner', **kwargs)
        # Training Parameters
        self.bce_weights = bce_weights
        self.loss_type = loss_type
        self.gdsc_normalization = gdsc_normalization
        self.balanced_sampler = balanced_sampler
        self.dsc_epsilon = dsc_epsilon

        if type(output_threshold)==float:
            assert output_threshold>=0.0 and output_threshold<=1.0
        elif type(output_threshold)==str:
            assert output_threshold in ['otsu', 'argmax']

        self.output_threshold = output_threshold
        if 'affine_aug' in kwargs.keys():
            self.affine_aug_y = kwargs['affine_aug']

from torch.utils.data.sampler import Sampler
class BalancedBatchSampler(Sampler):
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, y, batch_size, shuffle=True, sampling='oversample'):
        super(BalancedBatchSampler, self).__init__()
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        samples = [(idx, y[idx]) for idx in range(len(y))]
        labels = list(set(y))

        grouped = [[ex for ex in samples if ex[1]==lbl] for lbl in labels]
        if sampling=='oversample':
            mx_amt = max([len(gr) for gr in grouped])
        elif sampling=='undersample':
            mx_amt = min([len(gr) for gr in grouped])
        else:
            mx_amt = int(np.round(np.mean([len(gr) for gr in grouped])))
        balanced = []
        for gr in grouped:
            if len(gr)!=mx_amt:
                possible = int(np.ceil(mx_amt/len(gr))) * gr
                np.random.shuffle(possible)
                balanced.append(possible[:mx_amt])
            else:
                balanced.append(gr)
        new_y = [item for sublist in balanced for item in sublist]
        self.n_batches = int(len(new_y) / batch_size)
        self.y = new_y
        self.shuffle = shuffle
        self.btch_sz = batch_size

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.y)
        output_btchs = [self.y[i:i+self.btch_sz]
                        for i in range(0, len(self.y), self.btch_sz)]
        for train_idx in output_btchs:
            yield [x[0] for x in train_idx]

    def __len__(self):
        return self.n_batches

import matplotlib.pyplot as plt

def visualize_example(batch_data, pred, folder, epch,title, message=None):

    # dimensionality = 2 if dimension of batch is something....
    img = batch_data['X'][0,0].numpy()
    gt = batch_data['y'][0].numpy()
    pred = pred.detach().cpu().numpy()[0]

    fig, ax = plt.subplots(3, 2*gt.shape[0]+1, figsize=((2*gt.shape[0]+1)*5,15))
    fig.suptitle(str(title))

    mids = [int(sh/2) for sh in img.shape]
    ax[0,0].imshow(img[mids[0]], cmap='gray',aspect='auto')
    ax[1,0].imshow(img[:,mids[1]], cmap='gray',aspect='auto')
    ax[2,0].imshow(img[:,:,mids[2]], cmap='gray',aspect='auto')
    for rw_i in range(3):
        for cl_i, mp_i in zip(range(1, 2*gt.shape[0]+1, 2),
                              range(gt.shape[0])):
            ax[rw_i,cl_i].imshow(
                np.take(gt[mp_i],mids[rw_i], axis=rw_i),
                cmap='gray', aspect='auto')
            ax[rw_i,cl_i+1].imshow(
                np.take(pred[mp_i],mids[rw_i], axis=rw_i),
                cmap='gray', aspect='auto')

    if message:
        fig_name = folder + '/valexample_atepoch' + str(epch) + '_' +  message\
                   + '.pdf'
    else:
        fig_name = folder + '/valexample_atepoch' + str(epch) + '.pdf'
    plt.savefig(fig_name)
    plt.close(fig)

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
            OneHotEncode_seg(max_class=self.model.config.out_channels),
            ToTensor(field_oi='y')
        ])

    def setup_loss_functions(self):


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
        if type(self.config.loss_type)==str and 'CE' in self.config.loss_type:
            bce = self.bce_criterion(py, torch.argmax(y, dim=1))

        if type(self.config.loss_type)==str and 'DSC' in self.config.loss_type:
            dsc = self.dsc_criterion(py, y)

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
                else:
                    btch_x = data['X']
                btch_y = data['y']
                mdl_pred = mdl(btch_x)
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
                if len(dsc)!=self.model.config.out_channels:
                    print('ML Message: WARNING, a segmentation GT dosent have enough classes.')
                    continue
                vl_lss.append(dsc)
                epoch_iterator.set_postfix({'loss': [np.round(d, decimals=2)
                                                     for d in dsc]})
        if self.config.visualize_val:
            if self.config.gdsc_normalization=='softmax':
                vsl_pred = torch.softmax(mdl_pred, dim =1)
            elif self.config.gdsc_normalization=='sigmoid':
                vsl_pred = torch.sigmoid(mdl_pred)
            visualize_example(data, vsl_pred, self.io_manager.root,
                              self.config.trained_steps,
                              title = str(dsc),
                              message=str(batch_idx))
        vl_loss = np.array(vl_lss).mean(0)
        print(" ML Message: Validation Loss: " + str(vl_loss))
        vl_loss = hmean(vl_loss)
        return vl_loss

    def run_inference(self):
        trnsfrms = [
            Add_Channel(field_oi='X'),
            MnStdNormalize_Numpy(norm=list(self.config.normalization_channels),
                                 percentiles=self.config.normalization_percentiles,
                                 field_oi='X'),
            ToTensor(field_oi='X')
        ]

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
            res = data.copy()
            if torch.cuda.is_available():
                btch_x = data['X'][None,...].cuda()
            else:
                btch_x = data['X'][None,...]
            with torch.no_grad():
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
    #
    # def visualize_infexample(self, batch_data, pred, folder, message=None):
    #     img = batch_data['X'][0,0].numpy()
    #     fig, ax = plt.subplots(3, 4, figsize=(15,15))
    #     fig.suptitle(str(batch_data['AccNum'].numpy()[0]))
    #
    #     mids = [int(sh/2) for sh in img.shape]
    #     ax[0,0].imshow(img[mids[0]], cmap='gray')
    #     ax[0,1].imshow(img[:,mids[1]], cmap='gray')
    #     ax[0,2].imshow(img[:,:,mids[2]], cmap='gray')
    #     ax[0,3].hist(img[img!=-1].flatten(), bins=101)
    #     gt = batch_data['y'][0].numpy()
    #     ax[1,0].imshow(gt[0,mids[0]], cmap='gray')
    #     ax[1,1].imshow(gt[1,mids[0]], cmap='gray')
    #     ax[1,2].imshow(gt[2,mids[0]], cmap='gray')
    #     ax[1,3].hist(gt[:,mids[0]].flatten(), bins=51)
    #     pred = pred.detach().cpu().numpy()[0]
    #     ax[2,0].imshow(pred[0,mids[0]], cmap='gray')
    #     ax[2,1].imshow(pred[1,mids[0]], cmap='gray')
    #     ax[2,2].imshow(pred[2,mids[0]], cmap='gray')
    #     ax[2,3].hist(pred[:,mids[0]].flatten(), bins=51)
    #     if message:
    #         fig_name = folder + '/' + str(batch_data['AccNum']) + \
    #                    '_' +  message + '.pdf'
    #     else:
    #         fig_name = folder + '/' + str(batch_data['AccNum']) + \
    #                    '.pdf'
    #     plt.savefig(fig_name)
    #     plt.clf()