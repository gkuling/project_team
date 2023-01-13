from copy import deepcopy
from tqdm import tqdm
import numpy as np
import torch
import gc

from .UNet_Practitioner import UNet_Practitioner_config, UNet_Practitioner

class KDDA_UNet_Practitioner_config(UNet_Practitioner_config):
    def __init__(self, domain_adaptation='', T=2, a=0.5, retrain_teachers=True,
                 **kwargs):
        super(KDDA_UNet_Practitioner_config, self).__init__(**kwargs)
        self.domain_adaptation = domain_adaptation
        self.T = T
        self.a = a
        self.retrain_teachers = retrain_teachers

import matplotlib.pyplot as plt

# def visualize_example(batch_data, pred, tch_logits, folder, epch, message=None):
#     print('beginning visualization of ' + str(batch_data['AccNum'].numpy()[0]))
#     img = batch_data['X'][0,0].numpy()
#     fig, ax = plt.subplots(4, 4, figsize=(20,15))
#     fig.suptitle(str(batch_data['AccNum'].numpy()[0]))
#     ax[0,0].imshow(img[16], cmap='gray')
#     ax[0,1].imshow(img[:,64], cmap='gray')
#     ax[0,2].imshow(img[:,:,64], cmap='gray')
#     ax[0,3].hist(img[img!=-1].flatten(), bins=101)
#     gt = batch_data['y'][0].numpy()
#     ax[1,0].imshow(gt[0,16], cmap='gray')
#     ax[1,1].imshow(gt[1,16], cmap='gray')
#     ax[1,2].imshow(gt[2,16], cmap='gray')
#     ax[1,3].hist(gt[:,16].flatten(), bins=51)
#     pred = torch.softmax(pred,dim=1).detach().cpu().numpy()[0]
#     ax[2,0].imshow(pred[0,16], cmap='gray')
#     ax[2,1].imshow(pred[1,16], cmap='gray')
#     ax[2,2].imshow(pred[2,16], cmap='gray')
#     ax[2,3].hist(pred[:,16].flatten(), bins=51)
#     tchlg = torch.softmax(tch_logits,dim=1).detach().cpu().numpy()[0]
#     ax[3,0].imshow(tchlg[0,16], cmap='gray')
#     ax[3,1].imshow(tchlg[1,16], cmap='gray')
#     ax[3,2].imshow(tchlg[2,16], cmap='gray')
#     ax[3,3].hist(tchlg[:,16].flatten(), bins=51)
#     if message:
#         fig_name = folder + '/' + str(batch_data['AccNum'].numpy()[0]) + \
#                    '_atepoch' + str(epch) + '_' +  message + '.pdf'
#     else:
#         fig_name = folder + '/' + str(batch_data['AccNum'].numpy()[0]) + \
#                    '_atepoch' + str(epch) + '.pdf'
#     plt.savefig(fig_name)
#     plt.clf()


class KDDA_UNet_Practitioner(UNet_Practitioner):
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=KDDA_UNet_Practitioner_config()):
        super(KDDA_UNet_Practitioner, self).__init__(model,
                                                     io_manager,
                                                     data_processor,
                                                     trainer_config)
        assert trainer_config.domain_adaptation is not None

        self.da = trainer_config.domain_adaptation

    def train_teacher_models(self):
        teacher_names = list(set(
            [ex[self.da] for ex in self.data_processor.tr_dset.dfiles]
        ))

        teachers = {}
        mdls = {nm:deepcopy(self.model) for nm in teacher_names}
        for name in teacher_names:
            print('ML Message: Beginning Training ' + name + ' Teacher Model')
            kd_io_manager_config = deepcopy(self.io_manager.config)

            kd_input_output_manager = type(self.io_manager)(kd_io_manager_config)
            kd_input_output_manager.set_root(self.io_manager.root + \
                                             '/teacher_' + name)
            if not self.config.retrain_teachers and  not \
                    kd_input_output_manager.check_if_model_trained():
                raise ValueError("Cannot have KDDAUNet trainer config "
                                 "parameter retrain_teachers = 'False' and "
                                 "not have pretrained teachers. ")

            if self.config.retrain_teachers or not \
                    kd_input_output_manager.check_if_model_trained():

                self.data_processor.set_dataset_filter(
                    dataset_lambda_condition=lambda x: x[self.da]==name
                )
                practitioner = UNet_Practitioner(
                    model=mdls[name],
                    io_manager=kd_input_output_manager,
                    data_processor=self.data_processor,
                    trainer_config=deepcopy(self.config)
                )
                practitioner.set_custom_transforms(self.custom_transforms)
                practitioner.train_model()

                torch.cuda.empty_cache()

            print('ML Message: Finished Training ' + name + ' Teacher Model')
            teachers[name] = kd_input_output_manager.root
        self.data_processor.set_dataset_filter(
            dataset_lambda_condition=None
        )
        return teachers

    def train_model(self):
        teachers_locals = self.train_teacher_models()
        teachers = {}
        for key in teachers_locals.keys():
            temp = deepcopy(self.model)
            temp.load_state_dict(
                torch.load(teachers_locals[key] + '/final_model.pth')
            )
            temp.eval()
            if torch.cuda.is_available():
                temp.cuda()

            teachers[key] = temp
        tr_dtldr = self.setup_dataloader('training')
        if hasattr(self.data_processor, 'vl_dset'):
            vl_dtldr = self.setup_dataloader('validation')
        else:
            vl_dtldr = None

        self.setup_loss_functions()
        self.setup_training_accessories()
        self.kld_criterion = torch.nn.KLDivLoss(reduction='none')

        if torch.cuda.is_available():
            self.kld_criterion.cuda()
        self.config.set_n_epochs(len(self.data_processor.tr_dset))

        if torch.cuda.is_available():
            self.io_manager.set_best_model(
                self.model.cpu().state_dict())
            self.model = self.model.cuda()
            if self.config.data_parallel:
                print('ML Message: PT Practitioner putting the model onto '
                      'multiple GPU.')
                self.model = torch.nn.DataParallel(self.model)

        else:
            self.io_manager.set_best_model(
                self.model.state_dict())
        print("ML Message: ")
        print('-'*5 + ' KDDA UNet Practitioner Message:  The Beginning of '
                      'Student Training ')
        for epoch in range(1, self.config.n_epochs + 1):

            epoch_iterator = tqdm(tr_dtldr, desc="Epoch " + str(epoch)
                                                 + " Iteration: ",
                                  position=0, leave=True, ncols=100)
            epoch_iterator.set_postfix({'loss': 'Initialized'
                                        })
            tr_loss = []
            for batch_idx, data in enumerate(epoch_iterator):
                if self.config.trained_steps>=self.config.n_steps:
                    break
                self.config.trained_steps+=1
                self.model.train()
                self.optmzr.zero_grad()
                if torch.cuda.is_available():
                    btch_x = data['X'].cuda()
                    btch_y = data['y'].cuda()
                else:
                    btch_x = data['X']
                    btch_y = data['y']
                pred = self.model(btch_x)

                teacher_logits = []
                for i, nm in enumerate(data[self.da]):
                    teacher_logits.append(
                        teachers[nm].forward(btch_x[i][None,...])
                    )
                teacher_logits = torch.cat(teacher_logits, dim=0)
                teacher_logits = teacher_logits.detach()

                base_loss = self.calculate_loss(pred, btch_y)

                kld = self.kld_criterion(torch.log_softmax(
                    pred/self.config.T,
                    dim=1
                ),
                    torch.softmax(
                        teacher_logits/self.config.T,
                        dim=1
                    )).mean()

                loss = (1 - self.config.a) * base_loss + \
                       self.config.a * self.config.T * self.config.T * kld
                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.config.grad_clip)

                self.optmzr.step()
                if hasattr(self, 'scheduler'):
                    self.scheduler.step()
                losses = {nm: str(np.round(ls,2))
                          for nm, ls in zip(
                        ['loss','base','kd'],
                        [loss.item(), base_loss.item(), kld.item()]
                    )}
                epoch_iterator.set_postfix(losses)
                tr_loss.append(loss.item())
                if self.config.trained_steps%self.config.vl_interval==0:
                    if vl_dtldr:
                        vl_loss = self.validate_model(self.model, vl_dtldr)
                        if self.config.validation_criteria=='min':
                            if vl_loss<self.config.best_vl_loss:
                                self.config.best_vl_loss = vl_loss
                                if torch.cuda.is_available():
                                    self.io_manager.set_best_model(
                                        self.model.cpu().state_dict())
                                    self.model.cuda()
                                else:
                                    self.io_manager.set_best_model(
                                        self.model.state_dict())
                        else:
                            if vl_loss>self.config.best_vl_loss:
                                self.config.best_vl_loss = vl_loss
                                if torch.cuda.is_available():
                                    self.io_manager.set_best_model(
                                        self.model.cpu().state_dict())
                                    self.model.cuda()
                                else:
                                    self.io_manager.set_best_model(
                                        self.model.state_dict())
                    else:
                        if torch.cuda.is_available():
                            self.io_manager.set_best_model(
                                self.model.cpu().state_dict())
                            self.model.cuda()
                        else:
                            self.io_manager.set_best_model(
                                self.model.state_dict())
                    self.io_manager.save_model_checkpoint(self.config)
                if batch_idx==len(epoch_iterator)-1:
                    epoch_iterator.set_postfix({'Epoch loss': np.mean(tr_loss)})
        if vl_dtldr is None:
            if torch.cuda.is_available():
                self.io_manager.set_best_model(
                    self.model.cpu().state_dict())
                self.model = self.model.cuda()
            else:
                self.io_manager.set_best_model(
                    self.model.state_dict())
        self.io_manager.save_final_model(self.config,
                                         self.data_processor.config,
                                         self.model.config)
        torch.cuda.empty_cache()
        gc.collect()
        print('ML Message: Finished Training Student UNet')

    def visualize_example(self, batch_data, pred, tch_logits, folder,
                          message=None):
        img = batch_data['X'][0,0].numpy()
        fig, ax = plt.subplots(4, 4, figsize=(20,15))
        fig.suptitle(str(batch_data['AccNum'].numpy()[0]))

        mids = [int(sh/2) for sh in img.shape]
        ax[0,0].imshow(img[mids[0]], cmap='gray')
        ax[0,1].imshow(img[:,mids[1]], cmap='gray')
        ax[0,2].imshow(img[:,:,mids[2]], cmap='gray')
        ax[0,3].hist(img[img!=-1].flatten(), bins=101)
        gt = batch_data['y'][0].numpy()
        ax[1,0].imshow(gt[0,mids[0]], cmap='gray')
        ax[1,1].imshow(gt[1,mids[0]], cmap='gray')
        ax[1,2].imshow(gt[2,mids[0]], cmap='gray')
        ax[1,3].hist(gt[:,mids[0]].flatten(), bins=51)
        pred = torch.softmax(pred,dim=1).detach().cpu().numpy()[0]
        ax[2,0].imshow(pred[0,mids[0]], cmap='gray')
        ax[2,1].imshow(pred[1,mids[0]], cmap='gray')
        ax[2,2].imshow(pred[2,mids[0]], cmap='gray')
        ax[2,3].hist(pred[:,mids[0]].flatten(), bins=51)
        tchlg = torch.softmax(tch_logits,dim=1).detach().cpu().numpy()[0]
        ax[3,0].imshow(tchlg[0,mids[0]], cmap='gray')
        ax[3,1].imshow(tchlg[1,mids[0]], cmap='gray')
        ax[3,2].imshow(tchlg[2,mids[0]], cmap='gray')
        ax[3,3].hist(tchlg[:,mids[0]].flatten(), bins=51)
        if message:
            fig_name = folder + '/' + str(batch_data['AccNum']) + \
                       '_' +  message + '.pdf'
        else:
            fig_name = folder + '/' + str(batch_data['AccNum']) + \
                       '.pdf'
        plt.savefig(fig_name)
        plt.clf()