from src.project_config import project_config
from src.dt_project.dt_processing import *
import numpy as np
from copy import deepcopy
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import gc
from tqdm import tqdm

class PTPractitioner_config(project_config):
    def __init__(self,
                 batch_size=2,
                 n_epochs=1,
                 n_steps=1,
                 warmup=0.05,
                 n_saves=10,
                 validation_criteria='min',
                 optimizer='adamw',
                 lr=0.001,
                 lr_decay=None,
                 lr_decay_stepsize=0,
                 lr_decay_gamma=0.1,
                 lr_decay_step_timing='epoch',
                 grad_clip=None,
                 loss_type='MSE',
                 affine_aug=True,
                 add_Gnoise=True,
                 gaussian_std=1.0,
                 normalization_percentiles=(0.5,99.5),
                 normalization_channels=[(0.5,0.5)],
                 n_workers=0,
                 visualize_val=False,
                 data_parallel=False,
                 **kwargs):
        super(PTPractitioner_config, self).__init__('ML_PTPractitioner')
        # Training Parameters
        self.loss_type = loss_type
        self.batch_size=batch_size
        self.n_epochs = n_epochs
        self.n_saves = n_saves
        if n_steps is not None:
            self.n_steps = n_steps
            self.vl_interval = int(np.round(self.n_steps / self.n_saves))
        else:
            self.n_steps = None
            self.vl_interval = None
        self.warmup = warmup
        self.lr_decay = lr_decay
        self.lr_decay_stepsize = lr_decay_stepsize
        self.lr_decay_gamma = lr_decay_gamma
        self.lr_decay_step_timing = lr_decay_step_timing
        self.optimizer = optimizer
        self.lr = lr
        self.grad_clip = grad_clip
        self.trained_steps = 0
        self.data_parallel = data_parallel
        if validation_criteria=='min':
            self.best_vl_loss = np.inf
            self.best_vl_step = 0
        else:
            self.best_vl_loss = -np.inf
            self.best_vl_step = 0

        self.validation_criteria = validation_criteria
        self.n_workers = n_workers
        self.visualize_val = visualize_val

        # Augmentation Transforms
        self.normalization_percentiles = normalization_percentiles
        self.normalization_channels = normalization_channels
        self.add_Gnoise = add_Gnoise
        self.gaussian_std=gaussian_std
        self.affine_aug = affine_aug
        if affine_aug:
            self.initialize_augmentation_parameters()


    def initialize_augmentation_parameters(self):
        self.affine_aug_params = {
            'shift': (1,2,2),
            'rot': (1,2,2),
            'scale': 0.05,
            'order': 0
        }

    def set_augmentation_parameters(self,
                                    shift=None,
                                    rot=None,
                                    scale=None,
                                    uniform_scale=None,
                                    order=None):
        if shift:
            self.affine_aug_params['shift'] = shift
        if rot:
            self.affine_aug_params['rot'] = rot
        if scale:
            self.affine_aug_params['scale'] = scale
        if uniform_scale:
            self.affine_aug_params['uniform_scale'] = uniform_scale
        if order:
            self.affine_aug_params['order'] = order

    def set_n_epochs(self, len_tr_dset):
        self.n_epochs = int(
            np.ceil(
                (self.n_steps * self.batch_size)
                /len_tr_dset
            )
        )

class PT_Practitioner(object):
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=PTPractitioner_config()):
        self.model = deepcopy(model)
        self.io_manager = io_manager
        self.data_processor = data_processor
        self.config = trainer_config

        self.practitioner_name = 'base_PT'
        self.standard_transforms = []

        self.standard_transforms.extend([
            Add_Channel(field_oi='X'),
            MnStdNormalize_Numpy(norm=list(self.config.normalization_channels),
                                 percentiles=self.config.normalization_percentiles,
                                 field_oi='X'),
            ToTensor(field_oi='X')
        ])
        self.custom_transforms = None

    def validate_model(self, mdl, val_dataloader):
        raise NotImplementedError('This Parent class does not have a '
                                  'validate model function.')

    def run_inference(self, **kwargs):
        raise NotImplementedError('This Parent class does not have a '
                                  'run inference function.')

    def from_pretrained(self, model_folder=None):
        if model_folder:
            state = torch.load(
                    model_folder + '/final_model.pth'
                )

        else:
            state = torch.load(
                    self.io_manager.root + '/final_model.pth'
                )
        state = {k.split('module.')[1] if k.startswith('module.') else k: v for k,v in state.items()}
        self.model.load_state_dict(state)



    def set_custom_transforms(self, transforms_list):
        assert type(transforms_list)==list
        self.custom_transforms = transforms_list

    def extend_custom_transforms(self, transforms_list):
        assert type(transforms_list)==list
        self.custom_transforms.extend(transforms_list)

    def setup_dataloader(self, data_set):
        trnsfrms = []

        if self.custom_transforms:
            trnsfrms.extend(deepcopy(self.custom_transforms))

        if self.config.affine_aug and data_set=='training':
            trnsfrms.extend(
                [AffineAugmentation(**self.config.affine_aug_params)]
            )
            if hasattr(self.config, 'affine_aug_y'):
                trnsfrms.extend(
                    [AffineAugmentation(field_oi='y')]
                )
        if self.config.add_Gnoise and data_set=='training':
            trnsfrms.extend([
                AddGaussainNoise(self.config.gaussian_std)
            ])


        trnsfrms.extend(self.standard_transforms)
        if data_set=='training':
            self.data_processor.tr_dset.set_transforms(transforms.Compose(
                trnsfrms
            ))
            return DataLoader(
                    self.data_processor.tr_dset,
                    batch_size=self.config.batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=self.config.n_workers
                )

        elif data_set=='validation':
            self.data_processor.vl_dset.set_transforms(transforms.Compose(
                trnsfrms
            ))
            return DataLoader(self.data_processor.vl_dset,
                              batch_size=1,
                              shuffle=False)
        else:
            raise ValueError(data_set + ' is not an option for this '
                                        'practitioner. ')

    def setup_training_accessories(self):

        # setup the optimizer
        if self.config.optimizer=='adamw':
            self.optmzr = torch.optim.AdamW(self.model.parameters(),
                                            lr=self.config.lr)
        elif self.config.optimizer=='adam':
            self.optmzr = torch.optim.Adam(self.model.parameters(),
                                           lr=self.config.lr)
        elif self.config.optimizer=='sgd':
            self.optmzr = torch.optim.SGD(self.model.parameters(),
                                          lr=self.config.lr)
        elif self.config.optimizer=='adadelta':
            self.optmzr = torch.optim.Adadelta(self.model.parameters(),
                                               lr=self.config.lr)
        else:
            raise ValueError(str(self.config.optimizer) + ' is not a '
                                                          'recognizable '
                                                          'optimizer')

        # Set uplearning rate decay options
        if self.config.lr_decay is None:
            pass
        elif self.config.lr_decay=='cosine_schedule_with_warmup':
            self.scheduler = \
                get_cosine_schedule_with_warmup(
                    self.optmzr,
                    num_warmup_steps=self.config.warmup_steps,
                    num_training_steps=self.config.n_steps,
                    num_cycles=0.4
                )
        elif self.config.lr_decay=='steplr':
            scheduler = torch.optim.lr_scheduler.StepLR(self.optmzr,
                               step_size=self.config.lr_decay_stepsize ,
                               gamma = self.config.lr_decay_gamma)
        else:
            raise Exception('The scheduler you gave may not be implemented. '
                            'options are steplr, '
                            'and cosine_schedule_with_warmup. ')

    def setup_loss_functions(self):
        if self.config.loss_type=='MSE':
            self.loss_function = torch.nn.MSELoss()
        elif self.config.loss_type=='L1':
            self.loss_function = torch.nn.L1Loss()
        elif self.config.loss_type=='CE':
            self.loss_function = torch.nn.CrossEntropyLoss()
        elif self.config.loss_type=='BCE':
            self.loss_function = torch.nn.BCEWithLogitsLoss()
        elif self.config.loss_type=='KLD':
            self.loss_function = torch.nn.KLDivLoss(reduction='batchmean')
        elif self.config.loss_type=='NLL':
            self.loss_function = torch.nn.NLLLoss()
        elif callable(self.config.loss_type):
            self.loss_function = self.config.loss_type
        else:
            raise NotImplementedError('The ' + str(self.config.loss_type) +
                                      ' is not implemented. ')
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            if callable(getattr(self.loss_function,'cuda',None)):
                self.loss_function = self.loss_function.cuda()

    def setup_steps(self, tr_size):

        # Need to set all of
        # n_epochs, n_steps, vl_interval, and warmup steps
        # when setting n_epochs or n_steps
        if self.config.n_steps is None and not self.config.n_epochs is None:
            self.config.n_steps = tr_size * self.config.n_epochs
        elif self.config.n_epochs is None and not self.config.n_steps is None:
            self.config.set_n_epochs(len(self.data_processor.tr_dset))
        elif not self.config.n_epochs is None and not self.config.n_steps is None:
            pass
        else:
            raise Exception('n_epochs and n_steps cannot both be None in the '
                            'PT_Pracitioner_config')
        if self.config.vl_interval is None and not self.config.n_saves is None:
            self.config.vl_interval = \
                np.round(self.config.n_steps / self.config.n_saves).astype(int)
        elif not self.config.vl_interval is None and self.config.n_saves is None:
            self.config.n_saves = \
                np.round(self.config.n_steps / self.config.vl_interval).astype(
                    int)
        elif not self.config.vl_interval is None and not self.config.n_saves is None:
            pass
        else:
            raise Exception(
                'vl_interval and n_saves cannot both be None in the '
                'PT_Pracitioner_config')

        if self.config.warmup is None:
            self.config.warmup = 0
        elif self.config.warmup <= 1.0 and self.config.warmup >= 0.0:
            self.config.warmup_steps = int(
                np.round(self.config.n_steps * self.config.warmup))
        elif type(self.config.warmup) == int:
            pass
        else:
            raise Exception('warmup in the PT_Pracitioner_config must be '
                            'None, a float between [0.0,1.0], or int amount '
                            'of steps. ')

    def train_model(self):
        tr_dtldr = self.setup_dataloader('training')

        if hasattr(self.data_processor, 'vl_dset'):
            vl_dtldr = self.setup_dataloader('validation')
        else:
            vl_dtldr = None

        self.setup_steps(len(tr_dtldr))
        self.setup_loss_functions()
        self.setup_training_accessories()

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

        print('ML Message: ')
        print('-'*5 + ' ' + self.practitioner_name + ' Practitioner Message:  \
            The Beginning of Training ')
        for epoch in range(1, self.config.n_epochs + 1):

            epoch_iterator = tqdm(tr_dtldr, desc="Epoch " + str(epoch)
                                                 + " Iteration: ",
                                  position=0, leave=True)
            epoch_iterator.set_postfix({'loss': 'Initialized'})
            tr_loss = []
            for batch_idx, data in enumerate(epoch_iterator):
                if self.config.trained_steps>=self.config.n_steps:
                    break
                self.config.trained_steps+=1
                self.model.train()
                self.optmzr.zero_grad()

                btch_x, btch_y = self.organize_input_and_output(data)

                pred = self.model(btch_x)

                loss = self.calculate_loss(pred, btch_y)
                loss.backward()
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.config.grad_clip)

                self.optmzr.step()
                if hasattr(self, 'scheduler') and \
                        self.config.lr_decay_step_timing=='batch':
                    self.scheduler.step()
                epoch_iterator.set_postfix({'loss': loss.item()})
                tr_loss.append(loss.item())

                if self.config.trained_steps-1!=0 and \
                        (self.config.trained_steps-1)%self.config.vl_interval==0:
                    if vl_dtldr:
                        vl_loss = self.validate_model(self.model, vl_dtldr)
                        if self.config.validation_criteria=='min':
                            if vl_loss<self.config.best_vl_loss:
                                self.config.best_vl_loss = vl_loss
                                self.config.best_vl_step = self.config.trained_steps
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
                                self.config.best_vl_step = self.config.trained_steps
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

            if hasattr(self, 'scheduler') and \
                    self.config.lr_decay_step_timing == 'epoch':
                self.scheduler.step()

        if vl_dtldr:
            vl_loss = self.validate_model(self.model, vl_dtldr)
            if self.config.validation_criteria=='min':
                if vl_loss<self.config.best_vl_loss:
                    self.config.best_vl_loss = vl_loss
                    self.config.best_vl_step = self.config.trained_steps
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
                    self.config.best_vl_step = self.config.trained_steps
                    if torch.cuda.is_available():
                        self.io_manager.set_best_model(
                            self.model.cpu().state_dict())
                        self.model.cuda()
                    else:
                        self.io_manager.set_best_model(
                            self.model.state_dict())
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

        print('ML Message: Finished Training ' + self.practitioner_name)

    def calculate_loss(self, py, y):
        if self.config.loss_type=='KLD':
            return self.loss_function(torch.log_softmax(py, dim=1),
                                      y)
        elif self.config.loss_type=='NLL':
            if y.dim()==1:
                return self.loss_function(torch.log_softmax(py, dim=1),
                                          y.type(torch.cuda.LongTensor)
                                          if torch.cuda.is_available() else
                                          y.type(torch.LongTensor))
            elif y.shape[1]==1:
                return self.loss_function(torch.log_softmax(py, dim=1),
                                          y.type(torch.cuda.LongTensor)[:,0]
                                          if torch.cuda.is_available() else
                                          y.type(torch.LongTensor)[:,0])
            else:
                raise Exception('NLL loss tearget not being used properly. '
                                'Determine if this needs to be updated. ')
        else:
            return self.loss_function(py, y)

    def organize_input_and_output(self, batch_data):
        if type(batch_data['X'])!=dict:
            if torch.cuda.is_available():
                input_tensor = batch_data['X'].cuda()
            else:
                input_tensor = batch_data['X']
        else:
            if torch.cuda.is_available():
                input_tensor = {ky: v.cuda() if hasattr(v, 'cuda') else v
                                for ky, v, in batch_data['X'].items()}
            else:
                input_tensor = batch_data['X']
        if type(batch_data['y'])!=dict:
            if torch.cuda.is_available():
                output_tensor = batch_data['y'].cuda()
            else:
                output_tensor = batch_data['y']
        else:
            if torch.cuda.is_available():
                output_tensor = {ky: v.cuda() if hasattr(v, 'cuda') else v
                                for ky, v, in batch_data['y'].items()}
            else:
                output_tensor = batch_data['y']

        return input_tensor, output_tensor
