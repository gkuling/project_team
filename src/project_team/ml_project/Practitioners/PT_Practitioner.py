from project_team.project_config import project_config
from project_team.dt_project.dt_processing import functional as F
import numpy as np
from copy import deepcopy
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import gc
from tqdm import tqdm
from project_team.dt_project.dt_processing import Add_Channel, \
    MnStdNormalize_Numpy, AffineAugmentation, AddGaussianNoise, ToTensor, \
    Clip_Numpy

class PTPractitioner_config(project_config):
    # derived state written by __init__, restored via kwargs on reload
    _derived_config_keys = frozenset({'affine_aug_params'})

    def __init__(self,
                 batch_size=2,
                 n_epochs=1,
                 n_steps=None,
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
                 normalization_percentiles=[(0.5, 99.5)],
                 normalization_channels=[(0.5,0.5)],
                 n_workers=0,
                 data_parallel=False,
                 collate_function=None,
                 batch_spoofing=False,
                 trained_steps=0,
                 best_vl_loss=None,
                 best_vl_step=0,
                 vl_interval=None,
                 **kwargs):
        '''
        Config file for a pytorch practitioner
        :param batch_size: mini batch size
        :param n_epochs: amount of epochs to run
        :param n_steps: amount of training iterations/steps to run. Leave as
            None (the default) to train by epochs only
        :param warmup: Portion of training used for warmup steps
        :param n_saves: amount of times validation is run and the model saved
        :param validation_criteria: choice of whether the validation loss is
            minimized ('min') or maximized ('max')
        :param optimizer: str; the type of optimizer to be used, ex.. 'sgd',
            'adam' etc.
        :param lr: learning rate
        :param lr_decay: learning rate decay
        :param lr_decay_stepsize: learning rate decay stepsize
        :param lr_decay_gamma: learning rate decay gamma
        :param lr_decay_step_timing: learning rate decay step timing being
            either every 'batch' or every 'epoch'
        :param grad_clip: gradient clip value
        :param loss_type: loss function
        :param affine_aug: indicate affine augmentation is being used
        :param add_Gnoise: indicate whether gaussian noise will be added to
            the images
        :param gaussian_std: standard deviation of the gaussian noise added
            to the images
        :param normalization_percentiles: per-channel (min, max) percentile
            pairs that the input will be clipped by, e.g. [(0.5, 99.5)] —
            one tuple per input channel, matching normalization_channels
        :param normalization_channels: normalization mean and standard
            deviations used
        :param n_workers: number of worker that are used by the data loader
        :param data_parallel: indicator if data parallel is desired
        :param collate_function: collate function for the batch data. Either
            a callable (cannot be saved to json) or the string 'same_size'
        :param batch_spoofing: indicator of whether to use batch spoofing
            or not
        :param trained_steps: running count of completed training steps.
            A named parameter so a reloaded checkpoint config resumes
            training where it stopped
        :param best_vl_loss: best validation loss seen so far. None (the
            default) means no validation has run yet. None is used instead
            of +/-inf because inf is not valid json
        :param best_vl_step: the step at which best_vl_loss was recorded
        :param vl_interval: steps between validation/save points. Computed
            from n_steps/n_saves when None
        :param kwargs: any extra keywords are stored as config attributes
            (the huggingface convention); unrecognized names trigger a
            warning naming the closest real parameter
        '''
        # NOTE: trained_steps/best_vl_loss/best_vl_step live on the config,
        # not the practitioner, because the config is what gets checkpointed
        # — this is how training resumes mid-run. The config holds both
        # hyperparameters and mutable run state by design.
        kwargs.setdefault('config_type', 'ML_PTPractitioner')
        super(PTPractitioner_config, self).__init__(**kwargs)
        # Training Parameters
        self.loss_type = loss_type
        self.batch_size=batch_size
        self.n_epochs = n_epochs
        self.n_saves = n_saves
        # if n_steps are given and no n_epochs, the validation interval can
        # be easily calculated.
        if n_steps is not None:
            self.n_steps = n_steps
            if vl_interval is None:
                vl_interval = max(1, int(np.round(self.n_steps /
                                                  self.n_saves)))
            self.vl_interval = vl_interval
        else:
            self.n_steps = None
            self.vl_interval = vl_interval
        self.warmup = warmup
        self.lr_decay = lr_decay
        self.lr_decay_stepsize = lr_decay_stepsize
        self.lr_decay_gamma = lr_decay_gamma
        if lr_decay_step_timing not in ('epoch', 'batch'):
            raise ValueError(
                "lr_decay_step_timing must be 'epoch' or 'batch', got " +
                repr(lr_decay_step_timing))
        self.lr_decay_step_timing = lr_decay_step_timing
        self.optimizer = optimizer
        self.lr = lr
        self.grad_clip = grad_clip
        self.trained_steps = trained_steps
        self.data_parallel = data_parallel
        self.best_vl_loss = best_vl_loss
        self.best_vl_step = best_vl_step

        self.validation_criteria = validation_criteria
        self.n_workers = n_workers
        self.batch_spoofing = batch_spoofing

        # Augmentation Transforms
        self.normalization_percentiles = normalization_percentiles
        self.normalization_channels = normalization_channels
        self.add_Gnoise = add_Gnoise
        self.gaussian_std=gaussian_std
        self.affine_aug = affine_aug
        if affine_aug and not hasattr(self, 'affine_aug_params'):
            # hasattr guard: a reloaded config restores affine_aug_params
            # through kwargs, and re-initializing would clobber it
            self.initialize_augmentation_parameters()
        if collate_function is not None and \
                not callable(collate_function) and \
                collate_function != 'same_size':
            raise ValueError(str(collate_function) +
                             ' is not a recognized collate_function option. '
                             'Must be a custom function or "same_size".')
        # the spec (string or None) is stored so the config stays
        # json-serializable; get_collate_function() resolves it to a callable
        self.collate_function = collate_function

    def get_collate_function(self):
        '''
        resolve the stored collate_function spec into the callable handed to
        a DataLoader
        :return: a callable or None
        '''
        if self.collate_function is None or callable(self.collate_function):
            return self.collate_function
        if self.collate_function == 'same_size':
            return F.make_all_tensors_same_size
        raise ValueError(str(self.collate_function) +
                         ' is not a recognized collate_function option. '
                         'Must be a custom function or "same_size".')

    def initialize_augmentation_parameters(self):
        '''
        function used to initialize the affine augmentation parameters
        '''
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
        '''
        setting the affien augmentatiion parameters
        :param shift: tuple or list. the shift in each dimension
        :param rot: tuple or list. rotation on each axis
        :param scale: tuple or list. the portion fo scaling
        :param uniform_scale: bule whether to sample augmentation from a
            uniform distribution or a gaussian
        :param order: the level of interpolation in resampling. 0=nearest,
        1=linear, 3=cubic, ...  Follows skimage standards
        :return:
        '''
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
        '''
        determine the amount of epochs if it is not given.
        :param len_tr_dset: amount of training batches in the data
        '''
        self.n_epochs = int(
            np.ceil(
                (self.n_steps * self.batch_size)
                /len_tr_dset
            )
        )

class PT_Practitioner(object):
    def __init__(self, model, io_manager, data_processor,
                 trainer_config=None):
        '''
        constructor for the pytorch practitioner
        :param model: pytorch model to be trained
        :param io_manager: the manager of the project
        :param data_processor: the processor for the data
        :param trainer_config: configuration for the practitioner
        '''
        if trainer_config is None:
            trainer_config = PTPractitioner_config()
        self.model = deepcopy(model)
        self.io_manager = io_manager
        self.data_processor = data_processor
        self.config = trainer_config

        self.practitioner_name = 'base_PT'

        self.custom_transforms = None

        self.subclass_standard_transforms = []

        self.initialize_standard_pretransforms()

    def get_subclass_standard_transforms(self):
        '''
        Override in a subclass to add transforms (typically for the y field).
        Called every time the standard transforms are (re)built, so it is
        immune to __init__ ordering — an attribute assigned after
        super().__init__() would be silently missed.
        :return: list of transforms to append to standard_transforms
        '''
        return list(self.subclass_standard_transforms)

    def initialize_standard_pretransforms(self):
        # change standard_transforms based on the input data type
        self.standard_transforms = []
        if self.io_manager.config.X_dtype != 'Text':
            self.standard_transforms.extend([
                Clip_Numpy(field_oi='X',
                           max_min=self.config.normalization_percentiles),
                Add_Channel(field_oi='X'),
                MnStdNormalize_Numpy(norm=list(self.config.normalization_channels),
                                     field_oi='X'),
                ToTensor(field_oi='X')
            ])
        self.standard_transforms.extend(
            self.get_subclass_standard_transforms())

    def validate_model(self, val_dataloader):
        raise NotImplementedError('This Parent class does not have a '
                                  'validate model function.')

    def run_inference(self, **kwargs):
        raise NotImplementedError('This Parent class does not have a '
                                  'run inference function.')

    def from_pretrained(self, model_folder=None):
        '''
        load practitioner from a pretrained folder it was saved in.
        :param model_folder: folder for the model to be loaded. Optional,
            if not given the practitioner will ask the manager,
        '''
        self.io_manager.model_from_pretrained(self, model_folder)

    def save_pretrained(self, model_folder=None):
        '''
        save practitioner to a pretrained folder.
        :param model_folder: folder for the model to be saved. Optional,
            if not given the practitioner will ask the manager,
        '''
        self.io_manager.model_save_pretrained(self, model_folder)

    def set_custom_transforms(self, transforms_list):
        '''
        set custom_transforms used by the practitioner
        :param transforms_list: list of transforms
        '''
        if not isinstance(transforms_list, list):
            raise TypeError('transforms_list must be a list, got ' +
                            type(transforms_list).__name__)
        self.custom_transforms = transforms_list

    def extend_custom_transforms(self, transforms_list):
        '''
        extend the custom_transforms used by the practitioner
        :param transforms_list: list of transforms
        '''
        if not isinstance(transforms_list, list):
            raise TypeError('transforms_list must be a list, got ' +
                            type(transforms_list).__name__)
        self.custom_transforms.extend(transforms_list)

    def setup_dataloader(self, data_set):
        '''
        sets up data loaders for training or validation
        :param data_set: data set type either 'training' or 'validation '
        :return: data loader
        '''
        assert data_set=='training' or data_set=='validation'
        # set up transforms
        trnsfrms = []

        if self.custom_transforms:
            trnsfrms.extend(deepcopy(self.custom_transforms))

        # don't augment validation data
        if self.config.affine_aug and data_set=='training':
            trnsfrms.extend(
                [AffineAugmentation(**self.config.affine_aug_params)]
            )
            if hasattr(self.config, 'affine_aug_y'):
                trnsfrms.extend(
                    [AffineAugmentation(field_oi='y')]
                )

        # don't augment with noise for validation
        if self.config.add_Gnoise and data_set=='training':
            trnsfrms.extend([
                AddGaussianNoise(self.config.gaussian_std)
            ])

        # use dataset fingerprint for 'auto' parameters. Only these fields
        # are eligible — a static allowlist, not a scan of every config
        # string for the substring 'auto'
        AUTO_ELIGIBLE_FIELDS = ('normalization_percentiles',
                                'normalization_channels')
        params_to_auto = [key for key in AUTO_ELIGIBLE_FIELDS if
                          isinstance(getattr(self.config, key), str) and
                          'auto' in getattr(self.config, key)]
        if len(params_to_auto) > 0:
            if 'normalization_percentiles' in params_to_auto:
                if self.config.normalization_percentiles=='auto' or \
                    self.config.normalization_percentiles=='auto_05_995':
                    self.config.normalization_percentiles = (
                        self.data_processor.tr_dset.dataset_fingerprint
                        .get_percentiles('X', 0.5, 99.5))
                elif self.config.normalization_percentiles=='auto_1_99':
                    self.config.normalization_percentiles = (
                        self.data_processor.tr_dset.dataset_fingerprint
                        .get_percentiles('X', 1, 99))
                elif self.config.normalization_percentiles == 'auto_5_95':
                    self.config.normalization_percentiles = (
                        self.data_processor.tr_dset.dataset_fingerprint
                        .get_percentiles('X', 5, 95))
                elif self.config.normalization_percentiles == 'auto_min_max':
                    self.config.normalization_percentiles = (
                        self.data_processor.tr_dset.dataset_fingerprint
                        .get_min_max('X'))
            if 'normalization_channels' in params_to_auto:
                self.config.normalization_channels = (
                    self.data_processor.tr_dset.dataset_fingerprint
                    .get_mean_std('X'))
            self.initialize_standard_pretransforms()

        # custom transforms are always performed before standard transforms
        trnsfrms.extend(self.standard_transforms)

        # build specific data loaders depending on the dataset type
        if data_set=='training':
            self.data_processor.tr_dset.set_transforms(transforms.Compose(
                trnsfrms
            ))
            return DataLoader(
                self.data_processor.tr_dset,
                batch_size=self.config.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=self.config.n_workers,
                collate_fn=self.config.get_collate_function()
            )
        elif data_set=='validation':
            self.data_processor.vl_dset.set_transforms(transforms.Compose(
                trnsfrms
            ))
            # batch_size=1 is load-bearing: validate_model averages
            # per-batch losses unweighted, which is only the exact
            # per-sample mean when every batch has the same size
            return DataLoader(self.data_processor.vl_dset,
                              batch_size=1,
                              shuffle=False,
                              num_workers=self.config.n_workers)
        else:
            raise ValueError(data_set + ' is not an option for this '
                                        'practitioner. ')

    def setup_training_accessories(self):
        '''
        Set up the training accessories based on the parameters in the config.
        accessories are used for moving the parameters along the gradient,
        so the optimizer and learning rate scheduler.
        '''
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

        # Set up learning rate decay options
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
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optmzr,
                               step_size=self.config.lr_decay_stepsize ,
                               gamma = self.config.lr_decay_gamma)
        else:
            raise ValueError('The scheduler you gave may not be implemented. '
                             'options are steplr, '
                             'and cosine_schedule_with_warmup. ')

    def setup_loss_functions(self):
        '''
        Set up the practitioner loss function based on the config.
        Most are simple torch loss functions but the opportunity to use a
        custom one is there.
        '''
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
        # the model's own device move happens explicitly in train_model —
        # a loss-function setup method should not relocate the model
        if torch.cuda.is_available() and \
                callable(getattr(self.loss_function, 'cuda', None)):
            self.loss_function = self.loss_function.cuda()

    def setup_steps(self, tr_size):
        '''
        Function used to prepare the practitioner to know how long it needs
        to run on the data. This framework performs everything based on
        steps (or iterations), and not epochs, so epochs and dataset size
        needs to be used to translate to steps.
        :param tr_size: amount of batches in the training set
        '''
        # Need to set all of
        # n_epochs, n_steps, vl_interval, and warmup steps
        # when setting n_epochs or n_steps

        if self.config.n_steps is None and not self.config.n_epochs is None:
            # if no steps and epochs given
            self.config.n_steps = tr_size * self.config.n_epochs

        elif self.config.n_epochs is None and not self.config.n_steps is None:
            # if no epochs given and steps given
            self.config.set_n_epochs(len(self.data_processor.tr_dset))
        elif self.config.n_epochs is None and self.config.n_steps is None:
            raise ValueError('n_epochs and n_steps cannot both be None in '
                             'the PTPractitioner_config')
        else:
            if tr_size * self.config.n_epochs>=self.config.n_steps:
                self.config.n_steps = tr_size * self.config.n_epochs
            else:
                self.config.set_n_epochs(len(self.data_processor.tr_dset))

            print(
                'ML Message: n_steps and n_epochs are given. Taking the '
                'longer training time. n_steps: {}, n_epochs: {'
                '}'.format(str(self.config.n_steps),
                           str(self.config.n_epochs))
            )



        # set up the practitioner's  validation interval and save steps
        if self.config.vl_interval is None and not self.config.n_saves is None:
            # if no val_interval but n_saves given
            self.config.vl_interval = max(1, int(
                np.round(self.config.n_steps / self.config.n_saves)))

        elif not self.config.vl_interval is None and self.config.n_saves is None:
            # if no n_saves given but val_interval given
            self.config.n_saves = \
                np.round(self.config.n_steps / self.config.vl_interval).astype(
                    int)
        elif not self.config.vl_interval is None and not self.config.n_saves is None:
            pass
        else:
            raise ValueError(
                'vl_interval and n_saves cannot both be None in the '
                'PTPractitioner_config')

        # set up warmup steps if needed
        if self.config.warmup is None:
            self.config.warmup = 0
            self.config.warmup_steps = 0
        elif isinstance(self.config.warmup, float) and \
                0.0 <= self.config.warmup <= 1.0:
            self.config.warmup_steps = int(
                np.round(self.config.n_steps * self.config.warmup))
        elif isinstance(self.config.warmup, int) and \
                self.config.warmup < self.config.n_steps:
            self.config.warmup_steps = self.config.warmup
        else:
            raise ValueError('warmup in the PTPractitioner_config must be '
                             'None, a float between [0.0,1.0] (a fraction '
                             'of n_steps), or an int amount of steps less '
                             'than n_steps; got ' +
                             repr(self.config.warmup))

    def train_model(self):
        '''
        A function that calls all the tasks neccessary to train a model based
        on the parameters in the practitioner config.
        '''

        # Set up data
        tr_dtldr = self.setup_dataloader('training')

        if hasattr(self.data_processor, 'vl_dset'):
            vl_dtldr = self.setup_dataloader('validation')
        else:
            vl_dtldr = None

        # set up training tools
        self.setup_steps(len(tr_dtldr))
        self.setup_loss_functions()
        self.setup_training_accessories()

        # save the current model in the manager
        if torch.cuda.is_available():
            self.io_manager.set_final_model(
                self.model.cpu().state_dict())
            self.model = self.model.cuda()
            if self.config.data_parallel:
                print('ML Message: PT Practitioner putting the model onto '
                      'multiple GPU.')
                self.model = torch.nn.DataParallel(self.model)

        else:
            self.io_manager.set_final_model(
                self.model.state_dict())

        # batch_spoofing memory
        if self.config.batch_spoofing:
            split_multiple_memory = [1]
        else:
            split_multiple_memory = None
        print('ML Message: ')
        print('-'*5 + ' ' + self.practitioner_name + ' Practitioner Message:  \
            The Beginning of Training ')

        # beginning epoch loop
        for epoch in range(1, self.config.n_epochs + 1):

            epoch_iterator = tqdm(tr_dtldr, desc="Epoch " + str(epoch)
                                                 + " Iteration: ",
                                  position=0, leave=True,
                                  ncols=80
                                  )
            epoch_iterator.set_postfix({'loss': 'Initialized'})
            tr_loss = []
            # begin batch loop
            for batch_idx, data in enumerate(epoch_iterator):
                # Model will not be trained more steps than asked for
                if self.config.trained_steps>=self.config.n_steps:
                    break
                self.config.trained_steps+=1

                self.model.train()
                self.optmzr.zero_grad()

                btch_x, btch_y = self.organize_input_and_output(data)

                # Alter training if batch spoofing
                if self.config.batch_spoofing:
                    # batch spoofing scheme to try splitting the batch by a
                    # multiple of 2, if it doesn't work increase that
                    # multiple until it fits without exceeding the batchsize
                    continue_training = True
                    split_multiple = max(1, 2 ** np.round(
                        np.average(split_multiple_memory)
                    ))
                    while_cnt = 0
                    while continue_training:
                        while_cnt+=1
                        if split_multiple>self.config.batch_size:
                            raise RuntimeError(
                                "The GPU you are trying to spoof on cannot "
                                "hadle one training example. This "
                                "practitioner cannot calculate a single step "
                                "now. ")
                        try:
                            t_btch_x = torch.split(btch_x,
                                                   int(btch_x.shape[0] /
                                                       split_multiple))
                            t_btch_y = torch.split(btch_y,
                                                   int(btch_y.shape[0] /
                                                       split_multiple))
                            tracked_loss = []
                            for partition in zip(t_btch_x, t_btch_y):
                                pred = self.model(partition[0])
                                loss = self.calculate_loss(pred, partition[1])
                                loss.backward()
                                tracked_loss.append(loss.item())
                            continue_training = False
                            split_multiple_memory.append(np.log2(split_multiple))
                        except Exception as e:
                            if isinstance(e, torch.cuda.OutOfMemoryError):
                                split_multiple *= 2
                            else:
                                raise e
                else:
                    pred = self.model(btch_x)

                    loss = self.calculate_loss(pred, btch_y)
                    loss.backward()
                    tracked_loss = None
                if self.config.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   self.config.grad_clip)

                self.optmzr.step()
                if hasattr(self, 'scheduler') and \
                        self.config.lr_decay_step_timing=='batch':
                    self.scheduler.step()
                if tracked_loss is not None:
                    loss = torch.tensor(tracked_loss).mean()

                # report and record the loss
                epoch_iterator.set_postfix({'loss': loss.item()})
                tr_loss.append(loss.item())

                # perform validation or save interval
                if self.config.trained_steps-1!=0 and \
                        (self.config.trained_steps-1)%self.config.vl_interval==0:
                    if vl_dtldr:
                        vl_loss = float(self.validate_model(vl_dtldr))
                        # best_vl_loss is None until the first validation
                        if self.config.validation_criteria=='min':
                            if self.config.best_vl_loss is None or \
                                    vl_loss<self.config.best_vl_loss:
                                self.config.best_vl_loss = vl_loss
                                self.config.best_vl_step = self.config.trained_steps
                                if torch.cuda.is_available():
                                    self.io_manager.set_final_model(
                                        self.model.cpu().state_dict())
                                    self.model.cuda()
                                else:
                                    self.io_manager.set_final_model(
                                        self.model.state_dict())
                        else:
                            if self.config.best_vl_loss is None or \
                                    vl_loss>self.config.best_vl_loss:
                                self.config.best_vl_loss = vl_loss
                                self.config.best_vl_step = self.config.trained_steps
                                if torch.cuda.is_available():
                                    self.io_manager.set_final_model(
                                        self.model.cpu().state_dict())
                                    self.model.cuda()
                                else:
                                    self.io_manager.set_final_model(
                                        self.model.state_dict())
                    else:
                        if torch.cuda.is_available():
                            self.io_manager.set_final_model(
                                self.model.cpu().state_dict())
                            self.model.cuda()
                        else:
                            self.io_manager.set_final_model(
                                self.model.state_dict())
                    self.io_manager.save_model_checkpoint(
                        self
                    )

                if batch_idx==len(epoch_iterator)-1:
                    epoch_iterator.set_postfix({'Epoch loss': np.mean(tr_loss)})

            if hasattr(self, 'scheduler') and \
                    self.config.lr_decay_step_timing == 'epoch':
                self.scheduler.step()
        # perform one last validation run after all training is over
        if vl_dtldr:
            vl_loss = float(self.validate_model(vl_dtldr))
            if self.config.validation_criteria=='min':
                if self.config.best_vl_loss is None or \
                        vl_loss<self.config.best_vl_loss:
                    self.config.best_vl_loss = vl_loss
                    self.config.best_vl_step = self.config.trained_steps
                    if torch.cuda.is_available():
                        self.io_manager.set_final_model(
                            self.model.cpu().state_dict())
                        self.model.cuda()
                    else:
                        self.io_manager.set_final_model(
                            self.model.state_dict())
            else:
                if self.config.best_vl_loss is None or \
                        vl_loss>self.config.best_vl_loss:
                    self.config.best_vl_loss = vl_loss
                    self.config.best_vl_step = self.config.trained_steps
                    if torch.cuda.is_available():
                        self.io_manager.set_final_model(
                            self.model.cpu().state_dict())
                        self.model.cuda()
                    else:
                        self.io_manager.set_final_model(
                            self.model.state_dict())
        # final save of the trained model
        if vl_dtldr is None:
            if torch.cuda.is_available():
                self.io_manager.set_final_model(
                    self.model.cpu().state_dict())
                self.model = self.model.cuda()
            else:
                self.io_manager.set_final_model(
                    self.model.state_dict())
        self.io_manager.model_save_pretrained(self)
        # empty all memory
        torch.cuda.empty_cache()
        gc.collect()

        print('ML Message: Finished Training ' + self.practitioner_name)

    def calculate_loss(self, py, y):
        '''
        Some specific handling of pytorch losses were hard coded.
        :param py: prediction
        :param y: ground truth
        :return: calculated loss
        '''
        if self.config.loss_type=='KLD':
            # for KLD loss the prediction logits should be put through a
            # log_softmax before being evaluated.
            return self.loss_function(torch.log_softmax(py, dim=1),
                                      y)
        elif self.config.loss_type=='CE' and isinstance(y, torch.Tensor):
            # CrossEntropyLoss wants class-index targets as Long. A float
            # target with the prediction's own shape is a class-probability
            # target and passes through untouched.
            if y.dim()==1:
                return self.loss_function(py, y.long())
            elif y.dim()==2 and y.shape[1]==1 and py.shape[1]!=1:
                return self.loss_function(py, y[:, 0].long())
            else:
                return self.loss_function(py, y)
        elif self.config.loss_type=='NLL':
            # for NLL loss, the prediction logits should be put through a
            # log_softmax, and the groundtruth must be a LongTensor data
            # type, and a specific dimensionality.
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
                raise ValueError('NLL loss target not being used properly. '
                                 'Determine if this needs to be updated. ')
        else:
            return self.loss_function(py, y)

    def organize_input_and_output(self, batch_data):
        '''
        This will take the input and output for training and put it on cuda
        if it is available. Other wise it just hands back the tensors as they are
        :param batch_data: the data from the dataloader
        :return: input_tensor, and an output_tensor
        '''
        if not isinstance(batch_data['X'], dict):
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
        if not isinstance(batch_data['y'], dict):
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
