'''
Apologies, this needed to be pushed out asap for the sake of the project.
I will be back to clean this up later. - GCK
'''
import gc
import inspect
import json
import shutil
from copy import deepcopy

import torch
from typing import Tuple, List

from .PT_Practitioner import PTPractitioner_config
import torch.cuda
import os
import torch.multiprocessing as mp
from torch.backends import cudnn
from typing import Union

class nnUNet_Practitioner_config(PTPractitioner_config):
    def __init__(self,
                 labels,
                 channel_names,
                 nnunet_configuration='3d_fullres',
                 **kwargs):
        super(nnUNet_Practitioner_config, self).__init__(**kwargs)
        self.labels = labels
        self.channel_names = channel_names
        self.nnunet_configuration = nnunet_configuration
class nnUNet_Practitioner():
    def __init__(self,
                 io_manager,
                 dt_processor,
                 practitioner_config):
        self.io_manager = io_manager
        self.dt_processor = dt_processor
        self.config = practitioner_config

    def train_model(self):

        dataset_json = {
            'channel_names': self.config.channel_names,
            'labels': self.config.labels,
            'numTraining': len(
                os.listdir(
                    os.path.join(
                        self.io_manager.root,
                        'nnunetv2_raw',
                        'Dataset001_Current',
                        'imagesTr'
                    )
                )
            ),
            'file_ending': '.nii.gz'}
        # save dict as json
        with open(os.path.join(os.path.join(
            self.io_manager.root,
            'nnunetv2_raw'
        ),
                               'Dataset001_Current',
                               'dataset.json'), 'w') as fp:
            json.dump(dataset_json, fp)

        os.environ['nnUNet_raw'] = os.path.join(
            self.io_manager.root,
            'nnunetv2_raw'
        )
        os.environ['nnUNet_preprocessed'] = os.path.join(
            self.io_manager.root,
            'nnunetv2_preprocessed'
        )
        os.environ['nnUNet_results'] = os.path.join(
            self.io_manager.root,
            'nnunetv2_results'
        )
        os.environ['nnUNet_def_n_proc'] = "1"
        os.environ['nnUNet_n_proc_DA'] = "12"


        print('ML Message: nnUNetv2 training beginning.')
        print('ML Message: turning over printout to nnUNetv2.')
        # hafta import these here, because then the environment variables
        # will be set to None for nnUNetv2 and the above os lines will change
        # nothing for it
        import nnunetv2
        from nnunetv2.experiment_planning.plan_and_preprocess_entrypoints import (
            extract_fingerprints, plan_experiments, preprocess)
        # to solve some multithreading issues on titania
        os.environ['OMP_NUM_THREADS'] = '1'
        # planning and preprocessing
        # fingerprint extraction
        print("Fingerprint extraction...")
        extract_fingerprints([1], num_processes=1)

        # experiment planning
        print('Experiment planning...')
        plan_experiments([1])



        # manage default np
        default_np = {"2d": 1, "3d_fullres": 1, "3d_lowres": 1}
        np = [default_np[c] if c in default_np.keys() else 4 for c in
              [self.config.nnunet_configuration]]

        print('Preprocessing...')
        preprocess([1],
                   configurations=[self.config.nnunet_configuration],
                   num_processes=np)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # training the model
        assert device in ['cpu', 'cuda',
                               'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {device}.'
        if device == 'cpu':
            # let's allow torch to use hella threads
            torch.set_num_threads(1)
            device = torch.device('cpu')
        elif device == 'cuda':
            # multithreading in torch doesn't help nnU-Net if run on GPU
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            device = torch.device('cuda')
        else:
            device = torch.device('mps')

        self.run_training('Dataset001_Current',
                          configuration=self.config.nnunet_configuration,
                          device=device)
        print('ML Message: nnUNetv2 training finished.')

    def run_training(self,
                     dataset_name_or_id: Union[str, int],
                     configuration: str,
                     device):

        fold = 0
        trainer_class_name = 'nnUNetTrainer'
        plans_identifier = 'nnUNetPlans'
        pretrained_weights = None
        num_gpus = torch.cuda.device_count()
        use_compressed_data = True
        export_validation_probabilities = False
        continue_training = False
        only_run_validation = False
        disable_checkpointing = False
        assert use_compressed_data==True, ("use_compressed_data must be True. "
                                           "Otherwise, the data will be "
                                           "loaded from the raw folder, "
                                           "which is not supported by this "
                                           "implementation. When debugging I "
                                           "could not get their "
                                           "multiprocessing to work. ")
        from nnunetv2.run.run_training import (find_free_network_port,
                                               run_ddp,
                                               get_trainer_from_args,
                                               maybe_load_checkpoint)
        try:
            fold = int(fold)
        except ValueError as e:
            print(
                f'Unable to convert given value for fold to int: {fold}. '
                f'fold must bei either "all" or an integer!')
            raise e

        if num_gpus > 1:
            assert device.type == 'cuda', (f"DDP training (triggered by num_gpus "
                                           f"> 1) is only implemented for cuda "
                                           f"devices. Your device: {device}")

            os.environ['MASTER_ADDR'] = 'localhost'
            if 'MASTER_PORT' not in os.environ.keys():
                port = str(find_free_network_port())
                print(f"using port {port}")
                os.environ['MASTER_PORT'] = port  # str(port)

            mp.spawn(run_ddp,
                     args=(
                         dataset_name_or_id,
                         configuration,
                         fold,
                         trainer_class_name,
                         plans_identifier,
                         use_compressed_data,
                         disable_checkpointing,
                         continue_training,
                         only_run_validation,
                         pretrained_weights,
                         export_validation_probabilities,
                         num_gpus),
                     nprocs=num_gpus,
                     join=True)
        else:
            nnunet_trainer = get_trainer_from_args(dataset_name_or_id,
                                                   configuration, fold,
                                                   trainer_class_name,
                                                   plans_identifier,
                                                   use_compressed_data,
                                                   device=device)

            if disable_checkpointing:
                nnunet_trainer.disable_checkpointing = disable_checkpointing

            assert not (
                        continue_training and only_run_validation), \
                f'Cannot set --c and --val flag at the same time. Dummy.'

            maybe_load_checkpoint(nnunet_trainer, continue_training,
                                  only_run_validation, pretrained_weights)

            if torch.cuda.is_available():
                cudnn.deterministic = False
                cudnn.benchmark = True
            nnunet_trainer.batch_size = self.config.batch_size
            nnunet_trainer.num_epochs = self.config.n_epochs

            if not only_run_validation:
                nnunet_trainer.run_training()

    def run_inference(self,):

        os.environ['nnUNet_raw'] = os.path.join(
            self.io_manager.root,
            'nnunetv2_raw'
        )
        os.environ['nnUNet_preprocessed'] = os.path.join(
            self.io_manager.root,
            'nnunetv2_preprocessed'
        )
        os.environ['nnUNet_results'] = os.path.join(
            self.io_manager.root,
            'nnunetv2_results'
        )
        os.environ['nnUNet_def_n_proc'] = "1"
        os.environ['nnUNet_n_proc_DA'] = "12"
        print('ML Message: nnUNetv2 inference beginning.')
        print('ML Message: turning over printout to nnUNetv2.')

        self.predict_from_raw_data(
            os.path.join(self.io_manager.root,
                         'nnunetv2_raw','Dataset001_Current','imagesTs'),
            os.path.join(self.io_manager.root,
                         'nnunetv2_raw', 'Dataset001_Current',
                         'imagesTs_predlowres'),
            os.path.join(self.io_manager.root,'nnunetv2_results',
                         'Dataset001_Current',
                         'nnUNetTrainer__nnUNetPlans__3d_fullres'),
            (0,),
            0.5,
            use_gaussian=True,
            use_mirroring=False,
            perform_everything_on_gpu=True,
            verbose=True,
            save_probabilities=False,
            overwrite=False,
            checkpoint_name='checkpoint_final.pth',
            num_processes_preprocessing=1,
            num_processes_segmentation_export=1)
        print('ML Message: nnUNetv2 inference finished.')
        inf_results = []
        for _ in range(len(self.dt_processor.if_dset)):
            example = deepcopy(self.dt_processor.if_dset.dfiles[_])
            shutil.move(
                os.path.join(
                    self.io_manager.root,
                     'nnunetv2_raw', 'Dataset001_Current',
                     'imagesTs_predlowres',
                     'proteam_' + str(_).zfill(4) + '.nii.gz'),
                    example['X_location'][
                             0].replace('.nii.gz', '_' +
                                        self.io_manager.config.experiment_name
                                        + '.nii.gz')
            )
            example['pred_y'] = example['X_location'][
                             0].replace('.nii.gz', '_' +
                                        self.io_manager.config.experiment_name
                                        + '.nii.gz')
            inf_results.append(example)
        self.dt_processor.inference_results = inf_results
        torch.cuda.empty_cache()
        gc.collect()
        print("ML Message: Finished Running nnUNetV2 Segmentation")

    def predict_from_raw_data(
            self,
            list_of_lists_or_source_folder: Union[str, List[List[str]]],
            output_folder: str,
            model_training_output_dir: str,
            use_folds: Union[Tuple[int, ...], str] = None,
            tile_step_size: float = 0.5,
            use_gaussian: bool = True,
            use_mirroring: bool = True,
            perform_everything_on_gpu: bool = True,
            verbose: bool = True,
            save_probabilities: bool = False,
            overwrite: bool = True,
            checkpoint_name: str = 'checkpoint_final.pth',
            num_processes_preprocessing: int = 1,
            num_processes_segmentation_export: int = 1,
            folder_with_segs_from_prev_stage: str = None,
            num_parts: int = 1,
            part_id: int = 0,
            device: torch.device = torch.device('cuda')):
        # hafta import these here, because then the environment variables
        # will be set to None for nnUNetv2 and the above os lines will change
        # nothing for it
        import inspect
        import os
        import shutil
        import traceback
        from copy import deepcopy

        import numpy as np
        import torch
        from batchgenerators.dataloading.single_threaded_augmenter import \
            SingleThreadedAugmenter
        from batchgenerators.transforms.utility_transforms import NumpyToTensor
        from batchgenerators.utilities.file_and_folder_operations import \
            join, isfile, maybe_mkdir_p, \
            save_json
        from nnunetv2.inference.export_prediction import \
            export_prediction_from_softmax
        from nnunetv2.inference.sliding_window_prediction import \
            predict_sliding_window_return_logits, compute_gaussian
        from nnunetv2.utilities.json_export import recursive_fix_for_json_export
        from nnunetv2.utilities.utils import \
            create_lists_from_splitted_dataset_folder
        from nnunetv2.inference.predict_from_raw_data import (
            auto_detect_available_folds, load_what_we_need, PreprocessAdapter
        )

        if device.type == 'cuda':
            device = torch.device(type='cuda',
                                  index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!

        if device.type != 'cuda':
            perform_everything_on_gpu = False

        # let's store the input arguments so that its clear what was used to generate the prediction
        my_init_kwargs = {}
        for k in inspect.signature(
                self.predict_from_raw_data).parameters.keys():
            my_init_kwargs[k] = locals()[k]
        my_init_kwargs = deepcopy(
            my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
        # safety precaution.
        recursive_fix_for_json_export(my_init_kwargs)
        maybe_mkdir_p(output_folder)
        save_json(my_init_kwargs,
                  join(output_folder, 'predict_from_raw_data_args.json'))

        if use_folds is None:
            use_folds = auto_detect_available_folds(model_training_output_dir,
                                                    checkpoint_name)

        # load all the stuff we need from the model_training_output_dir
        parameters, configuration_manager, inference_allowed_mirroring_axes, \
            plans_manager, dataset_json, network, trainer_name = \
            load_what_we_need(model_training_output_dir, use_folds,
                              checkpoint_name)

        # sort out input and output filenames
        if isinstance(list_of_lists_or_source_folder, str):
            list_of_lists_or_source_folder = create_lists_from_splitted_dataset_folder(
                list_of_lists_or_source_folder,
                dataset_json['file_ending'])
        print(
            f'There are {len(list_of_lists_or_source_folder)} cases in the source folder')
        list_of_lists_or_source_folder = list_of_lists_or_source_folder[
                                         part_id::num_parts]
        caseids = [
            os.path.basename(i[0])[:-(len(dataset_json['file_ending']) + 5)] for
            i in list_of_lists_or_source_folder]
        print(
            f'I am process {part_id} out of {num_parts} (max process ID is {num_parts - 1}, we start counting with 0!)')
        print(f'There are {len(caseids)} cases that I would like to predict')

        output_filename_truncated = [join(output_folder, i) for i in caseids]
        seg_from_prev_stage_files = [join(folder_with_segs_from_prev_stage,
                                          i + dataset_json['file_ending']) if
                                     folder_with_segs_from_prev_stage is not None else None
                                     for i in caseids]
        # remove already predicted files form the lists
        if not overwrite:
            tmp = [isfile(i + dataset_json['file_ending']) for i in
                   output_filename_truncated]
            not_existing_indices = [i for i, j in enumerate(tmp) if not j]

            output_filename_truncated = [output_filename_truncated[i] for i in
                                         not_existing_indices]
            list_of_lists_or_source_folder = [list_of_lists_or_source_folder[i]
                                              for i in not_existing_indices]
            seg_from_prev_stage_files = [seg_from_prev_stage_files[i] for i in
                                         not_existing_indices]
            print(
                f'overwrite was set to {overwrite}, so I am only working on cases that haven\'t been predicted yet. '
                f'That\'s {len(not_existing_indices)} cases.')
            # caseids = [caseids[i] for i in not_existing_indices]

        # placing this into a separate function doesnt make sense because it needs so many input variables...
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        # hijack batchgenerators, yo
        # we use the multiprocessing of the batchgenerators dataloader to handle all the background worker stuff. This
        # way we don't have to reinvent the wheel here.
        num_processes = max(1, min(num_processes_preprocessing,
                                   len(list_of_lists_or_source_folder)))
        ppa = PreprocessAdapter(list_of_lists_or_source_folder,
                                seg_from_prev_stage_files, preprocessor,
                                output_filename_truncated, plans_manager,
                                dataset_json,
                                configuration_manager, num_processes)
        # mta = MultiThreadedAugmenter(ppa, NumpyToTensor(), num_processes, 1,
        #                              None, pin_memory=device.type == 'cuda')
        mta = SingleThreadedAugmenter(ppa, NumpyToTensor())

        # precompute gaussian
        inference_gaussian = torch.from_numpy(
            compute_gaussian(configuration_manager.patch_size)).half()
        if perform_everything_on_gpu:
            inference_gaussian = inference_gaussian.to(device)

        # num seg heads is needed because we need to preallocate the results in predict_sliding_window_return_logits
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_seg_heads = label_manager.num_segmentation_heads

        network = network.to(device)

        r = []
        with torch.no_grad():
            for preprocessed in mta:
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                print(f'\nPredicting {os.path.basename(ofile)}:')
                print(
                    f'perform_everything_on_gpu: {perform_everything_on_gpu}')

                properties = preprocessed['data_properites']

                # we have some code duplication here but this allows us to run with perform_everything_on_gpu=True as
                # default and not have the entire program crash in case of GPU out of memory. Neat. That should make
                # things a lot faster for some datasets.
                prediction = None
                overwrite_perform_everything_on_gpu = perform_everything_on_gpu
                if perform_everything_on_gpu:
                    try:
                        for params in parameters:
                            network.load_state_dict(params)
                            if prediction is None:
                                prediction = predict_sliding_window_return_logits(
                                    network, data, num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device)
                            else:
                                prediction += predict_sliding_window_return_logits(
                                    network, data, num_seg_heads,
                                    configuration_manager.patch_size,
                                    mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                    tile_step_size=tile_step_size,
                                    use_gaussian=use_gaussian,
                                    precomputed_gaussian=inference_gaussian,
                                    perform_everything_on_gpu=perform_everything_on_gpu,
                                    verbose=verbose,
                                    device=device)
                            if len(parameters) > 1:
                                prediction /= len(parameters)

                    except RuntimeError:
                        print(
                            'Prediction with perform_everything_on_gpu=True failed due to insufficient GPU memory. '
                            'Falling back to perform_everything_on_gpu=False. Not a big deal, just slower...')
                        print('Error:')
                        traceback.print_exc()
                        prediction = None
                        overwrite_perform_everything_on_gpu = False

                if prediction is None:
                    for params in parameters:
                        network.load_state_dict(params)
                        if prediction is None:
                            prediction = predict_sliding_window_return_logits(
                                network, data, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                                verbose=verbose,
                                device=device)
                        else:
                            prediction += predict_sliding_window_return_logits(
                                network, data, num_seg_heads,
                                configuration_manager.patch_size,
                                mirror_axes=inference_allowed_mirroring_axes if use_mirroring else None,
                                tile_step_size=tile_step_size,
                                use_gaussian=use_gaussian,
                                precomputed_gaussian=inference_gaussian,
                                perform_everything_on_gpu=overwrite_perform_everything_on_gpu,
                                verbose=verbose,
                                device=device)
                        if len(parameters) > 1:
                            prediction /= len(parameters)

                print('Prediction done, transferring to CPU if needed')
                prediction = prediction.to('cpu').numpy()
                print(
                    'sending off prediction to background worker for resampling and export')
                r.append(
                    # export_pool.starmap_async(
                        export_prediction_from_softmax(prediction,
                                                          properties,
                                                          configuration_manager,
                                                          plans_manager,
                                                          dataset_json,
                                                          ofile,
                                                          save_probabilities)
                    # )
                )
                print(f'done with {os.path.basename(ofile)}')

        # we need these two if we want to do things with the predictions like for example apply postprocessing
        shutil.copy(join(model_training_output_dir, 'dataset.json'),
                    join(output_folder, 'dataset.json'))
        shutil.copy(join(model_training_output_dir, 'plans.json'),
                    join(output_folder, 'plans.json'))