import os
import sys
import shutil
from functools import partial
from datetime import datetime
from shutil import copyfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
sys.path.append('./')
import utils.general as utils
import utils.plots as plt
from utils.hutils import get_rank, is_main_process
from utils.rotation_converter import *
import gzip
import json
import wandb
import numpy as np
from tqdm import tqdm
import copy
import natsort
import sys
sys.path.append('../submodules/diffmorpher-for-perse')
from diffmorpher_model import MultiDiffMorpherPipeline
from mmseg.apis import init_model
from collections import defaultdict
print = partial(print, flush=True)

def check_pth_files(checkpoints_path, optimizer_params_subdir):
    dir_path = os.path.join(checkpoints_path, optimizer_params_subdir)
    files = os.listdir(dir_path)
    pth_files = [f for f in files if f.endswith('.pth') and f != 'latest.pth']
    return bool(pth_files)

def find_checkpoint_file(directory, epoch_number):
    file_list = natsort.natsorted(os.listdir(os.path.join(directory, 'ModelParameters')))
    last_file_exists = False
    
    if epoch_number == 'latest':
        search_prefix = 'latest'
        last_files = []
    else:
        search_prefix = epoch_number

    for file_name in file_list:
        if file_name.startswith(search_prefix) and file_name.endswith('.pth'):
            if epoch_number == 'latest':
                last_files.append(file_name)
                last_file_exists = True
            else:
                return file_name

    if last_file_exists:
        last_files = natsort.natsorted(last_files)
        if len(last_files) > 1:
            return last_files[-2]
        else:
            return last_files[0]
    else:
        return None
    
def has_files(directory_path):
    for entry in os.listdir(directory_path):
        full_path = os.path.join(directory_path, entry)
        if os.path.isfile(full_path):
            return True
    return False

def state_dict_log(missing_keys, unexpected_keys):
    if is_main_process():
        if missing_keys:
            print("The following parameters were not found in the saved model and thus not loaded:")
            for key in missing_keys:
                print(key)
        else:
            print("All parameters from the saved model were loaded successfully.")
        if unexpected_keys:
            print("The following parameters were in the saved model but not expected by the current model:")
            for key in unexpected_keys:
                print(key)


class Runner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        opt = kwargs['opt']
        self.conf = kwargs['conf']
        self.is_test = opt.is_test
        self.batch_size = self.conf.get_int('train.batch_size')
        self.nepochs = kwargs['nepochs']
        self.exps_folder_name = self.conf.get_string('train.exps_folder')
        self.optimize_expression = self.conf.get_bool('train.optimize_expression')
        self.optimize_pose = self.conf.get_bool('train.optimize_camera')
        if self.is_test:
            self.optimize_expression = False
            self.optimize_pose = False
        self.subject = self.conf.get_string('dataset.subject_name')
        self.methodname = self.conf.get_string('train.methodname')

        # if 'WORLD_SIZE' in os.environ:
        #     distributed = (int(os.environ['WORLD_SIZE']) > 1)
        #     print('[INFO] distributed training: {}'.format(distributed))

        torch.cuda.set_device(opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        self.device = torch.device(f'cuda:{opt.local_rank}')
        dist.barrier()
        self.num_gpus = dist.get_world_size()
        
        # NOTE wandb를 rank 0에서만 설정하게 되면 다른rank들에게서 vram이 증가하는 문제가 발생한다.
        os.environ['WANDB_DIR'] = os.path.join(self.exps_folder_name)
        wandb.init(project=self.conf.get_string('train.projectname'), name='{0}_{1}'.format(self.subject, self.methodname), config=self.conf, tags=[f'rank_{opt.local_rank}'])
            
        self.is_val = self.conf.get_bool('val.is_val')
        if self.is_val:
            self.val_subdir = self.conf.get_config('val.val_subdir')
        self.optimize_scene_latent_code = self.conf.get_bool('train.optimize_scene_latent_code')
        self.accumulation_steps = self.conf.get_int('train.accumulation_steps')
        
        self.print_info = True
        if self.conf.dataset.train.sub_dir == 'all':
            data_folder = self.conf.get_string('dataset.data_folder')
            subject_name = self.conf.get_string('dataset.subject_name')
            dataset_name = self.conf.get_string('dataset.dataset_name')

            directory_path = os.path.join(data_folder, subject_name, dataset_name)

            self.dataset_train_subdir = natsort.natsorted([
                element for element in os.listdir(directory_path)
                if os.path.isdir(os.path.join(directory_path, element))
            ])
            self.dataset_train_subdir = [element for element in self.dataset_train_subdir if not element.startswith('test_')]

            self.conf.get_config('dataset.train')['sub_dir'] = self.dataset_train_subdir
        else:
            self.dataset_train_subdir = self.conf.get_list('dataset.train.sub_dir')

        self.scene_latent_dim = self.conf.get_int('model.scene_latent_dim')

        self.category_dict = self.conf.get_config('dataset.category_dict')                              # NOTE 할 수 있을만한 건 전부다 넣었음.
        self.category_latent_dim = len(self.category_dict)
        self.source_category = self.conf.get_list('dataset.source_category')
        self.source_category_dict = {key: self.category_dict[key] for key in self.source_category}      # NOTE source에 해당하는 category dict만 모아두었음.

        self.enable_upsample = self.conf.get_bool('train.enable_upsample')
        self.enable_prune = self.conf.get_bool('train.enable_prune')

        self.optimize_inputs = self.optimize_expression or self.optimize_pose or self.optimize_scene_latent_code
        self.expdir = os.path.join(self.exps_folder_name, self.subject, self.methodname)
        train_split_name = utils.get_split_name(self.dataset_train_subdir)

        self.eval_dir = os.path.join(self.expdir, train_split_name, 'eval')
        self.train_dir = os.path.join(self.expdir, train_split_name, 'train')
        
        if is_main_process():
            utils.mkdir_ifnotexists(self.train_dir)
            utils.mkdir_ifnotexists(self.eval_dir)
        dist.barrier()

        self.checkpoints_path = os.path.join(self.train_dir, 'checkpoints')
        if is_main_process():
            utils.mkdir_ifnotexists(self.checkpoints_path)
        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"
        self.scheduler_params_subdir = "SchedulerParameters"

        if is_main_process():
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.model_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))
            utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.scheduler_params_subdir))
        dist.barrier()

        if self.optimize_inputs:
            self.optimizer_inputs_subdir = "OptimizerInputs"
            self.input_params_subdir = "InputParameters"

            if is_main_process():
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir))
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.input_params_subdir))
            dist.barrier()
        
        is_continue = check_pth_files(self.checkpoints_path, self.model_params_subdir)
        
        if is_main_process():
            current_time = datetime.now()
            formatted_time = current_time.strftime('%Y%m%d_%H%M')
        else:
            formatted_time = None
        
        formatted_time_list = [formatted_time]
        dist.broadcast_object_list(formatted_time_list, src=0)
        self.formatted_time = formatted_time_list[0]  # 모든 프로세스가 동일한 시간 사용

        if is_main_process():
            self.file_backup(opt.exp_conf)
            print('shell command : {0}'.format(' '.join(sys.argv)))
            print('Loading data ...')

        if not self.is_test:
            if is_main_process():
                print('Loading train dataset ...')
            self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                              mode='train',
                                                                                              expr_dir=self.train_dir,
                                                                                              **self.conf.get_config('dataset.train'))
            self.num_train_dataset = len(self.train_dataset)

            self.diffmorpher_weight = self.conf.get_float('loss.diffmorpher_weight')
            self.diffmorpher_target_attrs = self.conf.get_list('train.diffmorpher_target_attrs')

            if self.diffmorpher_weight <= 0:
                dataset_config = self.conf.get_config('dataset.train')
                dataset_config.pop('sub_dir', None)  # sub_dir 키 제거

                self.train_single_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                                            mode='train_single',
                                                                                                            expr_dir=self.train_dir,
                                                                                                            sub_dir=[self.dataset_train_subdir[0]],
                                                                                                            **dataset_config)
                self.num_train_single_dataset = len(self.train_single_dataset)
        else:
            if is_main_process():
                print('Loading test dataset ...')
            self.test_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                             mode='test',
                                                                                             expr_dir=self.eval_dir,
                                                                                             **self.conf.get_config('dataset.test'))
        
        if self.is_val:
            if is_main_process():
                print('Loading val dataset ...')
            self.val_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(conf=self.conf,
                                                                                            mode='val',
                                                                                            expr_dir=self.eval_dir,
                                                                                            **self.conf.get_config('dataset.val'))
            
        
            
        if is_main_process():
            print('Finish loading data ...')

        latent_code_dim = (self.scene_latent_dim + self.category_latent_dim) * len(self.category_dict)

        # load mean and var expression
        if self.conf.get_bool('dataset.use_mean_expression'):
            mean_expression = torch.tensor(
                np.load(os.path.join(self.train_dir, 'mean_expression.npy'))
            ).float()
        if self.conf.get_bool('dataset.use_var_expression'):
            var_expression = torch.tensor(
                np.load(os.path.join(self.train_dir, 'var_expression.npy'))
            ).float()
        else:
            var_expression = None
        
        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf,
                                                                                img_res=self.conf.get_list('dataset.train.img_res'),
                                                                                canonical_expression=mean_expression,
                                                                                latent_code_dim=latent_code_dim,
                                                                                device=self.device)
        
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model.to(device=self.device), device_ids=[opt.local_rank], find_unused_parameters=True)
        if is_main_process():
            print('************************************************************')
            print(self.model)
            print('************************************************************')

        self._init_dataloader()

        if not self.is_test:
            pipelines = {}
            if self.diffmorpher_weight > 0:
                pipeline = MultiDiffMorpherPipeline.from_pretrained(
                    'stabilityai/stable-diffusion-2-1-base', 
                    torch_dtype=torch.float32).to(self.device)
                pipelines['diffmorpher'] = pipeline
                model_sapiens = init_model('../submodules/sapiens/seg/configs/sapiens_seg/seg_face/sapiens_1b_seg_face-1024x768.py', 
                                           '../submodules/sapiens/sapiens_host/seg/checkpoints/sapiens_1b/sapiens_1b_seg_face_epoch_200.pth', 
                                           device=self.device)
                pipelines['sapiens'] = model_sapiens

            self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(conf=self.conf,
                                                                                  pipelines=pipelines,
                                                                                  **self.conf.get_config('loss'), 
                                                                                  var_expression=var_expression,
                                                                                  optimize_scene_latent_code=self.optimize_scene_latent_code)

            self.lr = self.conf.get_float('train.learning_rate')
        
        self.train_from_scratch = self.conf.get_bool('train.training_method.train_from_scratch')
        if not self.train_from_scratch:
            self.from_scratch_prior_checkpoints_path = self.conf.get_string('train.training_method.prior_checkpoints_path')
            self.from_scratch_prior_checkpoints_epoch = self.conf.get_string('train.training_method.prior_checkpoints_epoch')

        self.lora_finetuning = self.conf.get_bool('train.lora.lora_finetuning')
        if self.lora_finetuning:
            self.lora_params_subdir = "LoRAParameters"

            self.accumulation_steps = 3
            if is_main_process():
                print('[INFO] accumulation step is changed because of LoRA finetuing stage.')
                utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.lora_params_subdir))
            dist.barrier()

            is_continue = False
            kwargs['checkpoint'] = self.conf.get_string('train.lora.prior_checkpoints_epoch')

            if is_main_process():
                print(f'[INFO] start checkpoints from {kwargs["checkpoint"]} epoch')
            self.prior_checkpoints_path = self.conf.get_string('train.lora.prior_checkpoints_path')
            self.lora_test_checkpoint = self.conf.get_string('test.lora.checkpoints')

        if not self.is_test:
            learnable_params = self.get_learnable_params()
            self.optimizer = torch.optim.Adam([
                {'params': learnable_params},
            ], lr=self.lr)
            
            self.sched_milestones = self.conf.get_list('train.sched_milestones', default=[])
            self.sched_factor = self.conf.get_float('train.sched_factor', default=0.0)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, self.sched_milestones, gamma=self.sched_factor)
            self.upsample_freq = self.conf.get_int('train.upsample_freq')

        if self.optimize_inputs:
            param = []

            if self.optimize_expression or self.optimize_pose:
                num_training_frames = self.num_train_dataset

            if self.optimize_expression:
                init_expression = torch.cat((self.train_dataset.data["expressions"], torch.randn(self.train_dataset.data["expressions"].shape[0], max(self.model.module.deformer_network.num_exp - 50, 0)).float()), dim=1)
                self.expression = torch.nn.Embedding(num_training_frames, self.model.module.deformer_network.num_exp, _weight=init_expression, sparse=False).to(device=self.device)
                param += list(self.expression.parameters())

            if self.optimize_pose:
                self.flame_pose = torch.nn.Embedding(num_training_frames, 15, _weight=self.train_dataset.data["flame_pose"], sparse=False).to(device=self.device)
                self.camera_pose = torch.nn.Embedding(num_training_frames, 3, _weight=self.train_dataset.data["world_mats"][:, :3, 3], sparse=False).to(device=self.device)
                param += list(self.flame_pose.parameters()) + list(self.camera_pose.parameters())
            
            if self.optimize_scene_latent_code:
                self.latent_code_std = self.conf.get_float('train.latent_code_std')
                # NOTE category에 없을 때 존재하는 latent code. 이걸로 일단 초기화한다.
                self.zero_latent_codes = torch.nn.Embedding(1, self.scene_latent_dim, sparse=False).to(device=self.device)
                torch.nn.init.normal_(
                    self.zero_latent_codes.weight.data,
                    0.0,
                    self.latent_code_std,
                )
                param += list(self.zero_latent_codes.parameters())

                # NOTE SH에 고유하게 존재하는 latent code.
                self.source_scene_latent_codes = torch.nn.Embedding(len(self.source_category_dict.keys()), self.scene_latent_dim, sparse=False).to(device=self.device)
                torch.nn.init.normal_(
                    self.source_scene_latent_codes.weight.data,
                    0.0,
                    self.latent_code_std,
                )
                param += list(self.source_scene_latent_codes.parameters())

                self.random_latent_code = torch.randn_like(self.zero_latent_codes(torch.tensor([0], device=self.device))) * self.latent_code_std
                if is_main_process():
                    print('[DEBUG] Scene latent code is used. The latent dimension is {0}x{1}.'.format(len(self.dataset_train_subdir), self.scene_latent_dim))

            # if self.lora_finetuning and self.latent_space_inversion:
            #     lr_cam = self.conf.get_float('train.lora.learning_rate_inversion')
            # else:
            #     lr_cam = self.conf.get_float('train.learning_rate_cam')
            lr_cam = self.conf.get_float('train.learning_rate_cam')

            if not self.is_test:
                self.optimizer_cam = torch.optim.Adam([{'params': param}], lr_cam)

        self.start_epoch = 0

        if self.lora_finetuning:
            checkpoints_path = self.prior_checkpoints_path
            saved_model_state = torch.load(os.path.join(checkpoints_path, 
                                                        self.model_params_subdir, 
                                                        '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_scheduler_state = torch.load(os.path.join(checkpoints_path, 
                                                            self.scheduler_params_subdir, 
                                                            '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_optimizer_inputs_state = torch.load(os.path.join(checkpoints_path, 
                                                                   self.optimizer_inputs_subdir, 
                                                                   '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            if self.is_test:
                saved_lora_state = torch.load(os.path.join(self.checkpoints_path, 
                                                           self.lora_params_subdir, 
                                                           f'{self.lora_test_checkpoint}.pth'), map_location=self.device)
            saved_input_state = torch.load(os.path.join(checkpoints_path, 
                                                        self.input_params_subdir, 
                                                        '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)

            self.load_checkpoints(saved_model_state, 
                                  saved_scheduler_state, 
                                  saved_optimizer_inputs_state, 
                                  saved_input_state,
                                  saved_lora_state)

        if not self.train_from_scratch:
            checkpoints_path = self.from_scratch_prior_checkpoints_path
            saved_model_state = torch.load(os.path.join(checkpoints_path, 
                                                        self.model_params_subdir, 
                                                        '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_scheduler_state = torch.load(os.path.join(checkpoints_path, 
                                                            self.scheduler_params_subdir, 
                                                            '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_optimizer_inputs_state = torch.load(os.path.join(checkpoints_path, 
                                                                   self.optimizer_inputs_subdir, 
                                                                   '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_input_state = torch.load(os.path.join(checkpoints_path, 
                                                        self.input_params_subdir, 
                                                        '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)

            self.load_checkpoints(saved_model_state, 
                                  saved_scheduler_state, 
                                  saved_optimizer_inputs_state, 
                                  saved_input_state,
                                  None)

        if is_continue:
            saved_model_state = torch.load(os.path.join(self.checkpoints_path, 
                                                        self.model_params_subdir, 
                                                        '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_scheduler_state = torch.load(os.path.join(self.checkpoints_path, 
                                                            self.scheduler_params_subdir, 
                                                            '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_optimizer_inputs_state = torch.load(os.path.join(self.checkpoints_path, 
                                                                self.optimizer_inputs_subdir, 
                                                                '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            saved_input_state = torch.load(os.path.join(self.checkpoints_path, 
                                                        self.input_params_subdir, 
                                                        '{}.pth'.format(kwargs['checkpoint'])), map_location=self.device)
            self.load_checkpoints(saved_model_state, 
                                  saved_scheduler_state, 
                                  saved_optimizer_inputs_state, 
                                  saved_input_state,
                                  None)

        if is_main_process() and not self.is_test:
            print('************************************************************')
            print('[INFO] learnable parameters of model')
            param_dict = {id(param): name for name, param in self.model.module.named_parameters()}

            # 옵티마이저를 통해 학습 가능한 파라미터 출력
            for param_group in self.optimizer.param_groups:
                for param in param_group['params']:
                    if param.requires_grad:
                        param_name = param_dict.get(id(param), "Unnamed parameter")
                        print(f"Parameter Name: {param_name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
            print('************************************************************')
            print('[INFO] learnable parameters of latent and camera')
            for param_group in self.optimizer_cam.param_groups:
                for param in param_group['params']:
                    if param.requires_grad:
                        print(f"Parameter: {param.shape}, Requires Grad: {param.requires_grad}")
            print('************************************************************')

    
        # self.interpolation_target_list = self.conf.get_list('test.interpolation.target_list')
        self.interpolation_ratio = self.conf.get_int('test.interpolation.num_frames')
        self.model_copied = copy.copy(self.model)

        self.pair_files = self.conf.get_string('test.interpolation.pair_files')

        if self.is_test:
            self.img_res = self.test_dataset.img_res
            self.dataset_test_subdir = self.conf.get_list('dataset.test.sub_dir')
            self.acc_loss = {}
            self.test_epoch = saved_model_state['epoch']
            print('[INFO] Loading checkpoint from {0} epoch'.format(self.test_epoch))
            # self.rendering_default = self.conf.get_bool('test.rendering_default')
            self.rendering_default = self.conf.get_bool('test.default.rendering')
            self.rendering_default_novel_view = self.conf.get_bool('test.default.rendering_novel_view')
            if self.rendering_default_novel_view:
                self.rendering_default_novel_view_euler_angle = self.conf.get_list('test.default.novel_view_euler_angle')
                self.rendering_default_novel_view_translation = self.conf.get_list('test.default.novel_view_translation')
            self.interpolation_rendering_same_time = self.conf.get_bool('test.interpolation.rendering_same_time')
            self.interpolation_rendering = self.conf.get_bool('test.interpolation.rendering')
            
            self.zero_shot_rendering = self.conf.get_bool('test.zero_shot.rendering')
            self.zero_shot_category = self.conf.get_string('test.zero_shot.category')
            self.random_sampling_rendering = self.conf.get_bool('test.random_sampling.rendering')
            self.random_sampling_category = self.conf.get_string('test.random_sampling.category')
            self.random_sampling_std = self.conf.get_float('test.random_sampling.std')
            self.random_sampling_n_samples = self.conf.get_int('test.random_sampling.n_samples')

            # self.lora_rendering_default = self.conf.get_bool('test.lora.rendering_default')
            # self.lora_interpolation_rendering = self.conf.get_bool('test.lora.interpolation.rendering')
            # if self.lora_interpolation_rendering:
            #     self.lora_interpolation_target_list = self.conf.get_list('test.lora.interpolation.target_list')
            #     self.lora_interpolation_ratio = int(1 / self.conf.get_float('test.lora.interpolation.ratio'))
        else:
            self.img_res = self.train_dataset.img_res

            self.plot_freq = self.conf.get_int('train.plot_freq')
            self.save_freq = self.conf.get_int('train.save_freq')
            self.log_freq = self.conf.get_int('train.log_freq')

            self.GT_lbs_milestones = self.conf.get_list('train.GT_lbs_milestones', default=[])
            self.GT_lbs_factor = self.conf.get_float('train.GT_lbs_factor', default=0.5)
            for acc in self.GT_lbs_milestones:
                if self.start_epoch > acc:
                    self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor
        
            self.start_time = torch.cuda.Event(enable_timing=True)
            self.end_time = torch.cuda.Event(enable_timing=True)

            self.start_time_step = torch.cuda.Event(enable_timing=True)
            self.end_time_step = torch.cuda.Event(enable_timing=True)

    def load_checkpoints(self, 
                         saved_model_state, 
                         saved_scheduler_state,
                         saved_optimizer_inputs_state, 
                         saved_input_state,
                         saved_lora_state):
        
        if saved_model_state is not None:
            self.start_epoch = saved_model_state['epoch']
            n_points = saved_model_state["model_state_dict"]['pc.points'].shape[0]
            if is_main_process():
                print("[INFO] n_points: {}".format(n_points))

            self.model.module.pc.init(n_points)
            self.model.module.pc = self.model.module.pc.to(device=self.device)

            missing_keys, unexpected_keys = self.model.module.load_state_dict(saved_model_state["model_state_dict"], strict=False)
            state_dict_log(missing_keys, unexpected_keys)

            self.model.module.radius = saved_model_state['radius']

            if saved_lora_state:
                self.model.module.load_state_dict(saved_lora_state["lora_state_dict"], strict=False)

            if not self.is_test:
                learnable_params = self.get_learnable_params()
                self.optimizer = torch.optim.Adam([
                    {'params': learnable_params},
                ], lr=self.lr)

        if (saved_scheduler_state is not None) and (not self.is_test):
            self.scheduler.load_state_dict(saved_scheduler_state["scheduler_state_dict"])

        if self.optimize_inputs:
            # if (saved_optimizer_inputs_state is not None) and (not self.is_test):
            #     self.optimizer_cam.load_state_dict(saved_optimizer_inputs_state["optimizer_cam_state_dict"])

            if self.optimize_expression:
                try:
                    missing_keys, unexpected_keys = self.expression.load_state_dict(saved_input_state["expression_state_dict"], strict=False)
                    state_dict_log(missing_keys, unexpected_keys)
                except:
                    print("[INFO] expression state dict is not loaded.")
            if self.optimize_pose:
                try:
                    missing_keys, unexpected_keys = self.flame_pose.load_state_dict(saved_input_state["flame_pose_state_dict"], strict=False)
                    state_dict_log(missing_keys, unexpected_keys)
                    missing_keys, unexpected_keys = self.camera_pose.load_state_dict(saved_input_state["camera_pose_state_dict"], strict=False)
                    state_dict_log(missing_keys, unexpected_keys)
                except:
                    print("[INFO] pose state dict is not loaded.")
            if (saved_input_state is not None) and self.optimize_scene_latent_code:
                self.zero_latent_codes.load_state_dict(saved_input_state["zero_latent_codes_state_dict"], strict=True)
                try:
                    self.source_scene_latent_codes.load_state_dict(saved_input_state["source_scene_latent_codes_state_dict"], strict=True)
                except:
                    print('[INFO] source scene latent codes are not loaded.')

    def get_learnable_params(self):
        if self.lora_finetuning:
            # freeze all of layers except named 'lora'
            freeze_lst = ['lora']
            learnable_params, _ = self.set_trainable_parameters(freeze_lst)
            if is_main_process():
                print('************************************************************')
                print(f'[INFO] make learnable parameters {freeze_lst}')
                print('************************************************************')
        else:
            if self.diffmorpher_weight > 0:
                freeze_lst = ['clip']       # clipregressionnetwork도 freeze한다. 기존 내용도 잘 유지하기 위함.
                _, learnable_params = self.set_trainable_parameters(freeze_lst)
                if is_main_process():
                    print('************************************************************')
                    print(f'[INFO] freeze parameters {freeze_lst}')
                    print('************************************************************')
            else:
                freeze_lst = ['clip_model']
                _, learnable_params = self.set_trainable_parameters(freeze_lst)
                if is_main_process():
                    print('************************************************************')
                    print(f'[INFO] freeze parameters {freeze_lst}')
                    print('************************************************************')
        return learnable_params

    def _init_dataloader(self):
        if not self.is_test:
            train_sampler = DistributedSampler(dataset=self.train_dataset, shuffle=True)
            self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
                                                                batch_size=self.batch_size,
                                                                collate_fn=self.train_dataset.collate_fn,
                                                                num_workers=0,
                                                                sampler=train_sampler)
            self.n_batches = len(self.train_dataloader)
            
            if self.diffmorpher_weight <= 0:
                train_single_sampler = DistributedSampler(dataset=self.train_single_dataset, shuffle=True)
                self.train_single_dataloader = torch.utils.data.DataLoader(self.train_single_dataset,
                                                                            batch_size=self.batch_size,
                                                                            collate_fn=self.train_single_dataset.collate_fn,
                                                                            num_workers=0,
                                                                            sampler=train_single_sampler)
                self.n_single_batches = len(self.train_single_dataloader)
        else:
            test_sampler = DistributedSampler(dataset=self.test_dataset, shuffle=False)
            self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset,
                                                            batch_size=self.batch_size,
                                                            collate_fn=self.test_dataset.collate_fn,
                                                            num_workers=0,
                                                            sampler=test_sampler)

        if self.is_val:
            val_sampler = DistributedSampler(dataset=self.val_dataset, shuffle=False)
            self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset,
                                                              batch_size=self.batch_size,
                                                              collate_fn=self.val_dataset.collate_fn,
                                                              num_workers=0,
                                                              sampler=val_sampler)

    def set_trainable_parameters(self, freeze_lst):
        freeze_params = [param for name, param in self.model.module.named_parameters()
                        if any(keyword in name for keyword in freeze_lst)]

        learnable_params = [param for name, param in self.model.module.named_parameters()
                        if not any(keyword in name for keyword in freeze_lst)]

        return freeze_params, learnable_params
    
    def save_checkpoints(self, epoch, only_latest=False):
        if self.lora_finetuning:
            # if self.latent_space_inversion:
            #     dict_to_save = {}
            #     dict_to_save["epoch"] = epoch
            #     dict_to_save["adaptive_latent_codes_state_dict"] = self.adaptive_latent_codes.state_dict()
            #     save_subdir = self.latent_inversion_params_subdir
                
            # else:
            #     # Saving the 'lora' parameters
            #     lora_state_dict = {name: param.cpu() for name, param in self.model.module.named_parameters() if 'lora' in name}
            #     dict_to_save = {}
            #     dict_to_save["epoch"] = epoch
            #     dict_to_save["lora_state_dict"] = lora_state_dict
            #     save_subdir = self.lora_params_subdir

            lora_state_dict = {name: param.cpu() for name, param in self.model.module.named_parameters() if 'lora' in name}
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            dict_to_save["lora_state_dict"] = lora_state_dict
            save_subdir = self.lora_params_subdir

            torch.save(dict_to_save, os.path.join(self.checkpoints_path, save_subdir, "latest.pth"))
            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, save_subdir, str(epoch) + ".pth"))
            return
     
        if not only_latest:
            torch.save(
                {"epoch": epoch, "radius": self.model.module.radius,
                "model_state_dict": self.model.module.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, str(epoch) + ".pth"))
            
            torch.save(
                {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(epoch) + ".pth"))
            torch.save(
                {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
                os.path.join(self.checkpoints_path, self.scheduler_params_subdir, str(epoch) + ".pth"))

        torch.save(
            {"epoch": epoch, "radius": self.model.module.radius,
            "model_state_dict": self.model.module.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))
        
        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))
        torch.save(
            {"epoch": epoch, "scheduler_state_dict": self.scheduler.state_dict()},
            os.path.join(self.checkpoints_path, self.scheduler_params_subdir, "latest.pth"))

        if self.optimize_inputs:
            torch.save(
                {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, "latest.pth"))
            
            dict_to_save = {}
            dict_to_save["epoch"] = epoch
            if self.optimize_expression:
                dict_to_save["expression_state_dict"] = self.expression.state_dict()
            if self.optimize_pose:
                dict_to_save["flame_pose_state_dict"] = self.flame_pose.state_dict()
                dict_to_save["camera_pose_state_dict"] = self.camera_pose.state_dict()
            if self.optimize_scene_latent_code:
                dict_to_save["zero_latent_codes_state_dict"] = self.zero_latent_codes.state_dict()
                # dict_to_save["scene_latent_codes_state_dict"] = self.scene_latent_codes.state_dict()
                dict_to_save["source_scene_latent_codes_state_dict"] = self.source_scene_latent_codes.state_dict()
            
            torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, "latest.pth"))
            
            if not only_latest:
                torch.save(dict_to_save, os.path.join(self.checkpoints_path, self.input_params_subdir, str(epoch) + ".pth"))
                torch.save(
                    {"epoch": epoch, "optimizer_cam_state_dict": self.optimizer_cam.state_dict()},
                    os.path.join(self.checkpoints_path, self.optimizer_inputs_subdir, str(epoch) + ".pth"))
            
    def upsample_points(self, epoch):
        current_radius = self.model.module.radius
        points = self.model.module.pc.points.data

        num_p = points.shape[0]

        # if epoch <= 100:
        if epoch <= 20 * self.upsample_freq:
            noise = (torch.rand(*points.shape).cuda() - 0.5) * current_radius
        else:
            noise = (torch.rand(*points.shape).cuda() - 0.5) * 0.004

        new_points = noise + points
        
        dist.all_reduce(new_points, op=dist.ReduceOp.SUM)
        new_points /= self.num_gpus

        # if epoch < 5:
        if epoch < self.upsample_freq:
            self.model.module.pc.upsample_400_points(new_points)
        # elif 5 <= epoch < 10:
        elif self.upsample_freq <= epoch < 2 * self.upsample_freq:
            self.model.module.pc.upsample_800_points(new_points)
        # elif 10 <= epoch < 15:
        elif 2 * self.upsample_freq <= epoch < 3 * self.upsample_freq:
            self.model.module.pc.upsample_1600_points(new_points)
        # elif 15 <= epoch < 20:
        elif 3 * self.upsample_freq <= epoch < 4 * self.upsample_freq:
            self.model.module.pc.upsample_3200_points(new_points)
        # elif 20 <= epoch < 25:
        elif 4 * self.upsample_freq <= epoch < 5 * self.upsample_freq:
            self.model.module.pc.upsample_6400_points(new_points)
        # elif 25 <= epoch < 30:
        elif 5 * self.upsample_freq <= epoch < 6 * self.upsample_freq:
            self.model.module.pc.upsample_10000_points(new_points)
        # elif 30 <= epoch < 40:
        elif 6 * self.upsample_freq <= epoch < 8 * self.upsample_freq:
            self.model.module.pc.upsample_20000_points(new_points)
        # elif 40 <= epoch < 50:
        elif 8 * self.upsample_freq <= epoch < 10 * self.upsample_freq:
            self.model.module.pc.upsample_40000_points(new_points)
        # elif 50 <= epoch < 60:
        elif 10 * self.upsample_freq <= epoch < 12 * self.upsample_freq:
            self.model.module.pc.upsample_80000_points(new_points)
        # elif epoch >= 60:
        elif 12 * self.upsample_freq <= epoch:
            self.model.module.pc.upsample_130000_points(new_points)

        # if epoch == 5:
        if epoch == self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 10:
        elif epoch == 2 * self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 15:
        elif epoch == 3 * self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 20:
        elif epoch == 4 * self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 25:
        elif epoch == 5 * self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 30:
        elif epoch == 6 * self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 40:
        elif epoch == 8 * self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 50:
        elif epoch == 10 * self.upsample_freq:
            self.model.module.radius = 0.75 * current_radius
        # elif epoch == 60:
        elif epoch == 12 * self.upsample_freq:
            self.model.module.radius = 0.9 * current_radius
        # elif epoch > 60 and epoch % 5 == 0:
        elif epoch > 12 * self.upsample_freq and epoch % self.upsample_freq == 0:
            self.model.module.radius = 0.75 * current_radius

        if is_main_process():
            # if epoch >= 100:
            if epoch >= 20 * self.upsample_freq:
                print("old radius: {}, new radius: {}, sample radius: {}".format(current_radius, self.model.module.radius, 0.004))
            else:
                print("old radius: {}, new radius: {}, sample radius: {}".format(current_radius, self.model.module.radius, current_radius))
            print("old points: {}, new points: {}".format(num_p, self.model.module.pc.points.data.shape[0]))

        learnable_params = self.get_learnable_params()
        self.optimizer = torch.optim.Adam([
            {'params': learnable_params},
        ], lr=self.lr)


    def file_backup(self, path_conf):
        dir_lis = ['./model', './scripts', './utils', './flame', './datasets']
        recording_path = os.path.join(self.train_dir, '{}_recording'.format(self.formatted_time))
        os.makedirs(recording_path, exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(recording_path, dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))
        
        source_path = os.path.join(os.getcwd(), path_conf)
        filename = "{}_{}".format(self.formatted_time, source_path.split("/")[-1])
        destination_path = os.path.join(self.train_dir, filename)
        shutil.copy(source_path, destination_path)

    def train_mode(self):
        self.model.train()
        if self.optimize_inputs:
            if self.optimize_expression:
                self.expression.train()
            if self.optimize_pose:
                self.flame_pose.train()
                self.camera_pose.train()
            if self.optimize_scene_latent_code:
                self.zero_latent_codes.train()
                # self.scene_latent_codes.train()
                self.source_scene_latent_codes.train()

    def eval_mode(self):
        self.model.eval()
        if self.optimize_inputs:
            if self.optimize_expression:
                self.expression.eval()
            if self.optimize_pose:
                self.flame_pose.eval()
                self.camera_pose.eval()
            if self.optimize_scene_latent_code:
                self.zero_latent_codes.eval()
                self.source_scene_latent_codes.eval()
                # self.scene_latent_codes.eval()

    # def novel_view(self, cam_pose, euler_angle, translation):
    #     x_euler, y_euler, z_euler = euler_angle

    #     ## for novel view synthesis
    #     cam_R = cam_pose[:, :3, :3]
    #     cam_t = cam_pose[:, :3, -1]
    #     cam_t = cam_t.reshape(-1, 1).unsqueeze(0)
    #     world_R, world_t = cam_R, cam_t

    #     quat = torch.stack(quaternion_from_euler(x_euler, y_euler, z_euler)).to(self.device)
        
    #     rotate_world_R = quaternion_to_rotation_matrix(rotation_matrix_to_quaternion(world_R)+(quat).unsqueeze(0))
    #     trans_world_t = world_t + translation.to(self.device).reshape(-1, 1).unsqueeze(0)

    #     rotate_cam_R, rotate_cam_t = rotate_world_R, trans_world_t
    #     cam_pose = Rt_to_matrix4x4(rotate_cam_R, rotate_cam_t)
    #     return cam_pose
        
    def run(self):
        # if self.lora_finetuning:
        #     self.run_lora_finetuning()
        # else:
        if not self.is_test:
            self.run_train()
        else:
            self.run_test()

    def run_train(self):
        acc_loss = {}

        for epoch in range(self.start_epoch, self.nepochs + 1):     
            self.train_dataloader.sampler.set_epoch(epoch)

            if epoch in self.GT_lbs_milestones:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor

            if is_main_process():
                if epoch % (self.save_freq * 1) == 0 and epoch != self.start_epoch:
                    self.save_checkpoints(epoch)
                else:
                    if epoch % self.save_freq == 0 and (epoch != self.start_epoch or self.start_epoch == 0):
                        self.save_checkpoints(epoch, only_latest=True)

            # NOTE 매 epoch마다 point를 동기화.
            # points = self.model.module.pc.points.data
            # dist.barrier()
            # dist.broadcast(points, src=0)

            # points = self.model.module.pc.points.data.clone()
            # dist.all_reduce(points, op=dist.ReduceOp.SUM)
            # points /= self.num_gpus
            # self.model.module.pc.update_points(points)

            # if self.is_val and((epoch % self.plot_freq == 0 and epoch < 5) or (epoch % (self.plot_freq) == 0)):
            #     dist.barrier()
            #     self.run_val(epoch)

            self.train_mode()

            self.start_time.record()

            if self.enable_prune and epoch != self.start_epoch and epoch % self.upsample_freq == 0:
                # visible_points = self.model.module.visible_points
                # dist.barrier()
                # dist.broadcast(visible_points, src=0)

                visible_points = self.model.module.visible_points.clone().float()
                dist.all_reduce(visible_points, op=dist.ReduceOp.SUM)
                visible_points /= self.num_gpus
                visible_points = visible_points >= 0.5

                self.model.module.pc.prune(visible_points)

                # self.optimizer = torch.optim.Adam([
                #     {'params': list(self.model.module.parameters())},
                # ], lr=self.lr)
                learnable_params = self.get_learnable_params()
                self.optimizer = torch.optim.Adam([
                    {'params': learnable_params},
                ], lr=self.lr)
            
            if epoch <= 1: #  self.upsample_freq * 4:
                if epoch == self.start_epoch:
                    freeze_lst = ['clip_model', 'gaussian', 'scaling', 'rotations', 'opacity', 'color']
                    freeze_params, learnable_params = self.set_trainable_parameters(freeze_lst)
                    self.optimizer = torch.optim.Adam([
                        {'params': learnable_params},
                    ], lr=self.lr)
                    if is_main_process():
                        print('************************************************************')
                        print(f'[INFO] freeze parameters {freeze_lst} at the beginning of training')
                        print('************************************************************')
                elif epoch == self.start_epoch + 1:
                    freeze_lst = ['clip_model']
                    freeze_params, learnable_params = self.set_trainable_parameters(freeze_lst)
                    self.optimizer = torch.optim.Adam([
                        {'params': learnable_params},
                    ], lr=self.lr)
                    if is_main_process():
                        print('************************************************************')
                        print('[INFO] make all of the gaussian parameters trainable')
                        print('************************************************************')

            if self.enable_upsample and epoch % self.upsample_freq == 0 and epoch != 0:   
                self.upsample_points(epoch)

            self.model.module.visible_points = torch.zeros(self.model.module.pc.points.shape[0]).bool().to(self.device)
            
            if epoch <= 1 and epoch == self.start_epoch and self.diffmorpher_weight <= 0:
                train_dataloader = self.train_single_dataloader
                num_train_dataset = self.num_train_single_dataset
            else:
                train_dataloader = self.train_dataloader
                num_train_dataset = self.num_train_dataset

            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(train_dataloader), desc='[INFO] training...', total=len(train_dataloader)):
                # dist.barrier()
                current_step = epoch * num_train_dataset + data_index * self.batch_size
                if data_index % self.save_freq == 0 or data_index == 0: # and not self.target_training and not self.multi_source_training:
                    if self.is_val and not (epoch <= 1 and epoch == self.start_epoch):
                        dist.barrier()
                        self.run_val(epoch, step=current_step)
                    self.train_mode()
                
                # if self.diffmorpher_weight > 0 and data_index % self.save_freq == 0:
                #     dist.barrier()
                #     self.run_val(epoch, step=current_step)
                self.train_mode()

                self.start_time_step.record()
                for k, v in model_input.items():
                    try:
                        model_input[k] = v.to(self.device)
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.to(self.device)
                    except:
                        ground_truth[k] = v

                if self.optimize_inputs:
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)

                category = model_input['sub_dir'][0].split('_')[1]
                if self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs):
                    alpha_list = list(torch.linspace(0, 1, self.accumulation_steps))
                    model_input['alpha_list'] = alpha_list
                    pseudo_gt_alphas, occlusion_masks = self.loss.get_diffmorpher_pseudo_gt(ground_truth, model_input)
                    ground_truth['pseudo_gt_alphas'] = pseudo_gt_alphas
                    ground_truth['occlusion_masks'] = occlusion_masks       # s가 붙었다.

                if self.lora_finetuning:
                    random_latent_code = torch.randn_like(self.zero_latent_codes(torch.tensor([0], device=self.device))) * self.latent_code_std

                for idx in range(self.accumulation_steps):
                    if self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs):
                        sub_dir = model_input['sub_dir'][0]
                        sub_dir_another_pair = model_input['sub_dir_another_pair'][0]
                        model_input['idx_accumulation_steps'] = idx
                        if idx == 0:          # 확인완료
                            model_input['dataset_type'] = ['pivot']
                        elif idx == self.accumulation_steps - 1:        # 확인완료
                            model_input['dataset_type'] = ['pivot_another_pair']
                            model_input['sub_dir'] = model_input['sub_dir_another_pair']
                        else:
                            model_input['dataset_type'] = ['diffmorpher']
                            alpha = alpha_list[idx].item()
                            model_input['interpolation_ratio'] = alpha
                    else:
                        if self.lora_finetuning:
                            model_input['lora_index'] = idx
                            if idx == 0:
                                # lora를 킨 상태로 새로운 attribute에 대해 RGB loss를 구한다.
                                model_input['lora_weight'] = 1.0
                            elif idx == 1:
                                # lora를 끄고 random latent sample에 대해 3d value를 구한다.
                                three_dim_values = 0
                                model_input['lora_weight'] = 0.0
                                model_input['random_latent_code'] = random_latent_code
                            elif idx == 2:
                                # lora를 키고 아까 그 random latent sample에 대해 3d loss를 구한다.
                                model_input['lora_weight'] = 1.0
                                model_input['random_latent_code'] = random_latent_code
                            else: 
                                assert False, "accumulation_steps should be 3"

                        model_input['dataset_type'] = ['pivot']
                        
                    model_input['category_dict'] = self.category_dict
                    model_input['source_category_dict'] = self.source_category_dict
                    model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                    model_input['zero_latent_codes'] = self.zero_latent_codes
                    model_input['data_index'] = data_index

                    model_outputs = self.model(model_input)
                    if self.lora_finetuning:
                        if idx == 1:
                            three_dim_values = model_outputs['three_dim_values'].copy()            # lora가 적용 안되어있음.
                            for key in three_dim_values.keys():
                                three_dim_values[key] = three_dim_values[key].detach()
                            del model_outputs
                            torch.cuda.empty_cache()
                            continue
                        elif idx == 2:
                            ground_truth['three_dim_values'] = three_dim_values
                    loss_output = self.loss(model_outputs, 
                                            ground_truth, 
                                            model_input)
                    loss = loss_output['loss']

                    self.optimizer.zero_grad()
                    if self.optimize_inputs and epoch > self.upsample_freq * 2:
                        self.optimizer_cam.zero_grad()
                    loss.backward()

                    self.optimizer.step()
                    if self.optimize_inputs and epoch > self.upsample_freq * 2:
                        self.optimizer_cam.step()
                    
                    # dist.barrier()
                    # dist.broadcast(self.zero_latent_codes.weight, src=0)
                    # dist.broadcast(self.source_scene_latent_codes.weight, src=0)

                    weight = self.zero_latent_codes.weight.clone()
                    dist.all_reduce(weight, op=dist.ReduceOp.SUM)
                    self.zero_latent_codes.weight = nn.Parameter(weight / self.num_gpus)

                    weight = self.source_scene_latent_codes.weight.clone()
                    dist.all_reduce(weight, op=dist.ReduceOp.SUM)
                    self.source_scene_latent_codes.weight = nn.Parameter(weight / self.num_gpus)

                    if self.optimize_expression:
                        weight = self.expression.weight.clone()
                        dist.all_reduce(weight, op=dist.ReduceOp.SUM)
                        self.expression.weight = nn.Parameter(weight / self.num_gpus)

                    if self.optimize_pose:
                        weight = self.flame_pose.weight.clone()
                        dist.all_reduce(weight, op=dist.ReduceOp.SUM)
                        self.flame_pose.weight = nn.Parameter(weight / self.num_gpus)

                        weight = self.camera_pose.weight.clone()
                        dist.all_reduce(weight, op=dist.ReduceOp.SUM)
                        self.camera_pose.weight = nn.Parameter(weight / self.num_gpus)

                for k, v in loss_output.items():
                    loss_output[k] = v.detach().item()
                    if k not in acc_loss:
                        acc_loss[k] = [v]
                    else:
                        acc_loss[k].append(v)

                acc_loss['visible_percentage'] = (torch.sum(self.model.module.visible_points)/self.model.module.pc.points.shape[0]).unsqueeze(0)

                self.end_time_step.record()
                if data_index % self.log_freq == 0:
                    for k, v in acc_loss.items():
                        acc_loss[k] = sum(v) / len(v)
                    print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                    for k, v in acc_loss.items():
                        print_str += '{}: {:.3g} '.format(k, v)
                    print_str += 'num_points: {} radius: {}'.format(self.model.module.pc.points.shape[0], self.model.module.radius)
                    if is_main_process():
                        print(print_str)
                    acc_loss['num_points'] = self.model.module.pc.points.shape[0]
                    acc_loss['radius'] = self.model.module.radius
                    acc_loss['lr'] = self.scheduler.get_last_lr()[0]
                    wandb.log(acc_loss, step=epoch * num_train_dataset + data_index * self.batch_size)
                    wandb.log({"timing_step": self.start_time_step.elapsed_time(self.end_time_step)}, step=epoch * num_train_dataset + data_index * self.batch_size)
                    acc_loss = {}

                if self.diffmorpher_weight > 0: # model_input['dataset_type'] == ['diffmorpher']:
                    # JSON 파일 경로 설정 (rank 0에서만 사용)
                    json_path = os.path.join(self.train_dir, f'train_seen_dataset_log_{self.formatted_time}.json.gz')

                    # # 이전 로그를 불러오기 (rank 0에서만 수행)
                    # if dist.get_rank() == 0 and os.path.exists(json_path):
                    #     with open(json_path, 'r') as f:
                    #         try:
                    #             final_log = json.load(f)
                    #         except:
                    #             final_log = {}  # 파일 손상 시 빈 dict로 초기화
                    # else:
                    #     final_log = {}  # 초기 로그 설정

                    # 이전 로그를 불러오기 (rank 0에서만 수행)
                    if dist.get_rank() == 0 and os.path.exists(json_path):
                        with gzip.open(json_path, 'rt') as f:  # gzip으로 파일 열기
                            try:
                                final_log = json.load(f)
                            except:
                                final_log = {}  # 파일 손상 시 빈 dict로 초기화
                    else:
                        final_log = {}  # 초기 로그 설정

                    # 각 GPU의 로컬 로그를 생성
                    local_log = {}

                    # model_input['sub_dir'][0]을 key로 사용해 로컬 로그에 데이터 추가
                    if category in self.diffmorpher_target_attrs:
                        key = '{0}_to_{1}'.format(sub_dir, sub_dir_another_pair)
                    else:
                        key = model_input['sub_dir'][0]
                    if key not in local_log:
                        local_log[key] = []  # 새로운 리스트로 초기화

                    # ground_truth['image_paths'][0]을 해당 key에 추가
                    local_log[key].append(os.path.basename(ground_truth['image_paths'][0]))

                    # 모든 GPU의 로그를 마스터 프로세스로 수집
                    gathered_logs = [None] * dist.get_world_size()
                    dist.all_gather_object(gathered_logs, local_log)

                    # rank 0에서 이전 로그와 새 로그 병합 및 저장
                    if dist.get_rank() == 0:
                        # 모든 GPU에서 수집한 로그 병합
                        for log in gathered_logs:
                            for k, v in log.items():
                                if k in final_log:
                                    final_log[k].extend(v)  # 기존 리스트에 데이터 추가
                                else:
                                    final_log[k] = v  # 새로운 key 추가

                        # # 병합된 로그를 JSON 파일에 저장
                        # with open(json_path, 'w') as f:
                        #     json.dump(final_log, f, indent=4)
                        # 병합된 로그를 JSON 파일에 압축해서 저장
                        with gzip.open(json_path, 'wt') as f:
                            json.dump(final_log, f, separators=(',', ':'))  # 최소한의 포맷으로 저장

                        # 체크포인트 저장
                        # self.save_checkpoints(epoch, only_latest=True)

                if data_index % self.save_freq == 0 and is_main_process():
                    self.save_checkpoints(epoch)

                # torch.cuda.synchronize()
                
            self.scheduler.step()
            self.end_time.record()
            torch.cuda.synchronize()
            wandb.log({"timing_epoch": self.start_time.elapsed_time(self.end_time)}, step=(epoch+1) * num_train_dataset)
            print("Epoch time: {} s".format(self.start_time.elapsed_time(self.end_time)/1000))

        if is_main_process():
            self.save_checkpoints(self.nepochs + 1)
        
        wandb.finish()
    
    def run_val(self, epoch, step=None):
        self.eval_mode()
        self.start_time.record()
        
        novel_view_type =  'val'

        indices, model_input, ground_truth = next(iter(self.val_dataloader))
        # model_input['cam_pose'][-1, -1, -1] += 1.5
        if step is None:
            data_index = 1
            step = epoch * self.num_train_dataset + data_index * self.batch_size

        for k, v in model_input.items():
            try:
                model_input[k] = v.to(self.device)
            except:
                model_input[k] = v
        for k, v in ground_truth.items():
            try:
                ground_truth[k] = v.to(self.device)
            except:
                ground_truth[k] = v
        
        if self.optimize_inputs:
            if self.optimize_scene_latent_code:
                model_input['dataset_type'] = ['pivot']
                model_input['category_dict'] = self.category_dict
                model_input['source_category_dict'] = self.source_category_dict
                model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                model_input['zero_latent_codes'] = self.zero_latent_codes

            if self.lora_finetuning:
                model_input['lora_weight'] = 1.0
                model_input['random_latent_code'] = self.random_latent_code

            if self.diffmorpher_weight > 0:
                model_input['rendering_type'] = ['interpolation']

                pair_files = self.pair_files
                with open(pair_files, 'r') as f:
                    lines = f.readlines()
                    pairs = [line.strip().split(',') for line in lines]
                pair_dict = dict()
                for k, pair in enumerate(pairs):
                    if pair[2] in pair_dict:
                        pair_dict[pair[2]].append((pair[0], pair[1], k+1))
                    else:
                        pair_dict[pair[2]] = [(pair[0], pair[1], k+1)]
                
                for idx, category in tqdm(enumerate(pair_dict.keys()), desc='[INFO] pair...', total=len(pair_dict.keys()), position=1):
                    if idx > 1:
                        break
                    sub_dir_pairs = pair_dict[category]
                    for j, (target_a, target_b, idx_txt) in tqdm(enumerate(sub_dir_pairs), desc='[INFO] pair...', total=len(sub_dir_pairs), position=0):
                        if j > 1:
                            break
                        target_a_category = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_a)), None).split('_')[1]
                        target_b_category = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_b)), None).split('_')[1]
                        assert target_a_category == target_b_category, "The target categories should be the same."
                        target_a_subdir = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_a)), None)
                        target_b_subdir = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_b)), None)

                        img_name = model_input['img_name'].item()

                        for l in range(self.interpolation_ratio):
                            novel_view_type = 'interpolation'
                            plot_dir = [os.path.join(self.eval_dir, 'interpolation_val') for i in range(len(model_input['sub_dir']))]
                            img_name = model_input['img_name'].item()

                            img_name = np.array('epoch-{}_step-{}_{}_{}_{}_{}-{}'.format(epoch, step, idx_txt, target_a, target_b, img_name, str(l+1).zfill(2)))

                            model_input['interpolation_ratio'] = l/self.interpolation_ratio
                            model_input['sub_dir_a'] = [target_a_subdir]
                            model_input['sub_dir_b'] = [target_b_subdir]

                            if self.lora_finetuning:
                                model_input['lora_weight'] = 1 - l/self.interpolation_ratio

                            with torch.set_grad_enabled(True):
                                model_outputs = self.model(model_input)

                            for k, v in model_outputs.items():
                                try:
                                    model_outputs[k] = v.detach()
                                except:
                                    model_outputs[k] = v

                            if is_main_process():
                                utils.mkdir_ifnotexists(plot_dir[0])
                            dist.barrier()
                            plt.plot(img_name,
                                    model_outputs,
                                    ground_truth,
                                    plot_dir,
                                    epoch,
                                    self.img_res,
                                    is_eval=True,
                                    first=True,
                                    custom_settings={'novel_view': novel_view_type})

                            # NOTE to save memory. but make slower.
                            del model_outputs, self.model
                            torch.cuda.empty_cache()
                            self.model = copy.copy(self.model_copied)

            elif not self.lora_finetuning:
                val_subdir = self.val_subdir
                target_subdir = val_subdir[str(get_rank())]
                model_input['sub_dir'] = [target_subdir]
                model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v

                # plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(str(epoch), novel_view_type)) for i in range(len(model_input['sub_dir']))]
                plot_dir = [os.path.join(self.eval_dir, 'rendering_val')]
                img_names = model_input['img_name'][:, 0].cpu().numpy()
                img_names = np.array(['epoch_{}_step_{}_img_{}_{}_{}'.format(epoch, step, img_names[0], target_subdir.split('_')[0], target_subdir.split('_')[1])])

                tqdm.write("Plotting images: {}.png".format(os.path.join(plot_dir[0], img_names[0])))
                if is_main_process():
                    utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))

                plt.plot(img_names,
                            model_outputs,
                            ground_truth,
                            plot_dir,
                            epoch,
                            self.img_res,
                            is_eval=False,
                            first=True,
                            custom_settings={'novel_view': novel_view_type, 'step': step, 'render_concat_gt': True})

                del model_outputs

            if self.lora_finetuning:
                novel_view_type += '_lora'
                del model_input['random_latent_code']
                torch.cuda.empty_cache()

                model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v

                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(str(epoch), novel_view_type)) for i in range(len(model_input['sub_dir']))]
                img_names = model_input['img_name'][:, 0].cpu().numpy()

                tqdm.write("Plotting images: {}".format(os.path.join(os.path.dirname(plot_dir[0]), 'rendering_validation')))
                if is_main_process():
                    utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))

                plt.plot(img_names,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        epoch,
                        self.img_res,
                        is_eval=False,
                        first=True,
                        custom_settings={'novel_view': novel_view_type, 'step': step, 'render_concat_gt': True})

                del model_outputs
            
            del ground_truth, model_input
        
        self.end_time.record()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        if is_main_process():
            tqdm.write("Plot time per image: {} ms".format(self.start_time.elapsed_time(self.end_time) / len(self.val_dataset)))
    
    def run_test(self):
        self.eval_mode()
        self.model.training = False

        is_first_batch = True

        eval_iterator = iter(self.test_dataloader)

        for img_index in tqdm(range(len(self.test_dataloader)), desc='[INFO] Rendering...'):
            indices, model_input, ground_truth = next(eval_iterator)

            for k, v in model_input.items():
                try:
                    model_input[k] = v.to(self.device)
                except:
                    model_input[k] = v

            for k, v in ground_truth.items():
                try:
                    ground_truth[k] = v.to(self.device)
                except:
                    ground_truth[k] = v

            if self.optimize_inputs:
                if self.optimize_expression:
                    model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                if self.optimize_pose:
                    model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                    model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
            
            is_first_batch = True

            if self.rendering_default:
                model_input['rendering_type'] = ['default']
                model_input['category_dict'] = self.category_dict
                model_input['source_category_dict'] = self.source_category_dict
                model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                model_input['zero_latent_codes'] = self.zero_latent_codes

                if self.lora_finetuning:
                    model_input['lora_weight'] = 1.0

                # 카테고리별 train_subdir 그룹화
                category_dict = defaultdict(list)

                for train_subdir in self.test_dataset.valid_dirs:
                    category = train_subdir.split('_')[1]
                    if category in ['test', 'source']:
                        continue
                    category_dict[category].append(train_subdir)

                # 각 카테고리별 최대 18개까지만 선택
                selected_train_subdirs = []
                for category, subdirs in category_dict.items():
                    selected_train_subdirs.extend(subdirs[:])  # 최대 18개까지만 사용

                for train_subdir in selected_train_subdirs:
                    if False:
                        if not any(category_idx in train_subdir for category_idx in ["0492"]):
                            continue
                    novel_view_type = 'rendering_default_{}'.format(train_subdir)
                    model_input['sub_dir'] = [train_subdir]
                    if self.rendering_default_novel_view:
                        plot_dir = [os.path.join(self.eval_dir, 'novel_view', '{}_novel_view_{}'.format(model_input['sub_dir'][0], "_".join(f"{num}" for num in self.rendering_default_novel_view_euler_angle)))]
                    else:
                        plot_dir = [os.path.join(self.eval_dir, 'default', model_input['sub_dir'][0])]
                    img_names = model_input['img_name'][:,0].cpu().numpy()[0]

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    if model_outputs is not None:
                        for k, v in model_outputs.items():
                            try:
                                model_outputs[k] = v.detach()
                            except:
                                model_outputs[k] = v

                        if is_main_process():
                            utils.mkdir_ifnotexists(plot_dir[0])
                        plt.plot(img_names,
                                model_outputs,
                                ground_truth,
                                plot_dir,
                                self.test_epoch,
                                self.img_res,
                                is_eval=True,
                                first=is_first_batch,
                                custom_settings={'novel_view': novel_view_type, 'render_concat_gt': False})

                        del model_outputs, self.model
                        torch.cuda.empty_cache()
                        self.model = copy.copy(self.model_copied)

            if self.interpolation_rendering:
                # if not self.lora_finetuning:
                model_input['rendering_type'] = ['interpolation']
                model_input['category_dict'] = self.category_dict
                model_input['source_category_dict'] = self.source_category_dict
                model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                model_input['zero_latent_codes'] = self.zero_latent_codes

                pair_files = self.pair_files
                with open(pair_files, 'r') as f:
                    lines = f.readlines()
                    pairs = [line.strip().split(',') for line in lines]
                pair_dict = dict()
                for k, pair in enumerate(pairs):
                    if pair[2] in pair_dict:
                        pair_dict[pair[2]].append((pair[0], pair[1], k+1))
                    else:
                        pair_dict[pair[2]] = [(pair[0], pair[1], k+1)]
                
                for category in pair_dict.keys():
                    sub_dir_pairs = pair_dict[category]
                    for j, (target_a, target_b, idx_txt) in tqdm(enumerate(sub_dir_pairs), desc='[INFO] pair...', total=len(sub_dir_pairs), position=0):
                        target_a_category = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_a)), None).split('_')[1]
                        target_b_category = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_b)), None).split('_')[1]
                        assert target_a_category == target_b_category, "The target categories should be the same."
                        target_a_subdir = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_a)), None)
                        target_b_subdir = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(target_b)), None)

                        img_name = model_input['img_name'].item()

                        for l in range(self.interpolation_ratio):
                            novel_view_type = 'interpolation'
                            plot_dir = [os.path.join(self.eval_dir, 'interpolation', 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                            img_name = model_input['img_name'].item()

                            img_name = np.array('{}_{}_{}_{}-{}'.format(idx_txt, target_a, target_b, img_name, str(l+1).zfill(2)))

                            model_input['interpolation_ratio'] = l/self.interpolation_ratio
                            model_input['sub_dir_a'] = [target_a_subdir]
                            model_input['sub_dir_b'] = [target_b_subdir]

                            if self.lora_finetuning:
                                model_input['lora_weight'] = 1 - l/self.interpolation_ratio

                            with torch.set_grad_enabled(True):
                                model_outputs = self.model(model_input)

                            for k, v in model_outputs.items():
                                try:
                                    model_outputs[k] = v.detach()
                                except:
                                    model_outputs[k] = v

                            if is_main_process():
                                utils.mkdir_ifnotexists(plot_dir[0])
                            dist.barrier()
                            plt.plot(img_name,
                                    model_outputs,
                                    ground_truth,
                                    plot_dir,
                                    self.test_epoch,
                                    self.img_res,
                                    is_eval=True,
                                    first=is_first_batch,
                                    custom_settings={'novel_view': novel_view_type})

                            # NOTE to save memory. but make slower.
                            # del model_outputs, self.model
                            # torch.cuda.empty_cache()
                            # self.model = copy.copy(self.model_free_memory)
                            del model_outputs
                        
                        for l in range(self.interpolation_ratio):
                            novel_view_type = 'interpolation'
                            plot_dir = [os.path.join(self.eval_dir, 'interpolation', 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                            img_name = model_input['img_name'].item()

                            img_name = np.array('{}_{}_{}_{}-{}'.format(idx_txt, target_a, target_b, img_name, str(self.interpolation_ratio+l+1).zfill(2)))

                            model_input['interpolation_ratio'] = 1 - l/self.interpolation_ratio
                            model_input['sub_dir_a'] = [target_a_subdir]
                            model_input['sub_dir_b'] = [target_b_subdir]

                            if self.lora_finetuning:
                                model_input['lora_weight'] = l/self.interpolation_ratio

                            with torch.set_grad_enabled(True):
                                model_outputs = self.model(model_input)

                            for k, v in model_outputs.items():
                                try:
                                    model_outputs[k] = v.detach()
                                except:
                                    model_outputs[k] = v

                            if is_main_process():
                                utils.mkdir_ifnotexists(plot_dir[0])
                            dist.barrier()
                            plt.plot(img_name,
                                    model_outputs,
                                    ground_truth,
                                    plot_dir,
                                    self.test_epoch,
                                    self.img_res,
                                    is_eval=True,
                                    first=is_first_batch,
                                    custom_settings={'novel_view': novel_view_type})

                            # NOTE to save memory. but make slower.
                            # del model_outputs, self.model
                            # torch.cuda.empty_cache()
                            # self.model = copy.copy(self.model_free_memory)
                            del model_outputs
                if False:
                    model_input['rendering_type'] = ['interpolation']
                    model_input['category_dict'] = self.category_dict
                    model_input['source_category_dict'] = self.source_category_dict
                    model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                    model_input['zero_latent_codes'] = self.zero_latent_codes

                    target_a, target_b = self.interpolation_target_list
                    target_a_category = target_a.split('_')[1]
                    target_b_category = target_b.split('_')[1]
                    assert target_a_category == target_b_category, "The target categories should be the same."

                    img_name = model_input['img_name'].item()

                    for l in range(self.interpolation_ratio):
                        # NOTE interpolation을 하는 코드.
                        novel_view_type = '{}_interpolation_{}_to_{}'.format(target_a_category, target_a, target_b)
                        plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                        img_name = model_input['img_name'].item()

                        img_name = np.array('{}-{}'.format(img_name, l))

                        model_input['interpolation_ratio'] = l/self.interpolation_ratio
                        model_input['sub_dir_a'] = [target_a]
                        model_input['sub_dir_b'] = [target_b]

                        if self.lora_finetuning:
                            model_input['lora_weight'] = 1 - l/self.interpolation_ratio

                        with torch.set_grad_enabled(True):
                            model_outputs = self.model(model_input)

                        for k, v in model_outputs.items():
                            try:
                                model_outputs[k] = v.detach()
                            except:
                                model_outputs[k] = v

                        print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
                        if is_main_process():
                            utils.mkdir_ifnotexists(os.path.join(self.eval_dir, model_input['sub_dir'][0]))
                            if eval_all:
                                for dir in plot_dir:
                                    utils.mkdir_ifnotexists(dir)
                        dist.barrier()
                        plt.plot(img_name,
                                model_outputs,
                                ground_truth,
                                plot_dir,
                                self.test_epoch,
                                self.img_res,
                                is_eval=eval_all,
                                first=is_first_batch,
                                custom_settings={'novel_view': novel_view_type})

                        # NOTE to save memory. but make slower.
                        # del model_outputs, self.model
                        # torch.cuda.empty_cache()
                        # self.model = copy.copy(self.model_free_memory)
                        del model_outputs
            
            if self.interpolation_rendering_same_time:
                model_input['rendering_type'] = ['same_time']
                model_input['category_dict'] = self.category_dict
                model_input['source_category_dict'] = self.source_category_dict
                model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                model_input['zero_latent_codes'] = self.zero_latent_codes

                pair_files = self.pair_files
                with open(pair_files, 'r') as f:
                    lines = f.readlines()
                    pairs = [line.strip().split(',') for line in lines]

                # idx_txt, category = pairs[???]
                self.interpolation_ratio = len(self.test_dataset)//(len(pairs)*2)
                intervals = [
                    (i * self.interpolation_ratio * 2, (i + 1) * self.interpolation_ratio * 2 - 1)
                    for i in range(len(pairs))
                ]
                # Find the current pair and interval
                current_pair = None
                for pair_idx, (start, end) in enumerate(intervals):
                    if start <= indices.item() <= end:
                        current_pair = pairs[pair_idx]
                        break

                if current_pair is None:
                    continue

                idx_txt, category = current_pair
                if category == 'source':
                    train_subdir = '9999_source'
                else:
                    train_subdir = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(idx_txt)), None)

                    if train_subdir.split('_')[1] in ['test', 'source']:
                        continue
                
                novel_view_type = 'rendering_same_time_{}'.format(train_subdir)
                model_input['sub_dir'] = [train_subdir]

                # Calculate l
                local_idx = indices.item() - intervals[pair_idx][0]  # Adjust index within the pair's interval
                half_interval = self.interpolation_ratio
                if local_idx < half_interval:
                    l = local_idx / half_interval  # 0 to 1
                else:
                    l = 1 - (local_idx - half_interval) / half_interval  # 1 to 0

                plot_dir = [os.path.join(self.eval_dir, 'same_time')]
                img_name = model_input['img_name'].item()

                img_name = np.array('{}_{}_{}_{}'.format(str(img_name).zfill(3), str(local_idx).zfill(3), idx_txt, category))

                model_input['interpolation_ratio'] = l

                with torch.set_grad_enabled(True):
                    model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v

                print("Plotting images: {}".format(os.path.join(plot_dir[0], 'rgb', '{}.png'.format(img_name))))
                if is_main_process():
                    for dir in plot_dir:
                        utils.mkdir_ifnotexists(dir)
                dist.barrier()
                plt.plot(img_name,
                        model_outputs,
                        ground_truth,
                        plot_dir,
                        self.test_epoch,
                        self.img_res,
                        is_eval=True,
                        first=is_first_batch,
                        custom_settings={'novel_view': novel_view_type})

                # NOTE to save memory. but make slower.
                # del model_outputs, self.model
                # torch.cuda.empty_cache()
                # self.model = copy.copy(self.model_free_memory)
                del model_outputs

                # pair_files = self.pair_files
                # with open(pair_files, 'r') as f:
                #     lines = f.readlines()
                #     pairs = [line.strip().split(',') for line in lines]

                # for idx_txt, category in pairs:
                #     train_subdir = next((item for i, item in enumerate(self.dataset_train_subdir) if item.startswith(idx_txt)), None)

                #     if train_subdir.split('_')[1] in ['test', 'source']:
                #         continue
                    
                #     novel_view_type = 'rendering_same_time_{}'.format(train_subdir)
                #     model_input['sub_dir'] = [train_subdir]

                #     for l in range(self.interpolation_ratio):
                #         plot_dir = [os.path.join(self.eval_dir, 'same_time')]
                #         img_name = model_input['img_name'].item()

                #         img_name = np.array('{}_{}_{}-{}'.format(idx_txt, category, img_name, str(l+1).zfill(2)))

                #         model_input['interpolation_ratio'] = l/self.interpolation_ratio

                #         with torch.set_grad_enabled(True):
                #             model_outputs = self.model(model_input)

                #         for k, v in model_outputs.items():
                #             try:
                #                 model_outputs[k] = v.detach()
                #             except:
                #                 model_outputs[k] = v

                #         if is_main_process():
                #             utils.mkdir_ifnotexists(plot_dir[0])
                #         dist.barrier()
                #         plt.plot(img_name,
                #                 model_outputs,
                #                 ground_truth,
                #                 plot_dir,
                #                 self.test_epoch,
                #                 self.img_res,
                #                 is_eval=True,
                #                 first=is_first_batch,
                #                 custom_settings={'novel_view': novel_view_type})

                #         # NOTE to save memory. but make slower.
                #         # del model_outputs, self.model
                #         # torch.cuda.empty_cache()
                #         # self.model = copy.copy(self.model_free_memory)
                #         del model_outputs

            if self.zero_shot_rendering:
                model_input['rendering_type'] = ['zero_shot']
                model_input['category_zero_shot'] = self.zero_shot_category
                model_input['category_dict'] = self.category_dict
                model_input['source_category_dict'] = self.source_category_dict
                model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                model_input['zero_latent_codes'] = self.zero_latent_codes

                # novel_view_type = 'rendering_zero_shot_{}'.format('hair_buzz_cut_afro_auburn_fine_1--Guy')
                novel_view_type = 'rendering_zero_shot_{}'.format(model_input['sub_dir_zero_shot'][0])
                plot_dir = [os.path.join(self.eval_dir, model_input['sub_dir'][i], 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
                img_names = model_input['img_name'][:,0].cpu().numpy()[0]

                with torch.set_grad_enabled(True):
                    model_outputs = self.model(model_input)

                for k, v in model_outputs.items():
                    try:
                        model_outputs[k] = v.detach()
                    except:
                        model_outputs[k] = v

                if is_main_process():
                    utils.mkdir_ifnotexists(plot_dir[0])
                dist.barrier()
                plt.plot(img_names,
                         model_outputs,
                         ground_truth,
                         plot_dir,
                         self.test_epoch,
                         self.img_res,
                         is_eval=True,
                         first=is_first_batch,
                         custom_settings={'novel_view': novel_view_type})

                del model_outputs
            
            if self.random_sampling_rendering:
                model_input['category_random_sampling'] = self.random_sampling_category
                # 매번 달라지겠금.
                model_input['rendering_type'] = ['random_sampling']
                model_input['category_dict'] = self.category_dict
                model_input['source_category_dict'] = self.source_category_dict
                model_input['source_scene_latent_codes'] = self.source_scene_latent_codes
                model_input['zero_latent_codes'] = self.zero_latent_codes

                novel_view_type = f'rendering_random_sampling_{self.random_sampling_category}'
                plot_dir = [os.path.join(self.eval_dir, 'random_sampling', 'epoch_{}_{}'.format(self.test_epoch, novel_view_type)) for i in range(len(model_input['sub_dir']))]
            
                for n in range(self.random_sampling_n_samples):
                    img_names = model_input['img_name'].item()
                    img_names = np.array(['{}_img_{}_{}'.format(self.random_sampling_category, img_names, n)])

                    model_input['random_latent_code'] = torch.randn_like(self.zero_latent_codes(torch.tensor([0], device=self.device))) * self.random_sampling_std

                    with torch.set_grad_enabled(True):
                        model_outputs = self.model(model_input)

                    for k, v in model_outputs.items():
                        try:
                            model_outputs[k] = v.detach()
                        except:
                            model_outputs[k] = v

                    if is_main_process():
                        utils.mkdir_ifnotexists(plot_dir[0])
                    dist.barrier()
                    plt.plot(img_names,
                                model_outputs,
                                ground_truth,
                                plot_dir,
                                self.test_epoch,
                                self.img_res,
                                is_eval=True,
                                first=is_first_batch,
                                custom_settings={'novel_view': novel_view_type})

                    del model_outputs

            del ground_truth
    
    def run_lora_finetuning(self):
        acc_loss = {}
        
        for epoch in range(self.start_epoch, self.nepochs + 1):     
            self.train_dataloader.sampler.set_epoch(epoch)

            if epoch in self.GT_lbs_milestones:
                self.loss.lbs_weight = self.loss.lbs_weight * self.GT_lbs_factor

            if is_main_process():
                if epoch % (self.save_freq * 1) == 0 and epoch != self.start_epoch:
                    self.save_checkpoints(epoch)
                else:
                    if epoch % self.save_freq == 0 and (epoch != self.start_epoch or self.start_epoch == 0):
                        self.save_checkpoints(epoch, only_latest=True)

            # NOTE 매 epoch마다 point를 동기화.
            # points = self.model.module.pc.points.data
            # dist.barrier()
            # dist.broadcast(points, src=0)
            points = self.model.module.pc.points.clone()  # .data 대신 clone() 사용
            dist.all_reduce(points, op=dist.ReduceOp.SUM)
            points /= self.num_gpus
            self.model.module.pc.update_points(points)

            if self.is_val and ((epoch % self.plot_freq == 0 and epoch < 5) or (epoch % (self.plot_freq) == 0)):
                self.run_val(epoch)
                dist.barrier()

            self.train_mode()

            self.start_time.record()

            self.model.module.visible_points = torch.zeros(self.model.module.pc.points.shape[0]).bool().to(self.device)
            
            upsample_freq_iter = len(self.train_dataloader) // 4 # len(self.dataset_train_subdir) * 100 # NOTE 100번 같은 dataset에서 step이 이루어진 경우 train epoch내에서 upsample.
            for data_index, (indices, model_input, ground_truth) in tqdm(enumerate(self.train_dataloader), desc='[INFO] training...', total=len(self.train_dataloader)):
                current_step = epoch * self.num_train_dataset + data_index * self.batch_size
                if current_step % upsample_freq_iter == 0: # and not self.target_training and not self.multi_source_training:
                    if self.is_val:
                        self.run_val(epoch, step=current_step)
                    dist.barrier()
                    self.train_mode()

                self.start_time_step.record()
                for k, v in model_input.items():
                    try:
                        model_input[k] = v.to(self.device)
                    except:
                        model_input[k] = v
                for k, v in ground_truth.items():
                    try:
                        ground_truth[k] = v.to(self.device)
                    except:
                        ground_truth[k] = v

                if self.optimize_inputs:
                    if self.optimize_expression:
                        model_input['expression'] = self.expression(model_input["idx"]).squeeze(1)
                    if self.optimize_pose:
                        model_input['flame_pose'] = self.flame_pose(model_input["idx"]).squeeze(1)
                        model_input['cam_pose'][:, :3, 3] = self.camera_pose(model_input["idx"]).squeeze(1)
                    if self.optimize_scene_latent_code:
                        # step 1. input data에 맞는 scene latent code를 가져온다.
                        indices = [self.dataset_train_subdir.index(name) for name in model_input['sub_dir']]
                        scene_latent_code_sample = self.scene_latent_codes(torch.tensor(indices, 
                                                                                        dtype=torch.long, 
                                                                                        device=self.device))
                        adaptive_latent_code_sample = self.adaptive_latent_codes(torch.tensor(indices,
                                                                                              dtype=torch.long,
                                                                                              device=self.device))
                        
                        # step 2. category one hot vector를 앞에 붙여준다.
                        index_list = []
                        for sub_dir_item in model_input['sub_dir']:
                            category_idx = self.category_dict[sub_dir_item.split('_')[0]]
                            index_list.append(category_idx)
                        category_one_hot = F.one_hot(torch.tensor(index_list), num_classes=len(self.category_dict)).to(self.device)
                        scene_latent_code_sample = torch.cat((category_one_hot, scene_latent_code_sample), dim=1)
                        adaptive_latent_code_sample = torch.cat((category_one_hot, adaptive_latent_code_sample), dim=1)

                        batch_scene_latent, scene_latent_dim = scene_latent_code_sample.shape

                        # step 3. initial input latent code를 만든다. [B, 32*len(category_dict)]
                        input_latent_codes = torch.zeros(batch_scene_latent, scene_latent_dim*len(self.category_dict))
                        
                        # step 4. zero latent code로 전체를 초기화해준다. 똑같은 latent code가 category code와 함께 들어간다.
                        for i, v in self.category_dict.items():
                            category_one_hot = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).to(self.device)
                            start_idx = v*scene_latent_dim
                            end_idx = (v+1)*scene_latent_dim
                            input_latent_codes[:, start_idx:end_idx] = torch.cat((category_one_hot, self.zero_latent_codes(torch.tensor(0).to(self.device))), dim=0)
                                
                        # step 5. source에 관한 latent code들을 넣어준다.
                        for i, v in enumerate(self.source_category_dict.values()):
                            category_one_hot = F.one_hot(torch.tensor(v), num_classes=len(self.category_dict)).to(self.device)
                            source_start_idx = v*scene_latent_dim
                            source_end_idx = (v+1)*scene_latent_dim
                            input_latent_codes[:, source_start_idx:source_end_idx] = torch.cat((category_one_hot, self.source_scene_latent_codes(torch.tensor(i).to(self.device))), dim=0)
                        
                        # step 6. 마지막으로 db human의 latent code를 넣어준다.
                        for i in range(len(index_list)):
                            start_idx = index_list[i]*scene_latent_dim
                            end_idx = (index_list[i]+1)*scene_latent_dim
                            input_latent_codes[i, start_idx:end_idx] = scene_latent_code_sample[i]
                            if self.latent_space_inversion:                                                 # NOTE latent space inversion인 경우에는 scene latent를 아예 대체해서 학습시켜줘야 함.
                                input_latent_codes[i, start_idx:end_idx] = adaptive_latent_code_sample[i]
                            else:                                                                           # NOTE latent space inversion이 아닌 경우에는 짝수번째에만 adaptive latent code를 넣어준다.
                                if data_index % 2 == 0:                                                     
                                    input_latent_codes[i, start_idx:end_idx] = adaptive_latent_code_sample[i]

                        # Add to the model_input dictionary
                        model_input['scene_latent_code'] = input_latent_codes.to(self.device)
                        model_input['type_latent'] = 'scene_latent'

                        if self.model_full_weight:
                            continue                                                                        # NOTE ful weight인 경우 adaptive latent에 대해서만 update를 해준다.

                        if self.latent_space_inversion:
                            model_input['type_latent'] = 'adaptive_latent'
                        else:
                            if data_index % 2 == 0:
                                model_input['type_latent'] = 'adaptive_latent'
                            
                model_outputs = self.model(model_input)
                loss_output = self.loss(model_outputs, ground_truth, model_input)
                loss = loss_output['loss']

                self.optimizer.zero_grad()
                if self.optimize_inputs and epoch > 10:
                    self.optimizer_cam.zero_grad()
                    
                torch.cuda.empty_cache()
                loss.backward()
                
                self.optimizer.step()
                if self.optimize_inputs and epoch > 10:
                    self.optimizer_cam.step()

                for k, v in loss_output.items():
                    loss_output[k] = v.detach().item()
                    if k not in acc_loss:
                        acc_loss[k] = [v]
                    else:
                        acc_loss[k].append(v)

                log_freq = self.log_freq
                dist.barrier()
                dist.broadcast(self.scene_latent_codes.weight, src=0)
                dist.broadcast(self.zero_latent_codes.weight, src=0)
                dist.broadcast(self.source_scene_latent_codes.weight, src=0)
                dist.broadcast(self.adaptive_latent_codes.weight, src=0)

                acc_loss['visible_percentage'] = (torch.sum(self.model.module.visible_points)/self.model.module.pc.points.shape[0]).unsqueeze(0)

                torch.cuda.synchronize()
                self.end_time_step.record()
                if data_index % log_freq == 0 and is_main_process():
                    for k, v in acc_loss.items():
                        acc_loss[k] = sum(v) / len(v)
                    print_str = '{0} [{1}] ({2}/{3}): '.format(self.methodname, epoch, data_index, self.n_batches)
                    for k, v in acc_loss.items():
                        print_str += '{}: {:.3g} '.format(k, v)
                    print_str += 'num_points: {} radius: {}'.format(self.model.module.pc.points.shape[0], self.model.module.radius)
                    print(print_str)
                    acc_loss['num_points'] = self.model.module.pc.points.shape[0]
                    acc_loss['radius'] = self.model.module.radius
                    acc_loss['lr'] = self.scheduler.get_last_lr()[0]
                    wandb.log(acc_loss, step=epoch * self.num_train_dataset + data_index * self.batch_size)
                    wandb.log({"timing_step": self.start_time_step.elapsed_time(self.end_time_step)}, step=epoch * self.num_train_dataset + data_index * self.batch_size)
                    acc_loss = {}

                if data_index % self.save_freq == 0 and is_main_process():
                    self.save_checkpoints(epoch)
                
            self.scheduler.step()
            self.end_time.record()
            torch.cuda.synchronize()
            if is_main_process():
                wandb.log({"timing_epoch": self.start_time.elapsed_time(self.end_time)}, step=(epoch+1) * self.num_train_dataset)
                print("Epoch time: {} s".format(self.start_time.elapsed_time(self.end_time)/1000))

        if is_main_process():
            self.save_checkpoints(self.nepochs + 1)