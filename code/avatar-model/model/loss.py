import torch
from torch import nn
from model.vgg_feature import VGGPerceptualLoss
from utils.hutils import is_main_process
from torch.autograd import Variable
from math import exp
import torch.nn.functional as F
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import sys
sys.path.append('../submodules')
from skimage import img_as_float32
from mmseg.apis import inference_model

# image_tensor: [C, H, W] 형식의 Torch Tensor
def resize_tensor(image_tensor, size=224):
    return transforms.functional.resize(image_tensor, [size, size], interpolation=transforms.InterpolationMode.BICUBIC)

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, 3)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window=window, window_size=self.window_size,
                     channel=self.channel, size_average=self.size_average)


class Loss(nn.Module):
    def __init__(self, 
                 conf, 
                 pipelines,
                 mask_weight, 
                 var_expression=None, 
                 lbs_weight=0,
                 sdf_consistency_weight=0, 
                 eikonal_weight=0, 
                 ssim_weight=0,
                 vgg_feature_weight=0, 
                 optimize_scene_latent_code=False, 
                 deform_pcd_weight=0, 
                 latent_kl_divergence_weight=0, 
                 normal_weight=0,
                #  clip_weight=0,
                 diffmorpher_weight=0):
        super().__init__()
        self.lora_finetuning = conf.get_bool('train.lora.lora_finetuning')
        self.mask_weight = mask_weight
        self.lbs_weight = lbs_weight
        self.sdf_consistency_weight = sdf_consistency_weight
        self.eikonal_weight = eikonal_weight
        self.ssim_weight = ssim_weight
        self.vgg_feature_weight = vgg_feature_weight
        self.var_expression = var_expression        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')                                        
        if self.var_expression is not None:
            self.var_expression = self.var_expression.unsqueeze(1).expand(1, 3, -1).reshape(1, -1).to(self.device)

        if is_main_process():
            print("Expression variance: ", self.var_expression)

        # if self.vgg_feature_weight > 0:
        #     self.get_vgg_loss = VGGPerceptualLoss().to(self.device)        

        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='none')
        self.ssim_loss = SSIM()

        self.binary_cross_entropy_loss = nn.BCEWithLogitsLoss(reduction='mean')

        self.optimize_scene_latent_code = optimize_scene_latent_code
        self.deform_pcd_weight = deform_pcd_weight
        self.latent_kl_divergence_weight = latent_kl_divergence_weight
        self.normal_weight = normal_weight
        # self.clip_weight = clip_weight
        self.diffmorpher_weight = diffmorpher_weight

        # if self.clip_weight > 0:
        #     self.clip = pipelines['clip']
        if self.diffmorpher_weight > 0:
            # self.modnet = pipelines['modnet']
            self.pipeline = pipelines['diffmorpher']   
            self.accumulation_steps = conf.get_int('train.accumulation_steps')

            # self.im_transform = transforms.Compose(
            #     [
            #         transforms.ToTensor(),
            #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #     ]
            # )
            
            self.sapiens = pipelines['sapiens']
            self.diffmorpher_target_attrs = conf.get_list('train.diffmorpher_target_attrs')

        self.get_vgg_loss = VGGPerceptualLoss().to(self.device)    

        self.category_dict = conf.get_config('dataset.category_dict')                              # NOTE 할 수 있을만한 건 전부다 넣었음.
        self.category_latent_dim = len(self.category_dict)
        self.source_category = conf.get_list('dataset.source_category')
        self.source_category_dict = {key: self.category_dict[key] for key in self.source_category}
        self.latent_code_std = conf.get_float('train.latent_code_std')
        self.segment = conf.get_config('dataset.segment')

    def get_rgb_loss(self, rgb_values, rgb_gt, weight=None):
        if weight is not None:
            image = rgb_values
            image_gt = rgb_gt
            ssim_loss = self.ssim_loss(image.permute(0, 3, 1, 2),
                                       image_gt.reshape(rgb_gt.shape[0], image.shape[1], image.shape[2], 3).permute(0, 3, 1, 2))
            Lssim = 1.0 - ssim_loss
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3) * weight.reshape(-1, 1), rgb_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            image = rgb_values
            image_gt = rgb_gt
            ssim_loss = self.ssim_loss(image.permute(0, 3, 1, 2),
                                       image_gt.reshape(rgb_gt.shape[0], image.shape[1], image.shape[2], 3).permute(0, 3, 1, 2))
            Lssim = 1.0 - ssim_loss
            rgb_loss = self.l1_loss(rgb_values.reshape(-1, 3), rgb_gt.reshape(-1, 3))
        return rgb_loss, Lssim

    def get_normal_loss(self, normal_values, normal_gt, weight=None):
        if weight is not None:
            normal_loss = self.l1_loss(normal_values.reshape(-1, 3) * weight.reshape(-1, 1), normal_gt.reshape(-1, 3) * weight.reshape(-1, 1))
        else:
            normal_loss = self.l1_loss(normal_values.reshape(-1, 3), normal_gt.reshape(-1, 3))
        return normal_loss

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, use_var_expression=False):
        # the same function is used for lbs, shapedirs, posedirs.
        if use_var_expression and self.var_expression is not None:
            lbs_loss = torch.mean(self.l2_loss(lbs_weight, gt_lbs_weight) / self.var_expression.to(lbs_weight.device) / 50)
        else:
            lbs_loss = self.l2_loss(lbs_weight, gt_lbs_weight).mean()
        return lbs_loss

    def embedding_to_kl_divergence(self, embeddings, target_std=0.25):
        # 임베딩 벡터들의 평균과 분산 계산
        mu = torch.mean(embeddings, dim=0)
        var = torch.var(embeddings, dim=0) + 1e-6  # 분산의 안정성을 위해 작은 값 추가

        # 목표 분포의 매개변수 설정
        target_mu = torch.zeros_like(mu)  # 목표 평균은 0
        target_var = (target_std ** 2) * torch.ones_like(var)  # 목표 분산은 (0.25)^2

        # KL Divergence 계산
        kl_loss = 0.5 * torch.sum(
            (var / target_var) + ((mu - target_mu) ** 2) / target_var - 1 - torch.log(var / target_var)
        )
        return kl_loss

    def get_kl_divergence_loss(self, latents_dict, scene_latent_code_dict, target_std=0.25):
        total_kl_loss = 0.0

        # grouped_items = latents_dict['grouped_items']
        # scene_latent_codes = latents_dict['scene_latent_codes']
        source_scene_latent_codes = latents_dict['source_scene_latent_codes']
        zero_latent_codes = latents_dict['zero_latent_codes']
        
        source_scene_latent_code_dict = {}
        for i, (prefix, idx) in enumerate(self.source_category_dict.items()):
            embeddings = source_scene_latent_codes(torch.tensor(i, device=self.device))
            source_scene_latent_code_dict[prefix] = embeddings

        for key in scene_latent_code_dict.keys():
            if key not in list(self.source_category_dict.keys()):
                embeddings = torch.cat([torch.cat(list(scene_latent_code_dict[key].values()), dim=0),
                                        zero_latent_codes(torch.tensor(0, device=self.device)).unsqueeze(0)], dim=0)
            else:
                embeddings = torch.cat([torch.cat(list(scene_latent_code_dict[key].values()), dim=0), 
                                        source_scene_latent_code_dict[key].unsqueeze(0), 
                                        zero_latent_codes(torch.tensor(0, device=self.device)).unsqueeze(0)], dim=0)
            total_kl_loss += self.embedding_to_kl_divergence(embeddings, target_std=target_std)

        return total_kl_loss

    def get_latent_regularization_loss(self, latent_values):
        return torch.mean(latent_values * latent_values)

    def get_mask_loss(self, predicted_mask, object_mask):
        mask_loss = self.l1_loss(predicted_mask.reshape(-1).float(), object_mask.reshape(-1).float())
        return mask_loss

    def get_gt_blendshape(self, index_batch, flame_lbs_weights, flame_posedirs, flame_shapedirs, ghostbone):
        if ghostbone:
            # gt_lbs_weight = torch.zeros(len(index_batch), 6).cuda()                           # NOTE original code
            gt_lbs_weight = torch.zeros(len(index_batch), 6, device=flame_lbs_weights.device)   # NOTE because of lightning
            gt_lbs_weight[:, 1:] = flame_lbs_weights[index_batch, :]
        else:
            gt_lbs_weight = flame_lbs_weights[index_batch, :]

        gt_shapedirs = flame_shapedirs[index_batch, :, 100:]
        gt_posedirs = torch.transpose(flame_posedirs.reshape(36, -1, 3), 0, 1)[index_batch, :, :]
        # gt_beta_shapedirs = flame_shapedirs[index_batch, :, :100]
        output = {
            'gt_lbs_weights': gt_lbs_weight,
            'gt_posedirs': gt_posedirs,
            'gt_shapedirs': gt_shapedirs,
            # 'gt_beta_shapedirs': gt_beta_shapedirs,
        }
        return output

    def get_sdf_consistency_loss(self, sdf_values):
        return torch.mean(sdf_values * sdf_values)

    def get_eikonal_loss(self, grad_theta):
        assert grad_theta.shape[1] == 3
        assert len(grad_theta.shape) == 2
        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_pcd_center_reg_loss(self, pcd_center):
        zero_coord = torch.zeros_like(pcd_center)
        pcd_center_reg_loss = self.l1_loss(pcd_center, zero_coord)
        return pcd_center_reg_loss

    def get_lora_3d_loss(self, model_input, ground_truth):
        loss = 0.0
        for key in model_input.keys():
            loss += self.l1_loss(model_input[key], ground_truth[key])
        return loss

    def cosine_similarity_loss(self, tensor1, tensor2):
        # Normalize the tensors
        tensor1_normalized = F.normalize(tensor1, p=2, dim=-1)
        tensor2_normalized = F.normalize(tensor2, p=2, dim=-1)

        # Compute cosine similarity
        cosine_similarity = torch.sum(tensor1_normalized * tensor2_normalized, dim=-1)

        # Compute the loss (1 - cosine similarity)
        loss = 1 - cosine_similarity.mean()
        return loss

    def cosine_similarity_matching_loss(self, cos_sim1, cos_sim2):
        # Compute the MSE loss between the two cosine similarity values
        loss = torch.mean((cos_sim1 - cos_sim2) ** 2)
        return loss
    
    def get_diffmorpher_pseudo_gt(self, ground_truth, model_input):
        gt_rgb_path_a, gt_rgb_path_b = ground_truth['image_paths'][0], ground_truth['image_paths_another_pair'][0]

        # img_orig_size = Image.open(gt_rgb_path_a).size
        lora_path_0 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(gt_rgb_path_a))), 'lora_5.ckpt')
        lora_path_1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(gt_rgb_path_b))), 'lora_5.ckpt')

        alpha_list = model_input['alpha_list']
        output_image = self.pipeline(img_path_0=gt_rgb_path_a, 
                                     img_path_1=gt_rgb_path_b, 
                                     load_lora_path_0=lora_path_0,
                                     load_lora_path_1=lora_path_1,
                                     use_adain=True,
                                     use_reschedule=True,
                                     num_frames=len(alpha_list),
                                     alpha_list=alpha_list,
                                     use_lora=True)

        # background matting
        # unify image channels to 3
        category = model_input['sub_dir'][0].split('_')[1]

        result_images, occlusion_masks = [], []
        for image in output_image:
            image, occlusion_mask = self.background_matting(image, category, model_input['bg_color'])
            result_images.append(image.detach().cpu())
            occlusion_masks.append(occlusion_mask.detach().cpu())
        
        return result_images, occlusion_masks     
        # Image.fromarray((result.reshape(512, 512, 3) * 255).cpu().numpy().astype(np.uint8)).save('back.png')

    def background_matting(self, image, category, bg_color):
        rgb = torch.tensor(img_as_float32(np.array(image))).permute(2, 0, 1).cuda().reshape(3, -1).transpose(1, 0).float()
        bg_color = bg_color.view(1, 3).expand_as(rgb)

        # generate attribute mask image by face parsing network.
        # im = np.array(image.resize((512, 512), Image.BILINEAR))
        im = np.array(image)
        result = inference_model(self.sapiens, im)
        pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy()

        num_segments = len(self.segment)
        h, w = pred_sem_seg.shape
        segments = np.zeros((h, w, num_segments))

        for idx, (key, value) in enumerate(self.segment.items()):
            if isinstance(value, list):  # If it's a list, concatenate the corresponding indices
                temp_arr = np.zeros((h, w))
                for v in value:
                    temp_arr += (pred_sem_seg == v)
                segments[:, :, idx] = (temp_arr >= 1)
            else:  # Single index case
                segments[:, :, idx] = (pred_sem_seg == value) >= 1
        
        background_mask = torch.tensor(segments[:, :, 0], device=rgb.device).reshape(-1, 1).float()
        result = rgb * (1 - background_mask) + bg_color * background_mask

        target_idx = list(self.segment.keys()).index(category)
        mask_target = torch.tensor(segments[:, :, target_idx], device=rgb.device).reshape(-1, 1).float()
        occlusion_mask = mask_target + background_mask

        return result, occlusion_mask

    def forward(self, 
                model_outputs, 
                ground_truth, 
                model_input=None):

        category = model_input['sub_dir'][0].split('_')[1]

        if self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs):
            if model_input['dataset_type'] == ['pivot_another_pair']:
                ground_truth['rgb'] = ground_truth['rgb_another_pair']

            if model_input['dataset_type'] == ['diffmorpher']:
                ground_truth['rgb'] = ground_truth['pseudo_gt_alphas'][model_input['idx_accumulation_steps']].unsqueeze(0).to(self.device)
                ground_truth['occlusion_mask'] = ground_truth['occlusion_masks'][model_input['idx_accumulation_steps']].squeeze(-1).unsqueeze(0).to(self.device)

                bz = model_outputs['batch_size']
                img_res = model_outputs['img_res']
                gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

                predicted = model_outputs['rgb_image'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

                vgg_loss = self.get_vgg_loss(predicted, gt)
        
        if (self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs) and (model_input['dataset_type'] == ['diffmorpher'])) or self.lora_finetuning:
            rgb_loss, ssim_loss = self.get_rgb_loss(model_outputs['rgb_image'] * ground_truth['occlusion_mask'].reshape(1, model_outputs['img_res'][0], model_outputs['img_res'][1], 1), 
                                                    ground_truth['rgb'] * ground_truth['occlusion_mask'].unsqueeze(-1))
        else:
            rgb_loss, ssim_loss = self.get_rgb_loss(model_outputs['rgb_image'], ground_truth['rgb'])

        # loss = 50 * rgb_loss + 0.25 * ssim_loss
        loss = rgb_loss + self.ssim_weight * ssim_loss

        out = {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'ssim_loss': ssim_loss * self.ssim_weight
        }

        # if self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs) and (model_input['dataset_type'] == ['diffmorpher']):
        #     out['loss'] += vgg_loss * self.vgg_feature_weight
        #     out['vgg_loss'] = vgg_loss * self.vgg_feature_weight

        if self.vgg_feature_weight > 0:
            if self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs) and (model_input['dataset_type'] == ['diffmorpher']):
                pass
            else:
                bz = model_outputs['batch_size']
                img_res = model_outputs['img_res']
                
                gt = ground_truth['rgb'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)
                predicted = model_outputs['rgb_image'].reshape(bz, img_res[0], img_res[1], 3).permute(0, 3, 1, 2)

                vgg_loss = self.get_vgg_loss(predicted, gt)

            out['loss'] += vgg_loss * self.vgg_feature_weight
            out['vgg_loss'] = vgg_loss * self.vgg_feature_weight
        
        if self.lora_finetuning and 'lora_index' in model_input and model_input['lora_index'] > 0 and model_input.get('lora_weight') > 0.5:       # lora disabled.
            loss_loras = self.get_lora_3d_loss(model_outputs['three_dim_values'], ground_truth['three_dim_values']) * 1.0
            out = {
                'loss': loss_loras,
                'lora_loss': loss_loras
            }

        if self.sdf_consistency_weight > 0:
            assert self.eikonal_weight > 0
            sdf_consistency_loss = self.get_sdf_consistency_loss(model_outputs['sdf_values'])
            eikonal_loss = self.get_eikonal_loss(model_outputs['grad_thetas'])
            out['loss'] += sdf_consistency_loss * self.sdf_consistency_weight + eikonal_loss * self.eikonal_weight
            out['sdf_consistency'] = sdf_consistency_loss * self.sdf_consistency_weight
            out['eikonal'] = eikonal_loss * self.eikonal_weight

        if self.lbs_weight != 0:
            num_points = model_outputs['lbs_weights'].shape[0]
            ghostbone = model_outputs['lbs_weights'].shape[-1] == 6
            outputs = self.get_gt_blendshape(model_outputs['index_batch'], model_outputs['flame_lbs_weights'],
                                             model_outputs['flame_posedirs'], model_outputs['flame_shapedirs'],
                                             ghostbone)

            lbs_loss = self.get_lbs_loss(model_outputs['lbs_weights'].reshape(num_points, -1),
                                             outputs['gt_lbs_weights'].reshape(num_points, -1),
                                             )

            out['loss'] += lbs_loss * self.lbs_weight * 0.1
            out['lbs_loss'] = lbs_loss * self.lbs_weight * 0.1

            gt_posedirs = outputs['gt_posedirs'].reshape(num_points, -1)
            posedirs_loss = self.get_lbs_loss(model_outputs['posedirs'].reshape(num_points, -1) * 10,
                                              gt_posedirs* 10,
                                              )
            out['loss'] += posedirs_loss * self.lbs_weight * 10.0
            out['posedirs_loss'] = posedirs_loss * self.lbs_weight * 10.0

            gt_shapedirs = outputs['gt_shapedirs'].reshape(num_points, -1)
            shapedirs_loss = self.get_lbs_loss(model_outputs['shapedirs'].reshape(num_points, -1)[:, :50*3] * 10,
                                               gt_shapedirs * 10,
                                               use_var_expression=True,
                                               )
            out['loss'] += shapedirs_loss * self.lbs_weight * 10.0
            out['shapedirs_loss'] = shapedirs_loss * self.lbs_weight * 10.0
        
        if self.optimize_scene_latent_code and self.latent_kl_divergence_weight > 0:
            if category != 'source':
                latent_regularization_loss = self.get_latent_regularization_loss(model_outputs['scene_latent_code_sample'])
                out['loss'] += latent_regularization_loss * self.latent_kl_divergence_weight
                out['latent_regularization_loss'] = latent_regularization_loss * self.latent_kl_divergence_weight

            # if model_input['data_index'] == 0 and not self.lora_finetuning and not (self.diffmorpher_weight > 0):
            #     latent_kl_divergence_loss = self.get_kl_divergence_loss(model_input, model_outputs['scene_latent_code_dict'], target_std=self.latent_code_std)
            #     out['latent_kl_divergence_loss'] = latent_kl_divergence_loss
            #     out['loss'] += latent_kl_divergence_loss * self.latent_kl_divergence_weight
            
        
        if self.normal_weight > 0:          # NOTE 
            normal_output = model_outputs['normal_image'].squeeze().reshape(-1, 3)
            # object_mask_gt = ground_truth["object_mask"].reshape(-1, 1).float()
            normal_gt = ground_truth['normal'].squeeze().reshape(-1, 3)

            # normal_rendering_output = normal_image_output * object_mask_gt + (1 - object_mask_gt)
            # normal_gt = normal_image_gt * object_mask_gt + (1 - object_mask_gt)

            normal_loss = self.get_normal_loss(normal_output, normal_gt)
            out['loss'] += normal_loss * self.normal_weight
            out['normal_loss'] = normal_loss

        return out
    

