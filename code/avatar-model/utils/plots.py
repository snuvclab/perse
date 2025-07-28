import numpy as np
import torch
import torchvision
import trimesh
from PIL import Image
import os
import cv2
import wandb
from einops import rearrange
import warnings
warnings.filterwarnings("ignore")
from utils.hutils import is_dist_avail_and_initialized, save_on_master, get_rank, is_main_process

SAVE_OBJ_LIST = [1]

def save_pcl_to_ply(filename, points, colors=None, normals=None):
    save_dir=os.path.dirname(os.path.abspath(filename))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if colors is not None:
        colors = colors.cpu().detach().numpy()
    if normals is not None:
        normals = normals.cpu().detach().numpy()
    mesh = trimesh.Trimesh(vertices=points.detach().cpu().numpy(),vertex_normals = normals, vertex_colors = colors)
    #there is a bug in trimesh of it only saving normals when we tell the exporter explicitly to do so for point clouds.
    #thus we are calling the exporter directly instead of mesh.export(...)
    f = open(filename, "wb")
    data = trimesh.exchange.ply.export_ply(mesh, vertex_normal=True)
    f.write(data)
    f.close()
    return


def plot(img_index, model_outputs, ground_truth, path, epoch, img_res, is_eval=False, first=False, custom_settings=None):
    # arrange data to plot
    batch_size = model_outputs['batch_size']
    plot_images(model_outputs, ground_truth, path, epoch, img_index, 1, img_res, batch_size, is_eval, custom_settings)

    # canonical_color = torch.clamp(model_outputs['pnts_albedo'], 0., 1.)
    if not is_eval:
        return
    if custom_settings is not None and 'novel_view' not in custom_settings: # hyunsoo added
        canonical_color = torch.clamp(model_outputs['pnts_albedo'], 0., 1.)
        for idx, img_idx in enumerate(img_index):
            # wo_epoch_path = path[idx].replace('/epoch_{}'.format(epoch), '')
            wo_epoch_path = os.path.dirname(path[idx])
            if img_idx in SAVE_OBJ_LIST:
                deformed_color = model_outputs["pnts_color_deformed"].reshape(batch_size, -1, 3)[idx]
                filename = '{0}/{1:04d}_deformed_color_{2}.ply'.format(wo_epoch_path, epoch, img_idx)
                save_pcl_to_ply(filename, model_outputs['deformed_points'].reshape(batch_size, -1, 3)[idx],
                                normals=model_outputs["pnts_normal_deformed"].reshape(batch_size, -1, 3)[idx],
                                colors=deformed_color)

                filename = '{0}/{1:04d}_deformed_albedo_{2}.ply'.format(wo_epoch_path, epoch, img_idx)
                save_pcl_to_ply(filename, model_outputs['deformed_points'].reshape(batch_size, -1, 3)[idx],
                                normals=model_outputs["pnts_normal_deformed"].reshape(batch_size, -1, 3)[idx],
                                colors=canonical_color)
        if first:
            # wo_epoch_path = path[0].replace('/epoch_{}'.format(epoch), '')
            wo_epoch_path = os.path.dirname(path[0])
            filename = '{0}/{1:04d}_canonical_points_albedo.ply'.format(wo_epoch_path, epoch)
            save_pcl_to_ply(filename, model_outputs["canonical_points"], colors=canonical_color)

            if 'unconstrained_canonical_points' in model_outputs:
                filename = '{0}/{1:04d}_unconstrained_canonical_points.ply'.format(wo_epoch_path, epoch)
                save_pcl_to_ply(filename, model_outputs['unconstrained_canonical_points'],
                                colors=canonical_color)
        if epoch == 0 or is_eval:
            if first:
                # wo_epoch_path = path[0].replace('/epoch_{}'.format(epoch), '')
                wo_epoch_path = os.path.dirname(path[0])
                filename = '{0}/{1:04d}_canonical_verts.ply'.format(wo_epoch_path, epoch)
                save_pcl_to_ply(filename, model_outputs['canonical_verts'].reshape(-1, 3),
                                colors=get_lbs_color(model_outputs['flame_lbs_weights']))

def is_integer_string(s):
    try:
        # Python 3.6 이상에서는 int()가 언더스코어(_)를 허용합니다.
        int(s)
        return True
    except ValueError:
        return False

def plot_image(rgb, path, img_index, plot_nrow, img_res, type, fill=False):
    rgb_plot = lin2img(rgb, img_res)

    tensor = torchvision.utils.make_grid(rgb_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = np.clip(tensor, 0., 1.)
    tensor = (tensor * scale_factor).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    # img = Image.fromarray(tensor)
    # os.makedirs('{0}/{1}'.format(path, type), exist_ok=True)
    # img.save('{0}/{2}/{1}.jpg'.format(path, img_index, type))

    # img_index to %4d
    if is_integer_string(img_index):
        img_index = '{:04d}'.format(img_index)
    
    if fill:
        tensor = cv2.erode(tensor, kernel, iterations=1)            # NOTE 흰색 노이즈를 제거
        tensor = cv2.dilate(tensor, kernel, iterations=1)           # NOTE 구멍이나 간격을 메움.

        img = Image.fromarray(tensor)
        os.makedirs('{0}/{1}_erode_dilate'.format(path, type), exist_ok=True)
        img.save('{0}/{2}_erode_dilate/{1}.jpg'.format(path, img_index, type))
    else:
        
        img = Image.fromarray(tensor)
        os.makedirs('{0}/{1}'.format(path, type), exist_ok=True)
        img.save('{0}/{2}/{1}.jpg'.format(path, img_index, type))


def plot_mask(mask_tensor, path, img_index, plot_nrow, img_res, type):
    # Reshape the tensor into a square 2D tensor
    mask_tensor = mask_tensor.cpu()
    side_length = int(torch.sqrt(torch.tensor(mask_tensor.numel())).item())
    mask_2d = mask_tensor.reshape(side_length, side_length)

    # Convert the 2D tensor to a NumPy array
    mask_np = mask_2d.numpy().astype(np.uint8) * 255

    mask_np = cv2.resize(mask_np, img_res)

    img = Image.fromarray(mask_np)
    if not os.path.exists('{0}/{1}'.format(path, type)):
        os.mkdir('{0}/{1}'.format(path, type))
    img.save('{0}/{2}/{1}.png'.format(path, img_index, type))

def get_lbs_color(lbs_points):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap('Paired')
    red = cmap.colors[5]
    cyan = cmap.colors[3]
    blue = cmap.colors[1]
    pink = [1, 1, 1]

    if lbs_points.shape[-1] == 5:
        colors = torch.from_numpy(
            np.stack([np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[
                None]).cuda()
    else:
        colors = torch.from_numpy(
            np.stack([np.array(red), np.array(cyan), np.array(blue), np.array(pink), np.array(pink), np.array(pink)])[
                None]).cuda()
    lbs_points = (colors * lbs_points[:, :, None]).sum(1)
    return lbs_points


def plot_images(model_outputs, ground_truth, path, epoch, img_index, plot_nrow, img_res, batch_size, is_eval, custom_settings):
    num_samples = img_res[0] * img_res[1]

    device = ground_truth['rgb'].device
    wandb_image_num = 0
    if 'rgb' in ground_truth:
        wandb_image_num += 1
        rgb_gt = ground_truth['rgb']
        if 'rendered_landmarks' in model_outputs:
            rendered_landmarks = model_outputs['rendered_landmarks'].reshape(batch_size, num_samples, 3)
            rgb_gt = rgb_gt * (1 - rendered_landmarks) + rendered_landmarks * torch.tensor([1, 0, 0]).to(device) # .cuda()
    else:
        rgb_gt = None
    rgb_points = model_outputs['rgb_image']
    if rgb_points.shape[0] > batch_size:     # in fact, batch_size should be one.
        batch_size_rgb = rgb_points.shape[0]
    else:
        batch_size_rgb = batch_size
    # rgb_points = rgb_points.reshape(batch_size_rgb, num_samples, 3)
    rgb_points = rgb_points.reshape(batch_size_rgb, -1, 3)

    if 'rendered_landmarks' in model_outputs:
        rendered_landmarks = model_outputs['rendered_landmarks'].reshape(batch_size, num_samples, 3)
        rgb_points_rendering = rgb_points * (1 - rendered_landmarks) + rendered_landmarks * torch.tensor([1, 0, 0]).to(device) # .cuda()
        output_vs_gt = rgb_points_rendering
    else:
        output_vs_gt = rgb_points

    if 'normal_image' in model_outputs:
        wandb_image_num += 1
        normal_points = model_outputs['normal_image']
        normal_points = normal_points.reshape(batch_size, num_samples, 3)       # NOTE result: (1, 262144, 3)
        output_vs_gt = torch.cat((output_vs_gt, normal_points), dim=0)

    # if rgb_gt is not None:
    #     wandb_image_num += 2
    #     output_vs_gt = torch.cat((output_vs_gt, rgb_gt, normal_points), dim=0)
    # else:
    #     output_vs_gt = torch.cat((output_vs_gt, normal_points), dim=0)

    if 'shading_image' in model_outputs:
        wandb_image_num += 1
        output_vs_gt = torch.cat((output_vs_gt, model_outputs['shading_image'].reshape(batch_size, num_samples, 3)), dim=0)
    
    if 'albedo_image' in model_outputs:
        wandb_image_num += 1
        output_vs_gt = torch.cat((output_vs_gt, model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)), dim=0)

    # if 'segment_image' in model_outputs:
    #     segment_image = model_outputs['segment_image']
    #     segment_image = plot_segment_images(rgb_gt, segment_image).reshape(batch_size, num_samples, 3)
    #     output_vs_gt = torch.cat((output_vs_gt, segment_image), dim=0)

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)
    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=batch_size).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)           # 0 to 1 normalized color scale to 255
    
    if custom_settings is not None and 'wandb_logger' in custom_settings:
        wandb_logger = custom_settings['wandb_logger'] # hyunsoo added

        wandb_tensor = torchvision.utils.make_grid(output_vs_gt_plot[:wandb_image_num, ...],
                                                   scale_each=False,
                                                   normalize=False,
                                                   nrow=output_vs_gt.shape[0]).cpu().detach().numpy()
        scale_factor = 255
        wandb_tensor = (wandb_tensor * scale_factor).astype(np.uint8) # (516, 3600, 3)

        wandb_image = rearrange(wandb_tensor, 'c h w -> h w c')
        wandb_logger.experiment.log({"Eval":[wandb.Image((wandb_image).astype(np.uint8))], "global_step": custom_settings['global_step']})

    # hyunsoo added
    if custom_settings is None or 'novel_view' not in custom_settings:
        novel_view = ''
    else:
        novel_view = '_{}'.format(custom_settings['novel_view'])
    
    if custom_settings is not None and 'step' in custom_settings:
        idx = 'step_{0}_img_'.format(custom_settings['step'])
    else:
        idx = ''

    try:
        idx += str(img_index[0])
    except:
        idx += str(img_index)
    

    if 'render_concat_gt' in custom_settings and custom_settings['render_concat_gt']:
        img = Image.fromarray(tensor)           
        wo_epoch_path = os.path.dirname(path[0])
        os.makedirs('{0}/rendering{1}'.format(wo_epoch_path, novel_view), exist_ok=True)
        img.save('{0}/rendering{3}/epoch_{1}_{2}.png'.format(wo_epoch_path, epoch, idx, novel_view))

    if is_eval:
        if img_index.ndim > 0:
            for i, idx in enumerate(img_index):
                plot_image(rgb_points[[i]], path[i], idx, plot_nrow, img_res, 'rgb')
                if 'normal_image' in model_outputs:
                    plot_image(normal_points[[i]], path[i], idx, plot_nrow, img_res, 'normal', fill=True)
                if 'albedo_image' in model_outputs:
                    plot_image(model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'albedo', fill=True)
                if 'shading_image' in model_outputs:
                    plot_image(model_outputs['shading_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'shading', fill=True)
                # if 'segment_image' in model_outputs:
                #     plot_mask(model_outputs['segment_image'] > 0.5, path[i], idx, plot_nrow, img_res, 'segment')
                # if 'mask_hole' in model_outputs:
                #     plot_mask(model_outputs['mask_hole'] > 0.5, path[i], idx, plot_nrow, img_res, 'mask_hole')
        else:
            i = 0
            idx = img_index.item()
            plot_image(rgb_points[[i]], path[i], idx, plot_nrow, img_res, 'rgb')
            if 'normal_image' in model_outputs:
                plot_image(normal_points[[i]], path[i], idx, plot_nrow, img_res, 'normal', fill=True)
            if 'albedo_image' in model_outputs:
                plot_image(model_outputs['albedo_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'albedo', fill=True)
            if 'shading_image' in model_outputs:
                plot_image(model_outputs['shading_image'].reshape(batch_size, num_samples, 3)[[i]], path[i], idx, plot_nrow, img_res, 'shading', fill=True)
            # if 'segment_image' in model_outputs:
            #     plot_segment_images(rgb_points[[i]], model_outputs['segment_image'], path[i], idx, plot_nrow, img_res, True)
            # if 'mask_hole' in model_outputs:
            #     plot_mask(model_outputs['mask_hole'] > 0.5, path[i], idx, plot_nrow, img_res, 'mask_hole')
    del output_vs_gt


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    if img_res[0] * img_res[1] != num_samples:
        img_res = (int(np.sqrt(num_samples)), int(np.sqrt(num_samples)))
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])       # batch, 512*512, 3 -> batch, 3, 512, 512


def plot_segment_images(rgb_gt, segment_image, path=None, idx=0, plot_nrow=None, img_res=512, save_im=False):
    # Colors for all 20 parts
    part_colors_bgr = [[255, 0, 0],     # 0
                       [255, 85, 0],    # 1 skin
                       [255, 170, 0],   # 2 eyebrows
                       [255, 0, 170],   # 3 eyes
                       [85, 255, 0],    # 4 eyeglasses
                       [170, 255, 0],   # 5 ears
                       [0, 255, 170],   # 6 earrings
                       [0, 0, 255],     # 7 nose
                       [85, 0, 255],    # 8 mouth
                       [0, 170, 255],   # 9 neck
                       [255, 255, 0],   # 10 necklace
                       [255, 255, 85],  # 11 cloth
                       [255, 255, 170], # 12 hair
                       [255, 0, 255]]   # 13 hat
    part_colors = [[r, g, b] for b, g, r in part_colors_bgr]

    device = rgb_gt.device
    segment_label = torch.argmax(torch.softmax(segment_image, dim=-1), dim=-1).squeeze(0)

    batch_size, image_dim1, image_dim2, num_of_class = segment_image.shape
    segment_concat_image = torch.zeros_like(rgb_gt).reshape(batch_size, image_dim1, image_dim2, rgb_gt.shape[-1])

    # 0 represents the background.
    for pi in range(1, num_of_class + 1):
        index = torch.where(segment_label == pi)
        if len(index[0]) > 0:
            segment_concat_image[:, index[0], index[1], :] = torch.tensor(part_colors[pi], device=device).float() / 255.
    
    if not save_im:
        return segment_concat_image
    else:
        plot_image(segment_concat_image.reshape(batch_size, -1, rgb_gt.shape[-1]), path, idx, plot_nrow, img_res, 'segment')