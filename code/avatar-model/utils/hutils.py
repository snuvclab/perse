import cv2
import glob
import os
from tqdm import tqdm
import argparse
import numpy as np
import torch
from kornia.geometry.conversions import (
    camtoworld_to_worldtocam_Rt, 
    worldtocam_to_camtoworld_Rt, 
    Rt_to_matrix4x4, 
    matrix4x4_to_Rt
)
from utils.rotation_converter import *
import warnings
warnings.filterwarnings("ignore")
import torch.distributed as dist

#-------------for distributed training--------------
#----------borrowed from https://github.com/The-AI-Summer/pytorch-ddp.git -------------
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def createDirectory(directory): 
    try: 
        if not os.path.exists(directory): 
            os.makedirs(directory) 
    except OSError: 
        print("Error: Failed to create the directory.")

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def visual_concat(name_left, name_right, left, right, concat_result, resize_type):
    import natsort
    root_left, extension_left = os.path.splitext(os.listdir(left)[0])
    root_right, extension_right = os.path.splitext(os.listdir(right)[0])

    left_file_sort = natsort.natsorted(glob.glob(os.path.join(left, "*"+extension_left)), reverse=False) # glob.glob(os.path.join(left, "*.png")).sort(key=lambda f: int(filter(str.isdigit, f)))
    right_file_sort = natsort.natsorted(glob.glob(os.path.join(right, "*"+extension_right)), reverse=False)
    
    img_source = [cv2.imread(image) for image in tqdm(left_file_sort, desc="Source Images Loading...")]
    img_target = [cv2.imread(image) for image in tqdm(right_file_sort, desc="Target Images Loading...")]

    if len(img_source) != len(img_target):
        if len(img_source) > len(img_target):
            img_source = img_source[:len(img_target)]
        else:
            img_target = img_target[:len(img_source)]

    assert len(img_source) == len(img_target), f"The number of images is different. Source images got: {len(img_source)}, target images got: {len(img_target)}"

    for i in tqdm(range(len(img_source)), desc="Saving Image..."):
        left = cv2.putText(img=img_source[i], text=name_left, org=(10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=(0,0,0), thickness=1)
        right = cv2.putText(img=img_target[i], text=name_right, org=(10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=(0,0,0), thickness=1)
        if resize_type == 'h':
            resizer = hconcat_resize_min([left, right]) # 가로
        else:
            resizer = vconcat_resize_min([left, right]) # 세로
        # cv2.putText(img=resizer, text='Left: '+name_left+', Right: '+name_right, org=(10, 30), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.7, color=(0,0,0), thickness=1)
        cv2.imwrite(os.path.join(concat_result, 'iter_{0:04d}.png'.format(i)), resizer)

def make_grid(photo_dirs):
    photos = [cv2.imread(image) for image in tqdm(sorted(glob.glob(os.path.join(photo_dirs, "*.png"))), desc="Source Images Loading...")]
    temp_dir = os.path.join(photo_dirs, 'temp')
    createDirectory(temp_dir)
    for i in range(3):
        resizer = hconcat_resize_min(photos[i*3:(i+1)*3])
        cv2.imwrite(os.path.join(temp_dir, 'concat_{0:04d}.png'.format(i)), resizer)
    
    photos_concat = [cv2.imread(image) for image in tqdm(sorted(glob.glob(os.path.join(temp_dir, "*.png"))), desc="Source Images Loading...")]
    resizer = vconcat_resize_min(photos_concat)
    cv2.imwrite(os.path.join(photo_dirs, 'grid_photos.png'.format(i)), resizer)
    os.system('rm -rf ' + temp_dir)

def do_ffmpeg(video_path):
    file_name = video_path.split('/')[-1].split('.')[0]
    print("[INFO] {} Converting video to images...".format(file_name))
    save_dir = os.path.join('/'.join(video_path.split('/')[:-1]), 'images_'+file_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    os.system("ffmpeg -i " + video_path + " " + os.path.join(save_dir, file_name + "_%04d.png"))
    print('[INFO] ffmpeg done')

def main_example(args):
    # 
    if True:
        name_lst = [
            'original', 
            'rendering', 
            'depth',
            'reprojection'
        ]
        directory_lst = [
            os.path.join('dataset', 'nerface', 'person_1_ours', 'test'), 
            os.path.join('trial', 'ours_person_1_20220816_015108', 'validation'), 
            os.path.join('trial', 'ours_person_1_20220816_015108', 'depth'),
            os.path.join('dataset', 'nerface', 'person_1_ours', 'visualization', 'train', 'optim_images')
        ]
        save_dir = os.path.join('result_videos')

        makevideo(args.frame_rate, name_lst, directory_lst, save_dir)
        print("Finished making video!")
    else:
        photo_dirs = os.path.join('trial', 'ours_yonwoo_otheroptim', 'validation')
        make_grid(photo_dirs)
    


def depth2normal(depthmap):
    if True:
        h, w = np.shape(depthmap)
        normals = np.zeros((h, w, 3))
        phong = np.zeros((h, w, 3))
        cx = cy = h//2
        fx=fy=500
        fx = fy = 1150
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                #dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
                #dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

                p = np.array([(x*depthmap[x,y]-cx)/fx, (y*depthmap[x,y]-cy)/fy, depthmap[x,y]])
                py = np.array([(x*depthmap[x,y+1]-cx)/fx, ((y+1)*depthmap[x,y+1]-cy)/fy, depthmap[x,y+1]])
                px = np.array([((x+1)*depthmap[x+1,y]-cx)/fx, (y*depthmap[x+1,y]-cy)/fy, depthmap[x+1,y]])

                #n = np.array([-dzdx, -dzdy, 0.005])
                n = np.cross(px-p, py-p)
                n = n * 1/np.linalg.norm(n)
                dir = p # np.array([x,y,1.0])
                dir = dir *1/np.linalg.norm(dir)

                normals[x, y] = (n*0.5 + 0.5)
                phong[x, y] = np.dot(dir,n)*0.5+0.5

        normals *= 255
        normals = normals.astype('uint8')

        phong *= 255
        phong = phong.astype('uint8')
    else:
        h, w = np.shape(depthmap)
        normals = np.zeros((h, w, 3))
        phong = np.zeros((h, w, 3))
        for x in range(1, h - 1):
            for y in range(1, w - 1):
                dzdx = (float((depthmap[x + 1, y])) - float((depthmap[x - 1, y]))) / 2.0
                dzdy = (float((depthmap[x, y + 1])) - float((depthmap[x, y - 1]))) / 2.0

                n = np.array([-dzdx, -dzdy, 0.005])

                n = n * 1/np.linalg.norm(n)
                dir = np.array([x,y,1.0])
                dir = dir *1/np.linalg.norm(dir)

                normals[x, y] = (n*0.5 + 0.5)
                phong[x, y] = np.dot(dir,n)*0.5+0.5

        normals *= 255
        normals = normals.astype('uint8')
    # plt.imshow(depthmap, cmap='gray')
    # plt.imsave('depth.png', depthmap, cmap='gray')
    # plt.show()
    # plt.imshow(normals)
    # plt.imsave('normal.png', normals)
    # plt.show()
    # plt.imshow(phong)
    # plt.imsave('phong.png', phong)
    # plt.show()
    # print('a')
    return normals, phong

def visualize_mask_to_file(mask_tensor: torch.Tensor, output_file: str):
    import numpy as np
    import cv2
    # Reshape the tensor into a square 2D tensor
    side_length = int(torch.sqrt(torch.tensor(mask_tensor.numel())).item())
    mask_2d = mask_tensor.reshape(side_length, side_length)

    # Convert the 2D tensor to a NumPy array
    mask_np = mask_2d.numpy().astype(np.uint8) * 255

    # Visualize the mask and save it to a file
    cv2.imwrite(output_file, mask_np)

def visualize_rgb_to_file(rgb_tensor: torch.Tensor, output_file: str):
    import numpy as np
    import cv2
    # Reshape the tensor into a square 3D tensor with shape (height, width, 3)
    side_length = int(torch.sqrt(torch.tensor(rgb_tensor.shape[0])).item())
    rgb_3d = rgb_tensor.reshape(side_length, side_length, 3)

    # Convert the 3D tensor to a NumPy array
    rgb_np = (rgb_3d.numpy() * 255).astype(np.uint8)

    # Visualize the RGB image and save it to a file
    rgb_np = cv2.cvtColor(rgb_np, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_file, rgb_np)

if __name__ == "__main__":
    # 원본비교 코드
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_rate", 
                        type=int,
                        default='30'
                        )
    args = parser.parse_args()
    main_example(args)

def novel_view(cam_pose, euler_angle, translation):
    x_euler, y_euler, z_euler = euler_angle

    ## for novel view synthesis
    cam_R = cam_pose[:, :3, :3]
    cam_t = cam_pose[:, :3, -1]
    cam_t = cam_t.reshape(-1, 1).unsqueeze(0)
    world_R, world_t = cam_R, cam_t

    quat = torch.stack(quaternion_from_euler(x_euler, y_euler, z_euler)).to(world_R.device)
    
    rotate_world_R = quaternion_to_rotation_matrix(rotation_matrix_to_quaternion(world_R)+(quat).unsqueeze(0))
    trans_world_t = world_t + translation.to(cam_t.device).reshape(-1, 1).unsqueeze(0)

    rotate_cam_R, rotate_cam_t = rotate_world_R, trans_world_t
    cam_pose = Rt_to_matrix4x4(rotate_cam_R, rotate_cam_t)
    return cam_pose