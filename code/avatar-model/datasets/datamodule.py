import os, re
import torch
import numpy as np
import cv2
import json
import imageio
import skimage
from tqdm import tqdm
from utils.hutils import novel_view
import copy
from utils.hutils import is_main_process
import torch.distributed as dist
from model.cameras import MiniCam, getProjectionMatrix, getWorld2View2, focal2fov
import random
import csv
import pandas as pd
from natsort import natsorted

def find_index_closest_to_target(lst, target):
    # Since the list is sorted and increasing, we can use binary search
    low, high = 0, len(lst) - 1
    closest_index = -1

    while low <= high:
        mid = (low + high) // 2

        # Check if the mid element is less than or equal to the target
        if lst[mid] <= target:
            closest_index = mid
            low = mid + 1
        else:
            high = mid - 1

    return closest_index

# Updated function to find the index based on both name and number
def find_closest_index_by_name_and_number(file_paths, target_name, target_number):
    # Extracting the relevant parts from the file paths
    extracted_info = [(path.split('/')[5], int(path.split('/')[-1].split('.')[0])) for path in file_paths]

    # Finding the closest index
    closest_index = None
    for i, (name, num) in enumerate(extracted_info):
        if name == target_name and str(num) in target_number:
            return i
        if name == target_name and num < int(target_number.split('.')[0]):
            closest_index = i

    return closest_index

def find_indices_of_substring(lst, substring):
    start_index = end_index = -1

    for i, s in enumerate(lst):
        if substring in s:
            # If start_index is not set yet, set it to the current index
            if start_index == -1:
                start_index = i
            # Update end_index to the current index
            end_index = i

    return start_index, end_index

def is_valid(video_dir):
    image_dir = 'image'   
    image_path = os.path.join(video_dir, image_dir)
    if not os.path.isdir(image_path):            # image
        return False
    video_length = len(list(os.scandir(image_path)))

    mask_dir = 'mask'
    mask_path = os.path.join(video_dir, mask_dir)
    if not os.path.isdir(mask_path):             # mask
        return False
    # video_mask_length = len(list(os.scandir(mask_path))) // 2
    video_mask_length = len([f for f in os.scandir(mask_path) if f.name.endswith('.png')])
    if video_length != video_mask_length:
        return False

    mask_dir = 'segment'
    mask_path = os.path.join(video_dir, mask_dir)
    if not os.path.isdir(mask_path):             # mask
        return False
    video_mask_length = len(list(os.scandir(mask_path)))
    if video_length != video_mask_length:
        return False
    
    return True
# def is_valid(video_dir):
#     image_dir = 'image'   
#     image_path = os.path.join(video_dir, image_dir)
#     if not os.path.isdir(image_path):            # image
#         return False
#     video_length = len(list(os.scandir(image_path)))

#     # mask_dir = 'mask'
#     # mask_path = os.path.join(video_dir, mask_dir)
#     # if not os.path.isdir(mask_path):             # mask
#     #     return False
#     # # video_mask_length = len(list(os.scandir(mask_path))) // 2
#     # video_mask_length = len([f for f in os.scandir(mask_path) if f.name.endswith('.png')])
#     # if video_length != video_mask_length:
#     #     return False

#     # mask_dir = 'segment'
#     # mask_path = os.path.join(video_dir, mask_dir)
#     # if not os.path.isdir(mask_path):             # mask
#     #     return False
#     # video_mask_length = len(list(os.scandir(mask_path)))
#     # if video_length != video_mask_length:
#     #     return False
    
#     return True

def is_valid_pegasus(video_dir):
    image_dir = 'image'   
    image_path = os.path.join(video_dir, image_dir)
    if not os.path.isdir(image_path):            # image
        return False
    video_length = len(list(os.scandir(image_path)))

    mask_dir = 'mask'
    mask_path = os.path.join(video_dir, mask_dir)
    if not os.path.isdir(mask_path):             # mask
        return False
    # video_mask_length = len(list(os.scandir(mask_path))) // 2
    video_mask_length = len([f for f in os.scandir(mask_path) if f.name.endswith('.png')])
    if video_length != video_mask_length:
        return False

    mask_dir = 'segment'
    mask_path = os.path.join(video_dir, mask_dir)
    if not os.path.isdir(mask_path):             # mask
        return False
    video_mask_length = len(list(os.scandir(mask_path)))
    if video_length != video_mask_length:
        return False

    # mask_dir = 'normal'
    # mask_path = os.path.join(video_dir, mask_dir)
    # if not os.path.isdir(mask_path):             # mask
    #     return False
    # video_mask_length = len(list(os.scandir(mask_path)))
    # if video_length != video_mask_length:
    #     return False
    
    return True

# CSV 파일을 읽고 데이터를 처리하는 함수
def process_csv(file_path):
    processed_data = []
    
    # CSV 파일 열기
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        # 첫 번째 행은 헤더이므로 건너뜀
        next(reader)
        
        # 각 행을 처리
        for row in reader:
            sample_a = row[0]
            sample_b = row[1]
            dir_name = row[2]
            # ratio = float(row[3])  # ratio를 실수로 변환
            
            # 예시로 sample_a와 sample_b에 대한 추가 작업 수행
            # 여기서는 간단히 각 값들을 출력하는 예시를 보여줌
            # print(f"Processing: {sample_a} to {sample_b}, dir_name: {dir_name}, ratio: {ratio}")
            
            # 필요한 처리 후 데이터 저장 (예: 특정 조건을 만족하는 데이터만 필터링)
            processed_data.append({
                "sample_a": sample_a,
                "sample_b": sample_b,
                "dir_name": dir_name,
            })
    return processed_data

# def extract_middle(s):
#     pattern = r'^\d+_(.*?)_youngman_hyunsoo_img_\d+$'
#     match = re.match(pattern, s)
#     if match:
#         return match.group(1)
#     else:
#         return None  # 매칭되지 않는 경우
    
def extract_middle(s, pattern_type):
    # 'guy', 'girl', 'youngman' 등 어떤 단어가 와도 매칭되도록 정규식 수정
    # pattern = r'^\d+_(.*?)_(?:guy|girl|youngman)_hyunsoo_img_\d+$'
    # pattern_type을 동적으로 포함한 정규식
    pattern = r'^\d+_(.*?)_' + pattern_type + r'$'
    match = re.match(pattern, s)
    if match:
        return match.group(1)  # 중간 부분 반환
    else:
        return None  # 매칭되지 않는 경우

def read_interp_pairs(pair_files):
    with open(pair_files, 'r') as f:
        lines = f.readlines()
        pairs = [line.strip().split(',') for line in lines]
    pair_dict = dict()
    for pair in pairs:
        if pair[2] in pair_dict:
            pair_dict[pair[2]].append((pair[0], pair[1]))
        else:
            pair_dict[pair[2]] = [(pair[0], pair[1])]

    return pair_dict
    

def load_rgb(path, img_res=None):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    if img_res is not None:
        img = cv2.resize(img, (int(img_res[0]), int(img_res[1])))
    img = img.transpose(2, 0, 1)
    return img


def load_mask(path, img_res=None):
    alpha = imageio.imread(path, as_gray=True)
    alpha = skimage.img_as_float32(alpha)

    if img_res is not None:
        alpha = cv2.resize(alpha, (int(img_res[0]), int(img_res[1])))
    object_mask = alpha / 255

    return object_mask

# # Define the classes and palette
# palette = [[0, 0, 0],
#            [128, 200, 255], 
#            [255, 200, 150], 
#            [0, 255, 127], 
#            [255, 99, 71], 
#            [30, 144, 255], 
#            [255, 140, 0], 
#            [238, 130, 238], 
#            [255, 215, 0], 
#            [82, 21, 114], 
#            [115, 227, 112], 
#            [235, 205, 119], 
#            [255, 182, 193], 
#            [255, 0, 109], 
#            [169, 169, 169], 
#            [0, 128, 72], 
#            [0, 70, 130], 
#            [255, 215, 0], 
#            [75, 0, 130]]

# classes = ['BACKGROUND', 
#            'SKIN', 
#            'NOSE', 
#            'RIGHT_EYE', 
#            'LEFT_EYE', 
#            'RIGHT_BROW', 
#            'LEFT_BROW', 
#            'RIGHT_EAR', 
#            'LEFT_EAR', 
#            'MOUTH_INTERIOR', 
#            'TOP_LIP', 
#            'BOTTOM_LIP', 
#            'NECK', 
#            'HAIR', 
#            'BEARD', 
#            'CLOTHING', 
#            'GLASSES', 
#            'HEADWEAR', 
#            'FACEWEAR']

# # Create a lookup dictionary from palette to class index
# palette_dict = {tuple(color): idx for idx, color in enumerate(palette)}

# def load_segment(path, img_res, segment_dict):
#     # Read the input image as uint8 format
#     img = imageio.imread(path, as_gray=True)
#     h, w = img.shape
#     segments = np.zeros((h, w, 14))
#     segment_values = segment_dict.values()
#     # follow the classes
#     # concatenate left and right for each facial attributes
#     segments[:, :, 0] = (img == 0) >= 1                                 # 0) background
#     segments[:, :, 1] = (img == 1) >= 1                                 # 1) skin
#     segments[:, :, 2] = (img == 2) >= 1                                 # 2) nose
#     segments[:, :, 3] = (img == 3) + (img == 4) >= 1                    # 3) eyes
#     segments[:, :, 4] = (img == 5) + (img == 6) >= 1                    # 4) eyebrows
#     segments[:, :, 5] = (img == 7) + (img == 8) >= 1                    # 5) ears
#     segments[:, :, 6] = (img == 9) + (img == 10) + (img == 11) >= 1     # 6) mouth
#     segments[:, :, 7] = (img == 12) >= 1                                 # 7) neck
#     segments[:, :, 8] = (img == 13) >= 1                                 # 8) hair
#     segments[:, :, 9] = (img == 14) >= 1                                 # 9) beard
#     segments[:, :, 10] = (img == 15) >= 1                                # 10) cloth
#     segments[:, :, 11] = (img == 16) >= 1                                # 11) eyeglasses
#     segments[:, :, 12] = (img == 17) >= 1                                # 12) headwear
#     segments[:, :, 13] = (img == 18) >= 1                                # 13) facewear
    
#     segments = cv2.resize(segments, (int(img_res[0]), int(img_res[1])))
#     segments = segments.transpose(2, 0, 1)
#     return segments

def load_segment(path, segment_dict, img_res=None):
    # Read the input image as uint8 format
    img = imageio.imread(path, as_gray=True)
    h, w = img.shape
    num_segments = len(segment_dict)
    segments = np.zeros((h, w, num_segments))
    
    # Iterate over the segment_dict to handle each segment
    for idx, (key, value) in enumerate(segment_dict.items()):
        if isinstance(value, list):  # If it's a list, concatenate the corresponding indices
            temp_arr = np.zeros((h, w))
            for v in value:
                temp_arr += (img == v)
            segments[:, :, idx] = (temp_arr >= 1)
        else:  # Single index case
            segments[:, :, idx] = (img == value) >= 1
    
    # Resize the segments to the desired resolution
    if img_res is not None:
        segments = cv2.resize(segments, (int(img_res[0]), int(img_res[1])))
    segments = segments.transpose(2, 0, 1)  # Change to the (channel, height, width) format
    return segments


class Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 conf,
                 mode,
                 expr_dir,
                 sub_dir,
                 img_res,
                 subsample_type=None,
                 subsample=1,
                 load_images=False,
                 shape_test=None,
                 hard_mask=False,
                 only_json=False
                 ):
        
        self.conf = conf
        data_folder = conf.get_string('dataset.data_folder')
        subject_name = conf.get_string('dataset.subject_name')
        dataset_name = conf.get_string('dataset.dataset_name')
        pattern_type = conf.get_string('dataset.pattern_type')
        flame_json_name = conf.get_string('dataset.flame_json_name')
        prompt_csv_name = conf.get_string('dataset.prompt_csv_name')
        use_mean_expression = conf.get_bool('dataset.use_mean_expression', default=False)
        use_var_expression = conf.get_bool('dataset.use_var_expression', default=False)
        bg_color_rendering = conf.get_string('dataset.bg_color_rendering')
        
        sub_dir = natsorted([str(dir) for dir in sub_dir])
        self.img_res = img_res
        self.use_background = conf.get_bool('dataset.use_background', default=False)
        self.load_images = load_images
        self.hard_mask = hard_mask

        self.mode = mode
        self.shape_test = shape_test
        # self.normal = True if conf.get_float('loss.normal_weight') > 0 and mode == 'train' else False
        self.segment_dict = conf.get_config('dataset.segment')
        self.category_dict = conf.get_config('dataset.category_dict')
        self.source_category = conf.get_list('dataset.source_category')
        self.source_category_dict = {key: self.category_dict[key] for key in self.source_category}

        self.diffmorpher_weight = conf.get_float('loss.diffmorpher_weight') 
        self.lora_finetuning = conf.get_bool('train.lora.lora_finetuning')

        num_sample_frames = conf.get_int('dataset.num_sample_frames')
        num_samples = conf.get_int('dataset.num_samples')

        self.reference_image_frame = conf.get_int('dataset.reference_image_frame')
        self.prompts = pd.read_csv(os.path.join(data_folder, subject_name, 'prompt', prompt_csv_name))

        self.test_default_rendering = conf.get_bool('test.default.rendering')
        self.test_default_rendering_novel_view = conf.get_bool('test.default.rendering_novel_view')
        self.test_default_rendering_novel_view_euler_angle = conf.get_list('test.default.novel_view_euler_angle')
        self.test_default_rendering_novel_view_translation = conf.get_list('test.default.novel_view_translation')

        self.data = {
            "image_paths": [],
            "mask_paths": [],
            "world_mats": [],
            "expressions": [],
            "flame_pose": [],
            "img_name": [],
            "sub_dir": [],
            "sub_dir_index": [],
            "shapes":[],
            "segment_paths": [],
            "world_view_transform": [],
            "full_proj_transform": [],
            "camera_center": [],
            "tanfovx": [],
            "tanfovy": [],
            "bg_color": [],
        }
        if self.mode == 'train' and self.diffmorpher_weight > 0:
            self.data['sub_dir_another_pair'] = []
            self.data["image_paths_another_pair"] = []
            self.data["mask_paths_another_pair"] = []
            self.data["segment_paths_another_pair"] = []

            self.diffmorpher_target_attrs = conf.get_list('train.diffmorpher_target_attrs')# ['hair', 'hat']

        # if self.normal:
        #     self.data["normal_original_paths"] = []
        #     self.data["normal_paths"] = []

        # black_list_dir = os.path.join(data_folder, subject_name, subject_name, "blacklist.json")
        # if os.path.exists(black_list_dir):
        #     with open(black_list_dir, "r") as file:
        #         self.blacklist = json.load(file)
        # else:
        #     self.blacklist = None
        
        # if self.mode == 'test':
        #     cam_file = os.path.join(data_folder, subject_name, subject_name, sub_dir[0], flame_json_name)

        num_images = []
        valid_dirs = set()
        for i, dir in tqdm(enumerate(sub_dir), desc='Loading data pivot type', total=len(sub_dir)):
            if self.mode == 'train':
                if dir.split('_')[1] == 'test':
                    continue
            instance_dir = os.path.join(data_folder, subject_name, dataset_name, dir)
            cam_file = os.path.join(data_folder, subject_name, dataset_name, dir, flame_json_name)

            if not os.path.exists(cam_file):
                print('[WARN] {} does not exist'.format(cam_file))
                continue

            with open(cam_file, 'r') as f:
                camera_dict = json.load(f)

            if not os.path.exists(instance_dir):
                continue

            if self.mode == 'train' and not is_valid(instance_dir):
                continue

            valid_dirs.add(dir)

            length_cam = len(camera_dict['frames'])
            length_img = len(os.listdir(os.path.join(instance_dir, 'image')))
            num_images.append(length_cam)
            if length_img != length_cam and is_main_process():
                print("[WARN] In directory {}, number of camera_dict {} != number of real image {}".format(dir, length_cam, length_img))

            # NOTE add gaussian intrincis related
            intrinc_ga = camera_dict['intrinsics']
            intrinsics_ga = np.zeros((3, 3))
            intrinsics_ga[0, 0] = -intrinc_ga[0] * self.img_res[0]
            intrinsics_ga[1, 1] = intrinc_ga[1] * self.img_res[1]
            intrinsics_ga[2, 2] = 1
            intrinsics_ga[0, 2] = intrinc_ga[2] * self.img_res[0]
            intrinsics_ga[1, 2] = intrinc_ga[3] * self.img_res[1]
            self.intrinsics_ga = intrinsics_ga

            if self.mode == 'train' or self.mode == 'train_single':
                category = dir.split('_')[1]
                if num_sample_frames > 0:
                    if self.diffmorpher_weight > 0:
                        sub_dirs_same_category = [d for d in sub_dir if d != dir and d.split('_')[1] == category]
                        if num_samples == -1:
                            num_samples = len(sub_dirs_same_category) - 1
                        if category in self.diffmorpher_target_attrs:
                            sampled_camera_dict = random.sample(camera_dict['frames'], num_sample_frames)                                      # 현 directory에 대해 num_sample_frames만큼의 frame을 샘플링
                            another_pair_dir = random.sample(sub_dirs_same_category, num_samples)              # 같은 category이고 다른 directory에 대해 num_sample_frames만큼의 frame을 샘플링
                        else:
                            sampled_camera_dict = random.sample(camera_dict['frames'], num_sample_frames)
                            
                            if category == 'source':            # 이건 반드시 유지되어야 한다.
                                another_pair_dir = [dir] * len(sampled_camera_dict)
                            else:
                                another_pair_dir = random.sample(sub_dirs_same_category, num_samples)
                    else:
                        if category == 'source':
                            sampled_camera_dict = camera_dict['frames']
                        else:
                            sampled_camera_dict = random.sample(camera_dict['frames'], num_sample_frames)
                else:
                    sampled_camera_dict = camera_dict['frames']

            elif self.mode == 'val':
                sampled_camera_dict = camera_dict['frames']

            elif self.mode == 'test':
                selected = self.conf.test.selected_frames

                if selected == 'all':
                    sampled_camera_dict = camera_dict['frames']
                elif isinstance(selected, str) and ':' in selected:
                    # 문자열 형식의 범위 e.g., "100:200"
                    start_str, end_str = selected.split(':')
                    test_start_frame = int(start_str)
                    test_end_frame = int(end_str)
                    sampled_camera_dict = camera_dict['frames'][test_start_frame:test_end_frame]
                else:
                    # 리스트 형식 e.g., [100, 200, 250, 300]
                    test_selected_frames = self.conf.get_list('test.selected_frames')
                    sampled_camera_dict = [camera_dict['frames'][i] for i in test_selected_frames]

            for j, frame in enumerate(sampled_camera_dict):
                world_mat = np.array(frame['world_mat']).astype(np.float32)
                # camera to world matrix
                self.data["world_mats"].append(world_mat)

                # NOTE add gaussian related
                camera_pose = np.vstack((np.array(frame['world_mat']), np.array([[0, 0, 0, 1]])))
                if self.test_default_rendering_novel_view:
                    # -----------------------------------
                    # 2) Novel view: y축으로 15도 회전
                    # -----------------------------------
                    # 2-1) camera_pose를 torch로 변환
                    #      batch 차원 맞추기 (여기서는 batch=1)
                    camera_pose_torch = torch.from_numpy(camera_pose).float().unsqueeze(0)  # shape: [1,4,4]
                    
                    angle_in_degrees = np.array(self.test_default_rendering_novel_view_euler_angle)
                    euler_angle = torch.tensor(np.deg2rad(angle_in_degrees), dtype=torch.float32)
                    translation = torch.tensor(self.test_default_rendering_novel_view_translation, dtype=torch.float32)

                    # 2-4) novel_view 함수로 새로운 pose 구하기
                    novel_pose = novel_view(camera_pose_torch, euler_angle, translation)  
                    # -> shape: [1,4,4]
                    camera_pose = novel_pose.squeeze(0).cpu().numpy()  # 다시 numpy [4,4]
                # camera_pose: w2c
                c2w = np.linalg.inv(camera_pose)
                c2w[:3, 1:3] *= -1                  # edit for new coordinate.
                w2c = np.linalg.inv(c2w)            # world2camera coordinate for new coordinate.

                R = np.transpose(w2c[:3, :3])                   # camera2world rotation.
                T = w2c[:3, 3]                                  # world2camera translation.
                trans = np.array([0.0, 0.0, 0.0], np.float32)   # no variation.
                scale = 1.0

                world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(trans), scale)).transpose(0, 1)       # world2camera matrix for new coordinate.

                if self.mode == 'train' or self.mode == 'train_single':
                    bg_color = torch.rand(3, dtype=torch.float32)
                else:
                    if bg_color_rendering == 'black':
                        bg_color = torch.zeros(3, dtype=torch.float32)
                    elif bg_color_rendering == 'white':
                        bg_color = torch.ones(3, dtype=torch.float32)
                    else:
                        assert False, f"bg_color_rendering: {bg_color_rendering}"
                    
                self.bg_color = bg_color
                fx = self.intrinsics_ga[0, 0]
                fy = self.intrinsics_ga[1, 1]
                # bg_mask = torch.ones(3, 512, 512)
                FovX = focal2fov(fx, self.img_res[1])
                FovX = np.array(FovX, np.float32)
                FovX = torch.from_numpy(FovX)
                tanfovx = torch.tan(FovX * 0.5).item()
                FovY = focal2fov(fy, self.img_res[0])
                FovY = np.array(FovY, np.float32)
                FovY = torch.from_numpy(FovY)
                tanfovy = torch.tan(FovY * 0.5).item()
                znear = 0.01
                zfar = 100

                projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY,
                                                        K=self.intrinsics_ga,
                                                        h=self.img_res[1], w=self.img_res[0]).transpose(0, 1)

                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                world_view_transform = world_view_transform
                self.data["tanfovx"].append(tanfovx)
                self.data["tanfovy"].append(tanfovy)
                self.data["bg_color"].append(bg_color)
                # self.data["bg_mask"].append(bg_mask)
                self.data["world_view_transform"].append(world_view_transform)
                self.data["full_proj_transform"].append(full_proj_transform)
                self.data["camera_center"].append(camera_center)
                #################################################################################

                self.data["expressions"].append(np.array(frame['expression']).astype(np.float32))
                self.data["flame_pose"].append(np.array(frame['pose']).astype(np.float32))
                self.data["sub_dir"].append(dir)
                self.data["sub_dir_index"].append(i)
                image_path = '{0}/{1}.jpg'.format(instance_dir, frame["file_path"])
                self.data["image_paths"].append(image_path)
                self.data["mask_paths"].append(image_path.replace('image', 'mask').replace('jpg', 'png'))
                if self.mode == 'train' or self.mode == 'train_single':
                    self.data["segment_paths"].append(image_path.replace('jpg', 'png').replace('image', 'segment')) 
                    # if self.normal:
                    #     self.data["normal_paths"].append(image_path.replace('image', 'normal'))

                    if self.diffmorpher_weight > 0:
                        # random sampling of another pair
                        another_sub_dir = random.choice(another_pair_dir)
                        self.data["sub_dir_another_pair"].append(another_sub_dir)
                        image_path_another_pair = '{0}/{1}.jpg'.format(os.path.join(data_folder, subject_name, dataset_name, another_sub_dir), frame["file_path"])
                        self.data["image_paths_another_pair"].append(image_path_another_pair)               # same view, different dataset subdir
                        self.data["mask_paths_another_pair"].append(image_path_another_pair.replace('image', 'mask').replace('jpg', 'png'))
                        self.data["segment_paths_another_pair"].append(image_path_another_pair.replace('jpg', 'png').replace('image', 'segment'))
                    
                self.data["img_name"].append(int(frame["file_path"].split('/')[-1]))
        
        if self.mode == 'train' or self.mode == 'train_single':
            self.valid_dirs = list(valid_dirs)
        else:
            with open(os.path.join(expr_dir.replace('eval', 'train'), 'train_dataset.txt'), 'r') as f:
                self.valid_dirs = f.read().splitlines()
            if self.lora_finetuning and self.mode == 'test':
                interpolation_target_list = conf.get_list('test.interpolation.target_list')
                self.valid_dirs.append(interpolation_target_list[-1])


        self.reference_image = {}
        self.reference_text = {}
        for vd in self.valid_dirs:
            if vd.split('_')[1] == 'source':
                continue
            reference_image_path = os.path.join(data_folder, subject_name, dataset_name, vd, 'image', '{}.jpg'.format(str(self.reference_image_frame).zfill(4)))
            reference_mask_path = reference_image_path.replace('image', 'mask').replace('jpg', 'png')
            reference_object_mask = torch.from_numpy(load_mask(reference_mask_path, self.img_res).reshape(-1)).unsqueeze(1).float()
            reference_rgb = torch.from_numpy(load_rgb(reference_image_path, self.img_res).reshape(3, -1).transpose(1, 0)).float()
            self.reference_image[vd] = reference_rgb * reference_object_mask + (1 - reference_object_mask)

            # dataset_name = vd.split('--')[0].replace('_1', '')            # for gafni version
            # dataset_name = 
            self.reference_text[vd] = self.prompts.loc[self.prompts['dataset_name'] == extract_middle(vd, pattern_type), 'prompt'].values
        
        if self.mode == 'test':
            self.zero_shot_rendering = conf.get_bool('test.zero_shot.rendering')
            if self.zero_shot_rendering:
                zero_shot_image_path = os.path.join(conf.get_string('test.zero_shot.image_path'), '{}.jpg'.format(str(self.reference_image_frame).zfill(4)))
                zero_shot_mask_path = zero_shot_image_path.replace('image', 'mask').replace('jpg', 'png')
                zero_shot_text_prompt = conf.get_string('test.zero_shot.text_prompt')

                rgb_zero_shot = torch.from_numpy(load_rgb(zero_shot_image_path,
                                                          self.img_res).reshape(3, -1).transpose(1, 0)).float()
                mask_zero_shot = torch.from_numpy(load_mask(zero_shot_mask_path,
                                                            self.img_res).reshape(-1))
                self.image_zero_shot = rgb_zero_shot * mask_zero_shot.unsqueeze(1).float() + (1 - mask_zero_shot.unsqueeze(1).float())
                self.text_zero_shot = zero_shot_text_prompt
                self.sub_dir_zero_shot = os.path.dirname(zero_shot_image_path).split('/')[-2]

        if self.mode == 'train':
            train_array = np.array(list(set(self.data['sub_dir'])))
            with open(os.path.join(expr_dir, 'train_dataset.txt'), 'w') as f:
                for item in train_array:
                    f.write("%s\n" % item)
            print(f'[INFO] {self.mode} dataset listup complete!')

            if self.diffmorpher_weight > 0:
                train_array_another_pair = np.array(list(set(self.data['image_paths_another_pair'])))
                with open(os.path.join(expr_dir, 'train_dataset_another_pair.txt'), 'w') as f:
                    for item in train_array_another_pair:
                        f.write("%s\n" % item)
                print(f'[INFO] {self.mode} another pair dataset listup complete!')
                
        if self.mode == 'test':     # NOTE test를 할 때는 shape parameter를 따로 지정해주도록 하자.
            shape_standard_instance_dir = os.path.join(data_folder, subject_name, dataset_name, self.shape_test)
            assert os.path.exists(shape_standard_instance_dir), "Data directory {} is empty".format(shape_standard_instance_dir)
            
            shape_standard_cam_file = '{0}/{1}'.format(shape_standard_instance_dir, flame_json_name)
            with open(shape_standard_cam_file, 'r') as f:
                shape_standard_camera_dict = json.load(f)

            self.shape_params = torch.tensor(shape_standard_camera_dict['shape_params']).float().unsqueeze(0)
            if is_main_process():
                print(f'[INFO] {self.mode} shape parameter from {shape_standard_cam_file}')
        else:
            self.shape_params = torch.tensor(camera_dict['shape_params']).float().unsqueeze(0)

        self.gt_dir = instance_dir
        focal_cxcy = camera_dict['intrinsics']

        def cumulative_sum(lst):
            result = []
            cumsum = 0
            for num in lst:
                cumsum += num
                result.append(cumsum)
            return result
        index_list = cumulative_sum(num_images)

        subsample_type = str(subsample_type)
        assert subsample_type in ['frames', 'ratio'], 'subsample_type should be either "frames" or "ratio"'
        if subsample_type == 'ratio':
            if is_main_process():
                print(f'[INFO] {self.mode} subsampling the data by a ratio of {subsample}')
            if isinstance(subsample, list):
                assert False, 'Need to revisit frame sampling'
                subsample = subsample*len(num_images)
            else:
                if is_main_process():
                    print('[INFO] no blacklist applied.')
        elif subsample_type == 'frames':
            assert False, 'Need to revisit frame sampling'
            subsampled_frames = subsample
            if isinstance(subsampled_frames, list) and len(subsampled_frames) > 0:
                if is_main_process():
                    print('[DEBUG] subsampling the data by a list of frames: {}'.format(subsampled_frames))
                if len(num_images) != len(subsampled_frames):
                    raise ValueError('The number of subdirectories and the number of subsampled frames should be the same')
                subsample = [0]*len(num_images)
                for idx, num_img in enumerate(num_images):
                    subsample[idx] = num_img // subsampled_frames[idx]
                    if subsample[idx] == 0:
                        subsample[idx] = 1
            elif isinstance(subsampled_frames, int) and subsampled_frames > 1:
                if is_main_process():
                    print('[DEBUG] all of the frame should be the same as {}'.format(subsampled_frames))
                subsampled_frames = [subsampled_frames] * len(num_images)
                subsample = [0]*len(num_images)
                for idx, num_img in enumerate(num_images):
                    subsample[idx] = num_img // subsampled_frames[idx]
            else:
                raise ValueError('subsampled_frames should be a list of integers')

        def pop_blacklist(data_list, blacklist):
            '''
            if data_list has the blacklist's item, then remove it and return
            '''
            sub_dir_name = data_list[0].split('/')[-4]
            black_file_list = blacklist[sub_dir_name]
            new_data_list = copy.deepcopy(data_list)
            removed_indices = []  # List to store the indices of removed items

            for idx, value in enumerate(data_list):
                if value.split('/')[-1] in black_file_list:
                    # print(value)
                    new_data_list.remove(value)
                    removed_indices.append(idx)
            return new_data_list, removed_indices
        
        def pop_blacklist_using_indices(data_list, removed_indices):
            '''
            if data_list has the blacklist's item, then remove it and return
            '''
            new_data_list = copy.deepcopy(data_list)
            for idx in reversed(removed_indices):
                new_data_list.pop(idx)
            return new_data_list
        
        if is_main_process():
            print('*****************************************************************')

        if isinstance(subsample, int) and subsample > 1:
            for k, v in self.data.items():
                self.data[k] = v[::subsample]
            if is_main_process():
                print('[INFO] subsampling the data by a factor of {} (int type)'.format(subsample))
                for i in zip(sub_dir, num_images):
                    print('[INFO] sub directory: {} | frames: {}'.format(i[0], i[1]))
                print('[INFO] total frames: {} -> subsampled: {}'.format(index_list[-1], len(self.data['image_paths'])))
        elif isinstance(subsample, list): #  and (self.mode != 'test'): # NOTE 이 코드는 blacklist를 적용해서 mask_obj나 mask_rgb에서 blacklist가 제대로 제거된다. train의 경우에만 해주고 test일 경우에는 그냥 다 해라.
            assert False, 'Need to revisit frame sampling'
            if is_main_process():
                print('[DEBUG] subsampling the data by a factor of {} (list type)'.format(subsample))
            if len(subsample) != len(sub_dir):
                raise ValueError('[ERROR] subsample list length should be equal to the number of subdirectories')
            
            removed_indices_list = []
            for k, v in self.data.items():
                temp_list = []
                for i, s in enumerate(subsample):
                    assert subsample[i] != 0, 'Please except the train dataset: {}'.format(sub_dir[i])
                    if s == 1:
                        if i == 0:
                            sampled_list = v[:index_list[i]]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if is_main_process():
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {}'.format(sub_dir[i], s, len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[:index_list[i]]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[:index_list[i]]), len(sampled_list)))
                        else:
                            sampled_list = v[index_list[i-1]:index_list[i]]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if is_main_process():
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {}'.format(sub_dir[i], s, len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[index_list[i-1]:index_list[i]]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[index_list[i-1]:index_list[i]]), len(sampled_list)))
                    elif s > 1:
                        if i == 0:
                            sampled_list = v[:index_list[i]][::s]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if is_main_process():
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {} -> {}'.format(sub_dir[i], s, len(v[:index_list[i]]), len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[:index_list[i]][::s]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[:index_list[i]][::s]), len(sampled_list)))
                        else:
                            sampled_list = v[index_list[i-1]:index_list[i]][::s]
                            if self.blacklist is not None:
                                if k == 'image_paths':
                                    sampled_list, removed_indices = pop_blacklist(sampled_list, self.blacklist)
                                    removed_indices_list.append(removed_indices)
                                else:
                                    sampled_list = pop_blacklist_using_indices(sampled_list, removed_indices_list[i])
                            temp_list += sampled_list
                            if is_main_process():
                                print('[DEBUG] sub directory: {} | sampling rate: {} | frames: {} -> {}'.format(sub_dir[i], s, len(v[index_list[i-1]:index_list[i]]), len(sampled_list)))
                                print('[DEBUG] first frame: {}, last frame: {}'.format(sampled_list[0], sampled_list[-1]))
                                if self.blacklist is not None and len(sampled_list) != len(v[index_list[i-1]:index_list[i]][::s]):
                                    print('[DEBUG] before black list: {} -> after black list: {}'.format(len(v[index_list[i-1]:index_list[i]][::s]), len(sampled_list)))
                    if k == 'image_paths':
                        assert sampled_list[0].split('/')[-4] == sampled_list[-1].split('/')[-4], 'sub directory name should be the same'
                    
                self.data[k] = temp_list
            if is_main_process():
                print('[DEBUG] total frames: {} -> subsampled: {}'.format(index_list[-1], len(self.data['image_paths'])))
        if is_main_process():
            print('*****************************************************************')

        self.data["expressions"] = torch.from_numpy(np.stack(self.data["expressions"], 0))
        self.data["flame_pose"] = torch.from_numpy(np.stack(self.data["flame_pose"], 0))
        self.data["world_mats"] = torch.from_numpy(np.stack(self.data["world_mats"], 0)).float()
        
        # construct intrinsic matrix
        intrinsics = np.zeros((4, 4))

        # from whatever camera convention to pytorch3d
        intrinsics[0, 0] = focal_cxcy[0] * 2
        intrinsics[1, 1] = focal_cxcy[1] * 2
        intrinsics[0, 2] = (focal_cxcy[2] * 2 - 1.0) * -1
        intrinsics[1, 2] = (focal_cxcy[3] * 2 - 1.0) * -1

        intrinsics[3, 2] = 1.
        intrinsics[2, 3] = 1.
        self.intrinsics = intrinsics

        if intrinsics[0, 0] < 0:
            intrinsics[:, 0] *= -1
            self.data["world_mats"][:, 0, :] *= -1
        self.data["world_mats"][:, :3, 2] *= -1
        self.data["world_mats"][:, 2, 3] *= -1

        if use_mean_expression:
            self.mean_expression = torch.mean(self.data["expressions"], 0, keepdim=True)
        else:
            self.mean_expression = torch.zeros_like(self.data["expressions"][[0], :])
        if use_var_expression:
            self.var_expression = torch.var(self.data["expressions"], 0, keepdim=True)
        else:
            self.var_expression = None

        self.intrinsics = torch.from_numpy(self.intrinsics).float()
        self.only_json = only_json

        images = []
        masks = []
        if load_images and not only_json:
            if is_main_process():
                print("[INFO] Loading all images, this might take a while.")
            for idx in tqdm(range(len(self.data["image_paths"]))):
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1,0)).float()
                object_mask = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                if not self.use_background:         
                    if not hard_mask:           
                        rgb = rgb * object_mask.unsqueeze(1).float() + (1 - object_mask.unsqueeze(1).float())
                    else:
                        rgb = rgb * (object_mask.unsqueeze(1) > 0.5) + ~(object_mask.unsqueeze(1) > 0.5)
                images.append(rgb)
                masks.append(object_mask)

        self.data['images'] = images
        self.data['masks'] = masks

        if is_main_process():
            if self.mode == 'train':
                # # save dataset configuration
                # train_array = np.array(list(set(self.data['sub_dir'])))
                # with open(os.path.join(expr_dir, 'train_dataset.txt'), 'w') as f:
                #     for item in train_array:
                #         f.write("%s\n" % item)
                # print(f'[INFO] {self.mode} dataset listup complete!')     

                # save mean and var expression
                np.save(os.path.join(expr_dir, 'mean_expression.npy'), self.mean_expression.cpu().numpy())
                if use_var_expression:
                    np.save(os.path.join(expr_dir, 'var_expression.npy'), self.var_expression.cpu().numpy())
        dist.barrier()

    def __len__(self):
        return len(self.data["image_paths"])

    def __getitem__(self, idx):
        sample = {
            "idx": torch.LongTensor([idx]),
            "img_name": torch.LongTensor([self.data["img_name"][idx]]),
            "sub_dir": self.data["sub_dir"][idx],
            "sub_dir_index": self.data["sub_dir_index"][idx],
            "intrinsics": self.intrinsics,
            "expression": self.data["expressions"][idx],
            "flame_pose": self.data["flame_pose"][idx],
            "cam_pose": self.data["world_mats"][idx],
            # "object_mask": torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1)),
            "shape": self.shape_params.squeeze(),
            # NOTE add gaussian-related
            "world_view_transform": self.data["world_view_transform"][idx],
            "full_proj_transform": self.data["full_proj_transform"][idx],
            "camera_center": self.data["camera_center"][idx],
            "tanfovx": self.data["tanfovx"][idx],
            "tanfovy": self.data["tanfovy"][idx],
            "bg_color": self.data["bg_color"][idx],
            "reference_text": self.reference_text,
            "reference_image": self.reference_image,
            "image_paths": self.data["image_paths"][idx],
            # # temp
            # "world2camera": self.data["world2camera"][idx],
            }
        
        if self.mode == 'test' and self.zero_shot_rendering:
            sample['image_zero_shot'] = self.image_zero_shot
            sample['text_zero_shot'] = self.text_zero_shot
            sample['sub_dir_zero_shot'] = self.sub_dir_zero_shot

        ground_truth = {}
        
        if not self.only_json:                    
            if not self.load_images:        
                ground_truth["object_mask"] = torch.from_numpy(load_mask(self.data["mask_paths"][idx], self.img_res).reshape(-1))
                sample["object_mask"] = ground_truth["object_mask"].clone()
                rgb = torch.from_numpy(load_rgb(self.data["image_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()

                bg_color = sample['bg_color']
                bg_color = bg_color.view(1, 3).expand_as(rgb)

                if not self.use_background:      
                    if not self.hard_mask:        
                        mask = ground_truth["object_mask"].unsqueeze(1).float()
                        ground_truth['rgb'] = rgb * mask + bg_color * (1 - mask)
                        # ground_truth['rgb'] = rgb * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float()) # rgb.shape: torch.Size([262144, 3]), ground_truth["object_mask"].unsqueeze(1).float().shape: torch.Size([262144, 1])
                    else:
                        ground_truth['rgb'] = rgb * (ground_truth["object_mask"].unsqueeze(1) > 0.5) + ~(ground_truth["object_mask"].unsqueeze(1) > 0.5)
                else:
                    ground_truth['rgb'] = rgb
                
                if self.mode == 'train':
                    category = sample['sub_dir'].split('_')[1]
                    # if category in ['hair', 'hat', 'beard']:
                    # # if category != 'source':            # bg와 같이 넣어준다.
                    #     background_idx, category_idx = list(self.segment_dict.keys()).index('background'), list(self.segment_dict.keys()).index(category)
                    #     object_mask_category = torch.from_numpy(load_segment(self.data["segment_paths"][idx], self.segment_dict, self.img_res)[category_idx].reshape(-1))
                    #     object_mask_background = torch.from_numpy(load_segment(self.data["segment_paths"][idx], self.segment_dict, self.img_res)[background_idx].reshape(-1))
                    #     ground_truth['occlusion_mask'] = object_mask_background + object_mask_category         # hair의 update영역을 변경함. hair를 파먹지 않게 하기 위해서.

                    if self.lora_finetuning:            # bg와 같이 넣어준다.
                        background_idx, category_idx = list(self.segment_dict.keys()).index('background'), list(self.segment_dict.keys()).index(category)
                        object_mask_category = torch.from_numpy(load_segment(self.data["segment_paths"][idx], self.segment_dict, self.img_res)[category_idx].reshape(-1))
                        object_mask_background = torch.from_numpy(load_segment(self.data["segment_paths"][idx], self.segment_dict, self.img_res)[background_idx].reshape(-1))
                        ground_truth['occlusion_mask'] = object_mask_background + object_mask_category         # hair의 update영역을 변경함. hair를 파먹지 않게 하기 위해서.

                if self.mode == 'train' and self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs):
                    # another pair를 만드는 곳.
                    sample['sub_dir_another_pair'] = self.data['sub_dir_another_pair'][idx]
                    assert sample['sub_dir'].split('_')[1] == sample['sub_dir_another_pair'].split('_')[1], f"sub_dir: {sample['sub_dir']}, sub_dir_another_pair: {sample['sub_dir_another_pair']}"

                    ground_truth["object_mask_another_pair"] = torch.from_numpy(load_mask(self.data["mask_paths_another_pair"][idx], self.img_res).reshape(-1))
                    rgb_another_pair = torch.from_numpy(load_rgb(self.data["image_paths_another_pair"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()
                    mask = ground_truth["object_mask_another_pair"].unsqueeze(1).float()
                    ground_truth['rgb_another_pair'] = rgb_another_pair * mask + bg_color * (1 - mask)
                
                    # background_idx, category_idx = list(self.segment_dict.keys()).index('background'), list(self.segment_dict.keys()).index(category)
                    # object_mask_background = torch.from_numpy(load_segment(self.data["segment_paths_another_pair"][idx], self.segment_dict, self.img_res)[background_idx].reshape(-1))
                    # object_mask_category = torch.from_numpy(load_segment(self.data["segment_paths_another_pair"][idx], self.segment_dict, self.img_res)[category_idx].reshape(-1))
                    # ground_truth['occlusion_mask_another_pair'] = object_mask_background + object_mask_category
                
                # if self.normal: # and all(key not in self.data["sub_dir"][idx] for key in ["target", "source"]) and self.mask_object:
                #     normal = torch.from_numpy(load_rgb(self.data["normal_paths"][idx], self.img_res).reshape(3, -1).transpose(1, 0)).float()
                #     ground_truth['normal'] = normal * ground_truth["object_mask"].unsqueeze(1).float() + (1 - ground_truth["object_mask"].unsqueeze(1).float())

            else:
                ground_truth = {
                    'rgb': self.data['images'][idx],
                    'object_mask': self.data['masks'][idx],
                }

        ground_truth['image_paths'] = self.data["image_paths"][idx]
        if self.mode == 'train' and self.diffmorpher_weight > 0 and (category in self.diffmorpher_target_attrs):
            sample['sub_dir_another_pair'] = self.data['sub_dir_another_pair'][idx]
            ground_truth['image_paths_another_pair'] = self.data["image_paths_another_pair"][idx]
            
        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # this function is borrowed from https://github.com/lioryariv/idr/blob/main/code/datasets/scene_dataset.py
        # get list of dictionaries and returns sample, ground_truth as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    try:
                        ret[k] = torch.stack([obj[k] for obj in entry])
                    except:
                        ret[k] = [obj[k] for obj in entry]
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)
    