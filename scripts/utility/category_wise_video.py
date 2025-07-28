import os
import cv2
import numpy as np
import ffmpeg
from natsort import natsorted as sorted
from collections import defaultdict
import argparse

def get_category_dirs(base_path):
    category_dict = defaultdict(list)
    
    for folder in sorted(os.listdir(base_path)):
        parts = folder.split('_')
        if len(parts) <= 2:
            continue
        category = parts[1]  # category 추출
        category_dict[category].append(os.path.join(base_path, folder))
    
    return dict(category_dict)

def load_images_from_dirs(dirs):
    image_sets = []
    
    for dir_path in dirs:
        dir_path = os.path.join(dir_path, 'rgb')
        image_files = sorted([f for f in os.listdir(dir_path) if f.endswith('.jpg')])
        images = [cv2.imread(os.path.join(dir_path, img_file)) for img_file in image_files if cv2.imread(os.path.join(dir_path, img_file)) is not None]
        if images:
            image_sets.append(images)
    
    if not image_sets:
        return []
    
    min_images = min(len(imgs) for imgs in image_sets)  # 가장 작은 이미지 개수 찾기
    return [imgs[:min_images] for imgs in image_sets]

def create_grid_frames(image_sets, grid_size=(3, 6), img_size=(256, 256)):
    rows, cols = grid_size
    num_frames = len(image_sets[0])
    
    frames = []
    for i in range(num_frames):
        images = [imgs[i] for imgs in image_sets]
        if len(images) < rows * cols:
            images += [np.zeros((img_size[1], img_size[0], 3), dtype=np.uint8)] * (rows * cols - len(images))
        
        resized_images = [cv2.resize(img, img_size) for img in images]
        grid_rows = [np.hstack(resized_images[j * cols:(j + 1) * cols]) for j in range(rows)]
        grid_image = np.vstack(grid_rows)
        frames.append(grid_image)
    
    return frames

def create_video(frames, output_path, fps=30):
    h, w, _ = frames[0].shape
    video_path = f"{output_path}.mp4"
    
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved at {video_path}")

def main(args):
    base_path = args.base_path
    output_folder = args.output_folder
    fps = args.fps

    os.makedirs(output_folder, exist_ok=True)
    category_dirs = get_category_dirs(base_path)
    
    for idx, (category, dirs) in enumerate(category_dirs.items()):
        if idx >= 18:
            break  # 최대 18개 카테고리만 처리
        image_sets = load_images_from_dirs(dirs)
        if image_sets:
            frames = create_grid_frames(image_sets, grid_size=(2, 5))
            output_path = os.path.join(output_folder, f"{category}_video")
            os.makedirs(output_folder, exist_ok=True)
            create_video(frames, output_path, fps=fps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, default="/home/hyunsoocha/GitHub/perse-dev/data/experiments/supp_video/stage2_supp_video/0000_beard_balbo_unkempt_black_soft_portrait_to_0089_nose_snub_broad_slightly_crooked_average_width_normal_bridge_rounded_nostrils_portrait_num_90/eval/default", help="Base directory containing category-wise folders")
    parser.add_argument("--output_folder", type=str, default="./", help="Output directory to save videos")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for the video")
    args = parser.parse_args()

    main(args)