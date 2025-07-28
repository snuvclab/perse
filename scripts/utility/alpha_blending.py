import argparse
from PIL import Image
import numpy as np
import math

def blend_images(img_a, img_b, boundary, blend_width=80, theta=45):
    """
    두 이미지를 불러와, 경계를 오른쪽으로 theta 도 기울어진 사선으로 설정하여 alpha blending을 수행합니다.
    - 경계선은 이미지 최하단에서 x좌표가 boundary인 점을 지나고,
      위로 올라갈수록 x 좌표가 tan(theta) (도 단위) 만큼씩 증가합니다.
    - 각 픽셀에 대해, 경계선으로부터의 수평 거리(diff)를 계산하여
      diff < -blend_width/2이면 image_a, diff > blend_width/2이면 image_b의 픽셀을 사용하고,
      그 사이에서는 선형 보간합니다.
    """
    width, height = img_b.size
    arr_a = np.array(img_a).astype(np.float32)
    arr_b = np.array(img_b).astype(np.float32)
    blended = np.zeros_like(arr_b)
    
    # theta는 도 단위이므로 radians로 변환 후 tan 계산
    slope = math.tan(math.radians(theta))
    
    for y in range(height):
        # 이미지 최하단(y = height-1)에서 boundary를 기준으로, 위로 갈수록 x좌표가 slope 만큼씩 증가
        row_boundary = boundary + (height - 1 - y) * slope
        for x in range(width):
            diff = x - row_boundary
            if diff < -blend_width / 2:
                blended[y, x] = arr_a[y, x]
            elif diff > blend_width / 2:
                blended[y, x] = arr_b[y, x]
            else:
                alpha = (diff + blend_width/2) / blend_width
                blended[y, x] = (1 - alpha) * arr_a[y, x] + alpha * arr_b[y, x]
    
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)

def main():
    parser = argparse.ArgumentParser(
        description="두 이미지를 불러와, 경계를 오른쪽으로 theta 도 기울어진 사선으로 설정하여 alpha blending 합니다."
    )
    parser.add_argument("--image_a", type=str, help="512x512 크기의 이미지 A 경로")
    parser.add_argument("--image_b", type=str, help="768x768 크기의 이미지 B 경로")
    parser.add_argument("--output", type=str, help="블렌딩된 결과 이미지 저장 경로")
    parser.add_argument("--boundary", type=int, default=430,
                        help="이미지 최하단에서 경계선의 x 좌표 (기본값: 440)")
    parser.add_argument("--blend_width", type=int, default=70,
                        help="블렌딩 영역의 폭 (픽셀 단위, 기본값: 80)")
    parser.add_argument("--theta", type=float, default=0,
                        help="경계선이 오른쪽으로 기울어지는 각도 (도 단위, radian이 아님, 기본값: 45)")
    args = parser.parse_args()
    
    # 예제 경로 (필요시 주석 처리하고 argparse 인자로 전달하세요)
    string_pattern = '1-9-54115'
    args.image_a = '{}.png'.format(string_pattern)
    args.image_b = '{}-gaussian.jpg'.format(string_pattern)
    args.output = '{}-edit.png'.format(string_pattern)
    # args.image_a = '/home/hyunsoocha/GitHub/perse-dev/scripts/utility/125-0-53660.png'
    # args.image_b = '/home/hyunsoocha/GitHub/perse-dev/data/experiments/paper/stage1_paper/0000_beard_balbo_medium_brown_curly_youngman_hyunsoo_img_6948_to_9999_test_hyunsoo_num_767/eval/default/2523_hat_bucket_hat_felt_burgundy_solid_vintage_youngman_hyunsoo_img_6948/rgb/0125.jpg'
    # args.output = '125-0-53660-edit.png'
    
    # 이미지 로드 및 RGB 모드 변환
    img_a = Image.open(args.image_a).convert("RGB")
    img_b = Image.open(args.image_b).convert("RGB")
    
    # 두 이미지의 크기가 다르면, image_a를 image_b 크기로 리사이즈
    if img_a.size != img_b.size:
        img_a = img_a.resize(img_b.size, Image.Resampling.LANCZOS)
    
    blended_image = blend_images(img_a, img_b, args.boundary, args.blend_width, args.theta)
    blended_image.save(args.output)
    print("Blended image saved to:", args.output)

if __name__ == "__main__":
    main()
