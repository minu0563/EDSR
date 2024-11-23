import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from edsr import EDSR
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def load_model(model_path, scale_factor=4):
    model = EDSR(scale_factor)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

def calculate_metrics(original_image, upscaled_image):
    # 원본 이미지와 업스케일된 이미지의 크기를 맞춤
    if original_image.shape != upscaled_image.shape:
        upscaled_image = cv2.resize(upscaled_image, (original_image.shape[1], original_image.shape[0]))
    
    psnr = cv2.PSNR(original_image, upscaled_image)
    try:
        height, width, _ = original_image.shape
        win_size = min(height, width, 7)
        ssim_value = ssim(original_image, upscaled_image, channel_axis=2, win_size=win_size)
    except ValueError as e:
        print(f"SSIM 계산 중 오류 발생: {e}")
        ssim_value = None
    return psnr, ssim_value

def upscale_image(model, low_res_path):
    img = Image.open(low_res_path).convert('RGB')
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output_tensor = model(img_tensor)

    output_image = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    output_image = (output_image * 255).clip(0, 255).astype('uint8')
    output_image_pil = Image.fromarray(output_image)

    # Gaussian Blur 적용
    output_image_blurred = output_image_pil.filter(ImageFilter.GaussianBlur(radius=.0))

    return output_image_blurred

def main():     
    scale_factor = 4
    model_path = r'C:\Users\User\Desktop\pyth\EDSR for SR\weights\EDSR_epoch_100.pth'
    low_res_path = r"C:\Users\User\Desktop\pyth\datasets\OGQ_LR\000000575815.jpg"  # 저화질 이미지 경로
    high_res_path = r"C:\Users\User\Desktop\pyth\datasets\OGQ_HR\000000575815.jpg"  # 고화질 이미지 경로

    # 모델 로드
    model = load_model(model_path, scale_factor)
    
    # 저화질 이미지 업스케일링
    upscaled_image = upscale_image(model, low_res_path)

    # 원본 고화질 이미지를 OpenCV로 로드
    original_image = cv2.imread(high_res_path)
    if original_image is None:
        print(f"원본 고화질 이미지를 로드할 수 없습니다: {high_res_path}")
        return
    
    # 업스케일된 이미지를 OpenCV 형식으로 변환
    upscaled_image_cv2 = cv2.cvtColor(np.array(upscaled_image), cv2.COLOR_RGB2BGR)
    
    # PSNR 및 SSIM 계산
    psnr, ssim_value = calculate_metrics(original_image, upscaled_image_cv2)

    # PSNR 및 SSIM 출력
    print(f"PSNR: {psnr:.2f} dB")
    if ssim_value is not None:
        print(f"SSIM: {ssim_value:.4f}")
    else:
        print("SSIM을 계산할 수 없음")

    # 이미지 출력
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title('Original High-Resolution Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(upscaled_image)
    plt.title('Upscaled Image from Low-Resolution (Blurred)')
    plt.axis('off')

    plt.show()

if __name__ == '__main__':
    main()
