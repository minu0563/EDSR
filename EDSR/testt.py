import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from edsr import EDSR

def load_model(model_path, scale_factor=4):
    model = EDSR(scale_factor)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
    model.eval()
    return model

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
    output_image_blurred = output_image_pil.filter(ImageFilter.GaussianBlur(radius=2.0))

    return output_image_blurred

def upscale_images_in_folder(input_folder, output_folder, model):
    # 입력 폴더의 모든 이미지 파일을 처리
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            low_res_path = os.path.join(input_folder, filename)
            upscaled_image = upscale_image(model, low_res_path)

            # 업스케일된 이미지 저장
            output_path = os.path.join(output_folder, filename)
            upscaled_image.save(output_path)
            print(f"업스케일된 이미지가 {output_path}에 저장되었습니다.")

def main():
    scale_factor = 4
    model_path = r"C:\Users\User\Desktop\pyth\EDSR for SR\weights\EDSR_epoch_100.pth"
    input_folder = r"C:\Users\chris\Downloads\OGQ_LR"  # 저화질 이미지 폴더
    output_folder = r"C:\Users\chris\Downloads\OGQ_SR"  # 저장할 폴더

    # 출력 폴더가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)

    model = load_model(model_path, scale_factor)

    # 폴더 내의 모든 이미지 업스케일링
    upscale_images_in_folder(input_folder, output_folder, model)

if __name__ == '__main__':
    main()
