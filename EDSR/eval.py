import torch
from edsr import Net
from dataset import DIV2KDataset
from torch.utils.data import DataLoader

# 모델 로드 및 평가 함수 정의
def evaluate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for lr_imgs, hr_imgs in data_loader:
            if torch.cuda.is_available():
                lr_imgs = lr_imgs.cuda()
                hr_imgs = hr_imgs.cuda()

            outputs = model(lr_imgs)

if __name__ == "__main__":
    # 경로 설정 및 데이터셋 로드
    images_dir = r"C:\Users\User\Desktop\pyth\DIV2K\DIV2K_train_HR"
    scale = 2
    dataset = DIV2KDataset(images_dir=images_dir, scale=scale)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 모델 초기화 및 평가
    model = Net()
    if torch.cuda.is_available():
        model.cuda()

    evaluate(model, data_loader)
