import os
import time
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from edsr import EDSR
from tqdm import tqdm

# SSIM 손실 함수 정의
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size)

    def gaussian(self, window_size, sigma):
        gauss = torch.tensor([-(x - window_size // 2)**2 / (2 * sigma**2) for x in range(window_size)])
        gauss = torch.exp(gauss)
        return gauss / gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        return _2D_window.expand(channel, 1, window_size, window_size).contiguous()

    def forward(self, img1, img2):
        device = img1.device
        channel = img1.size(1)

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window.to(device)
        else:
            window = self.create_window(self.window_size, channel=channel).to(device)
            self.window = window
            self.channel = channel

        mu1 = nn.functional.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = nn.functional.conv2d(img2, window, padding=self.window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1 * mu2
        sigma1_sq = nn.functional.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = nn.functional.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = nn.functional.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean() if self.size_average else ssim_map.mean([1, 2, 3])

# 데이터셋 클래스 정의
class DIV2KDataset(Dataset):
    def __init__(self, images_dir, scale):
        self.images_dir = images_dir
        self.scale = scale
        self.image_names = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        self.fixed_size = (512, 512)
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        img_path = os.path.join(self.images_dir, self.image_names[index])
        img = Image.open(img_path).convert('RGB').resize(self.fixed_size, Image.BICUBIC)
        lr_img = img.resize((img.width // self.scale, img.height // self.scale), Image.BICUBIC)
        return self.transform(lr_img), self.transform(img)

# 모델 학습 함수 정의
def train_model(data_loader, num_epochs=100, learning_rate=1e-4, save_dir="weights"):
    model = EDSR(scale_factor=4).to(device)
    criterion = SSIM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        epoch_loss = 0

        with tqdm(total=len(data_loader), desc=f"Epoch [{epoch + 1}/{num_epochs}]") as pbar:
            for lr_imgs, hr_imgs in data_loader:
                lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)

                optimizer.zero_grad()
                outputs = model(lr_imgs)

                loss = 1 - criterion(outputs, hr_imgs)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        epoch_duration = time.time() - start_time
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss: {epoch_loss / len(data_loader):.4f}, Time: {epoch_duration:.2f}s")

        torch.save(model.state_dict(), os.path.join(save_dir, f"EDSR_epoch_{epoch + 1}.pth"))

# 경로 설정 및 데이터 로더 설정
images_dir = "data/DataJumble"
scale = 4
dataset = DIV2KDataset(images_dir=images_dir, scale=scale)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 학습 시작
train_model(data_loader, num_epochs=100, save_dir="weights")