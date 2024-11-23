import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        res = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x + res

class EDSR(nn.Module):
    def __init__(self, scale_factor, num_residuals=16):
        super(EDSR, self).__init__()
        self.input_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.residuals = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residuals)])
        
        # 출력 채널 수를 48로 설정
        self.output_conv = nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = self.input_conv(x)
        x = self.residuals(x)
        x = self.output_conv(x)
        
        # pixel_shuffle로 업샘플링
        return F.pixel_shuffle(x, self.scale_factor)

def main():
    scale_factor = 4  # 또는 2로 조정 가능
    model = EDSR(scale_factor)

if __name__ == '__main__':
    main()
