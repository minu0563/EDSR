import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import Dataset

class DIV2KDataset(Dataset):
    def __init__(self, images_dir, scale):
        self.images_dir = images_dir
        self.scale = scale
        self.image_names = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        self.fixed_size = (512, 512)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        img_path = os.path.join(self.images_dir, image_name)
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.fixed_size, Image.BICUBIC)
        lr_img = img.resize((img.width // self.scale, img.height // self.scale), Image.BICUBIC)

        hr_tensor = self.transform(img)
        lr_tensor = self.transform(lr_img)

        return lr_tensor, hr_tensor
