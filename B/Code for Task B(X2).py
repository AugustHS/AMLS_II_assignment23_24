# import necessary libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from piq import psnr,ssim

# load and preprocess the data
class DIV2KDataset(Dataset):
    def __init__(self, lr_path, hr_path, crop_size_hr=192, scale_factor=2):
        self.lr_path = lr_path
        self.hr_path = hr_path
        self.crop_size_hr = crop_size_hr
        self.scale_factor = scale_factor
        
        self.lr_images = glob.glob(f'{lr_path}/*.png')
        self.hr_images = glob.glob(f'{hr_path}/*.png')
        
        self.crop_size_lr = crop_size_hr // self.scale_factor

        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.lr_transform = transforms.Compose([
            transforms.Resize((crop_size_hr,crop_size_hr), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.lr_images)

    def __getitem__(self, idx):
        lr_img_path = self.lr_images[idx]
        hr_img_path = self.hr_images[idx]

        lr_image = Image.open(lr_img_path)
        hr_image = Image.open(hr_img_path)

        i, j, h, w = transforms.RandomCrop.get_params(
            hr_image, output_size=(self.crop_size_hr, self.crop_size_hr))
        hr_image = TF.crop(hr_image, i, j, h, w)
        
        lr_image = TF.crop(lr_image, i // self.scale_factor, j // self.scale_factor, self.crop_size_lr, self.crop_size_lr)
        
        lr_image = self.lr_transform(lr_image)
        hr_image = self.hr_transform(hr_image)

        return lr_image, hr_image

# prepare the dataset for training   
TRAIN_HR_PATH = './DIV2K_train_HR'
TRAIN_LR_PATH = './DIV2K_train_LR_unknown/X2'
VAL_HR_PATH = './DIV2K_valid_HR'
VAL_LR_PATH = './DIV2K_valid_LR_unknown_X2/X2'
train_dataset = DIV2KDataset(lr_path=TRAIN_LR_PATH, hr_path=TRAIN_HR_PATH)
val_dataset = DIV2KDataset(lr_path=VAL_LR_PATH, hr_path=VAL_HR_PATH)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# construct the model B
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return x

# use GPU for training   
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
train_losses = []
val_losses = []

# train the model
for epoch in range(num_epochs):
    model.train()
    running_train_loss = 0.0

    for lr_imgs, hr_imgs in train_loader:
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        optimizer.zero_grad()
        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item()
    
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for lr_imgs_val, hr_imgs_val in val_loader:
            lr_imgs_val = lr_imgs_val.to(device)
            hr_imgs_val = hr_imgs_val.to(device)
            
            outputs_val = model(lr_imgs_val)
            val_loss = criterion(outputs_val, hr_imgs_val)
            
            running_val_loss += val_loss.item()
    
    epoch_train_loss = running_train_loss / len(train_loader)
    epoch_val_loss = running_val_loss / len(val_loader)
    train_losses.append(epoch_train_loss)
    val_losses.append(epoch_val_loss)
    
    print(f"Epoch {epoch+1}, Training Loss: {epoch_train_loss}, Validation Loss: {epoch_val_loss}")

# calculate PSNR and SSIM
psnr_values = []
ssim_values = []

model.eval()
with torch.no_grad():
    for lr_imgs_val, hr_imgs_val in val_loader:
        lr_imgs_val = lr_imgs_val.to(device)
        hr_imgs_val = hr_imgs_val.to(device)

        outputs_val = model(lr_imgs_val)
        
        outputs_val_clamped = torch.clamp(outputs_val, min=0.0, max=1.0)

        psnr_value = psnr(outputs_val_clamped, hr_imgs_val, data_range=1.0).item()
        ssim_value = ssim(outputs_val_clamped, hr_imgs_val, data_range=1.0).mean().item()

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

avg_psnr = sum(psnr_values) / len(psnr_values)
avg_ssim = sum(ssim_values) / len(ssim_values)
print(f"Average PSNR: {avg_psnr}, Average SSIM: {avg_ssim}")