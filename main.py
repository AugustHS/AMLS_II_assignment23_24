import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import glob
from piq import psnr,ssim
from model import Model_A,Model_B

# ======================================================================================================================
# Data preprocessing
# The training set part is omitted due to the model is already trained in advance
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
 

VAL_HR_PATH = './DIV2K_valid_HR'
VAL_LR_PATH_A = './DIV2K_valid_LR_bicubic_X2/X2'
VAL_LR_PATH_B = './DIV2K_valid_LR_unknown_X2/X2'

val_dataset_a = DIV2KDataset(lr_path=VAL_LR_PATH_A, hr_path=VAL_HR_PATH)
val_dataset_b = DIV2KDataset(lr_path=VAL_LR_PATH_B, hr_path=VAL_HR_PATH)

val_loader_a = DataLoader(val_dataset_a, batch_size=64, shuffle=True)
val_loader_b = DataLoader(val_dataset_b, batch_size=64, shuffle=True)

# ======================================================================================================================
# Task A(Bicubic x2)
# the model has been pre-trained to save time
model_a = Model_A()
model_a.load_state_dict(torch.load('Model_A_X2_state_dict.pth'))
psnr_values_a = []
ssim_values_a = []

model_a.eval()
with torch.no_grad():
    for lr_imgs_val, hr_imgs_val in val_loader_a:

        outputs_val = model_a(lr_imgs_val)
        
        outputs_val_clamped = torch.clamp(outputs_val, min=0.0, max=1.0)

        psnr_value = psnr(outputs_val_clamped, hr_imgs_val, data_range=1.0).item()
        ssim_value = ssim(outputs_val_clamped, hr_imgs_val, data_range=1.0).mean().item()

        psnr_values_a.append(psnr_value)
        ssim_values_a.append(ssim_value)

avg_psnr_a = sum(psnr_values_a) / len(psnr_values_a)
avg_ssim_a = sum(ssim_values_a) / len(ssim_values_a)

# ======================================================================================================================
# Task B(Unknown x2)
# the model has been pre-trained to save time
model_b = Model_B()
model_b.load_state_dict(torch.load('Model_B_X2_state_dict.pth'))
psnr_values_b = []
ssim_values_b = []

model_b.eval()
with torch.no_grad():
    for lr_imgs_val, hr_imgs_val in val_loader_b:

        outputs_val = model_b(lr_imgs_val)
        
        outputs_val_clamped = torch.clamp(outputs_val, min=0.0, max=1.0)

        psnr_value = psnr(outputs_val_clamped, hr_imgs_val, data_range=1.0).item()
        ssim_value = ssim(outputs_val_clamped, hr_imgs_val, data_range=1.0).mean().item()

        psnr_values_b.append(psnr_value)
        ssim_values_b.append(ssim_value)

avg_psnr_b = sum(psnr_values_b) / len(psnr_values_b)
avg_ssim_b = sum(ssim_values_b) / len(ssim_values_b)

# ======================================================================================================================
## Print out your results with following format:
print(f"Model A(Bicubic x2): Average PSNR: {avg_psnr_a}, Average SSIM: {avg_ssim_a}")
print(f"Model B(Unknown x2): Average PSNR: {avg_psnr_b}, Average SSIM: {avg_ssim_b}")
