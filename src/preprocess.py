
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
from math import sqrt
from image_utils import saveTensorImage, openImage, tensor_to_pil


def center_crop_nearest_multiple_of_64(img_tensor):
    # Get the current dimensions of the image tensor
    height, width = img_tensor.shape[-2:]

    # Round the dimensions to the nearest multiple of 64
    new_height = round(height // 64) * 64
    new_width = round(width // 64) * 64
    

    # Apply the center crop
    transform = transforms.CenterCrop((new_height, new_width))
    cropped_tensor = transform(img_tensor)

    return cropped_tensor
  
  
# SUPPORTED_RATIOS = [1, 1.25, 1.33, 1.5, 1.66, 2]  # List of supported aspect ratios

def crop_to_supported_ratio(img_tensor, ratios = [1, 1.25, 1.33, 1.5, 1.66, 2]):
    # Get the current aspect ratio of the image tensor
    height, width = img_tensor.shape[-2:]
    aspect_ratio = width / height

    # Find the supported aspect ratio that is closest to the current aspect ratio
    closest_ratio = min(ratios, key=lambda x: abs(x - aspect_ratio))
    print("closest ratio", closest_ratio)
    # Calculate the new dimensions of the image tensor that match the closest supported aspect ratio
    new_width = round(closest_ratio * height)
    new_height = round(new_width / closest_ratio)
    # print('new width', new_width, 'new_height', new_height)
    # Apply the center crop with the new dimensions
    transform = transforms.CenterCrop((new_height, new_width))
    cropped_tensor = transform(img_tensor)

    return cropped_tensor

  
def preprocess(image):
  image = image.convert('RGB')
  
  image = transforms.ToTensor()(image)

  downsized = transforms.Resize(768, max_size=1024)(image)  

  downsized = center_crop_nearest_multiple_of_64(downsized)
  
  aspect = downsized.shape[2] / downsized.shape[1]

  diff_factor = round(image.shape[2] / downsized.shape[2])

  recon_levels = round(sqrt(diff_factor))
  upscale_factor = 2 ** recon_levels
  high_image_width =  downsized.shape[2] * upscale_factor
  high_image_height = downsized.shape[1] * upscale_factor
  
  high_res_image = crop_to_supported_ratio(image, [aspect])
  # print("high_image match ar", high_res_image.shape)
  
  high_res_image = transforms.Resize((high_image_height, high_image_width))(high_res_image)
  
  saveTensorImage(downsized, 'downsized.png')
  saveTensorImage(high_res_image, 'high_image.png')
  
  return (tensor_to_pil(downsized), tensor_to_pil(high_res_image), recon_levels)
  
if __name__ == "__main__":
  image = openImage('./samples/me2100.png')

  