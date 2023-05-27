import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as vision_transforms
import cv2
import os
from blend_modes import multiply
import blend_modes  
from image_utils import is_pil_image
    
tensorToImage = transforms.Compose([
  transforms.ToPILImage()
])
to768 = transforms.Compose([
    transforms.Resize(768),
])

def saveTensorImage(image, filename="image.png"):
  pil_image = tensorToImage(image)
  pil_image.save(filename) 
  
def saveBatch(batchImages, name="result"):
  pilImages = [tensorToImage(img) for img in batchImages]
  for (i, pil_image) in enumerate(pilImages):
    pil_image.save(f"./src/laplace-output/{name}-" + str(i) + ".png") 

def openImage(image_path, mode="RGB"):
  image = Image.open(image_path)
  image = image.convert(mode)
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])
  image = transform(image)
  # image = image.unsqueeze(0)
  return image

# Load image as numpy float arr to be used in blend_modes package
def loadImageAsArray(image_path):
  image = Image.open(image_path)  # RGBA image
  image = np.array(image)  # Inputs to blend_modes need to be numpy arrays.
  image = image.astype(float)  # Inputs to blend_modes need to be floats.
  return image

# Converts (H, W, C) to (C, H, W) and scales from [0, 1] to [0, 255]
def convertTensorToNumpy(image, add_alpha=False):
  image = image.permute(1, 2, 0).numpy()
  image = (image * 255).astype(np.float64) # Scale from [0, 1] to [0, 255]
  if add_alpha:
    alpha_array = np.ones((image.shape[0], image.shape[1], 1)) * 255
    image = np.concatenate((image, alpha_array), axis=2)

  return image

# Converts (H, W, C) to (C, H, W) and scales from [0, 1] to [0, 255]
def convertNumpyToTensor(image):
  image = image.astype(np.uint8)
  image = torch.from_numpy(image)
  image = image.permute(2, 0, 1)
  return image

def convertNumpyToPIL(image):
  image = image.astype(np.uint8)
  image = Image.fromarray(image)
  return image
  
# use blend_modes for loading and returning as numpy arr.
def blendNPImages(image_path, blend_image_path):
  background_img_float = loadImageAsArray( "./src/img/bravo/me768.png")
  foreground_img_float = loadImageAsArray("./src/img/bravo/orangeblue.png")

  img_out = multiply(background_img_float,foreground_img_float,0.8)
  
  blended_img_raw = convertNumpyToPIL(img_out)
  blended_img_raw.save("./src/blends/multiply_opacity.png")
  
  return img_out


def adjustImage(image, brightness=1.5, saturation = 0):
  image = vision_transforms.adjust_brightness(image, brightness)
  image = vision_transforms.adjust_saturation(image, saturation)
  
  return image


blend_functions = {
  'soft_light':blend_modes.soft_light,
  'lighten':blend_modes.lighten_only,
  'dodge' :blend_modes.dodge,
  'addition' :blend_modes.addition,
  'darken' :blend_modes.darken_only,
  'multiply' :blend_modes.multiply,
  'hard light' :blend_modes.hard_light,
  'difference' :blend_modes.difference,
  'subtract' :blend_modes.subtract,
  'grain_extract':blend_modes.grain_extract,
  'grain_merge':blend_modes.grain_merge,
  'divide' :blend_modes.divide,
  'overlay':blend_modes.overlay,
  'normal':blend_modes.normal,
}

def laplace_fix(base_image_path, layer_image_path):
  layer_image = openImage(layer_image_path, 'RGB')

  layer_image_np = convertTensorToNumpy(layer_image, add_alpha=True) # Converts (H, W, 3) to (4, H, W) and scales from [0, 1] to [0, 255]
  
  base_image = openImage(base_image_path, 'RGB')
  base_image =  transforms.Resize((layer_image.shape[1], layer_image.shape[2]))(base_image)
  base_image_np = convertTensorToNumpy(base_image, add_alpha=True) # Converts (H, W, 3) to (4, H, W) and scales from [0, 1] to [0, 255]

  img_out = blend_functions['difference'](base_image_np, layer_image_np, 1.0)
  
  img_out = img_out[:,:,:3] # Drop alpha channel
  img_out = convertNumpyToTensor(img_out)
  img_out = img_out.float()
  threshold = (img_out.mean() + img_out.std()) * 2.5

  mask = (img_out > threshold).float()

  # Create a boolean mask where any of the channels has a non-zero value
  binary_mask = torch.any(mask != 0, dim=0) #.float()
  
  # # # Set all channels to 1.0 for each pixel where the mask is True
  # diff = torch.where(binary_mask, torch.tensor(1.0), mask)
  # saveTensorImage(diff, "./src/diff_outputs/diff.png")
  
  fixed_image = torch.where(binary_mask, base_image, layer_image)
  
  # Convert to PIL Image
  pil_image = tensorToImage(fixed_image)
  
  return pil_image

def laplace_fix_tensor(base_image_path, layer_image):
  layer_image_np = convertTensorToNumpy(layer_image, add_alpha=True) # Converts (H, W, 3) to (4, H, W) and scales from [0, 1] to [0, 255]
  
  base_image = openImage(base_image_path, 'RGB')
  base_image =  transforms.Resize(layer_image.shape[1])(base_image)
  base_image_np = convertTensorToNumpy(base_image, add_alpha=True) # Converts (H, W, 3) to (4, H, W) and scales from [0, 1] to [0, 255]

  
  img_out = blend_functions['difference'](base_image_np, layer_image_np, 1.0)
  img_out = img_out[:,:,:3] # Drop alpha channel
  img_out = convertNumpyToTensor(img_out)
  img_out = img_out.float()
  threshold = (img_out.mean() + img_out.std()) * 2.5
  mask = (img_out > threshold).float()

  # Create a boolean mask where any of the channels has a non-zero value
  binary_mask = torch.any(mask != 0, dim=0) #.float()
  
  # # # Set all channels to 1.0 for each pixel where the mask is True
  # diff = torch.where(binary_mask, torch.tensor(1.0), mask)
  # saveTensorImage(diff, "./src/diff_outputs/diff.png")
  
  fixed_image = torch.where(binary_mask, base_image, layer_image)
  
  # Convert to PIL Image
  pil_image = tensorToImage(fixed_image)
  
  return pil_image

def blend(base_image_path, layer_image_path, blend_mode="multiply", blend_opacity=1.0, base_brightness=1, base_saturation=1):
  if is_pil_image(base_image_path):
    base_image = transforms.ToTensor()(base_image_path)
  else:
    base_image = openImage(base_image_path, 'RGB')
  # Resize base image to Wx768 or Hx768
  # base_image = to768(base_image)
  # Brighten and desaturate base image
  base_image = adjustImage(base_image, brightness=base_brightness, saturation=base_saturation)
  # blend_modes uses numpy arrays with 4 channels (RGBA)
  # saveTensorImage(base_image, "./src/blends/mb_base_image_adjusted.png")
  
  base_image = convertTensorToNumpy(base_image, add_alpha=True) # Converts (H, W, 3) to (4, H, W) and scales from [0, 1] to [0, 255]

  layer_image = openImage(layer_image_path, 'RGB')
  # layer_image = to768(layer_image)
  layer_image = convertTensorToNumpy(layer_image, add_alpha=True) # Converts (H, W, 3) to (4, H, W) and scales from [0, 1] to [0, 255]

  img_out = blend_functions[blend_mode](base_image, layer_image, blend_opacity)
  
  img_out = img_out[:,:,:3] # Drop alpha channel
  img_out = convertNumpyToTensor(img_out)
    
  # Convert to PIL Image
  pil_image = tensorToImage(img_out)
  
  return pil_image


def testAllBlends():
  blend_settings = [
    { "blend_mode": "multiply", "blend_opacity": 1, "base_brightness": 1, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": 0.9, "base_brightness": 1, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": 0.8, "base_brightness": 1, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": 0.7, "base_brightness": 1, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": 1, "base_brightness": 1.25, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": .9, "base_brightness": 1.25, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": .8, "base_brightness": 1.25, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": .7, "base_brightness": 1.25, "base_saturation": 1 },
    { "blend_mode": "multiply", "blend_opacity": 1, "base_brightness": 1.25, "base_saturation": 0.25 },
    { "blend_mode": "multiply", "blend_opacity": .9, "base_brightness": 1.25, "base_saturation": 0.25 },
    { "blend_mode": "multiply", "blend_opacity": .8, "base_brightness": 1.25, "base_saturation": 0.25 },
    { "blend_mode": "multiply", "blend_opacity": .7, "base_brightness": 1.25, "base_saturation": 0.25 },
  ]
  for setting in blend_settings:
    blended_image = blend(
      "./src/img/bravo/me3072.png", 
      "./src/img/bravo/orangeblue.png",
      setting['blend_mode'],
      blend_opacity=setting['blend_opacity'],
      base_brightness=setting['base_brightness'],
      base_saturation=setting['base_saturation']
    )
    print("blended_image", blended_image.shape)
    saveTensorImage(blended_image, "./src/blends/blend_" + setting['blend_mode'] + "_" + str(setting['blend_opacity']) + "_" + str(setting['base_brightness']) + "_" + str(setting['base_saturation']) + ".png")

if __name__ == '__main__':
  blended_image = blend(
    "./src/img/bravo/me3072.png", 
    "./src/img/bravo/orangeblue.png",
    "multiply",
    blend_opacity=1.0,
    base_brightness=1,
    base_saturation=1
  )
  print("blended_image", blended_image.shape)

  saveTensorImage(blended_image, "./src/blends/blend.png")
  
  # testAllBlends()
  # blended_image = blendImages("./src/img/bravo/sophie1536.png", "./src/img/bravo/sophieblend.png")