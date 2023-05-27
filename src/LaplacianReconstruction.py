import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
from LaplacianPyramid import Lap_Pyramid
import torchvision
import torchvision.transforms as transforms
from image_utils import is_pil_image

tensorToImage = transforms.Compose([
  transforms.ToPILImage()
])

def saveBatch(batchImages, name="result"):
  pilImages = [tensorToImage(img) for img in batchImages]
  for (i, pil_image) in enumerate(pilImages):
    pil_image.save(f"./outputs/{name}-" + str(i) + ".png") 
        
def openImage(image_path):
  image = Image.open(image_path)
  image = image.convert('RGB')
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])
  image = transform(image)
  image = image.unsqueeze(0)
  return image
  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def laplacianReconstruct(image_path, alt_image_path=None, pyramid_levels=2):
  # image = openImage(image_path)
  if is_pil_image(image_path):
    image = transforms.ToTensor()(image_path)
    if len(image.shape) == 3:
      image = image.unsqueeze(0)
  else:
    image = openImage(image_path)
  
  alt_image = None
  if alt_image_path:
    # alt_image = openImage(alt_image_path)
    if is_pil_image(alt_image_path):
      alt_image = transforms.ToTensor()(alt_image_path)
    else:
      alt_image = openImage(alt_image_path)

  image = image.to(device=device)
  alt_image = alt_image.to(device=device)
  
  lapPyramid = Lap_Pyramid(num_high=pyramid_levels).to(device)
  pyr, gyr = lapPyramid.pyramid_decom(image)
    
  if alt_image is not None:
    pyr[-1] = alt_image
  pyr_recon = lapPyramid.pyramid_recons(pyr)
  
  
  if (pyr_recon.shape[2] > 2048):
    pyr_recon = transforms.Resize(2048)(pyr_recon)
  
  # saveBatch(pyr_recon, "recon")
  pilImages = [tensorToImage(img) for img in pyr_recon]
  
  if len(pilImages) == 1:
    return pilImages[0]
  
  return pilImages

def getLaplacian(image_path):
  if is_pil_image(image_path):
    image = transforms.ToTensor()(image_path)
    
    if len(image.shape) == 3:
      image = image.unsqueeze(0)
  else:
    image = openImage(image_path)
  

  image = image.to(device=device)
  
  lapPyramid = Lap_Pyramid(num_high=1).to(device)
  pyr, gyr = lapPyramid.pyramid_decom(image)
  
  pilImage = tensorToImage(pyr[0].squeeze(0))

  return pilImage

if __name__ == '__main__':
  lap = getLaplacian("./samples/me768.png")
  lap.save('lap.png')
