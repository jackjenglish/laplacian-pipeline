import torch
import torch.nn as nn
from PIL import Image, PngImagePlugin
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as vision_transforms
import cv2
import os
import io
import piexif
import piexif.helper
import base64

def saveTensorImage(image, filename="image.png"):
  pil_image = transforms.ToPILImage()(image)
  pil_image.save(filename) 
  
def openImage(image_path):
  image = Image.open(image_path)
  image = image.convert('RGB')
  transform = transforms.Compose([
    transforms.ToTensor(),
  ])
  image = transform(image)

  return image

 
def get_pil_metadata(pil_image):
  # Copy any text-only metadata
  metadata = PngImagePlugin.PngInfo()
  for key, value in pil_image.info.items():
      if isinstance(key, str) and isinstance(value, str):
          metadata.add_text(key, value)

  return metadata

# https://github.com/gradio-app/gradio/blob/main/gradio/processing_utils.py
def encode_pil_to_base64(pil_image):
  with io.BytesIO() as output_bytes:
    pil_image.save(output_bytes, "PNG", pnginfo=get_pil_metadata(pil_image))
    bytes_data = output_bytes.getvalue()
  base64_str = str(base64.b64encode(bytes_data), "utf-8")
  return "data:image/png;base64," + base64_str

def tensor_to_pil(tensor_image):
  return transforms.ToPILImage()(tensor_image)

def is_pil_image(input):
  return isinstance(input, Image.Image)