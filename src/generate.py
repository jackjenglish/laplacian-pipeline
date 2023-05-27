import requests
import shortuuid
from PIL import Image
import io
import base64
from base64 import b64encode
import cv2
from datetime import datetime
import numpy as np
import json
import os
from blending import blend, laplace_fix, laplace_fix_tensor
from LaplacianReconstruction import laplacianReconstruct

def readImage(path):
    img = cv2.imread(path)
    retval, buffer = cv2.imencode('.jpg', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img

prompts = [
  "headshot portrait photo of (jack) in space, sharp focus, detailed, shot by Fred Herzog, cinematic, great skin|super HD face",
  "portrait of (Jack) as Christian Grey, fifty shades of grey, key art, falling, opulent, highly detailed, digital painting, artstation, concept art, cinematic lighting, sharp focus, illustration, by gaston bussiere alphonse mucha , Jack illustration, symmetrical face, sharp jawline, gigachad, ultimate alpha male, full body, ((clear highly detailed photorealistic)), handsome, chad",
  "portrait illustration of jack in a suit, cinematic lighting, sharp jawline, ultimate alpha male, full body, ((clear highly detailed photorealistic)), falling, dominant, highly detailed, digital, highly detailed photorealistic key art, photorealistic illustration",
  "professional side profile of jack man in a biker jacket, cinematic lighting, sharp jawline, ultimate alpha male, full body, ((clear highly detailed photorealistic)), falling, dominant, highly detailed, digital, highly detailed photorealistic key art, photorealistic illustration"
  "headshot portrait photo of (jack) shot by Fred Herzog, cinematic",
  "portrait photo of (jack) shot by Fred Herzog, movie still, contest winner, great skin, oscar winner"
]
negativeA = "low quality, fake, painting, greyscale, night, beard, deformed, out of frame, (((cleft lip))), ((((ugly)))), (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), ((blurry)), ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), (((text))), (((title))), (((smooth))), (((deformed lips))), (((wrinkles))), (((feminine)))"

control_models = {
  'canny': 'control_canny-fp16 [e3fe7712]', 
  'depth': 'control_depth-fp16 [400750f6]',
  'hed': 'control_hed-fp16 [13fee50b]', 
  'laplace': 'control_laplace-fp16 [8f2576c7]',
  'mlsd': 'control_mlsd-fp16 [e3705cfa]',
  'normal': 'control_normal-fp16 [63f96f7c]',
  'openpose': 'control_openpose-fp16 [9ca67cc5]',
  'scribble': 'control_scribble-fp16 [c508311e]',
  'seg': 'control_seg-fp16 [b9c1cc12]',
  't2i_canny': 't2iadapter_canny-fp16 [f2e7f7cd]',
  't2i_color': 't2iadapter_color-fp16 [743b5c62]',
  't2i_depth': 't2iadapter_depth-fp16 [2c829a81]',
  't2i_keypose': 't2iadapter_keypose-fp16 [e3943bb9]',
  't2i_openpose': 't2iadapter_openpose-fp16 [4286314e]',
  't2i_seg': 't2iadapter_seg-fp16 [0e677718]',
  't2i_sketch': 't2iadapter_sketch-fp16 [75b15924]',
  't2i_style': 't2iadapter_style-fp16 [0e2e8330]',
  't2i_style_sd14v1': 't2iadapter_style_sd14v1 [202e85cc]'
}

def fileUUID():
    ts = datetime.timestamp(datetime.now())
    image_uuid = shortuuid.uuid()
    return f'{datetime.timestamp(datetime.now())}-{image_uuid[:10]}'

def getControlNetList():
  CNListResponse = requests.get(url=f'http://127.0.0.1:7860/controlnet/model_list')
  print(CNListResponse.json())

def txt2img(data):
  response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/txt2img', json=data)
  r = response.json()
  # Save Grid Only
  control_count = len(data['controlnet'])
  print('number of controls', control_count)
  excludeControlMasks = False
  ts = datetime.timestamp(datetime.now())
  # print(r['parameters'])
  results = []
  
  def saveImage(i):
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    # save each image with unique id
    image_uuid = shortuuid.uuid()
    print('save Image', image_uuid)
    image_name = f'./outputs/{datetime.timestamp(datetime.now())}-{image_uuid[:10]}.png'
    results.append(image_name)
    image.save(image_name)

  if excludeControlMasks:
    for i in r['images'][:-control_count]:
      saveImage(i)
  else:
    for i in r['images']:
      saveImage(i)

  return results

def img2img(data):
  response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/img2img', json=data)
  r = response.json()
  # Save Grid Only
  control_count = len(data['controlnet'])
  print('number of controls', control_count)
  excludeControlMasks = True
  ts = datetime.timestamp(datetime.now())
  # print(r['parameters'])
  results = []
  
  def saveImage(i):
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    # save each image with unique id
    image_uuid = shortuuid.uuid()
    print('save Image', image_uuid)
    image_name = f'./outputs/{datetime.timestamp(datetime.now())}-{image_uuid[:10]}.png'
    results.append(image_name)
    image.save(image_name)

  if excludeControlMasks:
    for i in r['images'][:-control_count]:
      saveImage(i)
  else:
    for i in r['images']:
      saveImage(i)

  return results

def decode_base64_to_image(encoding):
  if encoding.startswith("data:image/"):
      encoding = encoding.split(";")[1].split(",")[1]
  try:
      image = Image.open(io.BytesIO(base64.b64decode(encoding)))
      return image
  except Exception as err:
      raise err
  
def to_base64_nparray(encoding: str):
  return np.array(decode_base64_to_image(encoding)).astype('uint8')


def txt2imgBlend(result_path, downsized, full_res, recon_levels):
  blended_image = blend(
    downsized, 
    result_path,
    "multiply",
    blend_opacity=0.95,
    base_brightness=2.5,
    base_saturation=0.15
  )
  blended_image_path = f"./outputs/blended-{os.path.basename(result_path)}"
  blended_image.save(blended_image_path)
  
  recons = laplacianReconstruct(full_res, blended_image_path, pyramid_levels=recon_levels)
  recon_file = f"./outputs/recon-{os.path.splitext(os.path.basename(result_path))[0]}.jpg"
  recons.save(recon_file, quality=90)
  
  fixed_recon = laplace_fix(
    blended_image_path, 
    recon_file
  )
  
  return fixed_recon

  
if __name__ == '__main__':
  input_image_path = "./samples/raw1024.png"
  highres_image_path = "./samples/raw4096.png"
  upscale_levels = 2

  input_image = readImage(input_image_path)

  payload = {
    'prompt': 'photo, cinematic depth, blood on face, brutal warrior, canon eos r3, movie poster, young white man with dark hair and blood on his face, dark gritty movie, dark film look, cut up face',
    'prompt': 'photo, cinematic depth, cool blue tint, canon eos r3, movie poster, young white man with dark hair, tungsten subject, dark bluish background, dark gritty movie, dark film look, cut up face',
    "negative_prompt": negativeA,
    "steps": 15,
    "batch_size": 3,
    "cfg_scale": 12,
    "sampler_index": "DDIM", 
    "width": 1024,
    "height": 768,
    "controlnet": [{
      "input_image": input_image,
      "model": control_models['canny'],
      "module": "canny", 
      "weight" : 0.95, 
      "processor_res" : 1024,
      "threshold_a": 32,
      "threshold_b": 64,
    }],
  }

  # Text to image generation with laplacian controlnet
  results = txt2img(payload)

  for result in results:
    # Multiply blend the generation with the grayscale input image
    blended_image = blend(
      input_image_path, 
      result,
      "multiply",
      blend_opacity=0.9,
      base_brightness=2,
      base_saturation=0.2
    )
    blended_image_path = f"./outputs/blended-{os.path.basename(result)}"
    blended_image.save(blended_image_path)
    
    # Reconstruction of the blended image using laplacian reconstruction on the blended image.
    recons = laplacianReconstruct(highres_image_path, blended_image_path, pyramid_levels=upscale_levels)
    
    recon_file = f"./outputs/recon-{os.path.splitext(os.path.basename(result))[0]}.jpg"
    recons.save(recon_file, quality=90)
    
    # Screen out the laplacian reconstruction errors.
    fixed_recon = laplace_fix(
      blended_image_path, 
      recon_file
    )
    fixed_recon_file = f"./outputs/recon-fixed-{os.path.splitext(os.path.basename(result))[0]}.jpg"
    
    fixed_recon.save(fixed_recon_file, quality=90)
