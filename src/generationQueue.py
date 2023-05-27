

# https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on
# if len(sys.argv[1:]) != 0:
#     print("Job Handler: Setting CUDA_VISIBLE_DEVICES to {}".format(sys.argv[1]))
#     os.environ["CUDA_VISIBLE_DEVICES"] =  sys.argv[1]
# else:
#     sys.exit("Job Handler: Missing CUDA_VISIBLE_DEVICES!")
  
import traceback

import pymongo
import requests
import shortuuid
from PIL import Image
import io
import base64
import os
import subprocess
import sys
import GPUtil
from pynvml import *
import time
from s3Manager import upload_file, upload_fileobj, upload_fileobj_async
from generate import txt2imgBlend
from preprocess import preprocess
from image_utils import encode_pil_to_base64
from LaplacianReconstruction import getLaplacian

uri = "mongodb+srv://jjenglish:Zo0WsTJOIoZbhu7x@mirrorcluster.fmo1e.mongodb.net/?retryWrites=true&w=majority"

# MongoDB Quick Start Guide, find, insert, update, delete
# https://docs.mongodb.com/drivers/python/
def connect():
    client = pymongo.MongoClient(uri)
    db = client.mirror
    return db

def fetchQueue():
  db = connect()
  queue = db['generation-queue']
  result = queue.find_one_and_delete({
  }, sort=[('_id', pymongo.ASCENDING)])
  return result

settings_keys = [
  'prompt',
  'negative_prompt',
  'model_name',
  'steps',
  'width',
  'height',
  'batch_size',
  'n_iter',
  'enable_hr',
  'firstphase_width',
  'firstphase_height',
  'init_images',
  'denoising_strength',
  'seed',
  "grid",
  "grid_x_type",
  "grid_x_values",
  "grid_x_values",
  "grid_y_type",
  "grid_y_values",
  "grid_y_values",
  "prompt_matrix",
  "controlnet"
]

def base64ToImage(b64_image):
  return Image.open(io.BytesIO(base64.b64decode(b64_image)))


def downloadImage(url):
  response = requests.get(url)
  # Encode the image as a base64 string
  pil_image = Image.open(io.BytesIO(response.content))
  
  if hasattr(pil_image, '_getexif'):
    exif = pil_image._getexif()
    if exif:
      orientation = exif.get(0x0112)
      if orientation == 3:
        pil_image = pil_image.transpose(Image.ROTATE_180)
      elif orientation == 6:
        pil_image = pil_image.transpose(Image.ROTATE_270)
      elif orientation == 8:
        pil_image = pil_image.transpose(Image.ROTATE_90)
  
  return pil_image

def saveLocally(images):
  saveGridOnly = True
  if saveGridOnly:
    i = r['images'][0]
    image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
    # save each image with unique id
    image_uuid = shortuuid.uuid()
    print('save Image', image_uuid)
    image.save(f'outputs/{image_uuid}.png')
  else:
    for i in r['images']:
      image = Image.open(io.BytesIO(base64.b64decode(i.split(",",1)[0])))
      # save each image with unique id
      image_uuid = shortuuid.uuid()
      print('save Image', image_uuid)
      image.save(f'outputs/{image_uuid}.png')

import asyncio


def requestGeneration(settings, downsized, full_res, recon_levels, img2img= False):
  safe_keys = [key for key in settings if key in settings_keys]
  clean_settings = { key: settings[key] for key in safe_keys }
  endpoint = 'img2img' if img2img else 'txt2img'
  response = requests.post(url=f'http://127.0.0.1:7860/sdapi/v1/{endpoint}', json=clean_settings)
  r = response.json()
  job_id = settings['_id']
  start = time.time()
  results = r['images'][:-len(settings['controlnet'])] # Exclude control
  
  async def upload_result(index, image):
    image_obj = Image.open(io.BytesIO(base64.b64decode(image.split(",",1)[0])))
    image_obj = image_obj.convert("RGB")

    file_name = f'{job_id}/{index}.jpg'
    if index == len(results) - 1:
      file_name = f'{job_id}/final.jpg'
 
    image_name = f'./outputs/{job_id}-{index}.png'
    image_obj.save(image_name)
    final_image = txt2imgBlend(image_name, downsized, full_res, recon_levels)
    
    image_bytes_jpg = io.BytesIO()
    final_image.save(image_bytes_jpg, "JPEG", quality=90)
    image_bytes_jpg.seek(0)
    
    await upload_fileobj_async(image_bytes_jpg, 'mirror-results', file_name, {'ACL':'public-read', 'ContentType':'image/jpg'})

  # Create an asyncio event loop
  loop = asyncio.get_event_loop()
  # Create a list of tasks to upload the files
  tasks = [upload_result(index, image) for index, image in enumerate(results)]
  # Run the tasks concurrently
  loop.run_until_complete(asyncio.wait(tasks))
  print("jobId", job_id, 'uploaded', len(results), 'images', 'in', time.time() - start, 'seconds')


class Model:
  def __init__(self):
    self.isBusy = False
    
  def run(self, settings, downsized, full_res = None, recon_levels = 2, img2img = False):
    self.isBusy = True
    requestGeneration(settings, downsized, full_res, recon_levels, img2img)
    self.isBusy = False

model = Model()

def runModel(model, settings):
  try:
    start = time.time()
    print("Run:")

    if settings['input_image_src']:
      input_image = downloadImage(settings['input_image_src'])
  
      downsized, full_res, recon_levels = preprocess(input_image)

      settings['width'] = downsized.width
      settings['height'] = downsized.height
      # img2img
      if settings['img2img']:
        settings["init_images"] = [encode_pil_to_base64(downsized)]
      
      if settings['controlnet']:
        for control in settings['controlnet']:
          if control['module'] == 'canny':
            control['input_image'] = encode_pil_to_base64(downsized)
      
          if control['model'] == "control_laplace-fp16 [8f2576c7]":
            control['input_image'] = encode_pil_to_base64(getLaplacian(downsized))
      model.run(settings, downsized, full_res, recon_levels, settings['img2img'])
    else:
      model.run(settings)
    end = time.time()
    print("Job Handler: Job Took => ", end-start)
  except Exception as e:
    traceback.print_exc()
    print("Main loop of job handler failed: {}".format(e))
    model.isBusy = False

def schedule(model):
  while True:
    if not model.isBusy:
        try:
          # message = queue.dequeue()
          jobMessage = fetchQueue()
          if not jobMessage is None:
            # showUtilizationGPU()
            print("Job Handler: Dequeued message: {}".format(jobMessage))
            # print("Job Handler: Running on GPU : {}".format(sys.argv[1]))
            try:
              runModel(model, jobMessage)
            except Exception as e:
              print("GenerateJob Handler: Job Failed: {}".format(e))
              model.isBusy = False

            print('--------------------------')

          else:
            print("\nMongoQueue Job Handler: No message found in queue")
        except Exception as e:
          print("\nMongoQueue Job Handler Error:", e, jobMessage)
          model.isBusy = False
    time.sleep(0.75)


schedule(model)
