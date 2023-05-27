import boto3
import aioboto3
import os
from boto3.dynamodb.conditions import Key
from botocore.config import Config
from botocore.exceptions import NoCredentialsError, ClientError
from dotenv import load_dotenv, find_dotenv
import os
import sys
import threading
import time
import io

load_dotenv(find_dotenv())

ACCESS_KEY_ID = os.environ['AWS_ACCESS_KEY_ID']
SECRET_ACCESS_KEY = os.environ['AWS_SECRET_ACCESS_KEY']
AWS_REGION = os.environ["AWS_REGION"]
        
config = Config(
  connect_timeout=2, read_timeout=2,
  retries={'max_attempts': 10}
)

dynamodb = boto3.resource(
  'dynamodb',
  aws_access_key_id = ACCESS_KEY_ID,
  aws_secret_access_key = SECRET_ACCESS_KEY,
  region_name=AWS_REGION,
  config=config
)


def storeResult(id, result):
  table = dynamodb.Table("caption-results")
  response = table.put_item(
    Item={
      "id": id,
      "caption": result
    }
  )
  print("storeResult response: ", response)


class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(os.path.getsize(filename))
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()
            
class DownloadProgressPercentage(object):
    def __init__(self, client, bucket, filename):
        self._filename = filename
        # print('download %', bucket, filename, client.head_object(Bucket=bucket, Key=filename))
        self._size = client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        # To simplify, assume this is hooked up to a single filename
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s  %s / %s  (%.2f%%)" % (
                    self._filename, self._seen_so_far, self._size,
                    percentage))
            sys.stdout.flush()

def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """
    print("upload_file: ", file_name, bucket, object_name)
    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
    try:
        response = s3_client.upload_file(
          file_name,
          bucket,
          object_name,
          ExtraArgs={'ACL':'public-read'},
          Callback=ProgressPercentage(file_name)
        )
    except ClientError as e:
        print(e)
        return False
    return True
  
def upload_fileobj(file_data, bucket, object_name=None, extra_args={'ACL':'public-read'}):
    # Upload the file
    # Let's time this.
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
    try:
        response = s3_client.upload_fileobj(
          file_data,
          bucket,
          object_name,
          ExtraArgs=extra_args,
          # Callback=ProgressPercentage(file_name)
        )
    except ClientError as e:
        print(e)
        return False
    return True

async def upload_fileobj_async(file_data, bucket, object_name=None, extra_args={'ACL':'public-read'}):
  print("upload_file: ", bucket, object_name)
  # Upload the file
  # Let's time this.
  session = aioboto3.Session(aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
  async with session.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY) as s3:
    try:
      response = await s3.upload_fileobj(
        file_data,
        bucket,
        object_name,
        ExtraArgs=extra_args,
        # Callback=ProgressPercentage(file_name)
      )
    except ClientError as e:
      print(e)
      return False
  return True

def download_file(bucket, file_name, local_file_name=None):
    """Download a file from an S3 bucket

    :param bucket: Bucket to download from
    :param file_name: S3 object name. If not specified then file_name is used
    :param local_file_name: Local file name. If not specified then file_name is used
    :return: True if file was downloaded, else False
    """
    # If S3 object_name was not specified, use file_name
    if local_file_name is None:
        local_file_name = os.path.basename(file_name)

    # Download the file
    s3_client = boto3.client('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
    try:
        start = time.time()
        response = s3_client.download_file(bucket, file_name, local_file_name, Callback=DownloadProgressPercentage(s3_client, bucket, file_name))
        end = time.time()
        print(f"Download of {file_name}: Took => ", end-start)
    except ClientError as e:
        print(e)
        return False
    return True

def uploadThread(file_name, bucket, object_name=None):
  uploadWorker = threading.Thread(target=upload_file, args=(file_name, bucket, object_name))
  uploadWorker.start()
  # uploadWorker.join()
  

# Upload/Download Image Sets
from uuid import uuid4
import shortuuid


def uploadFolder():
  folder = shortuuid.uuid()
  for fileName in os.listdir('./inputImages'):
    print(fileName)
    file_extension = os.path.splitext(fileName)[1]
    # Does order of images the model trains on matter? uuid breaks that if so.
    upload_file(f"./inputImages/{fileName}", "mirror-uploads", f"{folder}/{shortuuid.uuid()}{file_extension}")
    print("uploaded", fileName)
    

def downloadBucketFolder(bucket, folder):
  s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
  bucket = s3.Bucket('mirror-uploads')
  for obj in bucket.objects.filter(Prefix=folder):
    print(obj.key)
    fileName = obj.key.split(folder + "/")[1]
    download_file("mirror-uploads", obj.key, f"./uploadedImages/{fileName}")
    # print("downloaded", obj.key)

def listBucketFolder(bucket, folder):
  s3 = boto3.resource('s3', aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=SECRET_ACCESS_KEY)
  bucket = s3.Bucket(bucket)
  for obj in bucket.objects.filter(Prefix=folder):
    fileName = obj.key
    if folder:
      fileName = obj.key.split(folder + "/")[1]
    print(fileName)
