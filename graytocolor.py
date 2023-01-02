# import required libraries
import glob
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from skimage import color
from skimage.transform import resize
import gdown
import os
import glob
import random

# import custom functions
import functions

# landscape gray & color data download
url = "https://drive.google.com/uc?id=1-K9o_YAGJbQeyFPR9CXhRLcgyex6bVjy"

gdown.download(url, 'landscape_data.zip', quiet = False)

with zipfile.ZipFile('landscape_data.zip', 'r') as z_fp:
  z_fp.extractall('./landscape_data')

# checking data type
temp = Image.open("./landscape_data/color/0.jpg")
temp = np.array(temp)
# image size = 150 x 150
print(f"image size : {temp.shape}")
# image max = 255 / min = 0
print(f"image max / min value : {temp.max(), temp.min()}")

# write directory address
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "landscape_data")
x_dir = os.path.join(data_dir, "gray")
y_dir = os.path.join(data_dir, "color")

# checking amount of data > 7129 / 7129
x_image_files = [fname for fname in os.listdir(x_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(x_image_files)
y_image_files = [fname for fname in os.listdir(y_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(y_image_files)

# remove Image of different size
for path in x_image_files:
  temp = np.array(Image.open(os.path.join(x_dir, path))).shape
  if temp != (150, 150):
    os.remove(os.path.join(x_dir, path))
  else:
    continue

x_image_files = [fname for fname in os.listdir(x_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(x_image_files)

for path in y_image_files:
  if path not in x_image_files:
    os.remove(os.path.join(y_dir, path))
  else:
    continue

y_image_files = [fname for fname in os.listdir(y_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(y_image_files)

for path in y_image_files:
  temp = np.array(Image.open(os.path.join(y_dir, path))).shape
  if temp != (150, 150, 3):
    print(path)
  else:
    continue

x = np.array([np.array(Image.open(os.path.join(x_dir, image))) for image in x_image_files])
x = x.reshape(-1, 150, 150, 1)
y = np.array([np.array(Image.open(os.path.join(y_dir, image))) for image in y_image_files])
print(f"x_shape / y_shape : {x.shape, y.shape}")

x, y = x/x.max(), y/y.max()
print(x.shape, x.max(), x.min(), y.shape, y.max(),y.min())

idx_shuffle_list = list(range(x.shape[0]))
random.shuffle(idx_shuffle_list)

