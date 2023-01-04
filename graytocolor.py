# Import required libraries
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
from sklearn.model_selection import train_test_split
import math
from tensorflow.keras.preprocessing import image_dataset_from_directory
from keras.layers import Conv2D, Conv2DTranspose, Input, Dropout, MaxPool2D, concatenate, Activation, \
  BatchNormalization, GlobalAvgPool2D, Flatten, Reshape, Dense
from keras.models import Model, Sequential, load_model
from tensorflow.keras.applications import *

# Import custom functions
import functions

# Setting value
AUTOTUNE = tf.data.AUTOTUNE
img_size = 150
n_train = 6500
n_val = 7106 - 6500
LR = 0.0001
n_batch = 32
n_epoch = 100


# Landscape gray & color datasets download
url = "https://drive.google.com/uc?id=1-K9o_YAGJbQeyFPR9CXhRLcgyex6bVjy"

gdown.download(url, 'landscape_data.zip', quiet = False)

with zipfile.ZipFile('landscape_data.zip', 'r') as z_fp:
  z_fp.extractall('./landscape_data')

# Checking data type
temp = Image.open("./landscape_data/color/0.jpg")
temp = np.array(temp)
# image size = 150 x 150
print(f"image size : {temp.shape}")
# image max = 255 / min = 0
print(f"image max / min value : {temp.max(), temp.min()}")

# Write directory address
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "landscape_data")
x_dir = os.path.join(data_dir, "gray")
y_dir = os.path.join(data_dir, "color")

# Checking amount of data > 7129 / 7129
x_image_files = [fname for fname in os.listdir(x_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(x_image_files)
y_image_files = [fname for fname in os.listdir(y_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(y_image_files)

# Remove Images of different size
for path in x_image_files:
  temp = np.array(Image.open(os.path.join(x_dir, path))).shape
  if temp != (img_size, img_size):
    os.remove(os.path.join(x_dir, path))
  else:
    continue

# Update gray data list and check amount
x_image_files = [fname for fname in os.listdir(x_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(x_image_files)

# Contrast the datasets of color and gray images to prevent datasets are different situation
for path in y_image_files:
  if path not in x_image_files:
    os.remove(os.path.join(y_dir, path))
  else:
    continue

# Update color data list and check amount
y_image_files = [fname for fname in os.listdir(y_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(y_image_files)

# Additionally, reconfirm that there are files with different image sizes in the color datasets
for path in y_image_files:
  temp = np.array(Image.open(os.path.join(y_dir, path))).shape
  if temp != (img_size, img_size, 3):
    print(path)
  else:
    continue

# Load data
x = np.array([np.array(Image.open(os.path.join(x_dir, image))) for image in x_image_files])
# To match with the shape of the color datasets, adjust shape of gray datasets
x = x.reshape(-1, img_size, img_size, 1)
y = np.array([np.array(Image.open(os.path.join(y_dir, image))) for image in y_image_files])
print(f"x_shape / y_shape : {x.shape, y.shape}")

# Data scailing between 0 and 1
x, y = x/x.max(), y/y.max()
print(x.shape, x.max(), x.min(), y.shape, y.max(),y.min())

# Split train dataset and validation dataset
train_x, val_x, train_y, val_y = train_test_split(x, y, test_size = 0.1, shuffle = True, random_state = 32)

# Checking image
functions.show(train_x[:5].transpose(1,0,2,3).reshape(150,-1,1))
functions.show(train_y[:5].transpose(1,0,2,3).reshape(150,-1,3))

# Make a Customdataset
train_ds = functions.Dataloader(train_x, train_y, n_batch, shuffle = True)
validation_ds = functions.Dataloader(val_x, val_y, n_batch)

# Using Customschedule class to make lr warmup, cosine decay
steps_per_epoch = n_train // n_batch
lr_schedule = functions.CustomSchedule(LR, 3 * steps_per_epoch, n_epoch * steps_per_epoch)

### Make first model U-Net
def conv2d_block(x, channel):
  x = Conv2D(channel, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  x = Conv2D(channel, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  return x

def unet_like():
  inputs = Input((150, 150, 1))

  con_1 = conv2d_block(inputs, 32)
  pool_1 = MaxPool2D((2, 2))(con_1)
  pool_1 = Dropout(0.1)(pool_1)

  con_2 = conv2d_block(pool_1, 64)
  pool_2 = MaxPool2D((2, 2))(con_2)
  pool_2 = Dropout(0.1)(pool_2)

  con_3 = conv2d_block(pool_2, 128)
  pool_3 = MaxPool2D((2, 2))(con_3)
  pool_3 = Dropout(0.1)(pool_3)

  con_4 = conv2d_block(pool_3, 256)
  pool_4 = MaxPool2D((2, 2))(con_4)
  pool_4 = Dropout(0.1)(pool_4)

  con_5 = conv2d_block(pool_4, 512)

  unite_1 = Conv2DTranspose(256, 2, 2, output_padding=(0, 0))(con_5)
  unite_1 = concatenate([unite_1, con_4])
  unite_1 = Dropout(0.1)(unite_1)
  unite_1_con = conv2d_block(unite_1, 256)

  unite_2 = Conv2DTranspose(128, 2, 2, output_padding=(1, 1))(unite_1_con)
  unite_2 = concatenate([unite_2, con_3])
  unite_2 = Dropout(0.1)(unite_2)
  unite_2_con = conv2d_block(unite_2, 128)

  unite_3 = Conv2DTranspose(64, 2, 2, output_padding=(1, 1))(unite_2_con)
  unite_3 = concatenate([unite_3, con_2])
  unite_3 = Dropout(0.1)(unite_3)
  unite_3_con = conv2d_block(unite_3, 64)

  unite_4 = Conv2DTranspose(32, 2, 2, output_padding=(0, 0))(unite_3_con)
  unite_4 = concatenate([unite_4, con_1])
  unite_4 = Dropout(0.1)(unite_4)
  unite_4_con = conv2d_block(unite_4, 32)

  outputs = Conv2D(3, 1, activation="sigmoid")(unite_4_con)

  model = Model(inputs, outputs)
  return model

model_unet = unet_like()

model_unet.compile(
  loss = "mae",
  optimizer = keras.optimizers.Adam(lr_schedule),
  metrics = ["accuracy"]
)

model_unet.fit(
  train_ds,
  validation_data = validation_ds,
  epochs = n_epoch,
  verbose = 1
)

loss, acc = model_unet.evaluate(validation_ds, verbose = 1)
print(f"loss : {loss} / acc : {acc}")

model_unet.save("./test_model.h5", include_optimizer = False)

# Need to change image size to standard of phone camera