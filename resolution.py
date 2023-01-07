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

# Write directory address
cur_dir = os.getcwd()
data_dir = os.path.join(cur_dir, "landscape_data")
y_dir = os.path.join(data_dir, "color")

# Checking amount of data > 7129 / 7129
y_image_files = [fname for fname in os.listdir(y_dir) if os.path.splitext(fname)[-1]=='.jpg']
len(y_image_files)

# Remove Images of different size
for path in y_image_files:
  temp = np.array(Image.open(os.path.join(y_dir, path))).shape
  if temp != (img_size, img_size):
    os.remove(os.path.join(y_dir, path))
  else:
    continue

# Load data
y = np.array([np.array(Image.open(os.path.join(y_dir, image))) for image in y_image_files])
print(f"x_shape / y_shape : {x.shape, y.shape}")

# Data scailing between 0 and 1
y = y/y.max()
print(y.shape, y.max(),y.min())

# Make a low-quality datasets
train_down = np.array([resize(img, (75, 75)) for img in y])
train_up = np.array([resize(img, (150, 150)) for img in train_down])

# Split train dataset and validation dataset
train_x, val_x, train_y, val_y = train_test_split(train_up, y, test_size = 0.1, shuffle = True, random_state = 32)

# Make a Customdataset
train_ds = functions.Dataloader(train_x, train_y, n_batch, shuffle = True)
validation_ds = functions.Dataloader(val_x, val_y, n_batch)

# Using Customschedule class to make lr warmup, cosine decay
steps_per_epoch = n_train // n_batch
lr_schedule = functions.CustomSchedule(LR, 3 * steps_per_epoch, n_epoch * steps_per_epoch)

