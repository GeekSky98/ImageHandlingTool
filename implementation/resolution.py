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
  BatchNormalization, GlobalAvgPool2D, Flatten, Reshape, Dense, Average, Lambda, Add, Rescaling
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
n_epoch = 50
channel = 3

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
print(f"y_shape : {y.shape}")

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

# Make a super resolution model using Unet
def conv2d_block(x, channel):
  x = Conv2D(channel, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  x = Conv2D(channel, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  return x

def unet_resolution():
  inputs = Input((150, 150, 3))

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

model_resolution = unet_resolution()

model_resolution.compile(
  loss = "mae",
  optimizer = keras.optimizers.Adam(lr_schedule),
  metrics = ["accuracy"]
)

model_resolution.fit(
  train_ds,
  validation_data = validation_ds,
  epochs = n_epoch,
  verbose = 1
)

model_resolution.save("./resolution1.h5", include_optimizer = False)

# Extract second data using the first model
x_second = np.array([model_resolution.predict(np.expand_dims(img, 0)).reshape(150, 150, 3) for img in train_up])
x_second = np.array(x_second)

# Split train dataset and validation dataset
train_x_second, val_x_second, train_y_second, val_y_second = train_test_split(
  x_second, y, test_size = 0.1, shuffle = True, random_state = 32
)

# Make a Customdataset
train_ds_second = functions.Dataloader(train_x_second, train_y_second, n_batch, shuffle = True)
validation_ds_second = functions.Dataloader(val_x_second, val_y_second, n_batch)

def unet_resolution2():

  inputs = Input((150, 150, 3))

  x = Conv2D(64, 9, activation="relu", padding = "same")(inputs)
  x1 = Conv2D(32, 1, activation="relu", padding="same")(x)
  x2 = Conv2D(32, 3, activation="relu", padding="same")(x)
  x3 = Conv2D(32, 5, activation="relu", padding="same")(x)

  x = Average()([x1,x2,x3])

  outputs = Conv2D(3, 5, activation="relu", padding="same")(x)

  model = Model(inputs, outputs)
  return model

model_resolution2 = unet_resolution2()

model_resolution2.compile(
  loss = "mae",
  optimizer = keras.optimizers.Adam(lr_schedule),
  metrics = ["accuracy"]
)

model_resolution2.fit(
  train_ds_second,
  validation_data = validation_ds_second,
  epochs = n_epoch,
  verbose = 1
)



class EDSR_model_builder(Model):
  def train_step(self, data):
    x, y = data

    with tf.GradientTape() as tape:
      y_pred = self(x, training=True)
      loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

    trainable_vars = self.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)

    self.optimizer.apply_gradients(zip(gradients, trainable_vars))
    self.compiled_metrics.update_state(y, y_pred)

    return {m.name: m.result() for m in self.metrics}

  def predict_step(self, data):
    x = tf.cast(tf.expand_dims(data, axis=0), tf.float32)
    pred = self(x, training=False)
    pred = tf.clip_by_value(pred, 0, 255)
    pred = tf.round(pred)
    pred = tf.cast(tf.squeeze(pred, axis=0), tf.uint8)

    return pred


def res_block(Input_layer, filter=64):
  res_x = Conv2D(filters=filter, kernel_size=3, activation='relu', padding='same')(Input_layer)
  res_x = Conv2D(filters=filter, kernel_size=3, padding='same')(res_x)
  res_x = Lambda(function=lambda x: x * 0.1)(res_x)
  result = Add()([Input_layer, res_x])

  return result


def upsampling(Input_layer, filter=64, scale=2, **sky):
  up_x = Conv2D(filter * (scale ** 2), kernel_size=3, padding='same', **sky)(Input_layer)
  up_x = tf.nn.depth_to_space(up_x, block_size=scale)
  up_x = Conv2D(filter * (scale ** 2), kernel_size=3, padding='same', **sky)(up_x)
  up_x = tf.nn.depth_to_space(up_x, block_size=scale)

  return up_x


def build_edsr(filter=64, num_res=16):
  input = Input((None, None, channel))
  x = Rescaling(scale=1.0 / 255.0)(input)
  x = res_x = Conv2D(filter, kernel_size=3, padding='same')(x)

  for _ in range(num_res):
    res_x = res_block(res_x)

  res_x = Conv2D(filter, kernel_size=3, padding='same')(res_x)
  x = Add()([x, res_x])

  up_x = upsampling(x)
  output = Conv2D(channel, kernel_size=3, padding='same')(up_x)
  output = Rescaling(255)(output)

  model = EDSR_model_builder(input, output)

  return model


model = build_edsr()

optimizer = keras.optimizers.Adam(
  learning_rate=keras.optimizers.schedules.PiecewiseConstantDecay(boundaries=[5000], values=[1e-4, 5e-5])
)

def PSNR(y_true, y_pred):
  return tf.image.psnr(y_true, y_pred, max_val=255)[0]

def SSIM(y_true, y_pred):
  return tf.image.ssim(y_true, y_pred, max_val=255)[0]

model.compile(
  loss='mae',
  optimizer=optimizer,
  metrics=[PSNR, SSIM]
)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=n_epoch,
    steps_per_epoch=300,
    verbose=1
)