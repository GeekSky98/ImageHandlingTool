# Import required libraries
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import math
from tensorflow.keras.utils import Sequence
import tensorflow as tf
from skimage import color
from skimage.transform import resize

# Import custom functions
import functions

# Temporary def / Modification required(grid, print, etc)
def predict_color(model, image, img_height = 150, img_width = 150):
    if image.mode == "RGB":
        image = np.array(image)
        if image.shape[-1] == 3:
            image = color.rgb2gray(image)
        image = resize(image, (img_height, img_width))
        image = image.reshape(1, img_height, img_width, 1)
    else:
        print("This image isn't RGB form")
        pass
    res = model.predict(image)
    functions.show(res[0])

# Make learning rate warmup and cosine decay
class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, max_lr, warmup_steps, decay_steps):
    super(CustomSchedule, self).__init__()
    self.max_lr = max_lr
    self.warmup_steps = warmup_steps
    self.decay_steps = decay_steps

  def __call__(self, step):
    lr = tf.cond(step < self.warmup_steps,
                 lambda: self.max_lr / self.warmup_steps * step,
                 lambda: 0.5 * (1 + tf.math.cos(math.pi * (step - self.warmup_steps) / self.decay_steps)) * self.max_lr)
    return lr

# Make a Custom dataset
class Dataloader(Sequence):

    def __init__(self, data_x, data_y, batch_size, shuffle=False):
        self.x, self.y = data_x, data_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]

        batch_x = [self.x[i] for i in indices]
        batch_y = [self.y[i] for i in indices]

        return np.array(batch_x), np.array(batch_y)

    def on_epoch_end(self):
        self.indices = np.arange(len(self.x))
        if self.shuffle == True:
            np.random.shuffle(self.indices)

# Show result image fastly
def show(image):
    plt.imshow(image)
    plt.grid(True)
    plt.colorbar()
    plt.show()

def gray_show(image):
    plt.imshow(image, cmap = "gray")
    plt.grid(True)
    plt.colorbar()
    plt.show()

def convert_lab_result(model, image, img_height = 150, img_width = 150):
    if image.mode == "RGB":
        image = np.array(image)
        if image.shape[-1] != 3:
            image = color.gray2rgb(image)
        image_color_resized = resize(image, (img_height, img_width))
        image_lab = color.rgb2lab(image_color_resized)[...,:1] / 100
        res = model.predict(np.expand_dims(image_lab, 0))
        mold_image = np.zeros((img_height, img_width, 3))
        mold_image[:,:,0] = image_lab.reshape(img_height, img_width)
        mold_image[:,:,1:] = res[0]
        mold_image_denorm = (mold_image * [100, 255, 255]) - [0, 128, 128]
        result = color.lab2rgb(mold_image_denorm)
    else:
        print("This image isn't RGB form")
        pass
    functions.show(result)

def resolution_color(model1, model2, image, img_height = 150, img_width = 150):
    if image.mode =="RGB":
        image = np.array(image)
        if image.shape[-1] != 3:
            print("This image isn't color image")
        image_resized = resize(image, (img_height, img_width))
        image_resized_expand = np.expand_dims(image_resized, 0)
        result1 = model1.predict(image_resized_expand)
        result2 = model2.predict(result1)
        plt.imshow(np.concatenate([image, result1, result2]).transpose(1, 0, 2, 3).reshape(150, -1, 3))
        plt.show()

        return result2

