import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import math
from tensorflow.keras.utils import Sequence

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