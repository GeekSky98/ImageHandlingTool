import matplotlib.pyplot as plt
from tensorflow import keras

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