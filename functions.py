import matplotlib.pyplot as plt

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