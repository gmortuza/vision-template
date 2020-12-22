import math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg


def load_image(path: str) -> np.ndarray:
    """
    convert any image to numpy array
    :param path: path of the image
    :return:
    """
    return np.asarray(mpimg.imread(path))


def show_image(images_path: list, label: list,  columns: int = 2, fig_size: tuple = (8, 16), dpi: int = 300):
    """
    Display the image
    :param images_path: list containing all the image path
    :param label: list containing the labels of the images
    :param columns: Number of images per column
    :param fig_size: size of the figure
    :param dpi: Depth per inch of the figure
    :return: None
    """
    # number of row needed
    rows = math.ceil(len(images_path) / columns)
    # create matplotlib figures
    fig, axs = plt.subplots(ncols=columns, nrows=rows, figsize=fig_size, dpi=dpi)
    for i, path in enumerate(images_path):
        row = i // columns
        col = i % columns
        axs[row, col].imshow(load_image(path=path))
        axs[row, col].set_axis_off()
        axs[row, col].set_title(label[i])
    plt.show()
