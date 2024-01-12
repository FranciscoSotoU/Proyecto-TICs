import numpy as np
from matplotlib import pyplot as plt

def get_color_histogram(img):
    """
    Returns the color histogram of the image.
    :param img: image
    :return: color histogram
    """
    # Assuming the image is in RGB format
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]

    # Compute the histogram for each color channel
    r_hist, r_bins = np.histogram(r, bins=256, range=[0,256])
    g_hist, g_bins = np.histogram(g, bins=256, range=[0,256])
    b_hist, b_bins = np.histogram(b, bins=256, range=[0,256])

    return r_hist, g_hist, b_hist

def plot_color_histograms(img):
    """
    Plots the color histogram of the image.
    :param img: image
    """
    # Get the histograms for each color channel
    r_hist, g_hist, b_hist = get_color_histogram(img)

    # Create a new figure
    plt.figure()

    # Plot the histogram of the R channel
    plt.subplot(3, 1, 1)  # 3 rows, 1 column, 1st subplot = top subplot
    plt.bar(range(256), r_hist, color='red')
    plt.title('Red Channel')

    # Plot the histogram of the G channel
    plt.subplot(3, 1, 2)  # 3 rows, 1 column, 2nd subplot = middle subplot
    plt.bar(range(256), g_hist, color='green')
    plt.title('Green Channel')

    # Plot the histogram of the B channel
    plt.subplot(3, 1, 3)  # 3 rows, 1 column, 3rd subplot = bottom subplot
    plt.bar(range(256), b_hist, color='blue')
    plt.title('Blue Channel')

    # Display the plot
    plt.tight_layout()
    plt.show()