# Edgedetection
# Tries to find edges in Images and enstronges them.
# Example Algorithms for this are Sobel & Canny Edge Detection

import matplotlib.pyplot as plt
import numpy as np

"""For information look at: https://en.wikipedia.org/wiki/Kernel_(image_processing)"""
def convolution(image_ausschnitt, kernel):
    filtered_pixel = 0
    for r in range(kernel.shape[0]):
        for c in range(kernel.shape[1]):
            filtered_pixel = filtered_pixel + image_ausschnitt[kernel.shape[0] -1 - r ,kernel.shape[1] -1 - c ] \
                             * kernel[r,c]
    return filtered_pixel


def sobel_filter(image):
    kernel= np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    kernel_t = kernel.T
    """ Goal: Detect Edges based on length of Gradientvector.
    to determine gradient vector, calculate sobelfilter (See: https://en.wikipedia.org/wiki/Sobel_operator)"""
    filtered_image = np.zeros(image.shape)
    for r in range(3,image.shape[0]-3):
        for c in range(3, image.shape[1]-3):
            filtered_image[r,c] = np.sqrt(convolution(image[(r-1):(r+2),(c-1):(c+2)], kernel)**2 +
                                          convolution(image[(r-1):(r+2),(c-1):(c+2)], kernel.T)**2)
    return filtered_image


def load_picture(image_path):
    image = plt.imread(image_path)
    return image

# https://tannerhelland.com/2011/10/01/grayscale-image-algorithm-vb6.html
def black_with_picture(picture):
    black_white = np.zeros((picture.shape[0], picture.shape[1]))
    for x in range((picture.shape[0])):
        for y in range((picture.shape[1])):
            black_white[x,y] = (picture[x,y][0] + picture[x,y][1] + picture[x,y][2])/3
    return black_white

def sobel_edge_detection():
    image = load_picture('probe_bild.jpeg')
    image = black_with_picture(image)
    image = sobel_filter(image)
    plt.imshow(image)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sobel_edge_detection()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
