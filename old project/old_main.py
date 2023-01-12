from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def ImagetoMatrix(array):
    return np.array(array)

def BlackWhite(array):
    newImage = np.zeros((len(array), len(array[0])))
    for i in range(len(array)):
        for j in range(len(array[0])):
            newImage[i][j] = 0.2989 * array[i][j][0] + 0.5870 * array[i][j][1] + 0.1140 * array[i][j][2]
    return newImage

def filter(array):
    filter = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
    newImage = np.zeros((len(array), len(array[0])))
    for i in range(1, len(array) - 1):
        for j in range(1, len(array[0]) - 1):
            newImage[i][j] = array[i - 1][j - 1] * filter[0][0] + array[i - 1][j] * filter[0][1] + array[i - 1][j + 1] * filter[0][2] + \
                             array[i][j - 1] * filter[1][0] + array[i][j] * filter[1][1] + array[i][j + 1] * filter[1][2] + \
                             array[i + 1][j - 1] * filter[2][0] + array[i + 1][j] * filter[2][1] + array[i + 1][j + 1] * filter[2][2]
    return newImage

def crop(array, l,r,t,b):
    newImage = np.zeros((len(array) - t - b, len(array[0]) - l - r))
    newImage = array[t:len(array) - b, l:len(array[0]) - r]
    return newImage

def main():
    image = Image.open('array.png')
    array = ImagetoMatrix(image)

    newImage = BlackWhite(array)
    plt.imshow(newImage, cmap ="gray")
    plt.show()

    newImage0 = filter(newImage)
    plt.imshow(newImage0, cmap="gray")
    plt.show()

    newimage1 = crop(array,100,100,100,100)
    plt.imshow(newimage1, cmap ="gray")
    plt.show()