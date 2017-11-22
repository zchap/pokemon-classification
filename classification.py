import numpy as np
import scipy
import csv  # used to read/write/investigate csv files
import os
import matplotlib.pyplot as plt

from PIL import Image
import glob


def load_pokemon_images():
    image_list = []
    pixel_list = []

    for filename in glob.glob('PokemonData/TrainingImages/*.png'):  # assuming gif
        im = Image.open(filename)
        rgb_im = im.convert('RGB')  # convert each image to 'RGB' from default format (greyscale?)

        # for each pixel in the rgb image
        width, height = rgb_im.size
        for x in range(0, width):
            for y in range(0, height):
                pixel = rgb_im.getpixel((x, y))  # this is a pixel. Each pixel is a tuple object containing the 3 different rgb values.
                pixel_list.append(pixel)  # this has every pixel for every image. Each row in the pixel_list matrix containings a long x*y row vector.
                
        # make a list for testing purposes
        image_list.append(rgb_im)

        # now image_list contains every image object

    return image_list, pixel_list




