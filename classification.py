import numpy as np
import scipy
import csv  # used to read/write/investigate csv files
import os
import matplotlib.pyplot as plt

from PIL import Image
import glob


def load_pokemon_images(folder):
    image_list = []

    for filename in glob.glob('PokemonData/' + folder + '/*.png'):
        im = Image.open(filename)
        rgb_im = im.convert('RGB')  # convert each image to 'RGB' from default format (greyscale?)
        pixel_list = []

        # for each pixel in the rgb image
        width, height = rgb_im.size
        for x in range(0, width):
            for y in range(0, height):
                pixel = rgb_im.getpixel((x, y))  # this is a pixel. Each pixel is a tuple object: (r, g, b)
                pixel_list.append(pixel)  # this has every pixel for current image. Each row in the pixel_list matrix
                # containing a long x*y row vector.

        # now image_list contains every image object in the current folder in the form of lists of all pixels for that
        # image (i.e. [[(r1,g1,b1),(r2,g2,b2),...,(r(x*y),g(x*y),b(x*y)],
        # [(r1,g1,b1),(r2,g2,b2),...,(r(x*y),g(x*y),b(x*y)], ..., [(r1,g1,b1),(r2,g2,b2),...,(r(x*y),g(x*y),b(x*y)]]
        # the size of the list above is the number of images (test: 200, training: 601)
        image_list.append(pixel_list)

    return image_list  
