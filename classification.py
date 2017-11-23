import numpy as np
import scipy
import csv  # used to read/write/investigate csv files
import os
import matplotlib.pyplot as plt

from PIL import Image
import glob


def load_pokemon_images(folder_name):
    image_list = []

    for filename in glob.glob('PokemonData/' + folder_name + '/*.png'):
        im = Image.open(filename)
        rgb_im = im.convert('RGB')  # convert each image to 'RGB' from default format (greyscale?)
        pixel_list = []

        # for each pixel in the rgb image
        width, height = rgb_im.size
        for x in range(0, width):
            for y in range(0, height):
                pixel = rgb_im.getpixel((x, y))  # this is a pixel. Each pixel is a tuple object: (r, g, b)
                pixel_list.append(pixel)  # this has every pixel for current image. Each row in the pixel_list matrix
                # contains a long x*y row vector.

        # now image_list contains every image object in the current folder in the form of lists of all pixels for that
        # image (i.e. [[(r1,g1,b1),(r2,g2,b2),...,(r(x*y),g(x*y),b(x*y)],
        # [(r1,g1,b1),(r2,g2,b2),...,(r(x*y),g(x*y),b(x*y)], ..., [(r1,g1,b1),(r2,g2,b2),...,(r(x*y),g(x*y),b(x*y)]]
        # the size of the list above is the number of images (test: 200, training: 601)
        image_list.append(pixel_list)

    return image_list


def load_pokemon_stats(csv_file_path):
    # you can have both integers and strings in the same list
    # again, we have a list of lists where the size of the outer list is the number of pokemon we are looking at
    # (test = 200, training = 601)
    # the size of each inner list (list of pokemon stats) is the number of columns in Test & Training Metadata
    # (number of stats representing a pokemon, 17 including pokemon number)
    # the first column can be ignored because it is just the number
    with open(csv_file_path, 'r') as f:
        read_input = csv.reader(f)
        stats_list = list(read_input)

    print(stats_list)
    return stats_list


load_pokemon_stats('PokemonData/TrainingMetadata.csv')
