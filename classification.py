import numpy as np
import scipy
import csv  # used to read/write/investigate csv files
import os
import matplotlib.pyplot as plt

from PIL import Image
import glob


def load_pokemon_images():
    image_list = []

    for filename in glob.glob('PokemonData/TrainingImages/*.png'):  # assuming gif
        im = Image.open(filename)
        image_list.append(im)
        # now image_list contains every image object

    return image_list


print(load_pokemon_images())  # this prints every
print("end")

