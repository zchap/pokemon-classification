"""
Matthew Mutammara (mjm13)
Ronaldo Sanchez (rbs4)
Emmanuel Loredo (el21)
Ayush Chapagain (ac91)

Multilayer perceptron implementation for Pokemon type classification
"""
import numpy
import keras.utils
import classification
import glob
import csv  # used to read/write/investigate csv files
from scipy.misc import imread
from PIL import Image, ImageOps
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Merge, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization

def rgb2gray(rgb):
    """
    Convert RGB value to grayscale value
    """
    return numpy.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def load_poke_images(folder_name):
    """
    Load RGB values of Pokemon images
    """
    image_list = []

    for filename in glob.glob('PokemonData/' + folder_name + '/*.png'):
        im = imread(filename)[:, :, :3]
        image_list.append(im)
    return image_list


def load_poke_images_grayscale(folder_name):
    """
    Load grayscale values of Pokemon images
    """
    image_list = []

    for filename in glob.glob('PokemonData/' + folder_name + '/*.png'):
        im = imread(filename)[:,:,:3]
        gray = rgb2gray(im)
        gray = gray.flatten()
        image_list.append(gray)
    return image_list


def load_test_stats(csv_file_path):
    """
    Load stats for test Pokemon

    Reads in a csv file of pokemon stats and returns a list of the stats for every pokemon included in the file
    List of stats starts with "type" because the first element has been removed. Each stat in the list is still a
    string, so we have to convert to float before performing operations such as gradient descent on them.
    """
    # We have a list of lists where the size of the outer list is the number of pokemon we are looking at
    # (test = 200, training = 601)
    # the size of each inner list (list of pokemon stats) is the number of columns in Test & Training Metadata
    # (number of stats representing a pokemon, 17 including pokemon number)
    # the first column can be ignored because it is just the number
    with open(csv_file_path, 'r') as f:
        read_input = csv.reader(f)
        stats_list = list(read_input)

    # we can ignore the first element in stats_list
    truncated_stats_list = stats_list[1:]
    for i in range(0, len(truncated_stats_list)):
        # truncated_stats_list[i] = (truncated_stats_list[i])  # this includes the pokemon number
        truncated_stats_list[i] = (truncated_stats_list[i])[1:]  # this removes the pokemon number

    return truncated_stats_list


def load_training_stats(csv_file_path):
    """
    Load stats from training data. Exclude Pokemon number and type.
    """
    with open(csv_file_path, 'r') as f:
        read_input = csv.reader(f)
        stats_list = list(read_input)

    # we can ignore the first element in stats_list (as it is just the header labels)
    truncated_stats_list = stats_list[1:]

    for i in range(0, len(truncated_stats_list)):
        temp_list = truncated_stats_list[i]
        # temp_list.pop(1)
        # truncated_stats_list[i] = (temp_list)  # this removes type from every inner list
        truncated_stats_list[i] = (truncated_stats_list[i])[2:]  # this removes number and type from every inner list

    return truncated_stats_list


def load_training_labels(csv_file_path):
    """
    Load types for Pokemon from training data
    """
    with open(csv_file_path, 'r') as f:
        read_input = csv.reader(f)
        stats_list = list(read_input)

    # we can ignore the first element in stats_list (as it is just the header labels)
    training_labels = stats_list[1:]

    for i in range(0, len(training_labels)):
        training_labels[i] = (training_labels[i])[1]  # this takes the label out from every inner list

    return training_labels


def bigX_mlp(x_train, y_train, x_test):
    """
    Multilayer perceptron implementation - Prediction accuracy between 27% and 33%
    Input: x_train - Big X matrix consisting of training stats and PCA'd grayscale image data
           y_train - Labels for training data
           x_test - Big X matrix consisting of test stats and PCA'd grayscale image data
    """
    batch_size = 16
    num_classes = 18
    epochs = 500
    dense = 601

    x_train = numpy.array(x_train).astype('float32')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes + 1)

    # Initialize model with first layer
    model = Sequential()
    model.add(Dense(dense, input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Second layer
    model.add(Dense(dense))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Third layer
    model.add(Dense(dense))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Fourth layer
    model.add(Dense(dense))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    # Output layer
    model.add(Dense(num_classes + 1, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size)

    return model.predict(x_test)


# Loading image vector, stats vector, and labels.
train_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
train_i = numpy.array(classification.load_poke_images('TrainingImages'))
train_y = numpy.array(list(map(int, classification.load_training_labels('PokemonData/TrainingMetadata.csv'))))

test_s = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')
test_i = numpy.array(classification.load_poke_images('TestImages'))

# Converting stats matrix to float from string, so that classifiers can be used on them.
for p in range(0, len(train_s)):  # indexing each pokemon
    for s in range(0, len(train_s[p])):  # indexing each stat for each pokemon
        train_s[p][s] = float(train_s[p][s])
for p in range(0, len(test_s)):  # indexing each pokemon
    for s in range(0, len(test_s[p])):  # indexing each stat for each pokemon
        test_s[p][s] = float(test_s[p][s])

# Get grayscale values of image data
train_gray = numpy.array(classification.load_poke_images_grayscale('TrainingImages'))
train_gray.reshape([601,9216])
train_gray = train_gray / 255

test_gray = numpy.array(classification.load_poke_images_grayscale('TestImages'))
test_gray.reshape([201,9216])
test_gray = test_gray / 255

# PCA on x_train (giving error)
x_train = numpy.array(train_gray).astype('float32')
pca = PCA(n_components=3, whiten = True).fit(train_gray)
train_gray_pca = pca.transform(x_train)

# PCA on image_i (giving error)
test_gray = numpy.array(test_gray)
test_gray_pca = pca.transform(test_gray)

# Horizontal concatenation of S and I to form X = [S I]
train_x = []
for n in range(0, len(train_i)):
    train_x.append(numpy.hstack((numpy.array(train_s[n]), train_gray_pca[n])))
train_x = numpy.array(train_x)

test_x = []
for n in range(0, len(test_i)):
    test_x.append(numpy.hstack((numpy.array(test_s[n]), test_gray_pca[n])))
test_x = numpy.array(test_x)


# Now that we have all the data preprocessed, just run it through bigX_mlp to get a prediction!
labels = bigX_mlp(train_x, train_y, test_x)
prediction = labels.argmax(axis = -1)

print(prediction)
