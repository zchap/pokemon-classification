"""
Neural net classifier module using the Keras library for Convolutional Neural Networks.
This classifier will be used to generate predicted labels for our test data.
The model will be trained on a set of 601 pokemon, over 18 statistics.
"""

import numpy
import keras.utils
import classification
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, Convolution2D

train_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')  # list of training stats
train_i = numpy.array(classification.load_poke_images('TrainingImages'))
train_y = numpy.array(list(map(int, classification.load_training_labels('PokemonData/TrainingMetadata.csv'))))
# training images

test_i = numpy.array(classification.load_pokemon_images('TestImages'))  # list of list of integer triplets

# image preprocessing so that it can be inputted into our model
train_I = train_i.reshape(train_i.shape[0], 3, 96, 96)
test_I = test_i.reshape(test_i.shape[0], 3, 96, 96)

# TODO: Implement Keras for X (stats | vectors)
# TODO: Implement Keras for stats and compare to outputs for images
def keras_nn_classification():
    """
    Keras NN classification of Pokemon data.
    """
    model = Sequential()  # sequential NN layering
    model.add(Convolution2D(32, (3, 3), input_shape=(3, 96, 96), data_format='channels_first'))  # input layer

    # first layer
    model.add(Convolution2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))  # dropout is to prevent overfitting

    # second and third layers
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # final processing of model (flattening input data)
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(19))
    model.add(Activation('sigmoid'))

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # preprocess training data
    train_Y = train_y.reshape((-1, 1))
    model.fit(train_I, train_Y, epochs=30)
    labels = model.predict(test_I)

    return labels


def labels_from_sparse_matrix(input):
    """
    Helper function for keras_nn_classification problems that minimize a sparse categorical crossentropy loss function.
    Converts the sparse matrix to a label vector (index of maximum value of sparse matrix, as the model
    doesn't do 1s and 0s perfectly. Sometimes, values that should be represented as 0s are really small i.e. 6e-23,
    and values that should be represented as 1s are not quite 1, but are still the largest values in the
    matrix i.e. 9.9548e-01 (0.99548 -> 1).
    """
    result = []
    for i in range(0, len(input)):
        result.append(numpy.argmax(input[i]))

    return result

sparse_labels = keras_nn_classification()
labels = labels_from_sparse_matrix(sparse_labels)

for label in labels:
    print(label)


