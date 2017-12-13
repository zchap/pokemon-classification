"""
This file is to run the merged method
This is where we have two separate implementations for each set of data, then merge using Keras's Merge() method
"""

import numpy
import keras.utils
import classification
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Merge, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import RMSprop, SGD
from keras.layers.normalization import BatchNormalization


# Loading image vector, stats vector, and labels.
train_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
train_i = numpy.array(classification.load_poke_images('TrainingImages'))
train_y = numpy.array(list(map(int, classification.load_training_labels('PokemonData/TrainingMetadata.csv'))))
train_pixels = list(train_i.flatten().reshape(601, 27648))

test_s = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')
test_i = numpy.array(classification.load_poke_images('TestImages'))
test_pixels = list(test_i.flatten().reshape(201, 27648))
test_y = [2.0, 2.0, 12.0, 12.0, 10.0, 10.0, 8.0, 18.0, 2.0, 8.0, 12.0, 12.0, 12.0, 1.0, 7.0, 7.0, 7.0, 5.0, 3.0, 13.0, 13.0, 4.0, 1.0, 3.0, 8.0, 3.0, 7.0, 7.0, 9.0, 1.0, 1.0, 3.0, 3.0, 11.0, 1.0, 4.0, 2.0, 15.0, 15.0, 11.0, 2.0, 2.0, 10.0, 12.0, 3.0, 18.0, 4.0, 3.0, 1.0, 5.0, 1.0, 18.0, 12.0, 16.0, 3.0, 6.0, 3.0, 3.0, 9.0, 1.0, 6.0, 4.0, 1.0, 1.0, 13.0, 11.0, 2.0, 5.0, 5.0, 2.0, 3.0, 1.0, 1.0, 10.0, 10.0, 11.0, 12.0, 1.0, 1.0, 12.0, 12.0, 1.0, 1.0, 7.0, 4.0, 8.0, 11.0, 9.0, 5.0, 13.0, 13.0, 3.0, 3.0, 3.0, 9.0, 13.0, 14.0, 11.0, 16.0, 11.0, 6.0, 15.0, 17.0, 17.0, 6.0, 11.0, 5.0, 3.0, 12.0, 4.0, 13.0, 3.0, 3.0, 14.0, 1.0, 1.0, 17.0, 15.0, 1.0, 9.0, 8.0, 8.0, 3.0, 3.0, 16.0, 9.0, 5.0, 4.0, 11.0, 3.0, 1.0, 14.0, 5.0, 5.0, 5.0, 2.0, 2.0, 3.0, 10.0, 10.0, 13.0, 11.0, 9.0, 3.0, 12.0, 5.0, 5.0, 5.0, 9.0, 14.0, 3.0, 16.0, 3.0, 3.0, 3.0, 12.0, 4.0, 11.0, 15.0, 6.0, 15.0, 9.0, 16.0, 15.0, 3.0, 1.0, 3.0, 3.0, 2.0, 7.0, 1.0, 17.0, 18.0, 18.0, 16.0, 3.0, 3.0, 4.0, 13.0, 13.0, 17.0, 14.0, 13.0, 2.0, 12.0, 13.0, 5.0, 5.0, 8.0, 1.0, 1.0, 5.0, 14.0, 4.0, 3.0, 11.0, 11.0, 12.0, 4.0, 11.0, 7.0]

# Converting stats matrix to float from string, so that classifiers can be used on them.
for p in range(0, len(train_s)):  # indexing each pokemon
    for s in range(0, len(train_s[p])):  # indexing each stat for each pokemon
        train_s[p][s] = float(train_s[p][s])
for p in range(0, len(test_s)):  # indexing each pokemon
    for s in range(0, len(test_s[p])):  # indexing each stat for each pokemon
        test_s[p][s] = float(test_s[p][s])

# Converting labels vector to float from string, so that classifiers can be used on them.

def kaggle_submit(prediction):
    # These are the types for the test data
    test_y = [2.0, 2.0, 12.0, 12.0, 10.0, 10.0, 8.0, 18.0, 2.0, 8.0, 12.0, 12.0, 12.0, 1.0, 7.0, 7.0, 7.0, 5.0, 3.0, 13.0, 13.0, 4.0, 1.0, 3.0, 8.0, 3.0, 7.0, 7.0, 9.0, 1.0, 1.0, 3.0, 3.0, 11.0, 1.0, 4.0, 2.0, 15.0, 15.0, 11.0, 2.0, 2.0, 10.0, 12.0, 3.0, 18.0, 4.0, 3.0, 1.0, 5.0, 1.0, 18.0, 12.0, 16.0, 3.0, 6.0, 3.0, 3.0, 9.0, 1.0, 6.0, 4.0, 1.0, 1.0, 13.0, 11.0, 2.0, 5.0, 5.0, 2.0, 3.0, 1.0, 1.0, 10.0, 10.0, 11.0, 12.0, 1.0, 1.0, 12.0, 12.0, 1.0, 1.0, 7.0, 4.0, 8.0, 11.0, 9.0, 5.0, 13.0, 13.0, 3.0, 3.0, 3.0, 9.0, 13.0, 14.0, 11.0, 16.0, 11.0, 6.0, 15.0, 17.0, 17.0, 6.0, 11.0, 5.0, 3.0, 12.0, 4.0, 13.0, 3.0, 3.0, 14.0, 1.0, 1.0, 17.0, 15.0, 1.0, 9.0, 8.0, 8.0, 3.0, 3.0, 16.0, 9.0, 5.0, 4.0, 11.0, 3.0, 1.0, 14.0, 5.0, 5.0, 5.0, 2.0, 2.0, 3.0, 10.0, 10.0, 13.0, 11.0, 9.0, 3.0, 12.0, 5.0, 5.0, 5.0, 9.0, 14.0, 3.0, 16.0, 3.0, 3.0, 3.0, 12.0, 4.0, 11.0, 15.0, 6.0, 15.0, 9.0, 16.0, 15.0, 3.0, 1.0, 3.0, 3.0, 2.0, 7.0, 1.0, 17.0, 18.0, 18.0, 16.0, 3.0, 3.0, 4.0, 13.0, 13.0, 17.0, 14.0, 13.0, 2.0, 12.0, 13.0, 5.0, 5.0, 8.0, 1.0, 1.0, 5.0, 14.0, 4.0, 3.0, 11.0, 11.0, 12.0, 4.0, 11.0, 7.0]

    correct = 0

    for i in range(len(prediction)):
        if prediction[i] == test_y[i]:
            correct += 1

    return correct / len(prediction)


def merged(s_train, i_train, y_train, s_test, i_test):
    batch_size = 32
    num_classes = 18
    epochs = 60
    dense = 601

    s_train = numpy.array(s_train).astype('float32')
    s_test = numpy.array(s_test).astype('float32')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes + 1)

    stat_branch = Sequential()
    stat_branch.add(Dense(dense, input_shape=(15,)))
    stat_branch.add(BatchNormalization())
    stat_branch.add(Activation('relu'))
    stat_branch.add(Dropout(0.5))

    stat_branch.add(Dense(dense))
    stat_branch.add(BatchNormalization())
    stat_branch.add(Activation('relu'))
    stat_branch.add(Dropout(0.5))

    stat_branch.add(Dense(dense))
    stat_branch.add(BatchNormalization())
    stat_branch.add(Activation('relu'))
    stat_branch.add(Dropout(0.5))

    stat_branch.add(Dense(dense))
    stat_branch.add(Activation('relu'))
    stat_branch.add(BatchNormalization())
    stat_branch.add(Dropout(0.5))

    stat_branch.add(Dense(num_classes + 1, activation='softmax'))

    image_branch = Sequential()
    image_branch.add(Dense(dense, input_shape = (i_train.shape[1],), init = 'normal'))
    image_branch.add(BatchNormalization())
    image_branch.add(Activation('relu'))
    image_branch.add(Dropout(0.5))

    image_branch.add(Dense(dense))
    image_branch.add(BatchNormalization())
    image_branch.add(Activation('relu'))
    image_branch.add(Dropout(0.5))

    image_branch.add(Dense(dense))
    image_branch.add(BatchNormalization())
    image_branch.add(Activation('relu'))
    image_branch.add(Dropout(0.5))

    image_branch.add(Dense(dense))
    image_branch.add(BatchNormalization())
    image_branch.add(Activation('relu'))
    image_branch.add(Dropout(0.5))

    image_branch.add(Dense(num_classes + 1, activation='softmax'))


    model = Sequential()
    model.add(Merge([stat_branch, image_branch], mode='concat'))
    model.add(Dense(num_classes + 1, init = 'normal', activation = 'sigmoid'))
    sgd = SGD(lr = 0.001, momentum = 0.9, decay = 0, nesterov = False)

    model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(),
                    metrics=['accuracy'])

    #numpy.random.seed(2017)

    model.fit([s_train, i_train], y_train, epochs=epochs, verbose=1, batch_size=batch_size)

    return model.predict([s_test, i_test])



train_gray = numpy.array(classification.load_poke_images_grayscale('TrainingImages'))
train_gray.reshape([601,9216])

test_gray = numpy.array(classification.load_poke_images_grayscale('TestImages'))
test_gray.reshape([201,9216])


# PCA on x_train (giving error)
x_train = numpy.array(train_gray).astype('float32')
pca = PCA(n_components=.99, whiten = True).fit(train_gray)
train_gray_pca = pca.transform(x_train)

# PCA on image_i (giving error)
test_gray = numpy.array(test_gray)
test_gray_pca = pca.transform(test_gray)


mergedLabels = merged(train_s, train_gray_pca, train_y, test_s, test_gray_pca)
mergedMaxLabels = mergedLabels.argmax(axis = -1)
numpy.savetxt('mergedLabels.csv', mergedMaxLabels, delimiter = ',')

print(mergedMaxLabels)
print(kaggle_submit(mergedMaxLabels))

# NOTE: This yields like 16% when you run on the PCA'd grayscale, 20% when you run on the raw grayscale
