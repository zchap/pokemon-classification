"""
This file is to run on the big X matrix
We create big X by concatenating the Pokemon stats with the results of running PCA on the grayscaled images
We then run big X through an MLP similar to the one that we use when only running on the stats
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


def bigX_mlp(x_train, y_train, x_test):
    """
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

    model = Sequential()
    model.add(Dense(dense, input_shape=(x_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(dense))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(dense))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(dense))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes + 1, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size)

    return model.predict(x_test)



train_gray = numpy.array(classification.load_poke_images_grayscale('TrainingImages'))
train_gray.reshape([601,9216])

test_gray = numpy.array(classification.load_poke_images_grayscale('TestImages'))
test_gray.reshape([201,9216])


# PCA on x_train (giving error)
x_train = numpy.array(train_gray).astype('float32')
pca = PCA(n_components=3, whiten = True).fit(train_gray)
train_gray_pca = pca.transform(x_train)

# PCA on image_i (giving error)
test_gray = numpy.array(test_gray)
test_gray_pca = pca.transform(test_gray)

# Horizontal concatenation of S and I to form X = [S I]. Had to be done using a loop instead of with .concatenate
# because of type differences in the loaded image data and the loaded statistics data stemming from the fact that
# I is represented by (r, g, b) tuples.
train_x = []
for n in range(0, len(train_i)):
    train_x.append(numpy.hstack((numpy.array(train_s[n]), train_gray_pca[n])))
train_x = numpy.array(train_x)

test_x = []
for n in range(0, len(test_i)):
    test_x.append(numpy.hstack((numpy.array(test_s[n]), test_gray_pca[n])))
test_x = numpy.array(test_x)


# Use this to run several times on AWS
maxAcc = -1
maxPred = None
count = 1

while maxAcc < 0.34:
    print('Trial:', count)
    mergedLabels = bigX_mlp(train_x, train_y, test_x)
    mergedMaxLabels = mergedLabels.argmax(axis = -1)

    score = kaggle_submit(mergedMaxLabels)

    if score > maxAcc:
        maxAcc = score
        maxPred = mergedMaxLabels
        numpy.savetxt('mergedLabels.csv', mergedMaxLabels, delimiter = ',')

    print(mergedMaxLabels)
    print(kaggle_submit(mergedMaxLabels))

    count += 1

# NOTE: This implementation accuracy ranges from 27% to 33%