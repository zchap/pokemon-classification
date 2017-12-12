import csv

import keras
import numpy
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from keras.optimizers import SGD
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import classification
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization


def getlabels(csv_file_path):
    labels = []
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[0]))
    trainingLabels = labels
    return trainingLabels

train_i = numpy.array(classification.load_poke_images('TrainingImages'), dtype='float32')
train_y = numpy.array(getlabels('PokemonData/traininglabels.csv'), dtype='uint8')
test_i = numpy.array(classification.load_poke_images('TestImages'), dtype='float32')
print(train_i.shape)
print(train_y.shape)
num_cats = 19;
cat_y = keras.utils.to_categorical(numpy.transpose(train_y), num_cats)
# cat_y = numpy.delete(cat_y, 0, 1)
numpy.savetxt('labels.csv', cat_y, delimiter=',')


batch_size = 32
epochs = 30
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=(96, 96, 3),
                     activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same',
                     activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_cats, activation='softmax'))
    lr = 0.001
    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])
    model.fit(train_i, train_y,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2
              )
    y_pred = model.predict(test_i)
    return y_pred

labels = cnn_model()
demaxlabels = labels.argmax(axis=-1)
numpy.savetxt('testlabels.csv', demaxlabels, delimiter=",")

