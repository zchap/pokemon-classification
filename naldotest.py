import csv
import keras
import numpy
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import classification
from keras.models import Sequential
from keras.layers import Dense, Dropout


def getlabels(csv_file_path):
    labels = []
    with open(csv_file_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            labels.append(int(row[0]))
    trainingLabels = labels
    return trainingLabels

train_i = numpy.array(classification.load_poke_images('TrainingImages'), dtype='float32')
train_y = numpy.array(list(map(int, classification.load_training_labels('PokemonData/TrainingMetadata.csv'))))
test_i = numpy.array(classification.load_poke_images('TestImages'), dtype='float32')
train_i /= 255
test_i /= 255
num_cats = 18;
attempt_y = numpy.empty((601))
for i in range(0, train_y.size):
    attempt_y[i] = train_y[i] - 1
print(attempt_y)
cat_y = keras.utils.to_categorical(attempt_y, num_cats)

batch_size = 32
epochs = 75
def cnn_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',input_shape=(96,96,3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_cats))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    datagen.fit(train_i)
    model.fit_generator(datagen.flow(train_i, cat_y, batch_size=batch_size) ,epochs=epochs)

    y_pred = model.predict(test_i)
    return y_pred


labels = cnn_model()
demaxlabels = labels.argmax(axis=-1)
predictions = demaxlabels + 1
numpy.savetxt('cnnlabels.csv', predictions, delimiter=",")
for predict in predictions:
    print(predict)