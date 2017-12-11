import keras
import numpy
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.models import Sequential
from sklearn.model_selection import cross_validate

import classification

train_i = numpy.array(classification.load_poke_images('TrainingImages'))
train_y = numpy.array(list(map(int, classification.load_training_labels('PokemonData/TrainingMetadata.csv'))))
test_i = numpy.array(classification.load_poke_images('TestImages'))
p_numbers = classification.load_pokemon_numbers('PokemonData/UnlabeledTestMetadata.csv')
for label in p_numbers:
    print(label)
num_cats = 19;
cat_y = keras.utils.to_categorical(train_y, num_cats)
cat_y = numpy.delete(cat_y, 0, axis=1)
def cnn_classifier():
    model = Sequential()
    model.add(Conv2D(32, 3, 3, activation='relu', input_shape=(96, 96, 3)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(18, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_i, cat_y, batch_size=60, nb_epoch=60,verbose=1)
    y_pred = model.predict(test_i)
    return y_pred

labels = cnn_classifier()
# for label in labels:
#     print(label)
demaxlabels = labels.argmax(axis=-1)
numpy.savetxt('testlabels.csv', demaxlabels, delimiter=",")
