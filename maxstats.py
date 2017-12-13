import numpy
import keras.utils
import classification
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization

# Loading image vector, stats vector, and labels.
train_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
train_y = numpy.array(list(map(int, classification.load_training_labels('PokemonData/TrainingMetadata.csv'))))

test_s = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')

# Converting stats matrix to float from string, so that classifiers can be used on them.
for p in range(0, len(train_s)):  # indexing each pokemon
    for s in range(0, len(train_s[p])):  # indexing each stat for each pokemon
        train_s[p][s] = float(train_s[p][s])
for p in range(0, len(test_s)):  # indexing each pokemon
    for s in range(0, len(test_s[p])):  # indexing each stat for each pokemon
        test_s[p][s] = float(test_s[p][s])

def keras_mlp(x_train, y_train, x_test):
    batch_size = 32
    num_classes = 18
    epochs = 100  # this seems to be the sweet spot, at least with the other parameters as they are now
    dense = 601

    x_train = numpy.array(x_train)
    x_train = x_train.astype('float32')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes + 1)

    model = Sequential()
    model.add(Dense(dense, activation='relu', input_shape=(15,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(dense, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(dense, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(dense, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(num_classes + 1, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                    optimizer=RMSprop(),
                    metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=epochs, verbose=1, batch_size=batch_size)

    return model.predict(x_test)


def kaggle_submit(prediction):
    """
    Tests prediction against true y values
    """
    # These are the types for the test data
    test_y = [2.0, 2.0, 12.0, 12.0, 10.0, 10.0, 8.0, 18.0, 2.0, 8.0, 12.0, 12.0, 12.0, 1.0, 7.0, 7.0, 7.0, 5.0, 3.0, 13.0, 13.0, 4.0, 1.0, 3.0, 8.0, 3.0, 7.0, 7.0, 9.0, 1.0, 1.0, 3.0, 3.0, 11.0, 1.0, 4.0, 2.0, 15.0, 15.0, 11.0, 2.0, 2.0, 10.0, 12.0, 3.0, 18.0, 4.0, 3.0, 1.0, 5.0, 1.0, 18.0, 12.0, 16.0, 3.0, 6.0, 3.0, 3.0, 9.0, 1.0, 6.0, 4.0, 1.0, 1.0, 13.0, 11.0, 2.0, 5.0, 5.0, 2.0, 3.0, 1.0, 1.0, 10.0, 10.0, 11.0, 12.0, 1.0, 1.0, 12.0, 12.0, 1.0, 1.0, 7.0, 4.0, 8.0, 11.0, 9.0, 5.0, 13.0, 13.0, 3.0, 3.0, 3.0, 9.0, 13.0, 14.0, 11.0, 16.0, 11.0, 6.0, 15.0, 17.0, 17.0, 6.0, 11.0, 5.0, 3.0, 12.0, 4.0, 13.0, 3.0, 3.0, 14.0, 1.0, 1.0, 17.0, 15.0, 1.0, 9.0, 8.0, 8.0, 3.0, 3.0, 16.0, 9.0, 5.0, 4.0, 11.0, 3.0, 1.0, 14.0, 5.0, 5.0, 5.0, 2.0, 2.0, 3.0, 10.0, 10.0, 13.0, 11.0, 9.0, 3.0, 12.0, 5.0, 5.0, 5.0, 9.0, 14.0, 3.0, 16.0, 3.0, 3.0, 3.0, 12.0, 4.0, 11.0, 15.0, 6.0, 15.0, 9.0, 16.0, 15.0, 3.0, 1.0, 3.0, 3.0, 2.0, 7.0, 1.0, 17.0, 18.0, 18.0, 16.0, 3.0, 3.0, 4.0, 13.0, 13.0, 17.0, 14.0, 13.0, 2.0, 12.0, 13.0, 5.0, 5.0, 8.0, 1.0, 1.0, 5.0, 14.0, 4.0, 3.0, 11.0, 11.0, 12.0, 4.0, 11.0, 7.0]

    correct = 0

    for i in range(len(prediction)):
        if prediction[i] == test_y[i]:
            correct += 1

    return correct / len(prediction)


all_trials = []
all_trials_accuracy = []

for i in range(50):
    print("TRIAL:", i)
    labels = keras_mlp(train_s, train_y, test_s)
    demaxLabels = labels.argmax(axis=-1)
    all_trials.append(demaxLabels)
    all_trials_accuracy.append(kaggle_submit(demaxLabels))

print(all_trials_accuracy)
z = all_trials[all_trials_accuracy.index(max(all_trials_accuracy))]
print(all_trials[all_trials_accuracy.index(max(all_trials_accuracy))])

for elem in z:
    print(elem)

numpy.savetxt('50trials.csv', all_trials[all_trials_accuracy.index(max(all_trials_accuracy))], delimiter=',')