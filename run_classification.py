import numpy
import tensorflow as tf
import numpy.linalg as npl
import keras.utils
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import classification
import test_classification
from sklearn.neighbors import KNeighborsClassifier  # K-NN
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC  # linear-SVM
from sklearn.svm import SVC
from sklearn import svm, tree
from sklearn.model_selection import cross_validate
# New changes for feature extraction
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, recall_score, precision_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
from pandas import DataFrame
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.optimizers import RMSprop
from keras.layers.normalization import BatchNormalization
import neural_net_classification as nn

# Loading image vector, stats vector, and labels.
train_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
train_i = numpy.array(classification.load_poke_images('TrainingImages'))
train_y = numpy.array(list(map(int, classification.load_training_labels('PokemonData/TrainingMetadata.csv'))))
train_pixels = list(train_i.flatten().reshape(601, 27648))

test_s = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')
test_i = numpy.array(classification.load_poke_images('TestImages'))
test_pixels = list(test_i.flatten().reshape(201, 27648))

# Converting stats matrix to float from string, so that classifiers can be used on them.
for p in range(0, len(train_s)):  # indexing each pokemon
    for s in range(0, len(train_s[p])):  # indexing each stat for each pokemon
        train_s[p][s] = float(train_s[p][s])
for p in range(0, len(test_s)):  # indexing each pokemon
    for s in range(0, len(test_s[p])):  # indexing each stat for each pokemon
        test_s[p][s] = float(test_s[p][s])

# Converting labels vector to float from string, so that classifiers can be used on them.

# Horizontal concatenation of S and I to form X = [S I]. Had to be done using a loop instead of with .concatenate
# because of type differences in the loaded image data and the loaded statistics data stemming from the fact that
# I is represented by (r, g, b) tuples.
train_x = []
for n in range(0, len(train_i)):
    train_x.append(numpy.array(train_s[n] + list(train_pixels[n])))
train_x = numpy.array(train_x)

test_x = []
for n in range(0, len(test_i)):
    test_x.append(numpy.array(test_s[n] + list(test_pixels[n])))
test_x = numpy.array(test_x)

# PCA for train_i and test_i
y_train = keras.utils.to_categorical(train_y, 19)
scaler = StandardScaler()
scaler.fit(train_pixels)
fit_train_i = scaler.transform(train_pixels)
fit_test_i = scaler.transform(test_pixels)
pca = PCA(n_components=0.8)
i_pca_train = pca.fit_transform(fit_train_i)
i_pca_test = pca.transform(fit_test_i)
pca_std = numpy.std(i_pca_train)
print(i_pca_train.shape)
print(i_pca_test.shape)
# Inverse PCA (if needed to get original matrix dimensions)
inv_pca_train = pca.inverse_transform(i_pca_train)
new_train = scaler.inverse_transform(inv_pca_train)
inv_pca_test = pca.inverse_transform(i_pca_test)
new_test = scaler.inverse_transform(inv_pca_test)
print(new_train.shape)
print(new_test.shape)

def k_nn(k):
    """
    K-NN classifier.
    :param k: The K value for K-NN
    :return: A k-nearest neighbors classifier with K = k.
    """
    n_neighbors = k
    classifier = KNeighborsClassifier(n_neighbors, weights='distance')
    return classifier


def linear_svm():
    """
    Linear SVM classifier.
    """
    classifier = LinearSVC()
    return classifier


def linear_kernel_svm():
    """
    Linear kernel SVM classifier. Should be the same as the above.
    """
    classifier = svm.SVC(kernel='linear')
    return classifier


def rbf_kernel_svm():
    """
    A radial basis function (Gaussian) classifier.
    """
    classifier = svm.SVC(kernel='rbf')
    return classifier


def poly_kernel_svm():
    """
    Polynomial kernel SVM classifier.
    """
    classifier = svm.SVC(kernel='poly', degree=3)
    return classifier


def neural_net():
    """
    Multi-layer perceptron classifier using stochastic gradient descent and rectifier activation function.
    """
    classifer = MLPClassifier(activation='relu', solver='sgd', shuffle=True, max_iter=1500)
    return classifer


def decision_tree():
    """
    Decision tree classifier.
    """
    classifier = tree.DecisionTreeClassifier()
    return classifier


def cross_validation(classifier, vec_x, vec_y):
    """
    Checks the accuracy of a given classifier on X and y. Prints the CV scores which is used to determine model
    accuracy.
    :param classifier: The classifier that we want to use to predict labels from vec_x.
    :param vec_x: Input feature matrix.
    :param vec_y: Input labels vector.
    """
    cv_results = cross_validate(classifier, vec_x, vec_y, cv=10, return_train_score=False)
    sorted(cv_results.keys())
    print(cv_results['test_score'])  # scores our classifier's accuracy


def prediction_probs(classifier, x_train, y_train, x_test, log_prob=False):
    """
    Get probability for each classification to easily combine later
    Note: Will only work for classifiers with built-in 'predict_proba' and 'predict_log_proba' methods.
    Input: classifier - Classifier created by pre-existing sklearn method
           log_prob - Boolean True if want the log probability, False if not (default)
    Output: probs - List? of probabilities for each prediction
    """
    # First fit the classifier to the training data
    model = classifier.fit(x_train, y_train)
    # Then use the fitted classifier to make a prediction on the test data and print the probabilities
    if log_prob:
        y_probs = model.predict_log_proba(x_test)
    else:
        y_probs = model.predict_proba(x_test)

    return y_probs


def combine_probs(a_probs, b_probs):
    """
    Input: a_probs, b_probs - Numpy arrays of the form output by prediction_probs. They contain the
           prediction probabilities made by the classifier for each class.
    Output: prediction - List containing prediction for each Pokemon
    Helper function to combine the predictions made by the two classifiers
    """
    prediction = []
    for i in range(len(a_probs)):
        # avg will contain the average probability for each class
        avg = (a_probs[i] + b_probs[i]) / 2
        # add class with highest probability to prediction
        prediction.append(numpy.argmax(avg))
    return prediction


def matt_cross_validate(x, y, cv=3, log_prob=False):
    """
    Input: x - Training data x values
           y - Training data y values
           cv - Number of subsets to use for cross-validation
           log_prob - Boolean True if want the log probability, False if not (default)
    Output: scores - List of percentages indicating how accurate the classifiers
                     were (one percentage for each subset in the cross-validation)
    """

    # use LinearSVC() instead of svm.SVC(kernel = 'linear') because it should be faster this way
    linear_svc = LinearSVC()
    # do this extra step for the linear SVC because the LinearSVC() classifier doesn't have a predict_proba() method
    s_classifier = CalibratedClassifierCV(linear_svc, method='sigmoid', cv=3)
    i_classifier = neural_net()

    scores = []
    # Note: this isn't a true cross-validation. there's nothing to prevent it from double counting/having overlapping subsets.
    for count in range(cv):
        print('Starting CV ' + str(count))
        # Note: train_test_split step is instantaneous
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1 / cv))

        # Note: This step takes a long time
        # Note: Dimensions of s_probs are 601 x 18 on full data
        s_probs = prediction_probs(s_classifier, x_train, y_train, x_test, log_prob)
        i_probs = prediction_probs(i_classifier, x_train, y_train, x_test, log_prob)

        # Note: combine_probs step is pretty fast
        prediction = combine_probs(s_probs, i_probs)

        # check accuracy
        accuracy = accuracy_score(y_test, prediction)
        scores.append(accuracy)

    return scores


def matt_predict(x_train, y_train, x_test, y_test, log_prob=False):
    """
    This is for getting a prediction from two classifiers
    """

    # use LinearSVC() instead of svm.SVC(kernel = 'linear') because it should be faster this way
    linear_svc = linear_svm()
    # do this extra step for the linear SVC because the LinearSVC() classifier doesn't have a predict_proba() method
    s_classifier = CalibratedClassifierCV(linear_svc, method='sigmoid', cv=3)
    i_classifier = neural_net()

    # Note: This step takes a long time
    # Note: Dimensions of s_probs are 601 x 18 on full data
    s_probs = prediction_probs(s_classifier, x_train, y_train, x_test, log_prob)
    i_probs = prediction_probs(i_classifier, x_train, y_train, x_test, log_prob)

    # Note: combine_probs step is pretty fast
    prediction = combine_probs(s_probs, i_probs)

    # check accuracy
    accuracy = accuracy_score(y_test, prediction)

    return accuracy, prediction

    # cross_validation(k_nn(9), x_train_pca, y)
    # cross_validation(linear_svm(), x_train_pca, y)
    # cross_validation(linear_kernel_svm(), vector_x, y)
    # cross_validation(poly_kernel_svm(), x_train_pca, y)
    # cross_validation(rbf_kernel_svm(), x_train_pca, y)
    # cross_validation(neural_net(), x_train_pca, y)
    # cross_validation(decision_tree(), x_train_pca, y)

# Looking at reducing the dimensionality of X using PCA
# def pca_data(n_components,train_x):
#     pca = PCA(n_components=n_components, whiten = True).fit(train_x)
#     x_train_pca = pca.transform(train_x)
#     return x_train_pca

# #pca_data(.8,vector_x)
# print("start")
# pca = PCA(n_components=.8).fit(data_image)
# print("Done computing pca")
# train_pca = pca.transform(data_image)
# #print (x_train_pca)
# print ("Dimension of pca reduced data_image that conserves 80% of variance:", train_pca.shape)
# print ("Original dimensions of data_image:", data_image.shape)

# # PCA on image_i (giving error)
# pca = PCA(n_components=.8, whiten = True).fit(vector_i)
# i_train_pca = pca.transform(vector_i)
# #print (i_train_pca)
# print ("Dimension of pca reduced vector_i that conserves 80% of variance:", i_train_pca.shape)
# print ("Original dimensions of vector_i:", vector_i.shape)

# Using Keras Library for Convolutional Neural Network
# Define model architecture
# def keras_nn_classifier():
#     y_train = keras.utils.to_categorical(train_y, num_classes + 1)
#     scaler = StandardScaler()
#     scaler.fit(train_pixels)
#     fit_train_i = scaler.transform(train_pixels)
#     fit_test_i = scaler.transform(test_pixels)
#     pca = PCA(n_components=0.8)
#     i_pca_train = pca.fit_transform(fit_train_i)
#     i_pca_test = pca.transform(fit_test_i)
#     pca_std=numpy.std(i_pca_train)
#     print(i_pca_train.shape)
#     print(i_pca_test.shape)
#     inv_pca_train = pca.inverse_transform(i_pca_train)
#     new_train = scaler.inverse_transform(inv_pca_train)
#     inv_pca_test = pca.inverse_transform(i_pca_test)
#     new_test = scaler.inverse_transform(inv_pca_test)
#     print(new_train.shape)
#     print(new_test.shape)
#
#     model = Sequential()
#     model.add(Conv2D(32, (5, 5), input_shape=(96, 96, 3), use_bias = True))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     # model.add(Conv2D(32, (3, 3), use_bias = True))
#     # model.add(Activation('relu'))
#     # model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Conv2D(64, (3, 3), use_bias = True))
#     model.add(Activation('relu'))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(128))
#     model.add(Activation('relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(19))
#     model.add(Activation('softmax'))
#
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
#     data_aug = ImageDataGenerator(rotation_range = 10, shear_range = 0.1, zoom_range = 0.1, horizontal_flip = True)
#     # Fit the model
#     y_new = numpy.array(train_y).reshape((-1, 1))
#     model.fit(train_i, y_new, epochs=27, shuffle = True)
#     #model.fit_generator(data_aug.flow(train_i,y_new), epochs=27, shuffle = True)
#     y_pred = model.predict(train_i)
#     y_pred_labels = y_pred.argmax(axis=-1)
#     y_pred_test = model.predict(test_i)
#     y_pred_test_labels = y_pred_test.argmax(axis=-1)
#     print("These are predicted training labels")
#     print(y_pred_labels)
#     print("# of labels: " + str(y_pred_labels.size))
#
#     acc = numpy.sum(y_pred_labels == train_y) / numpy.size(y_pred_labels)
#     print("Test accuracy on training labels = {}".format(acc))
#     print("These are the predicted testing labels")
#     print(y_pred_test_labels)
#     for i in range(0, len(y_pred_test_labels)):
#         print(y_pred_test_labels[i])
#
#
#     #y_pred = model.predict(test_i)
#     return

# labels = nn.labels_from_sparse_matrix(keras_nn_classifier())
# for label in labels:
#     print(label)


# recall = recall_score(y_new,y_pred,average=None)
# precision=precision_score(y_new,y_pred,average=None)
# accuracy=accuracy_score(y_new,y_pred)
# print("Recall score="+str(recall))
# print("Precision score="+str(precision))
# print("Accuracy score="+str(accuracy))

def keras_mlp(x_train, y_train, x_test):
    batch_size = 32
    num_classes = 18
    epochs = 100 # this seems to be the sweet spot, at least with the other parameters as they are now
    dense = 601

    x_train = numpy.array(x_train)
    x_train = x_train.astype('float32')
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes + 1)

    model = Sequential()
    model.add(Dense(dense, activation='relu', input_shape=(16,)))
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
    #model.fit(x_train, y_train, epochs=epochs, verbose=1)

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

# labels = keras_mlp(train_s, train_y, test_s)
# demaxLabels = labels.argmax(axis = -1)
# numpy.savetxt('mlpLabels.csv', demaxLabels, delimiter = ',')
#
# print(demaxLabels)
# print(kaggle_submit(demaxLabels))

#Converts arrays to txt files
def array_to_text(train, test):
    a = numpy.array(train)
    b = numpy.array(test)
    return a.tofile('train.txt', sep=" ", format = "%s"), b.tofile('test.txt', sep=" ", format = "%s")


# k = neural_net()
# k.fit(i_pca_train, train_y)
# labels = k.predict(i_pca_test)
# for i in range(0,len(labels)):
#     print (labels[i])

array_to_text(train_i, test_i)

