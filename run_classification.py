
import numpy
import numpy.linalg as npl
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import classification
import test_classification
from sklearn.neighbors import KNeighborsClassifier  # K-NN
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC  # linear-SVM
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm, tree
from sklearn.model_selection import cross_validate, train_test_split

# Loading image vector, stats vector, and labels.
train_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
train_i = numpy.array(classification.load_pokemon_images('TrainingImages'))
train_y = classification.load_training_labels('PokemonData/TrainingMetadata.csv')
train_pixels = list(train_i.flatten().reshape(601, 27648))

test_s = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')
test_i = numpy.array(classification.load_pokemon_images('TestImages'))
test_pixels = list(test_i.flatten().reshape(201, 27648))

# Converting stats matrix to float from string, so that classifiers can be used on them.
for p in range(0, len(train_s)):  # indexing each pokemon
    for s in range(0, len(train_s[p])):  # indexing each stat for each pokemon
        train_s[p][s] = float(train_s[p][s])
for p in range(0, len(test_s)):  # indexing each pokemon
    for s in range(0, len(test_s[p])):  # indexing each stat for each pokemon
        test_s[p][s] = float(test_s[p][s])

# Converting labels vector to float from string, so that classifiers can be used on them.
for index in range(0, len(train_y)):
    train_y[index] = float(train_y[index])

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
    classifer = MLPClassifier(solver='sgd', max_iter=1500, tol=1e-6)
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
    model = classifier.fit(x_train,y_train)
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
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=(1/cv))

        # Note: This step takes a long time
        # Note: Dimensions of s_probs are 601 x 18 on full data
        s_probs = prediction_probs(s_classifier, x_train, y_train, x_test, log_prob)
        i_probs = prediction_probs(i_classifier, x_train, y_train, x_test, log_prob)

        # Note: combine_probs step is pretty fast
        prediction = combine_probs(s_probs,i_probs)

        # check accuracy
        accuracy = accuracy_score(y_test,prediction)
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
    prediction = combine_probs(s_probs,i_probs)

    # check accuracy
    accuracy = accuracy_score(y_test,prediction)

    return accuracy, prediction


# Using only S to predict y:
    # Question 2 part E: K-NN Classifier with K = 9
    # Result of cross validation:
    # [ 0.26470588  0.15384615  0.171875    0.11666667  0.21666667  0.22033898
    #   0.17241379  0.14035088  0.19642857  0.11111111]
    # Average is 17.65%
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=9, weights='distance')

    # Question 2 part F/G: Linear SVM (same as Linear Kernel SVM)
    # Result of cross validation: [ 0.16176471  0.15384615  0.203125    0.2         0.18333333  0.16949153
    #  0.25862069  0.15789474  0.14285714  0.12962963]
    # Average is 17.605%
    # classifier = svm.SVC(kernel='linear')

    # Question 2 part G: Polynomial Kernel SVM of Degree 3
    # Result of cross validation: [ 0.13235294  0.12307692  0.15625     0.21666667  0.21666667  0.15254237
    #  0.10344828  0.14035088  0.19642857  0.09259259]
    # Average is 15.303%
    # classifier = svm.SVC(kernel='poly', degree=3)

    # Question 2 part G: Gaussian (radial basis/RBF) Kernel
    # Result of cross validation: [ 0.13235294  0.15384615  0.125       0.13333333  0.13333333  0.15254237
    #  0.13793103  0.12280702  0.125       0.14814815]
    # Average is 13.642%
    # classifier = svm.SVC(kernel='rbf')

    # Question 2 part H: Neural Network (Multilayer Perceptron with stochastic gradient descent) using Rectifier
    # Activation Function and a tolerance of 1e-6
    # Result of cross validation: [ 0.13235294  0.15384615  0.125       0.13333333  0.08333333  0.11864407
    #  0.10344828  0.14035088  0.125       0.11111111]
    # Average is 12.264%
    # classifier = MLPClassifier(solver='sgd', max_iter=1500, tol=1e-6)

    # Question 2 part H: Decision Tree Classifier
    # Result of cross validation: [ 0.17647059  0.09230769  0.15625     0.1         0.2         0.13559322
    #  0.0862069   0.10526316  0.10714286  0.09259259]
    # Average is 12.518%
    # classifier = DecisionTreeClassifier()

# Using X (X = [S I]) to predict y:
    # Question 2 part E: K-NN Classifier with K = 9
    # Result of cross validation:
    # [0.10294118  0.10769231  0.15625     0.13333333  0.13333333  0.01694915
    #  0.0862069   0.10526316  0.08928571  0.09259259]
    # Average is 10.23%
    # classifier = neighbors.KNeighborsClassifier(n_neighbors=9, weights='distance')
    #
    # Since the data we are using to predict y is the full data matrix X, K-NN will not work too well here, as it is
    # best suited for smaller-scale classification problems. Additionally, I would expect the average CV score to be
    # higher, but I assume the way that I created X is not optimal.
    #
    # Question 2 part F: Linear SVM
    # Result of cross validation:
    # [ 0.08823529  0.03076923  0.125  0.08333333  0.11666667  0.06779661
    #   0.0862069   0.12280702  0.07142857  0.14814815]
    # Average is 9.40%
    # classifier = LinearSVC()
    #
    # Question 2 part F: Linear Kernel SVM
    # Result of cross validation:
    # [ 0.16176471  0.03076923  0.1875      0.06666667  0.08333333  0.05084746
    #   0.0862069   0.0877193   0.125       0.11111111]
    # Average is 9.91%
    # classifier = svm.SVC(kernel='linear')
    #
    # Using linear SVM, we get a model that is slightly less accurate than a K-NN model with K = 9
    # when using the full data matrix X to predict Y.
    # Overall, however, their accuracies do not differ too much, and both predict labels with similar accuracy.
    #
    # Question 2 part G: Polynomial Kernel SVM
    # Result of cross validation:
    # [ 0.08823529  0.06153846  0.125       0.08333333  0.16666667  0.05084746
    #   0.06896552  0.0877193   0.14285714  0.09259259]
    # Average is 9.68%
    # classifier = svm.SVC(kernel='poly', degree=3)
    #
    # Question 2 part G: Gaussian (RBF) Kernel SVM
    # Result of cross validation:
    # [ 0.13235294  0.13846154  0.125       0.13333333  0.13333333  0.13559322
    #   0.13793103  0.14035088  0.14285714  0.14814815]
    # Average is 13.67%
    # classifier = svm.SVC(kernel='rbf')
    #
    # The radial basis (Gaussian) kernel seems to be better than the other kernel SVMs on the full data matrix X.
    # However, the linear SVM kernel works better than the others on just the stats matrix S.
    #
    # Question 2 part H: Multi-layer perceptron with rectifier activator and stochastic gradient descent
    # Result of cross validation:
    # [0.02941176  0.06153846  0.09375     0.03333333  0.03333333  0.03389831
    #  0.06896552  0.01754386  0.05357143  0.03703704]
    # Average is 4.62% :(
    # classifier = MLPClassifier(solver='sgd', max_iter=1500, tol=1e-6)
    #
    # Question 2 part H: Decision Tree Classifier
    # Result of cross validation:
    # [0.07352941  0.09230769  0.109375    0.16666667  0.11666667  0.16949153
    #  0.15517241  0.12280702  0.10714286  0.12962963]
    # Average is 12.43%
    # classifier = DecisionTreeClassifier()
    #
    # The results of the decision tree and multi-layer perceptron with rectifier activator are much lower than I
    # expected. I believe the issue definitely lies in the way that X is created. The results are sub-par compared to
    # the results from the methods introduced in class.

# Question 2 part I:
# Since our highest cross validation averages were when we used only S to predict y (with K-NN for K = 9),
# the label predictions from this model for the test data are what we submitted to Kaggle. Even though we are low on
# the leaderboard as of now, the accuracy % given by Kaggle is very close to our predicted accuracy using
# cross validation. Our CV estimate is about .2 higher (17.6 as compared to 17.4), most likely due to overfitting of
# the data, since S is not full rank. Moving forward, we want to find the ideal format for X so that the classifiers
# above produce better CV scores than they do for just S. Doing this will require feature extraction and more
# into classification algorithms and the sklearn package.


# cross_validation(k_nn(9), train_x, y)
# cross_validation(linear_svm(), train_x, y)
# cross_validation(linear_kernel_svm(), train_x, y)
# cross_validation(poly_kernel_svm(), train_x, y)
# cross_validation(rbf_kernel_svm(), train_x, y)
# cross_validation(neural_net(), train_x, y)
# cross_validation(decision_tree(), train_x, y)

