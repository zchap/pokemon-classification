
import numpy
import numpy.linalg as npl
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

# Loading image vector, stats vector, and labels.
vector_i = numpy.array(classification.load_pokemon_images('TrainingImages'))
vector_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
y = classification.load_training_labels('PokemonData/TrainingMetadata.csv')
pixels = list(vector_i.flatten().reshape(601, 27648))
num_pokemon = len(vector_i)
test_data = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')

# Converting stats matrix to float from string, so that classifiers can be used on them.
for p in range(0, len(vector_s)):  # indexing each pokemon
    for s in range(0, len(vector_s[p])):  # indexing each stat for each pokemon
        vector_s[p][s] = float(vector_s[p][s])

# Converting labels vector to float from string, so that classifiers can be used on them.
for index in range(0, len(y)):
    y[index] = float(y[index])

# Horizontal concatenation of S and I to form X = [S I]. Had to be done using a loop instead of with .concatenate
# because of type differences in the loaded image data and the loaded statistics data stemming from the fact that
# I is represented by (r, g, b) tuples.
vector_x = []
for n in range(0, num_pokemon):
    vector_x.append(numpy.array(vector_s[n] + list(pixels[n])))
vector_x = numpy.array(vector_x)


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


# cross_validation(k_nn(9), vector_x, y)
# cross_validation(linear_svm(), vector_x, y)
# cross_validation(linear_kernel_svm(), vector_x, y)
# cross_validation(poly_kernel_svm(), vector_x, y)
# cross_validation(rbf_kernel_svm(), vector_x, y)
# cross_validation(neural_net(), vector_x, y)
# cross_validation(decision_tree(), vector_x, y)

