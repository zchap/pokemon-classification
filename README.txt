pokemon-classification
[0.23880597014925373, 0.24875621890547264, 0.22885572139303484, 0.3034825870646766, 0.22885572139303484, 0.263681592039801, 0.25870646766169153, 0.26865671641791045, 0.24378109452736318, 0.2537313432835821, 0.27860696517412936, 0.24875621890547264, 0.27860696517412936, 0.24875621890547264, 0.2537313432835821, 0.24875621890547264, 0.24378109452736318, 0.22885572139303484, 0.263681592039801, 0.24875621890547264, 0.23880597014925373, 0.25870646766169153, 0.21890547263681592, 0.25870646766169153, 0.26865671641791045, 0.26865671641791045, 0.24875621890547264, 0.24875621890547264, 0.24875621890547264, 0.208955223880597, 0.25870646766169153, 0.22388059701492538, 0.2736318407960199, 0.22885572139303484, 0.25870646766169153, 0.27860696517412936, 0.2537313432835821, 0.19402985074626866, 0.2736318407960199, 0.25870646766169153, 0.25870646766169153, 0.24378109452736318, 0.3034825870646766, 0.23880597014925373, 0.24875621890547264, 0.22388059701492538, 0.27860696517412936, 0.27860696517412936, 0.21890547263681592, 0.27860696517412936]
ELEC 301 Pokemon Classification Project

Team Members:

Classification of Pokémon based on training data in the form of their images and statistics (16 stats ranging from ATTACK to HEIGHT).

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
