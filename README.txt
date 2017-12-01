pokemon-classification

ELEC 301 Pokemon Classification Project

Team Members:

Classification of Pokémon based on training data in the form of their images and statistics (16 stats ranging from ATTACK to HEIGHT).


   # k_nn_stats()
    # linear_svm_stats() #gives an error for SVM
    # linear_kernel_svm_stats()
    # rbf_kernel_svm_stats()
    # poly_kernel_svm_stats()
    # expected: 201 vectors of floats, given 201 vectors of floats

    # Question 2 part E:
    # Result of cross validation: [ 0.22058824  0.13846154  0.125       0.15        0.21666667  0.20338983
    #  0.15517241  0.14035088  0.21428571  0.12962963]
    # Average is 16.935%
    # classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

    # Question 2 part F:
    # Result of cross validation: [ 0.16176471  0.15384615  0.203125    0.2         0.18333333  0.16949153
    #  0.25862069  0.15789474  0.14285714  0.12962963]
    # Average is 17.605%
    # classifier = svm.SVC(kernel='linear')

    # Question 2 part G:
    # Result of cross validation: [ 0.13235294  0.12307692  0.15625     0.21666667  0.21666667  0.15254237
    #  0.10344828  0.14035088  0.19642857  0.09259259]
    # Average is 15.303%
    # classifier = svm.SVC(kernel='poly', degree=3)

    # Question 2 part H:
    # Result of cross validation: [ 0.13235294  0.15384615  0.125       0.13333333  0.13333333  0.15254237
    #  0.13793103  0.12280702  0.125       0.14814815]
    # Average is 13.642%
    # classifier = svm.SVC(kernel='rbf')

    # Question 2 part I:
    # Neural Net:
    # Result of cross validation: [ 0.13235294  0.15384615  0.125       0.13333333  0.08333333  0.11864407
    #  0.10344828  0.14035088  0.125       0.11111111]
    # Average is 12.264%

    # Decision Tree:
    # Result of cross validation: [ 0.17647059  0.09230769  0.15625     0.1         0.2         0.13559322
    #  0.0862069   0.10526316  0.10714286  0.09259259]
    # Average is 12.518%

    # Cross Validation Results for big X (X = [S I]):
    # K-NN with K = 3
    # [0.10294118  0.10769231  0.15625     0.13333333  0.13333333  0.01694915
    #  0.0862069   0.10526316  0.08928571  0.09259259]
    # Average is 10.23%

    # Cross Validation Results for big X:
