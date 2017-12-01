import numpy
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import classification
import test_classification
from sklearn.neighbors import KNeighborsClassifier #K-NN
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC #linear-SVM
from sklearn.svm import SVC
from sklearn import svm, tree
from sklearn.model_selection import cross_validate

vector_i = numpy.array(classification.load_pokemon_images('TrainingImages'))
vector_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
y = classification.load_training_labels('PokemonData/TrainingMetadata.csv')
pixels = list(vector_i.flatten().reshape(601, 27648))
num_pokemon = len(vector_i)

vector_x = []
for n in range(0, num_pokemon):
    print(n)
    vector_x.append(vector_s[n] + list(pixels[n]))

print(vector_x[0])

test_data = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')
# converting vector_s from string to float
for p in range(0, len(vector_s)):  # indexing each pokemon
    for s in range(0, len(vector_s[p])):  # indexing each stat for each pokemon
        vector_s[p][s] = float(vector_s[p][s])

# converting y from string to float
for index in range(0, len(y)):
    y[index] = float(y[index])


def k_nn_stats():
    # This function runs our classification algorithm using the methods in 'classification'

    # Note: each inner vector of vector_s only contains 15 elements, since the type of the pokemon has not been
    # given (we are supposed to find these labels)

    # vector_x = vector_i + vector_s

    # doing k-nn on just the stats (with k = 3)
    n_neighbors = 3
    classifier = KNeighborsClassifier(n_neighbors)
    return classifier

    # for i in range(0, num_pokemon):
    #     combined_vec = vector_s[i] + vector_i[i]
    #     vector_x.append(combined_vec)
    #
    # return vector_x


def linear_svm_stats():
    classifier = LinearSVC()
    return classifier


def linear_kernel_svm_stats():
    classifier = svm.SVC(kernel='linear')
    return classifier


def rbf_kernel_svm_stats():
    classifier = svm.SVC(kernel='rbf')
    return classifier

def poly_kernel_svm_stats():
    classifier = svm.SVC(kernel='poly', degree=3)
    return classifier

def neural_net_classifier():
    classifer = MLPClassifier(solver='sgd', max_iter = 1500, tol = 1e-6)
    return classifer

def decision_tree():
    classifier = tree.DecisionTreeClassifier()
    return classifier

def cross_validation(classifier):
    cv_results = cross_validate(classifier, vector_s, y, cv=10, return_train_score=False)
    sorted(cv_results.keys())
    print(cv_results['test_score'])


#k_nn_stats()
#linear_svm_stats() #gives an error for SVM
#linear_kernel_svm_stats()
#rbf_kernel_svm_stats()
#poly_kernel_svm_stats()
# expected: 201 vectors of floats, given 201 vectors of floats



# Question 2 part E:
# Result of cross validation: [ 0.22058824  0.13846154  0.125       0.15        0.21666667  0.20338983
#  0.15517241  0.14035088  0.21428571  0.12962963]
# Average is 16.935%
#classifier = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')

# Question 2 part F:
# Result of cross validation: [ 0.16176471  0.15384615  0.203125    0.2         0.18333333  0.16949153
#  0.25862069  0.15789474  0.14285714  0.12962963]
# Average is 17.605%
#classifier = svm.SVC(kernel='linear')

# Question 2 part G:
# Result of cross validation: [ 0.13235294  0.12307692  0.15625     0.21666667  0.21666667  0.15254237
#  0.10344828  0.14035088  0.19642857  0.09259259]
# Average is 15.303%
#classifier = svm.SVC(kernel='poly', degree=3)

# Question 2 part H:
# Result of cross validation: [ 0.13235294  0.15384615  0.125       0.13333333  0.13333333  0.15254237
#  0.13793103  0.12280702  0.125       0.14814815]
# Average is 13.642%
#classifier = svm.SVC(kernel='rbf')

# Question 2 part I:
# Neural Net:
# Result of cross validation: [ 0.13235294  0.15384615  0.125       0.13333333  0.08333333  0.11864407
#  0.10344828  0.14035088  0.125       0.11111111]
# Average is 12.264%

# Decision Tree:
# Result of cross validation: [ 0.17647059  0.09230769  0.15625     0.1         0.2         0.13559322
#  0.0862069   0.10526316  0.10714286  0.09259259]
# Average is 12.518%

#cross_validation(decision_tree())