from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
import classification
import test_classification
from sklearn.neighbors import KNeighborsClassifier #K-NN
from sklearn.svm import LinearSVC #linear-SVM
from sklearn.svm import SVC
from sklearn import svm

vector_i = classification.load_pokemon_images('TrainingImages')
vector_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
y = classification.load_training_labels('PokemonData/TrainingMetadata.csv')

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
    neigh = KNeighborsClassifier(n_neighbors)
    neigh.fit(vector_s, y)
    print(neigh.predict(test_data))

    # for i in range(0, num_pokemon):
    #     combined_vec = vector_s[i] + vector_i[i]
    #     vector_x.append(combined_vec)
    #
    # return vector_x


def linear_svm_stats():
    classifier = LinearSVC()
    classifier.fit(vector_s, y)
    print(classifier.predict(test_data))

def linear_kernel_svm_stats():
    linear_svm = svm.SVC(kernel='linear')
    linear_svm.fit(vector_s, y)
    print(linear_svm.predict(test_data))

def rbf_kernel_svm_stats():
    rbf_svm = svm.SVC(kernel='rbf')
    rbf_svm.fit(vector_s, y)
    print(rbf_svm.predict(test_data))

def poly_kernel_svm_stats():
    poly_svm = svm.SVC(kernel='poly', degree=3)
    poly_svm.fit(vector_s, y)
    print(poly_svm.predict(test_data))

#k_nn_stats()
#linear_svm_stats() #gives an error for SVM
#linear_kernel_svm_stats()
#rbf_kernel_svm_stats()
#poly_kernel_svm_stats()
# expected: 201 vectors of floats, given 201 vectors of floats


n_neighbors = 18

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

classifier = MLPClassifier(solver='sgd')
print('stepa')
cv_results = cross_validate(classifier, vector_s, y, cv=10, return_train_score=False)
sorted(cv_results.keys())
print(cv_results['test_score'])
