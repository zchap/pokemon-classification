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
    #print(classifier.predict(test_data))


def linear_kernel_svm_stats():
    linear_svm = svm.SVC("linear")

def rbf_kernel_svm_stats():
    rbf_svm = svm.SVC("rbf")

def poly_kernel_svm_stats():
    poly_svm = svm.SVC("poly")

#k_nn_stats()
linear_svm_stats()
# expected: 201 vectors of floats, given 201 vectors of floats

