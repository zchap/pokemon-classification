import classification
import test_classification
from sklearn.neighbors import KNeighborsClassifier #K-NN

def run_pokemon_classification():
    # This function runs our classification algorithm using the methods in 'classification'
    vector_i = classification.load_pokemon_images('TrainingImages')
    vector_s = classification.load_training_stats('PokemonData/TrainingMetadata.csv')
    num_pokemon = len(vector_i)

    # Note: each inner vector of vector_s only contains 15 elements, since the type of the pokemon has not been
    # given (we are supposed to find these labels)

    # vector_x = vector_i + vector_s

    # converting vector_s from string to float
    for p in range(0, len(vector_s)):  # indexing each pokemon
        for s in range(0, len(vector_s[p])):  # indexing each stat for each pokemon
            vector_s[p][s] = float(vector_s[p][s])

    # doing k-nn on just the stats
    n_neighbors = 3
    y = classification.load_training_labels('PokemonData/TrainingMetadata.csv')

    test_data = classification.load_test_stats('PokemonData/UnlabeledTestMetadata.csv')

    neigh = KNeighborsClassifier(n_neighbors)
    neigh.fit(vector_s, y)
    print(neigh.predict(test_data))

    # for i in range(0, num_pokemon):
    #     combined_vec = vector_s[i] + vector_i[i]
    #     vector_x.append(combined_vec)
    #
    # return vector_x


run_pokemon_classification()
# expected: 201 vectors of floats, given 201 vectors of floats

