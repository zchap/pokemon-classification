import classification
import test_classification


def run_pokemon_classification():
    # This function runs our classification algorithm using the methods in 'classification'
    vector_i = classification.load_pokemon_images('TestImages')
    vector_s = classification.load_pokemon_stats('PokemonData/UnlabeledTestMetadata.csv')

    # vector_x = vector_i + vector_s

    for p in range(0, len(vector_s)):  # indexing each pokemon
        for s in range(0, len(vector_s[p])):  # indexing each stat for each pokemon
            vector_s[p][s] = float(vector_s[p][s])
        print(vector_s[p])  # here for testing purposes

    return vector_s  # after every inner vector of vector_s has been converted to a float for easy arithmetic
    # vector_s begins with the data for "Type" because the number has been removed in 'load_pokemon_stats' in
    # classification.py


run_pokemon_classification()  # expected: 201 vectors of floats, given 201 vectors of floats

