import classification
import test_classification


def run_pokemon_classification():
    # This function runs our classification algorithm using the methods in 'classification'
    vector_i = classification.load_pokemon_images('TestImages')
    vector_s = classification.load_pokemon_stats('PokemonData/UnlabeledTestMetadata.csv')
    num_pokemon = len(vector_i)

    # Note: each inner vector of vector_s only contains 15 elements, since the type of the pokemon has not been
    # given (we are supposed to find these labels)

    # vector_x = vector_i + vector_s

    for p in range(0, len(vector_s)):  # indexing each pokemon
        for s in range(0, len(vector_s[p])):  # indexing each stat for each pokemon
            vector_s[p][s] = float(vector_s[p][s])

    vector_x = []
    for i in range(0, num_pokemon):
        combined_vec = vector_s[i] + vector_i[i]
        vector_x.append(combined_vec)

    return vector_x


test_classification.print_vector(run_pokemon_classification()[0])
# expected: 201 vectors of floats, given 201 vectors of floats

