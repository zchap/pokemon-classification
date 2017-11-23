import classification
import unittest

# global variables for testing
images_list = classification.load_pokemon_images('TestImages')[0]
stats_list = classification.load_pokemon_stats('PokemonData/UnlabeledTestMetadata.csv')


# Print tests that should pass by basic observation
def print_stats_list():
    print(stats_list)


def print_vector(input_vector):
    for vector in input_vector:
        print(vector)


class TestClassification(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(len(images_list), 96 * 96)  # should pass
        self.assertNotEqual(len(images_list), 96*94)  # should pass

    def test_stats_load(self):
        # expected size of list below should be 601 (first row has been removed so it is just the # of pokemon)
        self.assertEqual(len(stats_list), 201)

        # expected size of each inner list is 15 (pokemon number has been removed, so data starts with "Type")
        self.assertEqual(len(stats_list[0]), 15)


print_stats_list()
if __name__ == '__main__':
    unittest.main()


