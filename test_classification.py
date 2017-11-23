import classification
import unittest


class TestClassification(unittest.TestCase):
    def test_classification(self):
        self.assertEqual(len(classification.load_pokemon_images('TestImages')[0]), 96 * 96)  # should pass
        self.assertNotEqual(len(classification.load_pokemon_images('TestImages')[0]), 96*94)  # should pass

    def test_stats_load(self):
        # expected size of list below should be 602 (as this is the # of rows in
        self.assertEqual(len(classification.load_pokemon_stats('PokemonData/TrainingMetadata.csv')), 602)


unittest.main()
