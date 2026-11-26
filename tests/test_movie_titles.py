import unittest
from src.dataset import display_movie_title


class MovieTitleTests(unittest.TestCase):
    def test_catalog_articles(self):
        examples = {
            'Usual Suspects, The (1995)': 'The Usual Suspects (1995)',
            'Empire Strikes Back, The (1980)': 'The Empire Strikes Back (1980)',
            'Shawshank Redemption, The (1994)': 'The Shawshank Redemption (1994)',
            'Close Shave, A (1995)': 'A Close Shave (1995)',
            'American in Paris, An (1951)': 'An American in Paris (1951)',
            'Boys, Les (1997)': 'Les Boys (1997)',
            "Avventura, L' (1960)": "L'Avventura (1960)",
            'City of Lost Children, The (1995)': 'The City of Lost Children (1995)',
            'Star Wars (1977)': 'Star Wars (1977)',
            '2001: A Space Odyssey (1968)': '2001: A Space Odyssey (1968)',
            'Good, the Bad and the Ugly (1966)': 'Good, the Bad and the Ugly (1966)',
            'Hello, Dolly! (1969)': 'Hello, Dolly! (1969)',
        }
        for raw, expected in examples.items():
            with self.subTest(raw=raw):
                self.assertEqual(display_movie_title(raw), expected)
                self.assertEqual(display_movie_title(expected), expected)
