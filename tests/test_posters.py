import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

import download_catalog_posters as downloader
from src import posters


class PosterTests(unittest.TestCase):
    def test_archive_and_offline_cards(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            catalog = root / 'catalog'
            archive = root / 'source.zip'
            image = BytesIO()
            Image.new('RGB', (120, 180), 'navy').save(image, 'JPEG')
            with ZipFile(archive, 'w') as z:
                z.writestr('repo/50.jpg', image.getvalue())
                z.writestr('repo/99.jpg', image.getvalue())
                z.writestr('repo/1.jpg', b'invalid')
                z.writestr('repo/50_99.jpg', image.getvalue())
            with patch.object(posters, 'CATALOG_DIR', catalog), \
                 patch.object(downloader, 'CATALOG_DIR', catalog), \
                 patch.object(downloader, 'POSTER_DIR', root), \
                 patch('urllib.request.urlopen', side_effect=AssertionError('Internet não permitida')):
                self.assertEqual(downloader.import_archive(archive, {1, 50}), 1)
                self.assertTrue(posters.valid_image(catalog / '50.jpg'))
                self.assertFalse((catalog / '99.jpg').exists())
                self.assertFalse((catalog / '1.jpg').exists())
                card = posters.movie_card(50, 'A & B <film>', 4.75, 20, 1)
                self.assertIn('data:image/jpeg;base64,', card)
                self.assertIn('A &amp; B &lt;film&gt;', card)
                self.assertIn('4.75', card)
                self.assertIn('20 vizinhos', card)
                self.assertIn('Sem capa', posters.movie_card(1, 'Sem arquivo'))
                downloader.prepare_page2({50: 'Star Wars (1977)'})
                self.assertTrue(posters.valid_image(root / 'star_wars.jpg'))
                self.assertEqual(downloader.import_archive(archive, {50}), 0)

    def test_corrupt_image_fallback(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / '1.jpg'
            path.write_bytes(b'not an image')
            self.assertIn('Sem capa', posters.image_html(path, 'Filme'))


if __name__ == '__main__':
    unittest.main()
