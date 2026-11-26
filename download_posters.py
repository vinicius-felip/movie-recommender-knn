"""Baixa os pôsteres uma única vez para uso local no seminário.

Imagens promocionais de terceiros; uso ilustrativo no seminário.
Não é necessário acessar a internet ao executar seminar.py após o download.
"""
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image, ImageOps

OUT = Path(__file__).resolve().parent / 'assets' / 'posters'
POSTERS = {
    'star_wars.jpg': 'https://www.originalfilmart.com/cdn/shop/products/star_wars_1977_style_A_linen_original_film_art_2000x.webp?v=1661377156',
    'terminator_2.jpg': 'https://filmartgallery.com/cdn/shop/files/Terminator-2-Vintage-Movie-Poster-Original_a7f7f2ca_5000x.jpg?v=1771933364',
    'titanic.jpg': 'https://www.originalfilmart.com/cdn/shop/products/titanic_1997_styleA_original_film_art_ff6e5718-c79d-4bfd-850b-1068970032f2_5000x.jpg?v=1668710472',
    'pretty_woman.jpg': 'https://filmartgallery.com/cdn/shop/files/Pretty-Woman-Vintage-Movie-Poster-Original_affefbcd_5000x.jpg?v=1771983421',
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    failures = []
    for name, url in POSTERS.items():
        path = OUT / name
        try:
            if path.is_file():
                with Image.open(path) as existing:
                    existing.verify()
                print(f'{name}: já disponível')
                continue
            request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(request, timeout=45) as response:
                data = response.read(30_000_001)
            if len(data) > 30_000_000:
                raise ValueError('Imagem maior que o limite de 30 MB')
            with Image.open(BytesIO(data)) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
                image.thumbnail((480, 720), Image.Resampling.LANCZOS)
                image.save(path, 'JPEG', quality=88, optimize=True)
            print(f'{name}: salvo')
        except Exception as exc:
            failures.append(name)
            print(f'{name}: {exc}. Salve o pôster manualmente em {path}.')
    if failures:
        print('Downloads pendentes: ' + ', '.join(failures))
        return 1
    print('Pôsteres disponíveis. O seminário pode ser executado offline.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
