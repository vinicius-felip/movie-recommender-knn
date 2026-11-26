"""Prepara as capas localmente, ANTES do seminário (internet só nesta etapa).

Fonte principal: https://github.com/mohanvpatta/movie-posters
Os nomes numéricos do arquivo correspondem aos IDs MovieLens 100K.
As imagens são material promocional de terceiros; não redistribua sem autorização.
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
import re
import shutil
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from urllib.parse import urlsplit, urlunsplit
from zipfile import ZipFile

from PIL import Image, ImageOps

from src.dataset import load_movies
from src.posters import CATALOG_DIR, POSTER_DIR, poster_path, valid_image

ARCHIVE_URL = 'https://codeload.github.com/mohanvpatta/movie-posters/zip/refs/heads/main'
MAPPING_URL = 'https://raw.githubusercontent.com/babu-thomas/movielens-posters/master/movie_poster.csv'
PAGE2 = {
    'star_wars.jpg': 'Star Wars (1977)',
    'terminator_2.jpg': 'Terminator 2: Judgment Day (1991)',
    'titanic.jpg': 'Titanic (1997)',
    'pretty_woman.jpg': 'Pretty Woman (1990)',
}


def fetch_archive(destination):
    request = Request(ARCHIVE_URL, headers={'User-Agent': 'MovieLens-Seminar/1.0'})
    with urlopen(request, timeout=60) as response, destination.open('wb') as output:
        total = 0
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > 350_000_000:
                raise ValueError('Arquivo excedeu o limite de 350 MB.')
            output.write(chunk)


def import_archive(archive, ids):
    """Importa somente arquivos <id>.jpg; não extrai caminhos do ZIP."""
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    imported = 0
    with ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            match = re.fullmatch(r'(\d+)\.jpg', Path(member.filename).name, re.I)
            if not match or member.is_dir() or member.file_size > 20_000_000:
                continue
            movie_id = int(match.group(1))
            if movie_id not in ids:
                continue
            path = poster_path(movie_id)
            if valid_image(path):
                continue
            try:
                with Image.open(BytesIO(zip_file.read(member))) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                    image.thumbnail((360, 540), Image.Resampling.LANCZOS)
                    temp = path.with_suffix('.tmp')
                    image.save(temp, 'JPEG', quality=85, optimize=True)
                    temp.replace(path)
                imported += 1
            except (OSError, ValueError, SyntaxError) as exc:
                print(f'ID {movie_id}: imagem ignorada ({exc}).')
    return imported



def fetch_bytes(url, limit=10_000_000):
    request = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urlopen(request, timeout=15) as response:
        data = response.read(limit + 1)
    if len(data) > limit:
        raise ValueError('Arquivo excede o limite de tamanho.')
    return data


def fill_missing(ids):
    """Fonte alternativa, só para IDs ainda sem imagem; execução prévia online."""
    rows = csv.reader(fetch_bytes(MAPPING_URL, 2_000_000).decode('utf-8-sig').splitlines())
    allowed = {'images-na.ssl-images-amazon.com', 'm.media-amazon.com', 'ia.media-imdb.com'}
    candidates = {}
    for row in rows:
        if len(row) < 2 or not row[0].isdigit():
            continue
        movie = int(row[0])
        url = row[1].strip()
        parsed = urlsplit(url)
        if movie in ids and parsed.scheme == 'https' and parsed.hostname in allowed:
            if not valid_image(poster_path(movie)):
                candidates[movie] = url
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    def download_one(item):
        movie, url = item
        parsed = urlsplit(url)
        # Os links antigos do IMDb podem ter migrado para o CDN atual.
        resized = re.sub(r'\.\.jpg$', '._V1_SX360.jpg', parsed.path, flags=re.I)
        modern = urlunsplit(('https', 'm.media-amazon.com', resized, '', ''))
        for candidate in dict.fromkeys([modern, url]):
            try:
                with Image.open(BytesIO(fetch_bytes(candidate))) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
                    image.thumbnail((360, 540), Image.Resampling.LANCZOS)
                    path = poster_path(movie)
                    temp = path.with_suffix('.tmp')
                    image.save(temp, 'JPEG', quality=85, optimize=True)
                    temp.replace(path)
                return True
            except Exception:
                continue
        return False

    print(f'Tentando completar {len(candidates)} capas da fonte alternativa...', flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        return sum(pool.map(download_one, candidates.items()))

def prepare_page2(titles):
    """Preserva capas do patch 013 e reutiliza o catálogo nas quatro restantes."""
    POSTER_DIR.mkdir(parents=True, exist_ok=True)
    by_title = {str(title).casefold(): movie for movie, title in titles.items()}
    for filename, title in PAGE2.items():
        path = POSTER_DIR / filename
        movie = by_title.get(title.casefold())
        if movie is None:
            continue
        catalog = poster_path(movie)
        if valid_image(path):
            # Aproveita a capa que o usuário já baixou, caso falte no catálogo.
            if not valid_image(catalog):
                CATALOG_DIR.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, catalog)
        elif valid_image(catalog):
            shutil.copyfile(catalog, path)


def report(titles):
    missing = [(int(movie), str(title)) for movie, title in titles.items()
               if not valid_image(poster_path(movie))]
    print(f'Capas locais válidas: {len(titles)-len(missing)}/{len(titles)}')
    absent = [name for name in PAGE2 if not valid_image(POSTER_DIR / name)]
    print('Página 2:', 'todas as quatro capas disponíveis' if not absent else ', '.join(absent))
    if missing:
        print('Filmes sem capa usarão o marcador "Sem capa". Exemplos:')
        for movie, title in missing[:8]:
            print(f'  {movie}: {title}')
        print('Para completar manualmente: assets/posters/catalog/<movie_id>.jpg')
    return missing, absent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Importa um ZIP já baixado, sem internet.')
    parser.add_argument('--check', action='store_true', help='Verifica os arquivos locais, sem internet.')
    parser.add_argument('--fill-missing', action='store_true', help='Tenta baixar apenas as capas faltantes de uma segunda fonte.')
    args = parser.parse_args()
    try:
        titles = load_movies().set_index('movie_id').title.to_dict()
    except FileNotFoundError:
        parser.error('Base ausente. Execute python download_data.py primeiro.')
    ids = set(titles)
    if args.archive and args.fill_missing:
        parser.error('Use --archive ou --fill-missing separadamente.')
    if args.check:
        report(titles)
        return 0
    prepare_page2(titles)
    if args.fill_missing:
        try:
            print(f'{fill_missing(ids)} capas adicionais importadas.')
        except Exception as exc:
            print(f'Fonte alternativa indisponível: {exc}')
            report(titles)
            return 1
    elif any(not valid_image(poster_path(movie)) for movie in ids):
        try:
            if args.archive:
                print(f'Importando {args.archive}...')
                count = import_archive(args.archive, ids)
            else:
                with TemporaryDirectory() as directory:
                    archive = Path(directory) / 'movie-posters.zip'
                    print('Baixando a coleção de capas MovieLens (uma única vez)...', flush=True)
                    fetch_archive(archive)
                    count = import_archive(archive, ids)
            print(f'{count} novas capas importadas.')
        except Exception as exc:
            print(f'Não foi possível preparar o ZIP: {exc}')
            print('Alternativa: baixe o ZIP em https://github.com/mohanvpatta/movie-posters e use --archive.')
            report(titles)
            return 1
    prepare_page2(titles)
    report(titles)
    print('O aplicativo lê apenas o cache local. Você já pode desconectar a internet.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
