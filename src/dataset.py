from pathlib import Path
import re
import pandas as pd
DATA_DIR = Path(__file__).resolve().parents[1] / 'data' / 'ml-100k'

def load_ratings():
    return pd.read_csv(DATA_DIR / 'u.data', sep='\t', names=['user_id','movie_id','rating','timestamp'])

def load_movies():
    return pd.read_csv(DATA_DIR / 'u.item', sep='|', encoding='latin-1', header=None, usecols=[0,1], names=['movie_id','title'])

# A MovieLens coloca alguns artigos no final do título para ordenação.
# Normalizar apenas para exibição, sem alterar o arquivo original nem os IDs.
_TRAILING_ARTICLE = re.compile(
    r"^(?P<title>.+?),\s*(?P<article>The|An|A|Les|Le|La|L['’]|Los|Las|El|Der|Die|Das|Il|Lo|I|Gli)"
    r"(?P<suffix>(?:\s+\([^()]*\))*)$", re.IGNORECASE
)


def display_movie_title(title):
    """Ex.: 'Usual Suspects, The (1995)' -> 'The Usual Suspects (1995)'."""
    text = str(title)
    match = _TRAILING_ARTICLE.fullmatch(text)
    if match is None:
        return text
    article = match['article']
    separator = '' if article.endswith(("'", '’')) else ' '
    return f"{article}{separator}{match['title']}{match['suffix']}"


def build_rating_matrix(ratings):
    return ratings.pivot(index='user_id', columns='movie_id', values='rating').sort_index().sort_index(axis=1)

def dataset_summary(ratings, movies):
    n, m = ratings.user_id.nunique(), ratings.movie_id.nunique()
    print(f'Usuários: {n}\nFilmes avaliados: {m}\nCatálogo: {len(movies)}\nAvaliações: {len(ratings)}\nEsparsidade: {1-len(ratings)/(n*m):.2%}')
    print(ratings.rating.value_counts().sort_index())

GENRES = ['unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western']

def load_genres():
    raw = pd.read_csv(DATA_DIR/'u.item', sep='|', encoding='latin-1', header=None)
    return raw.set_index(0).iloc[:,4:23].set_axis(GENRES, axis=1).drop(columns='unknown')

def load_users():
    return pd.read_csv(DATA_DIR/'u.user', sep='|', names=['user_id','age','sex','occupation','zip']).set_index('user_id')
