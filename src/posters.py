"""Pôsteres locais: nenhuma operação de rede neste módulo."""
from base64 import b64encode
from html import escape
from pathlib import Path

from PIL import Image

POSTER_DIR = Path(__file__).resolve().parents[1] / 'assets' / 'posters'
CATALOG_DIR = POSTER_DIR / 'catalog'


def poster_path(movie_id):
    return CATALOG_DIR / f'{int(movie_id)}.jpg'


def valid_image(path):
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError, SyntaxError):
        return False


def image_html(path, title, css_class='movie-poster'):
    """Incorpora os bytes locais; fallback explícito quando não houver capa."""
    title = escape(str(title), quote=True)
    try:
        with Image.open(path) as image:
            fmt = image.format
            image.verify()
        mime = {'JPEG': 'image/jpeg', 'PNG': 'image/png', 'WEBP': 'image/webp'}.get(fmt)
        if mime:
            encoded = b64encode(path.read_bytes()).decode('ascii')
            return f'<img class="{css_class}" src="data:{mime};base64,{encoded}" alt="Capa de {title}">'
    except (OSError, ValueError, SyntaxError):
        pass
    return f'<div class="{css_class} poster-unavailable" role="img" aria-label="Capa indisponível: {title}">Sem capa</div>'


def movie_card(movie_id, title, score=None, neighbors=None, rank=None):
    """HTML seguro para a seleção de filmes e para o ranking."""
    poster = image_html(poster_path(movie_id), title)
    title = escape(str(title))
    detail = ''
    if score is not None:
        detail = (f'<div class="movie-poster-score">{float(score):.2f} <small>/ 5</small></div>'
                  f'<div class="movie-poster-support">{int(neighbors)} vizinhos</div>')
    position = f'<span class="movie-poster-rank">{int(rank):02d}</span> ' if rank is not None else ''
    return (f'<div class="movie-poster-card">{poster}<div class="movie-poster-info">'
            f'<div class="movie-poster-title">{position}{title}</div>{detail}</div></div>')


POSTER_CSS = '''<style>
.movie-poster-card {display:flex;gap:12px;align-items:center;min-width:0;
    box-sizing:border-box;background:#191c24;border:1px solid #303540;border-radius:9px;padding:8px;}
.movie-poster {width:68px;height:100px;object-fit:contain;flex:none;display:block;
    border-radius:5px;background:#101218;box-sizing:border-box;}
.poster-unavailable {display:flex;align-items:center;justify-content:center;text-align:center;
    font-size:12px;color:#b6bac5;border:1px dashed #454b59;}
.movie-poster-info {min-width:0;flex:1;}
.movie-poster-title {font-size:1rem;font-weight:650;line-height:1.25;color:#f4f4f5;overflow-wrap:anywhere;}
.movie-poster-rank {color:#ff6977;font-size:.84rem;}
.movie-poster-score {color:#ff8790;font-size:1.36rem;font-weight:750;line-height:1.2;margin-top:6px;}
.movie-poster-score small {font-size:.75rem;color:#b6bac5;font-weight:400;}
.movie-poster-support {color:#b6bac5;font-size:.82rem;margin-top:3px;}
.movie-poster-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
    gap:12px;align-items:stretch;padding-bottom:18px;box-sizing:border-box;}
/* O último card precisa terminar antes dos controles/expanders do Streamlit.
   Padding no próprio HTML e no slider, não apenas gap dos wrappers externos. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(.movie-poster-card):not(:has(.movie-poster-grid)) {
    padding-bottom:16px;box-sizing:border-box;
}
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"][class*="st-key-rating-"],
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"][class*="st-key-seminar_rating-"] {
    padding-top:8px;padding-bottom:10px;box-sizing:border-box;
}
.st-key-seminar_recommendation_details {padding-top:8px;box-sizing:border-box;}
.st-key-seminar_recommendation_details [data-testid="stExpander"] {margin-bottom:8px;}
/* Mantém o botão Recomendar e os painéis dentro da altura da apresentação. */
@media(min-width:900px) and (min-height:651px) {
    .st-key-seminar_panel_results {max-height:min(495px,calc(100dvh - 310px)) !important;}
    .st-key-seminar_demo_inputs {max-height:min(405px,calc(100dvh - 385px)) !important;}
    .st-key-seminar_panel_results .movie-poster {width:60px;height:88px;}
    .st-key-seminar_controls_ratings .movie-poster {width:60px;height:88px;}
}

@media(max-width:700px) {.movie-poster-grid {grid-template-columns:1fr;}}
@media(max-height:650px) {.movie-poster {width:54px;height:80px;}}
</style>'''
