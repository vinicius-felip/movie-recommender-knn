"""Seminário interativo. Executar: python -m streamlit run seminar.py"""
import inspect
from html import escape
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st

from src import metrics
from src.dataset import load_ratings, build_rating_matrix
from src.knn import MovieKNN
from src.hybrid import HybridKNN
from src.posters import image_html, valid_image
from ranking_metrics import precision_recall_at_n

ROOT = Path(__file__).resolve().parent
PAGES = [
    'Introdução', 'Problema real e motivação', 'Sistemas de recomendação',
    'Aprendizado baseado em instâncias', 'Como funciona o k-NN',
    'Métricas de distância e similaridade', 'MovieLens 100K', 'Primeira implementação',
    'Evolução: k-NN híbrido', 'Metodologia experimental', 'Resultados experimentais',
    'Implementação do algoritmo', 'Demonstração prática', 'Conclusão',
]
LABELS = {'tradicional':'Tradicional','generos':'Gêneros','perfil':'Perfil ampliado','baseline':'Baseline'}

st.set_page_config(page_title='MovieLens | Seminário de IA', layout='wide', initial_sidebar_state='collapsed')
st.markdown('''<style>
.stApp {background:#101218;color:#f4f4f5;}
/* Modo apresentação: o cabeçalho nativo não deve cobrir a navegação. */
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background:transparent !important;
    box-shadow:none !important;
    backdrop-filter:none !important;
    pointer-events:none !important;
}
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"],
[data-testid="stMainMenu"], #MainMenu, [data-testid="stDecoration"] {
    display:none !important;
}
/* Mantém o acesso ao menu lateral nas diferentes versões do Streamlit. */
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"],
[data-testid="stExpandSidebarButton"], [data-testid="stSidebarCollapseButton"] {
    pointer-events:auto !important;
}
/* Um padrão para as 14 páginas. O espaço depende do grupo de conteúdo,
   e não do número da página. Cards têm sua própria grade; controles próximos
   usam containers nomeados, sem comprimir o restante do slide. */
.stApp {
    --section-gap:20px;
    --card-gap:24px;
    --panel-gap:12px;
    --control-gap:8px;
    --block-gap:16px;
}
[class*="st-key-seminar_slide_"] {--block-gap:var(--section-gap);}
.st-key-seminar_navigation, .st-key-seminar_heading {--block-gap:8px;}
[class*="st-key-seminar_slide_"] [data-testid="stColumn"] {--block-gap:var(--section-gap);}
[class*="st-key-seminar_panel_"] {--block-gap:var(--panel-gap);}
[class*="st-key-seminar_controls_"] {--block-gap:var(--control-gap);}
[data-testid="stMainBlockContainer"], .block-container {
    max-width:1360px;
    padding-top:1.8rem;
    padding-bottom:1.5rem;
}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"],
.block-container.stVerticalBlock {
    display:flex !important;
    flex-direction:column !important;
    gap:var(--block-gap) !important;
}
/* Inclui os grupos de cards e colunas que a versão atual envolve em wrappers. */
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] > :is([data-testid="stLayoutWrapper"], [data-testid="stElementContainer"], [data-testid="stHorizontalBlock"], [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stVerticalBlock"]) {
    margin-block:0 !important;
    flex-shrink:0 !important;
}
[data-testid="stMainBlockContainer"] [data-testid="stHorizontalBlock"] {
    /* O gap horizontal vem de st.columns: o Streamlit também calcula
       a largura das colunas a partir dele. Não sobrescrever só uma metade. */
    row-gap:var(--panel-gap) !important;
}
/* Retira somente os blocos técnicos invisíveis do fluxo de layout.
   O iframe de atalhos continua no DOM e pode executar seu script. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(style):not(:has(.slide-card, .slide-lead, .slide-key, .slide-flow, .slide-hero)),
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"][height="0px"]:has(iframe),
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"]:has(iframe[height="0"]) {
    position:absolute !important;
    width:0 !important;
    height:0 !important;
    min-height:0 !important;
    margin:0 !important;
    padding:0 !important;
    overflow:hidden !important;
    pointer-events:none !important;
}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {font-size:1.05rem;line-height:1.45;}
.slide-lead {
    font-size:clamp(1.2rem,2.3vw,2rem);line-height:1.25;font-weight:650;
    max-width:1100px;margin:0 0 .22rem;padding:.12rem 0 .3rem;color:#f4f4f5;
}
.slide-kicker {font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:#b6bac5;margin:0;}
.slide-cards {
    display:grid;grid-template-columns:repeat(var(--card-columns),minmax(0,1fr));
    gap:var(--card-gap);align-items:stretch;
}
.slide-card {
    box-sizing:border-box;background:#191c24;border:1px solid #303540;border-radius:12px;
    padding:1.15rem 1.3rem;min-height:136px;height:100%;margin:0;
}
.slide-cards.compact .slide-card {min-height:0;padding:.85rem 1rem;}
.slide-card.focus {border-color:#f04452;border-left:4px solid #f04452;}
.slide-card h3 {font-size:1.2rem !important;margin:0 0 .55rem !important;padding:0 !important;}
.slide-card p {font-size:1.03rem;line-height:1.45;color:#d5d7de;margin:0;}
.slide-card .card-tag {font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:#ff8790;margin-bottom:.65rem;}
.slide-key {border-left:3px solid #f04452;padding:.55rem .85rem;margin:.15rem 0 .25rem;color:#f4f4f5;font-size:1.1rem;font-weight:600;line-height:1.4;}
.slide-flow {display:flex;align-items:center;gap:.8rem;margin:.2rem 0 .35rem;flex-wrap:wrap;}
.slide-flow span {background:#191c24;border:1px solid #343842;padding:.7rem 1rem;border-radius:8px;font-weight:600;}
.slide-flow b {color:#f04452;}
.slide-hero {font-size:clamp(2rem,4vw,3.6rem);line-height:1.1;font-weight:750;max-width:1100px;margin:.7rem 0 .3rem;letter-spacing:-.04em;}
.slide-hero em {color:#ff6977;font-style:normal;}
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {font-size:.85rem;color:#b6bac5 !important;line-height:1.4;}
h1,h2,h3 {letter-spacing:-.035em;}
h1 {font-size:2rem !important;padding:.15rem 0 .35rem !important;}
h2 {font-size:1.65rem !important;padding:.2rem 0 !important;}
h3 {font-size:1.12rem !important;padding:.15rem 0 !important;}
[data-testid="stMetric"] {box-sizing:border-box;border:1px solid #393d48;border-radius:8px;padding:12px 16px;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
[data-testid="stSidebar"] {background:#191c24;}
[data-testid="stCode"] pre {font-size:.84rem !important;line-height:1.35 !important;padding:10px !important;}
.katex-display {margin:.45em 0 !important;}
.slide-table {width:100%;border-collapse:collapse;font-size:.95rem;}
.slide-table th, .slide-table td {
    padding:6px 10px;border-bottom:1px solid #343842;text-align:center;
}
.slide-table thead th {background:#191c24;color:#d5d7de;}
.slide-table tbody th {font-weight:500;}
.slide-table-frame {overflow-x:auto;}
/* Referência: 1366 x 768 em tela cheia. Sem escala artificial. */
.stApp {--section-gap:18px;--card-gap:24px;--panel-gap:10px;--control-gap:8px;--block-gap:12px;}
[data-testid="stMainBlockContainer"], .block-container {max-width:1320px;padding:20px 34px 22px 52px;}
.st-key-seminar_navigation {--block-gap:6px;}
.st-key-seminar_navigation button {min-height:36px;}
.st-key-seminar_heading {--block-gap:4px;}
.st-key-seminar_heading h1 {font-size:2.05rem !important;}
.slide-lead {font-size:1.5rem;line-height:1.25;}
.slide-card {min-height:110px;padding:16px 20px;}
.slide-card h3 {font-size:1.25rem !important;}
.slide-card p {font-size:1.05rem;line-height:1.4;}
.slide-card.pending {visibility:hidden;}
.slide-cards {margin-block:2px;}
.slide-table th,.slide-table td {padding:5px 10px;}
.slide-hero {font-size:clamp(2.2rem,4.4vw,3.8rem);line-height:1.15;max-width:1050px;}
.slide-cover {padding:clamp(24px,8vh,68px) 0 24px;}
.slide-cover .slide-kicker {margin-bottom:20px;color:#ff6977;}
.slide-cover p {font-size:1.35rem !important;color:#b6bac5;max-width:900px;margin-top:20px;}
.slide-cover .slide-hero {margin:0;}
.slide-poster {display:flex;flex-direction:column;justify-content:end;min-height:90px;padding:12px;border-radius:8px;background:linear-gradient(140deg,#343a50,#191c24);font-weight:700;}
.slide-poster small {color:#b6bac5;font-weight:400;}
.slide-poster-row {display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.slide-split {display:flex;align-items:center;gap:12px;flex-wrap:wrap;}
.slide-big {font-size:2.3rem;font-weight:750;color:#ff6977;line-height:1.15;}
.slide-stack {display:grid;gap:12px;}
.slide-compare {display:grid;grid-template-columns:1fr 1fr;gap:var(--card-gap);}
.slide-compare .slide-card {text-align:center;min-height:0;}
.slide-compare .steps {display:grid;gap:6px;justify-items:center;}
.slide-compare .steps b {color:#ff6977;}
.slide-compare .steps span {background:#242833;border-radius:6px;padding:6px 14px;width:90%;}
@media (max-width:760px) {.slide-compare {grid-template-columns:1fr;}}
.slide-conclusion {display:grid;grid-template-columns:46px 1fr;gap:16px;align-items:center;font-size:1.28rem;padding:10px 0;}
.slide-conclusion b {color:#ff6977;font-size:1.8rem;}
.st-key-seminar_source_code [data-testid="stCode"] {max-height:410px;overflow:auto;}
.st-key-seminar_demo_inputs {max-height:405px;overflow-y:auto;scrollbar-gutter:stable;}
.st-key-seminar_panel_results {max-height:495px;overflow-y:auto;scrollbar-gutter:stable;}
@media (max-height:650px) {
 .stApp {--section-gap:12px;--card-gap:16px;--panel-gap:8px;--block-gap:8px;}
 [data-testid="stMainBlockContainer"], .block-container {padding-top:12px;padding-bottom:12px;}
 .slide-card {min-height:0;padding:10px 14px;}
 .slide-cover {padding:26px 0;}
 .st-key-seminar_demo_inputs {max-height:240px;}
 .st-key-seminar_source_code [data-testid="stCode"] {max-height:300px;}
}
@media (max-width:760px) {
    .slide-cards {grid-template-columns:1fr;}
}
/* Distribuição do slide em fullscreen (1366 x 768).
   O bloco de página tem uma chave própria. As margens são aplicadas aos
   filhos reais (inclusive stLayoutWrapper), e não ao HTML dentro do card. */
#stMain [data-testid="stVerticalBlock"][class*="st-key-seminar_slide_"],
#stMain [class*="st-key-seminar_slide_"] > [data-testid="stVerticalBlock"] {
    --slide-section-space:24px;
    --slide-panel-space:12px;
    --slide-top-space:8px;
    --block-gap:var(--slide-section-space);
    gap:0 !important;
    row-gap:0 !important;
    padding-top:var(--slide-top-space);
}
#stMain [data-testid="stVerticalBlock"][class*="st-key-seminar_slide_"] > * + *,
#stMain [class*="st-key-seminar_slide_"] > [data-testid="stVerticalBlock"] > * + * {
    margin-block-start:var(--slide-section-space) !important;
}
#stMain [class*="st-key-seminar_slide_"] [data-testid="stColumn"] > [data-testid="stVerticalBlock"] {
    --block-gap:var(--slide-panel-space);
    gap:var(--slide-panel-space) !important;
}
#stMain [class*="st-key-seminar_slide_"] [data-testid="stVerticalBlock"][class*="st-key-seminar_controls_"],
#stMain [class*="st-key-seminar_slide_"] [data-testid="stVerticalBlock"][class*="st-key-seminar_panel_"] {
    gap:var(--slide-panel-space) !important;
}
#stMain [class*="st-key-seminar_slide_"] .slide-cards {gap:24px;margin:0;}
#stMain [class*="st-key-seminar_slide_"] .slide-card {height:auto;}
#stMain [class*="st-key-seminar_slide_"] .slide-key {margin:0;padding:12px 18px;}
#stMain [class*="st-key-seminar_slide_"] .slide-lead {margin:0;padding:0;}
#stMain [class*="st-key-seminar_slide_"] .slide-flow {margin:0;gap:14px;}
#stMain [class*="st-key-seminar_slide_"] [data-testid="stMetricValue"] p {
    font-size:1.85rem !important;line-height:1.15 !important;font-weight:700;
}
#stMain [class*="st-key-seminar_slide_"] [data-testid="stCaptionContainer"] p {color:#b6bac5 !important;}
/* Slides de leitura: mais respiro e tipografia de projeção. */
@media (min-width:1000px) and (min-height:700px) {
    #stMain [data-testid="stVerticalBlock"]:is(.st-key-seminar_slide_02,.st-key-seminar_slide_03,.st-key-seminar_slide_08,.st-key-seminar_slide_14),
    #stMain :is(.st-key-seminar_slide_02,.st-key-seminar_slide_03,.st-key-seminar_slide_08,.st-key-seminar_slide_14) > [data-testid="stVerticalBlock"] {
        --slide-section-space:28px;
        --slide-panel-space:16px;
        --slide-top-space:14px;
    }
    #stMain [class*="st-key-seminar_slide_"] .slide-lead {font-size:1.65rem;}
    #stMain [class*="st-key-seminar_slide_"] .slide-card h3 {font-size:1.35rem !important;}
    #stMain [class*="st-key-seminar_slide_"] .slide-card p {font-size:1.12rem;line-height:1.45;}
    #stMain [class*="st-key-seminar_slide_"] .slide-key {font-size:1.18rem;}
    #stMain :is(.st-key-seminar_slide_02,.st-key-seminar_slide_03,.st-key-seminar_slide_08) .slide-card {padding:22px 24px;min-height:145px;}
    #stMain :is(.st-key-seminar_slide_02,.st-key-seminar_slide_03,.st-key-seminar_slide_08) .slide-cards.compact .slide-card {min-height:100px;}
    #stMain .st-key-seminar_slide_02 .slide-poster {min-height:136px;padding:20px;font-size:1.15rem;}
    #stMain .st-key-seminar_slide_02 .slide-poster-row {gap:18px;}
    #stMain .st-key-seminar_slide_03 .slide-cards:not(.compact) .slide-card {min-height:170px;}
    #stMain .st-key-seminar_slide_08 .slide-table th,
    #stMain .st-key-seminar_slide_08 .slide-table td {padding:10px 12px;font-size:1.05rem;}
    #stMain .st-key-seminar_slide_14 .slide-conclusion {font-size:1.5rem;padding:12px 0;gap:22px;}
    #stMain .st-key-seminar_slide_14 .slide-conclusion b {font-size:2rem;}
    #stMain .st-key-seminar_slide_01 .slide-cover {padding:clamp(85px,15vh,130px) 0 34px;}
    #stMain .st-key-seminar_slide_01 .slide-hero {max-width:1160px;font-size:clamp(3rem,4.8vw,4.2rem);}
    #stMain .st-key-seminar_slide_01 .slide-cover p {margin-top:30px;font-size:1.5rem !important;}
    /* Páginas densas: seções separadas sem empurrar os cálculos para fora. */
    #stMain [data-testid="stVerticalBlock"]:is(.st-key-seminar_slide_04,.st-key-seminar_slide_05,.st-key-seminar_slide_07,.st-key-seminar_slide_09,.st-key-seminar_slide_10,.st-key-seminar_slide_13),
    #stMain :is(.st-key-seminar_slide_04,.st-key-seminar_slide_05,.st-key-seminar_slide_07,.st-key-seminar_slide_09,.st-key-seminar_slide_10,.st-key-seminar_slide_13) > [data-testid="stVerticalBlock"] {
        --slide-section-space:18px;
        --slide-panel-space:10px;
    }
    #stMain .st-key-seminar_slide_04 .slide-compare .slide-card {padding:16px 20px;}
    #stMain .st-key-seminar_slide_04 .slide-cards.compact .slide-card {padding:14px 20px;}
    #stMain .st-key-seminar_slide_06 .slide-card {padding:20px;}
    #stMain .st-key-seminar_slide_06 .slide-cards {gap:28px;}
    #stMain .st-key-seminar_slide_06 .katex-display {margin:.75em 0 !important;}
    #stMain .st-key-seminar_slide_12 [data-testid="stCode"] pre {font-size:.98rem !important;line-height:1.5 !important;padding:18px !important;}
}
/* Janela menor: mantém separação real, mas dá prioridade ao conteúdo. */
@media (max-height:699px) {
    #stMain [data-testid="stVerticalBlock"][class*="st-key-seminar_slide_"],
    #stMain [class*="st-key-seminar_slide_"] > [data-testid="stVerticalBlock"] {
        --slide-section-space:14px;--slide-panel-space:8px;--slide-top-space:4px;
    }
    #stMain [class*="st-key-seminar_slide_"] .slide-cards {gap:16px;}
    #stMain [class*="st-key-seminar_slide_"] .slide-key {padding:8px 12px;}
}
/* Páginas 2 e 9: pôsteres reais e painéis de cálculo alinhados. */
#stMain .st-key-seminar_slide_02 .slide-poster-row {
    display:grid;grid-template-columns:repeat(2,minmax(0,1fr));
    gap:24px;max-width:460px;margin:12px auto 0;
}
#stMain .st-key-seminar_slide_02 .film-poster {margin:0;text-align:center;min-width:0;}
#stMain .st-key-seminar_slide_02 .film-poster img,
#stMain .st-key-seminar_slide_02 .film-poster-placeholder {
    width:100%;height:clamp(140px,30vh,230px);object-fit:contain;
    background:#191c24;border:1px solid #303540;border-radius:10px;box-sizing:border-box;
}
#stMain .st-key-seminar_slide_02 .film-poster-placeholder {
    display:flex;align-items:center;justify-content:center;padding:12px;color:#b6bac5;
}
#stMain .st-key-seminar_slide_02 .film-poster figcaption {font-size:1rem;font-weight:650;margin-top:8px;}
#stMain .st-key-seminar_slide_02 .poster-group-title {font-size:1.18rem;font-weight:700;text-align:center;}
#stMain .st-key-seminar_slide_02 .poster-group-genre {color:#b6bac5;font-size:.95rem;text-align:center;margin-top:12px;}
#stMain .st-key-seminar_slide_09 .slide-card {min-height:92px;padding:14px 20px;}
#stMain .st-key-seminar_slide_09 .slide-card .card-tag {margin-bottom:6px;}
#stMain .st-key-seminar_slide_09 .slide-card h3 {margin-bottom:6px !important;}
#stMain .st-key-seminar_slide_09 [class*="st-key-seminar_hybrid_row_"] {
    padding-top:4px;padding-bottom:2px;
}
/* Cada fórmula e sua legenda ocupam UM bloco HTML com altura intrínseca.
   Não aplicar borda/padding ao wrapper de altura calculada do Streamlit. */
#stMain .st-key-seminar_slide_09 .hybrid-formula-card {
    background:#191c24;border:1px solid #303540;border-radius:12px;
    padding:16px 20px;box-sizing:border-box;min-height:0;height:auto;
    display:flex;flex-direction:column;gap:10px;
}
#stMain .st-key-seminar_slide_09 .hybrid-formula-card .hybrid-label,
#stMain .st-key-seminar_slide_09 .hybrid-formula-card .hybrid-note {margin:0;}
#stMain .st-key-seminar_slide_09 .hybrid-equation {display:flex;align-items:center;justify-content:center;gap:8px;font:italic 1.35rem Georgia,serif;line-height:1.3;}
#stMain .st-key-seminar_slide_09 .hybrid-frac {display:inline-grid;text-align:center;align-items:center;vertical-align:middle;}
#stMain .st-key-seminar_slide_09 .hybrid-frac > span:first-child {border-bottom:1px solid currentColor;padding:0 5px 3px;}
#stMain .st-key-seminar_slide_09 .hybrid-frac > span:last-child {padding:3px 5px 0;}
#stMain .st-key-seminar_slide_09 .hybrid-formula-card .hybrid-note {line-height:1.4;overflow-wrap:anywhere;}
#stMain .st-key-seminar_slide_09 .hybrid-label {font-size:1.08rem;font-weight:700;margin:0 0 10px;}
#stMain .st-key-seminar_slide_09 .hybrid-note {color:#b6bac5;font-size:.93rem;line-height:1.35;margin:10px 0 0;}
#stMain .st-key-seminar_slide_09 .hybrid-weights {display:flex;min-height:36px;border-radius:7px;overflow:hidden;text-align:center;font-weight:650;}
#stMain .st-key-seminar_slide_09 .hybrid-weights > span {display:flex;align-items:center;justify-content:center;padding:6px 3px;}
#stMain .st-key-seminar_slide_09 .hybrid-summary {
    display:grid;grid-template-columns:200px 1fr;align-items:center;gap:20px;
    border:1px solid #303540;border-left:4px solid #f04452;border-radius:10px;
    background:#191c24;padding:13px 20px;margin-top:6px;
}
#stMain .st-key-seminar_slide_09 .hybrid-summary p {margin:3px 0;}
#stMain .st-key-seminar_slide_09 .katex-display {margin:10px 0 !important;}
@media (max-height:650px) {
    #stMain .st-key-seminar_slide_09 [class*="st-key-seminar_hybrid_row_"] {padding-top:6px;}
    #stMain .st-key-seminar_slide_09 .hybrid-summary {margin-top:6px;padding:10px 14px;}
}
@media (max-width:760px) {
    #stMain .st-key-seminar_slide_09 .hybrid-summary {grid-template-columns:1fr;gap:8px;}
}
</style>''', unsafe_allow_html=True)

# Estado explícito das revelações. O controle de página é independente.
MAX_STAGE = [1,3,3,3,3,3,3,3,3,3,3,2,0,2]
if 'seminar_page' not in st.session_state:
    st.session_state.seminar_page = 0
if 'seminar_stage' not in st.session_state:
    st.session_state.seminar_stage = 0


def set_page(index, end=False):
    index = max(0, min(len(PAGES)-1, int(index)))
    st.session_state.seminar_page = index
    st.session_state.seminar_picker = index
    st.session_state.seminar_stage = MAX_STAGE[index] if end else 0


def navigate(delta):
    set_page(st.session_state.seminar_page + delta)


def reveal():
    p = st.session_state.seminar_page
    if st.session_state.seminar_stage < MAX_STAGE[p]:
        st.session_state.seminar_stage += 1
    elif p < len(PAGES)-1:
        set_page(p+1)


def undo_reveal():
    if st.session_state.seminar_stage > 0:
        st.session_state.seminar_stage -= 1
    elif st.session_state.seminar_page > 0:
        set_page(st.session_state.seminar_page-1, end=True)


def reset_stage():
    st.session_state.seminar_stage = 0


def pick_page():
    set_page(st.session_state.seminar_picker)


page = st.session_state.seminar_page
stage = st.session_state.seminar_stage
if 'seminar_picker' not in st.session_state:
    st.session_state.seminar_picker = page
with st.sidebar:
    st.radio('Ir para a página', range(len(PAGES)), format_func=lambda i:f'{i+1:02d} · {PAGES[i]}', key='seminar_picker', on_change=pick_page)
    st.caption('F11: tela cheia. Espaço/→: revelar. ←: desfazer. Page Up/Down: trocar de página.')


def navigation():
    brand, prev, undo, nxt, advance, count = st.columns([2.1,1.05,.65,1.75,1.1,.7],gap='small')
    brand.markdown('**MOVIELENS / IA**')
    prev.button('← Voltar', key='seminar_previous', on_click=navigate,args=(-1,),disabled=page==0,use_container_width=True)
    undo.button('↶', key='seminar_undo', on_click=undo_reveal,disabled=page==0 and stage==0,use_container_width=True,help='Revelação anterior')
    label = ('Próximo resultado' if page==10 else 'Revelar') if stage<MAX_STAGE[page] else 'Continuar →'
    nxt.button(label,key='seminar_reveal',on_click=reveal,disabled=page==len(PAGES)-1 and stage>=MAX_STAGE[page],use_container_width=True,type='primary')
    advance.button('Avançar →',key='seminar_next',on_click=navigate,args=(1,),disabled=page==len(PAGES)-1,use_container_width=True)
    count.markdown(f'**{page+1:02d} / {len(PAGES)}**')
    st.progress((page+1)/len(PAGES))


with st.container(key='seminar_navigation'):
    navigation()

# Atalhos não interferem nos widgets; botões funcionam sem JS.
import streamlit.components.v1 as components
components.html('''<script>
try {
  const host=window.parent, doc=host.document;
  const old=host.__movieSeminarKeyHandler;
  if(old) doc.removeEventListener('keydown',old);
  const handler=(e)=>{
    if(e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.repeat) return;
    if(e.target.closest('input,textarea,select,[contenteditable=true],[role=slider],[role=combobox],[role=listbox],[role=radio],[role=tab],[role=grid],a')) return;
    if(e.target.closest('button') && !e.target.closest('.st-key-seminar_navigation')) return;
    const keys={'ArrowRight':'seminar_reveal',' ':'seminar_reveal','ArrowLeft':'seminar_undo','PageDown':'seminar_next','PageUp':'seminar_previous'};
    if(!keys[e.key]) return;
    const b=doc.querySelector('.st-key-'+keys[e.key]+' button');
    if(b && !b.disabled){e.preventDefault();b.click();setTimeout(()=>b.blur(),0);}
  };
  host.__movieSeminarKeyHandler=handler;
  doc.addEventListener('keydown',handler);
  const slide="SLIDE_NUMBER";
  if(host.__movieSeminarSlide!==slide){
    host.__movieSeminarSlide=slide;
    const main=doc.querySelector('[data-testid="stMain"]'); if(main) main.scrollTop=0;
    host.scrollTo(0,0);
  }
}catch(_){}
</script>'''.replace('SLIDE_NUMBER',str(page)),height=0,width=0)

def takeaway(text):
    st.markdown(f'<div class="slide-key">{escape(text)}</div>', unsafe_allow_html=True)


def lead(text):
    st.markdown(f'<div class="slide-lead">{escape(text)}</div>', unsafe_allow_html=True)


def card_html(title, body, tag='', focus=False):
    cls = 'slide-card focus' if focus else 'slide-card'
    label = f'<div class="card-tag">{escape(tag)}</div>' if tag else ''
    return f'<div class="{cls}">{label}<h3>{escape(title)}</h3><p>{escape(body)}</p></div>'


def cards(items, columns=None, compact=False):
    """Mesma grade e mesmo espaço entre cards, em linha ou empilhados."""
    columns = columns or len(items)
    cls = 'slide-cards compact' if compact else 'slide-cards'
    content = ''.join(card_html(*item) for item in items)
    st.markdown(f'<div class="{cls}" style="--card-columns:{columns}">{content}</div>', unsafe_allow_html=True)


def slide_columns(widths):
    return st.columns(widths, gap='medium')


def small_table(frame, index=True):
    """Tabela de leitura: todas as linhas visíveis, sem altura fixa de canvas."""
    html=frame.to_html(index=index,classes='slide-table',border=0,na_rep='NaN',escape=True,float_format=lambda value:f'{value:g}')
    st.markdown(f'<div class="slide-table-frame">{html}</div>',unsafe_allow_html=True)


def flow(*items):
    content = '<b>→</b>'.join(f'<span>{escape(item)}</span>' for item in items)
    st.markdown(f'<div class="slide-flow">{content}</div>', unsafe_allow_html=True)


def compact_source(fn):
    # Exibe o código real, removendo apenas linhas em branco para caber na tela.
    return '\n'.join(line for line in inspect.getsource(fn).splitlines() if line.strip())


@st.cache_data
def result_file(name):
    path = ROOT / 'results_v2' / name
    if path.exists():
        return pd.read_csv(path)
    archive = ROOT / 'results_v2.zip'
    if archive.exists():
        with ZipFile(archive) as z:
            with z.open('results_v2/'+name) as f:
                return pd.read_csv(f)
    raise FileNotFoundError(name)

@st.cache_data
def ranking_table():
    df = result_file('predictions.csv')
    return pd.DataFrame([{'model':model, **precision_recall_at_n(group,5), **precision_recall_at_n(group,10)} for model,group in df.groupby('model')])


def metric_demo(stage):
    options = {
        'Euclidiana': ('euclidiana', metrics.euclidean_distance, r'd=\sqrt{\sum_i(a_i-b_i)^2}',
            'Distância em linha reta entre as notas.', 'Dá maior importância a divergências grandes (1 vs. 5).', 'Penaliza usuários que dão notas mais baixas; a distância cresce com o número de filmes em comum.'),
        'Manhattan': ('manhattan', metrics.manhattan_distance, r'd=\sum_i|a_i-b_i|',
            'Soma das diferenças absolutas entre as notas.', 'Interpretação direta; divergências grandes não são elevadas ao quadrado.', 'Diferenças de rigor ao avaliar e a quantidade de filmes em comum afetam a distância.'),
        'Hamming': ('hamming', metrics.hamming_distance, r'd=\frac{1}{n}\sum_i\mathbf{1}(a_i\ne b_i)',
            'Proporção de notas diferentes.', 'Valor normalizado entre 0 e 1; aproveita a concordância exata na escala discreta de estrelas.', 'Trata 4 vs. 5 igual a 1 vs. 5. Com poucos filmes, muitos vizinhos podem empatar.'),
        'Cosseno': ('cosseno', metrics.cosine_similarity, r's=\frac{A\cdot B}{\|A\|\|B\|}',
            'Compara a orientação dos vetores de notas.', 'Preserva a similaridade sob mudanças proporcionais na magnitude das notas.', 'Notas positivas podem gerar similaridades altas mesmo quando o gosto diverge; pouca sobreposição reduz a confiança.'),
        'Pearson': ('pearson', metrics.pearson_correlation, r'r=\frac{(A-\bar A)\cdot(B-\bar B)}{\|A-\bar A\|\|B-\bar B\|}',
            'Compara a variação das notas em relação às médias.', 'Captura tendências de gosto mesmo quando um usuário costuma dar notas menores.', 'Indefinida para vetores constantes; com apenas duas notas distintas pode resultar em +1 ou −1 sem evidência robusta.'),
    }
    a = np.array([5,4,2,3], dtype=float)
    with st.container(key='seminar_controls_metric'):
        chooser,editor=st.columns([4,1])
        with chooser:
            chosen = st.radio('Métrica de proximidade', list(options), horizontal=True, label_visibility='collapsed', key='seminar_metric', on_change=reset_stage)
        with editor, st.popover('Editar notas de B'):
            st.caption('A e B avaliam os mesmos quatro filmes. Notas de 1 a 5.')
            cols=st.columns(2)
            b=np.array([cols[i%2].number_input(f'B{i+1}',1,5,v,step=1,key=f'sem_b{i}') for i,v in enumerate([4,3,2,3])],dtype=float)
        metric, fn, formula, description, advantage, limitation = options[chosen]
        st.markdown(f'**{description}**')
    l,r,context = slide_columns([1.1,1.15,1])
    with l, st.container(key='seminar_controls_metric_math'):
        st.caption('FÓRMULA · SUBSTITUIÇÃO · RESULTADO')
        st.markdown('**A = (5, 4, 2, 3)**  \n'+f'**B = ({", ".join(str(int(v)) for v in b)})**')
        av,bv = a.astype(int),b.astype(int)
        diffs = av-bv
        if metric=='euclidiana':
            calc = r'\sqrt{'+ '+'.join(f'({x}-{y})^2' for x,y in zip(av,bv))+'}'
        elif metric=='manhattan':
            calc = '+'.join(f'|{x}-{y}|' for x,y in zip(av,bv))
        elif metric=='hamming':
            calc = r'\frac{'+str(np.count_nonzero(diffs))+'}{4}'
        elif metric=='cosseno':
            calc = rf'\frac{{{a@b:g}}}{{\sqrt{{{a@a:g}}}\sqrt{{{b@b:g}}}}}'
        else:
            ac,bc = a-a.mean(),b-b.mean()
            calc = rf'\frac{{{ac@bc:g}}}{{\sqrt{{{ac@ac:g}}}\sqrt{{{bc@bc:g}}}}}' if np.linalg.norm(bc)>0 else r'\text{Indefinida: norma zero}'
        st.latex(formula)
        val,weight,n = metrics.proximity(a,b,metric)
        if stage>=1:
            st.latex(calc)
            st.metric('Resultado',f'{val:.4f}' if np.isfinite(val) else 'Indefinido')
    with r, st.container(key='seminar_controls_metric_code'):
        if stage>=2:
            st.caption('IMPLEMENTAÇÃO REAL · src/metrics.py')
            st.code(compact_source(fn),language='python',wrap_lines=True)
            st.caption(f'Peso na previsão: {weight:.4f}')
            if metric=='pearson':
                with st.popover('Ver função reutilizada'):
                    st.caption(f'Médias: A = {a.mean():g}; B = {b.mean():g}.')
                    st.code(compact_source(metrics.cosine_similarity),language='python',wrap_lines=True)
    with context:
        if stage>=3:
            cards([('+ Vantagem',advantage),('− Limitação',limitation)],columns=1,compact=True)
            st.latex(r'w=1/(1+d)' if metric in ('euclidiana','manhattan','hamming') else r'w=\max(0,s)')
            st.caption('Para similaridades indefinidas, peso zero.' if metric in ('cosseno','pearson') else 'Distância menor → peso maior.')


def results_page(stage):
    try:
        final=result_file('test_metrics.csv')
        validation=result_file('validation_metrics.csv')
        if stage==0:
            choice,measure=st.columns(2)
            mode=choice.selectbox('Modelo',list(LABELS)[:3],format_func=LABELS.get,key='sem_val_model')
            error=measure.radio('Indicador',['rmse','mae'],format_func=str.upper,horizontal=True,key='sem_val_error')
            subset=validation[validation.model==mode]
            import altair as alt
            chart=alt.Chart(subset).mark_line(point=True,strokeWidth=2.5).encode(
                x=alt.X('k:Q',title='Quantidade de vizinhos (k)',axis=alt.Axis(values=sorted(subset.k.unique().tolist()),format='d',labelFontSize=14,titleFontSize=15)),
                y=alt.Y(f'{error}:Q',title=error.upper(),scale=alt.Scale(zero=False),axis=alt.Axis(labelFontSize=14,titleFontSize=15)),
                color=alt.Color('metric:N',title='Métrica',legend=alt.Legend(orient='bottom',labelFontSize=14)),
                tooltip=['metric:N','k:Q',alt.Tooltip(f'{error}:Q',format='.4f')]
            ).properties(height=340)
            st.altair_chart(chart,use_container_width=True)
            st.caption('Validação · quanto menor o erro, melhor.')
        elif stage==1:
            subset=validation[validation.model=='tradicional']
            best=subset.loc[subset.rmse.idxmin()]
            for col,label,value in zip(st.columns(3),['Melhor métrica','Vizinhos','RMSE · validação'],[str(best.metric).title(),f'k = {int(best.k)}',f'{best.rmse:.4f}']):col.metric(label,value)
            takeaway('Aumentar k nem sempre melhora o resultado.')
            st.write('Incluir vizinhos mais distantes pode reduzir a personalização.')
        elif stage==2:
            measure=st.radio('Indicador',['RMSE','MAE','Coverage'],horizontal=True,key='sem_test_error')
            display=final.copy();display['Modelo']=display.model.map(LABELS)
            field=measure.lower()
            if field=='coverage':
                display=display[display.model!='baseline'].copy();display[field]*=100
            l,r=slide_columns([1.3,1])
            with l: st.bar_chart(display.set_index('Modelo')[field],height=360)
            with r:
                small_table(display[['Modelo','metric','k',field]].round(4),index=False)
                st.caption('Coverage em %; menor MAE/RMSE é melhor.')
            winner=final.loc[final.rmse.idxmin()]
            takeaway(f'{LABELS[winner.model]} apresentou o menor RMSE no teste.')
        else:
            n=st.radio('Número de recomendações',[5,10],format_func=lambda v:f'Top {v}',horizontal=True,key='sem_rank_n')
            rank=ranking_table().set_index('model');rank.index=rank.index.map(LABELS)
            cols=[f'precision@{n}',f'recall@{n}']; values=rank[cols]*100
            l,r=slide_columns([1.4,1])
            with l: st.bar_chart(values,height=360)
            with r:
                small_table(values.round(2))
                st.caption('Valores em %. Relevante: nota real ≥ 4.')
            takeaway('Diferenças pequenas, sem teste de significância. Ranking restrito aos filmes do teste, não ao catálogo completo.')
    except (FileNotFoundError,KeyError) as exc:
        st.warning(f'Resultado indisponível: {exc}. Execute experiments.py ou disponibilize results_v2.zip.')


def staged_cards(items, stage, columns=None):
    # Reserva o lugar dos cards sem revelar sua explicação antes da hora.
    columns=columns or len(items)
    html=''.join(card_html(title, body if stage>=step else '',tag if stage>=step else '',focus and stage>=step)
                 for title,body,tag,focus,step in items)
    st.markdown(f'<div class="slide-cards" style="--card-columns:{columns}">{html}</div>',unsafe_allow_html=True)


POSTERS_DIR = ROOT / 'assets' / 'posters'
POSTER_FILES = {
    'Star Wars': 'star_wars.jpg', 'Terminator 2': 'terminator_2.jpg',
    'Titanic': 'titanic.jpg', 'Pretty Woman': 'pretty_woman.jpg',
}


def poster_pair(title1, title2, genre, user):
    """Pôsteres locais, sem requisições à internet durante a apresentação."""
    items = []
    for title in (title1, title2):
        path = POSTERS_DIR / POSTER_FILES[title]
        if valid_image(path):
            image = image_html(path, title, 'film-poster-image')
        else:
            image = '<div class="film-poster-placeholder">Pôster não disponível</div>'
        items.append(f'<figure class="film-poster">{image}<figcaption>{escape(title)}</figcaption></figure>')
    st.markdown(f'<div class="poster-group-title">{escape(user)}</div><div class="slide-poster-row">{"".join(items)}</div><div class="poster-group-genre">{escape(genre)}</div>', unsafe_allow_html=True)


def hybrid_formula_card(title, prefix, numerator, denominator, note):
    # Todos os argumentos são constantes da página, não dados do usuário.
    st.markdown(f'<div class="hybrid-formula-card"><div class="hybrid-label">{escape(title)}</div>'
                f'<div class="hybrid-equation"><span>{prefix}</span><span class="hybrid-frac">'
                f'<span>{numerator}</span><span>{denominator}</span></span></div>'
                f'<div class="hybrid-note">{escape(note)}</div></div>',unsafe_allow_html=True)


with st.container(key=f'seminar_slide_{page+1:02d}'):
    if page!=0:
        with st.container(key='seminar_heading'):
            st.title(PAGES[page])
    if page==0:
        st.markdown('<div class="slide-cover"><div class="slide-kicker">Aprendizado baseado em instâncias</div><div class="slide-hero">Sistema de recomendação de filmes utilizando <em>k-NN</em></div><p>Uma aplicação de filtragem colaborativa utilizando a MovieLens 100K.</p></div>',unsafe_allow_html=True)
        if stage>=1: st.caption('Seminário de Inteligência Artificial · 2026')
    elif page==1:
        lead('Milhares de filmes. Como escolher o próximo?')
        if stage>=1:
            cards([('Popularidade','Recomendar os filmes mais bem avaliados para todos ignora as preferências individuais.')],compact=True)
        if stage>=2:
            left,right=slide_columns(2)
            with left:
                poster_pair('Star Wars','Terminator 2','Ação e ficção científica','Usuário A')
            with right:
                poster_pair('Titanic','Pretty Woman','Romance e comédia','Usuário B')
            if any(not valid_image(POSTERS_DIR / filename) for filename in POSTER_FILES.values()):
                st.caption('Para carregar as capas locais, execute python download_catalog_posters.py antes do seminário.')
        if stage>=3: takeaway('Nosso objetivo: prever as notas de filmes ainda não avaliados e gerar recomendações personalizadas.')
    elif page==2:
        lead('Três abordagens para personalizar recomendações.')
        staged_cards([('Conteúdo','Compara características dos filmes com as preferências do usuário. Ex.: filmes do mesmo gênero.','',False,1),('Colaborativa','Utiliza padrões de avaliações e comportamento de outros usuários.','',True,2),('Híbrida','Combina diferentes fontes de informação.','',False,3)],stage)
        if stage>=2:
            cards([('User-Based','Compara usuários semelhantes.','Nossa escolha',True),('Item-Based','Compara itens com padrões de avaliação semelhantes.')],compact=True)
        if stage>=3: takeaway('Começamos pela filtragem colaborativa baseada em usuários e evoluímos para uma abordagem híbrida.')
    elif page==3:
        lead('Aprender utilizando exemplos conhecidos.')
        st.markdown('<div class="slide-compare"><div class="slide-card"><h3>Aprendizado baseado em modelo</h3><div class="steps"><span>Dados de treinamento</span><b>↓</b><span>Ajuste de parâmetros</span><b>↓</b><span>Modelo preditivo</span></div></div><div class="slide-card focus"><h3>Aprendizado baseado em instâncias</h3><div class="steps"><span>Dados de treinamento</span><b>↓</b><span>Armazenamento dos exemplos</span><b>↓</b><span>Busca dos vizinhos</span></div></div></div>',unsafe_allow_html=True)
        if stage>=1: takeaway('Lazy learning: grande parte do processamento acontece no momento da previsão.')
        if stage>=2: cards([('Classificação','Votação dos vizinhos → classe. Ex.: gosta / não gosta.'),('Regressão','Média das notas dos vizinhos → valor numérico. Ex.: 4,82.')],compact=True)
        if stage>=3: takeaway('Nosso problema é de regressão: prever notas numéricas.')
    elif page==4:
        lead('k-Nearest Neighbors · k vizinhos mais próximos')
        df=pd.DataFrame({'Filme A':[5,4,1,2,5],'Filme B':[4,4,5,5,3],'Filme C':[5,4,1,np.nan,3]},index=[1,2,3,4,5])
        l,r=slide_columns([1,1.15])
        with l:
            st.markdown('**Avaliações conhecidas · X = (5, 4, ?)**')
            shown=pd.concat([pd.DataFrame([[5,4,np.nan]],columns=df.columns,index=['X']),df])
            small_table(shown)
            if stage>=1: k=st.slider('Quantidade máxima de vizinhos (k)',1,5,3,key='knn_demo_k')
            else: k=3
        if stage>=1:
            model=MovieKNN(df,'euclidiana'); neighbors=model.neighbors(pd.Series({'Filme A':5,'Filme B':4})); score,details=model.predict('Filme C',neighbors,k)
            with r:
                st.markdown('**Vizinhos selecionados · avaliaram C**')
                table=pd.DataFrame(details,columns=['Usuário','Distância','Peso','Em comum','Nota C']).drop(columns='Em comum')
                small_table(table.round(3),index=False)
                st.caption('Distância euclidiana entre A e B; peso = 1/(1+d).')
                if stage>=2:
                    st.latex(r'\hat r=\frac{\sum_v w_vr_v}{\sum_v w_v}')
                    st.latex(r'\hat r\approx\frac{'+ '+'.join(rf'{row[2]:.2f}\cdot {row[4]:g}' for row in details)+'}{'+ '+'.join(f'{row[2]:.2f}' for row in details)+'}')
                    st.metric('Nota prevista para o filme C',f'{score:.2f}')
        if stage>=3: takeaway('Repetimos o cálculo para os filmes não avaliados e ordenamos as previsões.')
    elif page==5:
        metric_demo(stage)
    elif page==6:
        for c,label,value in zip(st.columns(4),['Usuários','Filmes','Avaliações','Notas'],['943','1.682','100.000','1–5']):c.metric(label,value)
        l,r=slide_columns([1,1.25])
        with l:
            if stage>=1: small_table(pd.DataFrame([('u.data','Usuário, filme, nota e timestamp'),('u.item','Título e gêneros'),('u.user','Idade, ocupação, sexo e CEP')],columns=['Arquivo','Informações']),index=False)
        with r:
            if stage>=2:
                st.markdown('**Matriz usuário-filme · amostra real**')
                try:
                    matrix=build_rating_matrix(load_ratings());start=st.slider('Primeiro usuário exibido',0,len(matrix)-5,0,key='sem_matrix_start');small_table(matrix.iloc[start:start+5,:5])
                except FileNotFoundError:
                    st.warning('Base local ausente. Execute download_data.py.')
        if stage>=3:
            l,r=slide_columns([.65,1.6])
            l.metric('Esparsidade','93,7%')
            with r:
                st.progress(.063,text='Apenas 6,3% das posições possuem avaliações.')
                takeaway('NaN significa ausência de avaliação, e não nota zero.')
    elif page==7:
        lead('Versão 1 · Filtragem colaborativa tradicional')
        flow('Avaliações','Matriz','Distâncias','Vizinhos','Previsão')
        if stage>=1:
            st.markdown('**Primeiros resultados da execução**')
            small_table(pd.DataFrame([('Legal Deceit','5,00',1),('Santa with Muscles','5,00',2),('A Great Day in Harlem','5,00',1)],columns=['Filme','Nota prevista','Vizinhos']),index=False)
        if stage>=2: cards([('Poucas avaliações em comum','Menos usuários elegíveis para comparação.'),('Pouco suporte','Notas máximas sustentadas por apenas um ou dois vizinhos.')],compact=True)
        if stage>=3: takeaway('Podemos aproveitar melhor os outros dados da MovieLens?')
    elif page==8:
        staged_cards([('Tradicional','Avaliações dos filmes.','Modelo 1',False,0),('Gêneros','Avaliações + preferências por gênero.','Modelo 2',False,1),('Perfil ampliado','Avaliações + gêneros + demografia.','Modelo 3',False,2)],stage)
        if stage>=1:
            with st.container(key='seminar_hybrid_row_genre'):
                l,r=slide_columns([1.15,1])
                with l:
                    st.markdown('<div class="hybrid-label">Comparação por gênero · exemplo fictício</div>',unsafe_allow_html=True)
                    small_table(pd.DataFrame([('Ação',4.8,4.5),('Romance',2.,2.5),('Comédia',3.5,3.8)],columns=['Gênero','A','B']),index=False)
                with r:
                    hybrid_formula_card('Similaridade por gênero',
                        'S<sub>g</sub> = 1 −', 'mean(|A<sub>g</sub> − B<sub>g</sub>|)', '4',
                        'Permite comparar gostos mesmo sem filmes em comum.')
        if stage>=2:
            with st.container(key='seminar_hybrid_row_weights'):
                l,r=slide_columns([1.15,1])
                with l:
                    st.markdown('<div class="hybrid-label">Pesos do perfil ampliado</div><div class="hybrid-weights"><span style="width:50%;background:#b92b3a">Notas · 50%</span><span style="width:35%;background:#4f6585">Gêneros · 35%</span><span style="width:15%;background:#636775">Perfil · 15%</span></div><div class="hybrid-note">Idade: diferença normalizada. Ocupação e sexo: igualdade. Pesos iniciais não otimizados.</div>',unsafe_allow_html=True)
                with r:
                    hybrid_formula_card('Similaridade combinada', 'S =',
                        '∑<sub>j∈D</sub> w<sub>j</sub>S<sub>j</sub>', '∑<sub>j∈D</sub> w<sub>j</sub>',
                        'D: blocos disponíveis. Exigimos evidência positiva de preferência.')
        if stage>=3:
            st.markdown('<div class="hybrid-summary"><span class="slide-big">505 → 943</span><div><p><strong>Usuários elegíveis no exemplo</strong></p><p>O filtro de suporte mínimo evita recomendações sustentadas por pouquíssimas avaliações.</p></div></div>',unsafe_allow_html=True)
    elif page==9:
        lead('Como saber se nossas previsões são boas?')
        st.markdown('<div style="display:flex;height:36px;border-radius:6px;overflow:hidden;text-align:center;line-height:36px;font-weight:650"><div style="width:64%;background:#b92b3a">64%</div><div style="width:16%;background:#4f6585">16%</div><div style="width:20%;background:#636775">20%</div></div>',unsafe_allow_html=True)
        for c,label,value in zip(st.columns(3),['Treinamento','Validação','Teste'],['64.000','16.000','20.000']):c.metric(label,value)
        if stage>=1: st.markdown('**3 modelos × 5 métricas × 6 valores de k = 90 configurações**');st.caption('k = 5, 10, 20, 40, 60, 80. Escolha pelo RMSE de validação.')
        if stage>=2:
            l,r=slide_columns([1,1.25])
            with l:
                st.markdown('**MAE · erro absoluto médio**');st.latex(r'\frac{1}{n}\sum_i|r_i-\hat r_i|')
                st.markdown('**RMSE · penaliza erros grandes**');st.latex(r'\sqrt{\frac{1}{n}\sum_i(r_i-\hat r_i)^2}')
            with r:
                if stage>=3:
                    small_table(pd.DataFrame([('Coverage','Previsões sustentadas por vizinhos'),('Precision@N','Proporção das recomendações relevantes'),('Recall@N','Proporção dos filmes relevantes recuperados'),('Baseline','Média das avaliações de cada filme')],columns=['Indicador','Finalidade']),index=False)
                    st.caption('Relevância: nota real ≥ 4. Treino + validação para prever o teste; ranking restrito ao teste.')
    elif page==10:
        results_page(stage)
    elif page==11:
        flow('dataset.py','metrics.py','knn.py','hybrid.py','experiments.py / app.py')
        if stage>=1:
            funcs={'Construção da matriz':(build_rating_matrix,'Transforma avaliações individuais em uma matriz usuário-filme.'),'Avaliações em comum':(metrics.get_common_ratings,'Compara apenas posições com avaliações válidas.'),'Métrica e peso':(metrics.proximity,'Retorna a proximidade, o peso e a quantidade de notas em comum.'),'Encontrar vizinhos':(MovieKNN.neighbors,'Compara o perfil com todos os usuários e ordena por peso.'),'Seleção dos vizinhos':(MovieKNN.select_neighbors,'Mantém até k vizinhos com nota no filme candidato.'),'Previsão da nota':(MovieKNN.weighted_average,'Calcula a média ponderada das avaliações.'),'Perfil por gênero':(HybridKNN.genre_profile,'Agrupa notas por gênero cinematográfico.'),'Similaridade híbrida':(HybridKNN.combined_similarity,'Combina os blocos disponíveis.'),'Ranking':(MovieKNN.recommend,'Filtra por suporte e ordena pelas notas previstas.')}
            l,r=slide_columns([.8,1.5])
            with l:
                name=st.selectbox('Função real do projeto',list(funcs),key='sem_code_fn');fn,desc=funcs[name]
                st.caption(Path(inspect.getsourcefile(fn)).name);st.markdown(f'**{name}**');st.write(desc)
            with r,st.container(key='seminar_source_code'):st.code(compact_source(fn),language='python')
        if stage>=2: st.latex(r'\hat r_{u,i}=\frac{\sum_v w_vr_{v,i}}{\sum_v w_v}');takeaway('O mesmo núcleo é utilizado nos experimentos e na interface.')
    elif page==12:
        from app import render_recommender
        render_recommender(presentation=True)
    else:
        lead('O que aprendemos com o projeto?')
        conclusions=['O k-NN superou o baseline na previsão de notas.','A inclusão dos gêneros ampliou as comparações e reduziu os erros.','Mais atributos não garantem melhores resultados.']
        for i,conclusion in enumerate(conclusions):
            if i==0 or stage>=1:st.markdown(f'<div class="slide-conclusion"><b>{i+1:02d}</b><span>{escape(conclusion)}</span></div>',unsafe_allow_html=True)
        if stage>=2:
            st.caption('Limitações: esparsidade, cold start, vieses, custo computacional e avaliação restrita do ranking. Diferenças pequenas, sem teste de significância.')
            takeaway('A qualidade da recomendação depende tanto dos dados quanto do algoritmo.')
            with st.popover('Referências'):
                st.markdown('- [GroupLens — MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)\n- [Cover & Hart (1967) — Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964)\n- [Sarwar et al. (2001) — Item-based collaborative filtering](https://doi.org/10.1145/371920.372071)')
