"""Seminário interativo. Executar: python -m streamlit run seminar.py"""
import inspect
from html import escape
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import streamlit as st

from src import metrics
from src.dataset import load_ratings, load_movies, build_rating_matrix
from src.knn import MovieKNN
from src.hybrid import HybridKNN
from ranking_metrics import precision_recall_at_n

ROOT = Path(__file__).resolve().parent
PAGES = [
    'Introdução', 'Problema e motivação', 'Sistemas de recomendação',
    'Aprendizado baseado em instâncias', 'Como funciona o k-NN',
    'Métricas de proximidade', 'MovieLens 100K', 'Solução tradicional',
    'Solução híbrida', 'Metodologia experimental', 'Resultados',
    'Explicação do código', 'Recomendador ao vivo', 'Conclusão e referências',
]
TIMES = [1,2,1,1,2,3,2,1,2,2,3,3,2,1]
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
[data-testid="stMainBlockContainer"], .block-container {max-width:1360px;padding-top:1.8rem;padding-bottom:1rem;}
[data-testid="stMain"] [data-testid="stMarkdownContainer"] p {font-size:1.05rem;line-height:1.4;}
.slide-lead {font-size:clamp(1.2rem,2.3vw,2rem);line-height:1.25;font-weight:650;max-width:1050px;margin:.3rem 0 .8rem;color:#f4f4f5;}
.slide-kicker {font-size:.78rem;letter-spacing:.12em;text-transform:uppercase;color:#a5a8b3;margin:.25rem 0;}
.slide-card {background:#191c24;border:1px solid #303540;border-radius:12px;padding:1.1rem;min-height:125px;margin:.15rem 0;}
.slide-card.focus {border-color:#f04452;border-left:4px solid #f04452;}
.slide-card h3 {font-size:1.2rem !important;margin:0 0 .55rem !important;}
.slide-card p {font-size:1.03rem;line-height:1.4;color:#d5d7de;margin:0;}
.slide-card .card-tag {font-size:.74rem;letter-spacing:.09em;text-transform:uppercase;color:#ff8790;margin-bottom:.55rem;}
.slide-key {border-left:3px solid #f04452;padding:.4rem .8rem;margin:.65rem 0;color:#f4f4f5;font-size:1.1rem;font-weight:600;}
.slide-flow {display:flex;align-items:center;gap:.65rem;margin:.5rem 0;flex-wrap:wrap;}
.slide-flow span {background:#191c24;border:1px solid #343842;padding:.65rem .9rem;border-radius:8px;font-weight:600;}
.slide-flow b {color:#f04452;}
.slide-hero {font-size:clamp(2rem,4vw,3.6rem);line-height:1.1;font-weight:750;max-width:1050px;margin:.65rem 0 1.25rem;letter-spacing:-.04em;}
.slide-hero em {color:#ff6977;font-style:normal;}
[data-testid="stMain"] [data-testid="stCaptionContainer"] p {font-size:.85rem;color:#a5a8b3;}

[data-testid="stMain"] [data-testid="stVerticalBlock"] {gap:.45rem;}
h1,h2,h3 {letter-spacing:-.035em;}
h1 {font-size:2rem !important;padding:.15rem 0 .3rem !important;}
h2 {font-size:1.65rem !important;padding:.2rem 0 !important;}
h3 {font-size:1.12rem !important;padding:.15rem 0 !important;}
p {margin-bottom:.25rem;}
[data-testid="stMetric"] {border:1px solid #393d48;border-radius:8px;padding:8px 12px;}
[data-testid="stMetricValue"] {font-size:1.65rem;}
[data-testid="stSidebar"] {background:#191c24;}
[data-testid="stCode"] pre {font-size:.84rem !important;line-height:1.35 !important;padding:10px !important;}
.katex-display {margin:.45em 0 !important;}
</style>''', unsafe_allow_html=True)

# A página é o estado principal; o seletor é sincronizado por callbacks.
# Nenhum widget é alterado depois de ser criado no mesmo rerun.
if 'seminar_page' not in st.session_state:
    st.session_state.seminar_page = 0


def set_page(index):
    index = max(0, min(len(PAGES)-1, int(index)))
    st.session_state.seminar_page = index
    st.session_state.seminar_picker = index


def navigate(delta):
    set_page(st.session_state.seminar_page + delta)


def pick_page():
    set_page(st.session_state.seminar_picker)


page = st.session_state.seminar_page
if 'seminar_picker' not in st.session_state:
    st.session_state.seminar_picker = page
with st.sidebar:
    st.caption('SEMINÁRIO DE INTELIGÊNCIA ARTIFICIAL')
    st.radio('Ir para a página', range(len(PAGES)), format_func=lambda i:f'{i+1:02d} · {PAGES[i]}', key='seminar_picker', on_change=pick_page)
    notes = st.checkbox('Mostrar roteiro de fala', False)
    st.caption('F11: tela cheia. Use os botões ou as setas ← → fora dos campos de edição.')


def navigation():
    left, mid, right = st.columns([1.15,5,1.15])
    left.button('← Voltar', key='seminar_previous', on_click=navigate, args=(-1,), disabled=page==0, use_container_width=True)
    mid.caption(f'MOVIELENS / IA  ·  {page+1:02d} DE {len(PAGES)}  ·  {TIMES[page]} min  ·  {"Apresentador 1" if page < 7 else "Apresentador 2" if page < 12 else "Ambos"}')
    right.button('Avançar →', key='seminar_next', on_click=navigate, args=(1,), disabled=page==len(PAGES)-1, use_container_width=True)
    st.progress((page+1)/len(PAGES))


navigation()
# O atalho é opcional: os botões e o seletor funcionam sem JavaScript.
# Não intercepta setas usadas para alterar notas, sliders ou outros controles.
import streamlit.components.v1 as components
components.html('''<script>
try {
  const host = window.parent;
  const doc = host.document;
  const old = host.__movieSeminarKeyHandler;
  if (old) doc.removeEventListener('keydown', old);
  const handler = (e) => {
    if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.shiftKey || e.repeat) return;
    if (!['ArrowLeft','ArrowRight'].includes(e.key)) return;
    if (e.target.closest('input,textarea,select,[contenteditable=true],[role=slider],[role=combobox],[role=listbox],[role=radio],[role=tab],a')) return;
    const label = e.key === 'ArrowRight' ? 'Avançar →' : '← Voltar';
    const button = [...doc.querySelectorAll('button')].find(b => b.textContent.trim() === label && !b.disabled);
    if (button) { e.preventDefault(); button.click(); }
  };
  host.__movieSeminarKeyHandler = handler;
  doc.addEventListener('keydown', handler);
  const slide = "SLIDE_NUMBER";
  if (host.__movieSeminarSlide !== slide) {
    host.__movieSeminarSlide = slide;
    const main = doc.querySelector('[data-testid="stMain"]');
    if (main) main.scrollTop = 0;
    const app = doc.querySelector('[data-testid="stAppViewContainer"]');
    if (app) app.scrollTop = 0;
    host.scrollTo(0,0);
  }
} catch (_) { /* navegação nativa permanece disponível */ }
</script>'''.replace('SLIDE_NUMBER', str(page)), height=0, width=0)
if page not in (0, 12):
    st.title(PAGES[page])


def takeaway(text):
    st.markdown(f'<div class="slide-key">{escape(text)}</div>', unsafe_allow_html=True)


def lead(text):
    st.markdown(f'<div class="slide-lead">{escape(text)}</div>', unsafe_allow_html=True)


def card(title, body, tag='', focus=False):
    cls = 'slide-card focus' if focus else 'slide-card'
    label = f'<div class="card-tag">{escape(tag)}</div>' if tag else ''
    st.markdown(f'<div class="{cls}">{label}<h3>{escape(title)}</h3><p>{escape(body)}</p></div>', unsafe_allow_html=True)


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


def metric_demo():
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
    chosen = st.radio('Métrica de proximidade', list(options), horizontal=True, label_visibility='collapsed')
    metric, fn, formula, description, advantage, limitation = options[chosen]
    st.markdown(f'**{description}**')
    a = np.array([5,4,2,3], dtype=float)
    l,r = st.columns([1,1.2])
    with l:
        st.caption('A = (5, 4, 2, 3) · Edite as notas de B nos mesmos filmes')
        cols = st.columns(4)
        b = np.array([c.number_input(f'B{i+1}',1,5,v,step=1,key=f'sem_b{i}') for i,(c,v) in enumerate(zip(cols,[4,3,2,3]))],dtype=float)
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
        result = f'{val:.4f}' if np.isfinite(val) else r'\text{indefinido}'
        st.latex(calc+'='+result if np.isfinite(val) else calc)
        st.markdown(f'**Peso do vizinho: {weight:.4f}**')
        st.caption('Peso = 1/(1+d) para distância; max(0,s) para similaridade válida.')
    with r:
        st.caption('IMPLEMENTAÇÃO REAL · src/metrics.py')
        st.code(compact_source(fn),language='python')
        if metric=='pearson':
            st.caption(f'Médias: A = {a.mean():g}; B = {b.mean():g}. Reutiliza a função cosseno.')
            with st.expander('Ver função reutilizada'):
                st.code(compact_source(metrics.cosine_similarity),language='python')
    c1,c2=st.columns(2)
    c1.markdown('**+ Vantagem**  \n'+advantage)
    c2.markdown('**− Limitação**  \n'+limitation)


def results_page():
    try:
        final = result_file('test_metrics.csv')
        validation = result_file('validation_metrics.csv')
        view=st.radio('Visualização',['Teste final','Validação de k','Precision e Recall'],horizontal=True,label_visibility='collapsed')
        if view=='Teste final':
            winner=final.loc[final.rmse.idxmin()]
            lead(f'{LABELS[winner.model]} apresentou o menor RMSE no teste.')
            choice=st.radio('Indicador',['RMSE','MAE','Coverage'],horizontal=True)
            measure=choice.lower()
            display=final.copy(); display['Modelo']=display.model.map(LABELS)
            if measure=='coverage':
                display=display[display.model!='baseline'].copy()
                display[measure]*=100
            l,r=st.columns([1.45,1])
            with l:
                st.bar_chart(display.set_index('Modelo')[measure],height=235)
            with r:
                table=display[['Modelo','metric','k',measure]].copy()
                table.columns=['Modelo','Métrica','k',choice]
                st.dataframe(table.round(4),hide_index=True,use_container_width=True,height=185)
                st.caption('Coverage em %; baseline sem vizinhos. Erros menores são melhores.')
            takeaway('Gêneros trouxe ganho modesto; incluir demografia não reduziu mais o RMSE.')
        elif view=='Validação de k':
            l,r=st.columns(2)
            mode=l.selectbox('Modelo',list(LABELS)[:3],format_func=LABELS.get)
            measure=r.radio('Erro',['rmse','mae'],horizontal=True)
            subset=validation[validation.model==mode]
            best=subset.loc[subset[measure].idxmin()]
            st.line_chart(subset.pivot(index='k',columns='metric',values=measure).sort_index(),height=260)
            takeaway(f'Menor {measure.upper()}: {best.metric} · k = {int(best.k)} · {best[measure]:.4f}.')
        else:
            n=st.radio('Número de recomendações',[5,10],horizontal=True,format_func=lambda v:f'Top {v}')
            rank=ranking_table().set_index('model')
            rank.index=rank.index.map(LABELS)
            cols=[f'precision@{n}',f'recall@{n}']
            st.bar_chart(rank[cols]*100,height=260)
            st.caption('Relevante: nota real ≥ 4. Ranking apenas dos filmes do teste; não do catálogo completo.')
            takeaway('Os três modelos têm desempenhos muito próximos no ranking restrito.')
        st.caption('Comparação exploratória; diferenças pequenas não comprovam significância estatística.')
    except (FileNotFoundError,KeyError) as exc:
        st.warning(f'Resultado indisponível: {exc}. Execute experiments.py ou disponibilize results_v2.zip.')


if page == 0:
    st.markdown('<div class="slide-kicker">Seminário de inteligência artificial · 2026</div><div class="slide-hero">Como prever o próximo <em>filme favorito</em> de um usuário?</div>',unsafe_allow_html=True)
    lead('k-NN e filtragem colaborativa')
    st.write('Da primeira implementação à avaliação de um recomendador híbrido.')
    for c,label,value in zip(st.columns(3),['Avaliações reais','Métricas de proximidade','Configurações avaliadas'],['100.000','5','90']): c.metric(label,value)
    flow('Problema','Algoritmo','Melhoria','Resultados','Demonstração')
elif page == 1:
    lead('Um catálogo enorme. Gostos diferentes.')
    l,r=st.columns([1,1.15])
    with l:
        card('Popularidade','A mesma lista de filmes para todos.')
        card('Personalização','Estimar o que cada usuário gostaria de assistir.','Nosso objetivo',True)
    with r:
        st.caption('Exemplo fictício · avaliações de 1 a 5')
        st.dataframe(pd.DataFrame({'Filme':['Star Wars','Titanic','Toy Story'],'A':[5,2,4],'B':[2,5,2],'X':[5,2,None]}),hide_index=True,use_container_width=True,height=145)
        takeaway('Qual nota X daria a Toy Story?')
    st.caption('Entrada: notas conhecidas. Saída: notas previstas e ranking de filmes não avaliados.')
elif page == 2:
    lead('Como personalizar uma recomendação?')
    for col,(title,body,tag,focus) in zip(st.columns(3),[
        ('Conteúdo','Compara características dos filmes com o gosto do usuário.','Alternativa',False),
        ('Colaborativa','Busca usuários com avaliações semelhantes.','Nossa primeira solução',True),
        ('Híbrida','Combina notas e atributos adicionais para comparar usuários.','Nossa evolução',False)]):
        with col: card(title,body,tag,focus)
    flow('Usuário X','Usuários semelhantes','Filmes recomendados')
    takeaway('Começamos pela filtragem colaborativa baseada em usuários.')
elif page == 3:
    lead('k-NN: aprender com exemplos conhecidos.')
    flow('Armazenar instâncias','Medir proximidade','Selecionar k vizinhos','Prever')
    l,r=st.columns(2)
    with l: card('Classificação','Votação dos vizinhos → uma classe. Ex.: gosta ou não gosta.')
    with r: card('Regressão','Média das notas dos vizinhos → valor numérico. Ex.: 4,82.','Nosso problema',True)
    takeaway('Lazy learning: o modelo consulta os exemplos em vez de ajustar parâmetros preditivos.')
elif page == 4:
    lead('Das notas conhecidas à previsão de uma nota.')
    df=pd.DataFrame({'Filme A':[5,4,1,2,5],'Filme B':[4,4,5,5,3],'Filme C':[5,4,1,np.nan,3]},index=[1,2,3,4,5])
    l,r=st.columns([1,1.2])
    with l:
        st.markdown('**1 · Perfil X: A = 5, B = 4, C = ?**')
        st.dataframe(df,use_container_width=True,height=215)
        k=st.slider('2 · Quantidade máxima de vizinhos (k)',1,5,3,key='knn_demo_k')
    model=MovieKNN(df,'euclidiana')
    neighbors=model.neighbors(pd.Series({'Filme A':5,'Filme B':4}))
    score,details=model.predict('Filme C',neighbors,k)
    with r:
        st.markdown('**3 · Mais próximos que avaliaram C**')
        table=pd.DataFrame(details,columns=['Usuário','Distância','Peso','Em comum','Nota']).drop(columns='Em comum')
        st.dataframe(table.round(3),hide_index=True,use_container_width=True,height=180)
        st.caption('Distância euclidiana entre A e B; peso = 1/(1+d).')
        st.latex(r'\hat r=\frac{\sum_v w_vr_v}{\sum_v w_v}'+r'\approx\frac{'+ '+'.join(f'{row[2]:.2f}\\cdot {row[4]:g}' for row in details)+'}{'+ '+'.join(f'{row[2]:.2f}' for row in details)+'}')
        st.metric('4 · Nota prevista para C',f'{score:.2f}')
    st.caption('Exemplo fictício. O ranking repete a previsão para cada filme não avaliado.')
elif page == 5:
    metric_demo()
elif page == 6:
    lead('A maioria das avaliações não existe.')
    for c,label,value in zip(st.columns(4),['Usuários','Filmes','Avaliações','Esparsidade'],['943','1.682','100.000','93,7%']):c.metric(label,value)
    st.progress(.063,text='Apenas 6,3% da matriz está preenchida')
    try:
        ratings=load_ratings(); matrix=build_rating_matrix(ratings)
        with st.expander('Explorar os dados reais',expanded=True):
            start=st.slider('Primeiro usuário exibido',0,len(matrix)-5,0)
            st.dataframe(matrix.iloc[start:start+5,:8],height=210,use_container_width=True)
    except FileNotFoundError: st.warning('Base local ausente. Execute download_data.py.')
    takeaway('NaN é ausência de avaliação, não nota zero.')
elif page == 7:
    lead('A primeira versão funcionou. Mas nem toda nota alta era confiável.')
    l,r=st.columns(2)
    with l:
        card('Nossa primeira solução','Comparar notas em comum e prever pela média ponderada.','k-NN tradicional')
        card('O problema','Filmes raros podiam receber 5,00 com apenas 1 ou 2 vizinhos.','Primeira execução',True)
    with r:
        st.metric('Usuários elegíveis no modelo tradicional','505 / 943')
        st.caption('Exemplo do perfil fictício executado na versão 2.')
        st.metric('Melhor configuração tradicional','Hamming · k = 20')
        st.caption('Seleção pelo RMSE de validação.')
    takeaway('Precisávamos ampliar a busca de vizinhos e exigir suporte mínimo para recomendar.')
elif page == 8:
    lead('Mais atributos para encontrar vizinhos, mesmo sem filmes em comum.')
    for col,(title,body,tag) in zip(st.columns(3),[
        ('Notas','Avaliações dos mesmos filmes.','50%'),('Gêneros','Média das notas por gênero.','35%'),('Perfil','Idade, ocupação e sexo registrado.','15%')]):
        with col: card(title,body,tag)
    l,r=st.columns([1.1,1])
    with l:
        st.latex(r'S=\frac{\sum_{j\in D}w_jS_j}{\sum_{j\in D}w_j}')
        st.caption('Pesos iniciais, não otimizados; D = blocos disponíveis.')
    with r:
        st.metric('Usuários elegíveis no exemplo','943 / 943',delta='438 a mais que o tradicional')
        st.caption('Elegível não significa igualmente semelhante.')
    takeaway('A nota continua sendo prevista pelos vizinhos que avaliaram o filme.')
    with st.expander('Detalhes: gêneros, demografia e suporte'):
        st.latex(r'S_g=1-\frac{\operatorname{mean}(|A_g-B_g|)}{4}')
        st.write('Demografia: média das comparações disponíveis. Idade usa diferença limitada a 60 anos; ocupação e sexo usam igualdade. Exigimos evidência positiva de gosto. Suporte mínimo filtra o ranking da interface.')
elif page == 9:
    lead('Como verificamos se a melhoria funcionou?')
    for c,label,value in zip(st.columns(3),['Treino · 64%','Validação · 16%','Teste · 20%'],['64.000','16.000','20.000']):c.metric(label,value)
    flow('3 modelos × 5 métricas × 6 valores de k','90 configurações','Escolha pelo RMSE')
    l,r=st.columns(2)
    with l: card('Previsão de notas','MAE e RMSE: menor erro é melhor. Coverage: previsões com vizinhos.')
    with r: card('Ranking','Precision: proporção de relevantes no Top N. Recall: proporção de relevantes recuperados.')
    st.caption('Baseline: média do filme. Relevância: nota real ≥ 4. Ranking restrito ao teste.')
    with st.expander('Fórmulas e protocolo experimental'):
        st.latex(r'MAE=\frac{1}{n}\sum_i|r_i-\hat r_i|\qquad RMSE=\sqrt{\frac{1}{n}\sum_i(r_i-\hat r_i)^2}')
        st.latex(r'Precision@N=\frac{acertos}{N}\qquad Recall@N=\frac{acertos}{total\ relevante}')
        st.write('Divisão aleatória por avaliação, semente 42. Escolhemos cada modelo pela validação, depois usamos treino + validação para prever o teste. O ranking só considera avaliações ocultas; média por usuário com pelo menos um relevante.')
elif page == 10:
    results_page()
elif page == 11:
    flow('dataset.py','metrics.py / hybrid.py','knn.py','experiments.py / app.py')
    funcs={
        'Encontrar vizinhos':(MovieKNN.neighbors,'Compara o perfil com todos os usuários e ordena por peso.'),
        'Selecionar vizinhos':(MovieKNN.select_neighbors,'Mantém até k vizinhos com nota no filme candidato.'),
        'Média ponderada':(MovieKNN.weighted_average,'Transforma notas dos vizinhos em uma previsão.'),
        'Perfil por gênero':(HybridKNN.genre_profile,'Agrupa notas conhecidas por gênero cinematográfico.'),
        'Similaridade combinada':(HybridKNN.combined_similarity,'Combina apenas os blocos disponíveis.'),
        'Prever nota':(MovieKNN.predict,'Seleciona vizinhos e usa fallback quando necessário.'),
        'Ranking':(MovieKNN.recommend,'Filtra por suporte e ordena pelas notas previstas.')}
    name=st.selectbox('Função real do projeto',list(funcs),index=2)
    fn,desc=funcs[name]
    st.markdown(f'**{desc}**')
    st.code(compact_source(fn),language='python')
    st.caption('O mesmo núcleo é utilizado nos experimentos e na interface.')
elif page == 12:
    st.caption('Demonstração ao vivo · base completa, fora da avaliação experimental.')
    from app import render_recommender
    render_recommender()
else:
    lead('O algoritmo é simples. A representação dos usuários faz diferença.')
    for col,(title,body) in zip(st.columns(3),[
        ('k-NN funciona','Erros menores que a média do filme no teste.'),
        ('Gêneros ajudam','Melhor RMSE observado na versão híbrida com gêneros.'),
        ('Mais não é sempre melhor','O perfil ampliado não melhorou o RMSE em relação a gêneros.')]):
        with col: card(title,body)
    takeaway('Limitações: esparsidade, usuários novos, filmes raros e avaliação restrita do ranking.')
    st.caption('Resultados exploratórios; diferenças pequenas, sem teste de significância.')
    with st.expander('Referências'):
        st.markdown('- [GroupLens — MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)\n- [Cover & Hart (1967) — Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964)\n- [Sarwar et al. (2001) — Item-based collaborative filtering](https://doi.org/10.1145/371920.372071)')

if notes:
    prompts=[
        'Apresente a questão central: prever notas e ordenar filmes. O projeto evoluiu após observar uma limitação prática; não começamos com o híbrido.',
        'Explique que popularidade não equivale a relevância pessoal. Na tabela, X concorda com A nos filmes conhecidos; queremos estimar sua nota para Toy Story.',
        'Conteúdo usa atributos dos filmes; pode restringir diversidade. Colaborativa explora o comportamento coletivo, mas depende de notas. Híbrida combina evidências e pode introduzir ruído. Nosso k-NN é user-based, não item-based.',
        'Lazy learning: treinamento consiste em organizar as instâncias e construir perfis com notas conhecidas. Na consulta calculamos proximidade e agregamos respostas. Gênero e ocupação são atributos, não classes. A saída é uma nota numérica.',
        'Mostre a matriz, o perfil X e a ausência de C. Compare A/B, ordene pela distância, filtre quem avaliou C, aplique até k vizinhos. A expressão numérica usa pesos arredondados; o resultado usa precisão integral. Mude k.',
        'Escolha uma métrica por vez; mostre conceito, fórmula, substituição, código real e resultado. Distâncias menores e similaridades positivas maiores implicam maior peso. Hamming é normalizada; Pearson pode ser indefinida. Testamos as cinco, sem assumir superioridade universal.',
        'MovieLens tem 100 mil avaliações de 943 usuários e 1.682 filmes. Mostre NaN na matriz real. Não significa zero nem que o usuário nunca assistiu ao filme; só que não existe nota registrada.',
        'Na execução inicial, o ranking tinha notas 5 com um ou dois vizinhos. k é um teto, não garantia. Pouca sobreposição dificulta comparar usuários. Filtro mínimo evita recomendações sem suporte; Hamming k=20 foi melhor na validação tradicional.',
        'Explique os três modelos. Gêneros são médias por categoria de filme (um filme pode ter vários gêneros). Demografia usa comparações normalizadas. Pesos são heurísticos e redistribuídos entre blocos disponíveis; demografia sozinha não torna um vizinho elegível. 505→943 no exemplo fictício não é uma média populacional.',
        'Separe treino, validação e teste. Escolhemos hiperparâmetros na validação e refazemos a matriz com 80 mil avaliações. Perfis de gênero usam apenas notas conhecidas. MAE/RMSE medem notas; ranking usa nota ≥4 e apenas candidatos do teste. O suporte mínimo só afeta a interface.',
        'Leia os resultados reais. Mostre as curvas de validação, a melhor configuração de cada modelo e a diferença para baseline. Teste anterior já era conhecido; análise exploratória. Precision/Recall não avaliam o catálogo completo e não temos evidência de diferença significativa.',
        'Siga o fluxo dos arquivos. Abra seleção de vizinhos e média ponderada; mostre a função combinada do híbrido. Não leia todas as linhas. A refatoração tornou os cálculos independentes.',
        'Selecione filmes, atribua notas, compare modelos e métricas com o mesmo perfil. Mostre o suporte de cada filme e abra uma explicação de vizinhos. A base completa é usada só na demonstração.',
        'Retome a pergunta inicial: o k-NN reduziu os erros diante da média do filme; gêneros deram ganho modesto; perfil ampliado não melhorou RMSE em relação a gêneros. Discuta limitações e abra para perguntas.'
    ]
    with st.sidebar:
        st.markdown('### Roteiro de fala')
        st.write(prompts[page])
