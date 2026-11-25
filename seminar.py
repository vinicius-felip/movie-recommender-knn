"""Seminário interativo. Executar: python -m streamlit run seminar.py"""
import inspect
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
[data-testid="stMainBlockContainer"], .block-container {max-width:1440px;padding-top:1.7rem;padding-bottom:1rem;}
[data-testid="stVerticalBlock"] {gap:.65rem;}
h1,h2,h3 {letter-spacing:-.035em;}
h1 {font-size:2.15rem !important;padding:.15rem 0 .3rem !important;}
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
st.title(PAGES[page])


def takeaway(text):
    st.markdown(f'**Ideia-chave:** {text}')


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
    st.caption(description + ' · A = perfil fixo; B = perfil editável. Apenas filmes em comum.')
    a = np.array([5,4,2,3], dtype=float)
    cols = st.columns(4)
    b = np.array([c.number_input(f'Filme {i+1}: A = {int(a[i])} | B',1,5,v,step=1,key=f'sem_b{i}') for i,(c,v) in enumerate(zip(cols,[4,3,2,3]))],dtype=float)
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
    l,r = st.columns([1,1.1])
    with l:
        st.markdown('**1. Fórmula matemática**')
        st.latex(formula)
        st.markdown('**2. Valores substituídos**')
        if metric=='pearson':
            st.caption(f'Médias: A = {a.mean():g} · B = {b.mean():g}')
        st.latex(calc)
        val,weight,n = metrics.proximity(a,b,metric)
        c1,c2=st.columns(2)
        c1.metric('Resultado',f'{val:.4f}' if np.isfinite(val) else 'Indefinido')
        c2.metric('Peso',f'{weight:.4f}')
        st.caption('Peso = 1/(1+d) para distâncias; max(0,s) para similaridades. Indefinida = 0.')
    with r:
        st.markdown('**3. Implementação real — `metrics.py`**')
        st.code(compact_source(fn),language='python')
        if metric=='pearson':
            with st.expander('Ver código de cosseno reutilizado'):
                st.code(compact_source(metrics.cosine_similarity),language='python')
        st.caption('Menor distância ou maior similaridade positiva → maior influência.')
    c1,c2=st.columns(2)
    c1.markdown('**Vantagem para nosso estudo**  \n'+advantage)
    c2.markdown('**Limitação para nosso estudo**  \n'+limitation)
    st.caption('Testamos as cinco alternativas na validação, sem presumir que uma seja universalmente superior.')


def results_page():
    try:
        final = result_file('test_metrics.csv')
        validation = result_file('validation_metrics.csv')
        tab1,tab2,tab3 = st.tabs(['Teste final','Escolha de k','Precision e Recall'])
        with tab1:
            display=final.copy(); display['model']=display.model.map(LABELS)
            choice=st.radio('Indicador',['rmse','mae','coverage'],horizontal=True)
            chart=display if choice!='coverage' else display[display.model!='Baseline']
            st.bar_chart(chart.set_index('model')[choice])
            st.dataframe(display,hide_index=True)
            winner=final.loc[final.rmse.idxmin()]
            st.success(f'Menor RMSE no teste: {LABELS[winner.model]} ({winner.rmse:.4f}).')
            st.caption('Resultados exploratórios: o teste da primeira versão já era conhecido. Diferenças pequenas não implicam significância estatística.')
        with tab2:
            mode=st.selectbox('Modelo',list(LABELS)[:3],format_func=LABELS.get)
            measure=st.radio('Erro',['rmse','mae'],horizontal=True)
            subset=validation[validation.model==mode]
            st.line_chart(subset.pivot(index='k',columns='metric',values=measure).sort_index())
            st.dataframe(subset.sort_values('rmse').head(6),hide_index=True)
        with tab3:
            rank=ranking_table().set_index('model')
            rank.index=rank.index.map(LABELS)
            st.bar_chart(rank[['precision@5','recall@5','precision@10','recall@10']]*100)
            st.dataframe(rank,hide_index=False)
            st.caption('Relevância: nota real ≥ 4. Ranking restrito às avaliações de teste; média por usuário com ao menos um relevante. Denominador fixo N na precisão. Não avalia o catálogo completo nem o filtro de suporte da interface.')
    except (FileNotFoundError,KeyError) as exc:
        st.warning(f'Resultado indisponível: {exc}. Execute experiments.py ou disponibilize results_v2.zip.')


if page == 0:
    st.header('Como prever o próximo filme favorito de um usuário?')
    st.write('Uma investigação de aprendizado baseado em instâncias: da filtragem colaborativa tradicional ao k-NN com atributos adicionais.')
    for c,label,value in zip(st.columns(3),['Avaliações reais','Métricas de proximidade','Configurações avaliadas'],['100.000','5','90']): c.metric(label,value)
    st.markdown('**Nossa trajetória:** problema real → algoritmo → experimento inicial → limitação → melhoria → avaliação → demonstração.')
    st.caption('Seminário de IA · apresentação em dupla · 2026')
elif page == 1:
    st.header('O catálogo é grande. A preferência é individual.')
    l,r=st.columns([1,1])
    with l:
        st.markdown('### O problema real')
        st.write('Plataformas de streaming precisam ajudar cada pessoa a encontrar conteúdo relevante em catálogos extensos. Um ranking de popularidade oferece a mesma lista a todos, mas gostos diferentes exigem recomendações diferentes.')
        st.markdown('### O desafio de IA')
        st.write('A plataforma conhece apenas parte das preferências de cada usuário. Como estimar a avaliação de um filme que ele ainda não avaliou, sem conhecer sua opinião?')
    with r:
        st.markdown('### Mesmos filmes, gostos diferentes')
        st.dataframe(pd.DataFrame({'Filme':['Star Wars','Titanic','Toy Story'],'Usuário A':[5,2,4],'Usuário B':[2,5,4],'Usuário X':[5,2,None]}),hide_index=True,use_container_width=True)
        st.caption('Exemplo fictício: o usuário X parece mais próximo de A do que de B. O que ele acharia de Toy Story?')
        st.markdown('**Entrada:** histórico de notas.  \n**Saída:** nota estimada e ranking Top 5 de filmes ainda não avaliados.')
    takeaway('O objetivo não é prever popularidade, mas personalizar a recomendação. Vamos medir se os vizinhos ajudam de verdade.')
elif page == 2:
    st.header('Três formas de personalizar uma recomendação')
    cols=st.columns(3)
    with cols[0]:
        st.markdown('### Baseada em conteúdo')
        st.write('Compara as características dos filmes com o perfil de gostos do usuário.')
        st.markdown('**Exemplo:** quem gosta de ficção científica recebe filmes desse gênero.')
        st.caption('Vantagem: não exige notas de outros usuários. Limitação: pode restringir a diversidade de recomendações.')
    with cols[1]:
        st.markdown('### Filtragem colaborativa')
        st.write('Usa padrões de avaliações de vários usuários. Na versão baseada em usuários, procura pessoas de gostos semelhantes.')
        st.markdown('**Exemplo:** A e X gostam dos mesmos filmes; A também gostou de outro título.')
        st.caption('Vantagem: explora o comportamento coletivo. Limitação: depende de avaliações compartilhadas.')
    with cols[2]:
        st.markdown('### Híbrida')
        st.write('Combina avaliações com atributos adicionais, como preferências por gênero e informações do usuário.')
        st.markdown('**Exemplo:** A e B gostam de ação, ainda que tenham avaliado títulos diferentes.')
        st.caption('Vantagem: mais evidências de proximidade. Limitação: atributos extras podem introduzir ruído.')
    st.markdown('**Nossa escolha inicial:** filtragem colaborativa user-based com k-NN. Não comparamos filmes diretamente (item-based); buscamos usuários semelhantes.')
    takeaway('Começamos com a abordagem mais simples e depois investigamos se uma representação híbrida melhora a previsão.')
elif page == 3:
    st.header('Aprender com exemplos, sem ajustar uma rede de parâmetros')
    l,r=st.columns(2)
    with l:
        st.markdown('### Aprendizado baseado em instâncias')
        st.write('O k-NN armazena as instâncias conhecidas e usa uma medida de proximidade para buscar exemplos semelhantes quando surge uma consulta. É chamado de lazy learning: o esforço principal acontece na busca e na previsão.')
        st.markdown('**Treinamento no nosso caso:** organizar a matriz de avaliações e, no modelo híbrido, construir perfis por gênero usando as notas conhecidas.')
        st.markdown('**Predição:** calcular proximidade, selecionar até k vizinhos e agregar suas respostas.')
    with r:
        st.markdown('### Classificação × regressão')
        st.table(pd.DataFrame({'Classificação':['Votação','Categoria','Gosta / não gosta'],'Regressão':['Média ponderada','Valor numérico','Nota prevista: 4,82']},index=['Agregação','Saída','Exemplo']))
        st.write('Cada instância é um usuário. As notas e os atributos são entradas; a variável-alvo é a nota de um filme candidato.')
    takeaway('Usamos regressão baseada em vizinhança: gêneros e ocupações são atributos, não classes de saída.')
elif page == 4:
    st.caption('Exemplo didático: o usuário X avaliou A = 5 e B = 4; queremos estimar sua nota para C. Distância euclidiana.')
    df=pd.DataFrame({'Filme A':[5,4,1,2,5],'Filme B':[4,4,5,5,3],'Filme C':[5,4,1,np.nan,3]},index=[1,2,3,4,5])
    k=st.slider('1. Defina k (máximo de vizinhos)',1,5,3,key='knn_demo_k')
    model=MovieKNN(df,'euclidiana')
    profile=pd.Series({'Filme A':5,'Filme B':4})
    neighbors=model.neighbors(profile)
    score,details=model.predict('Filme C',neighbors,k)
    l,r=st.columns([1,1.25])
    with l:
        st.markdown('**2. Compare as notas de A e B**')
        st.dataframe(df,use_container_width=True)
        st.caption('Cada linha é um usuário. C não participa do cálculo de distância. NaN indica nota ausente.')
    with r:
        st.markdown('**3. Selecione os mais próximos que avaliaram C**')
        table=pd.DataFrame(details,columns=['Usuário','Distância','Peso','Em comum','Nota'])
        st.dataframe(table.round(3),hide_index=True,use_container_width=True)
        st.caption('Ordenação pela proximidade; o peso é 1/(1+d). O usuário sem nota em C não contribui.')
    l,r=st.columns([2,1])
    with l:
        st.markdown('**4. Preveja a nota por média ponderada**')
        st.latex(r'\hat r=\frac{\sum_v w_vr_v}{\sum_v w_v}='+r'\frac{'+ '+'.join(f'{row[2]:.3f}\\cdot {row[4]:g}' for row in details)+'}{'+ '+'.join(f'{row[2]:.3f}' for row in details)+'}')
    with r:
        st.metric('Nota prevista para C',f'{score:.2f}')
        st.caption(f'{len(details)} de até {k} vizinhos utilizados.')
    takeaway('Repetimos o cálculo para todos os filmes ainda não avaliados e ordenamos as notas previstas. k é um máximo por filme.')
elif page == 5:
    metric_demo()
elif page == 6:
    for c,label,value in zip(st.columns(4),['Usuários','Filmes','Avaliações','Esparsidade'],['943','1.682','100.000','93,7%']):c.metric(label,value)
    st.progress(.063,text='6,3% das posições contêm notas')
    try:
        ratings=load_ratings(); matrix=build_rating_matrix(ratings)
        start=st.slider('Primeiro usuário exibido',0,len(matrix)-10,0)
        st.dataframe(matrix.iloc[start:start+10,:10])
        st.bar_chart(ratings.rating.value_counts().sort_index())
    except FileNotFoundError: st.warning('Base local ausente. Execute download_data.py.')
    takeaway('NaN não é nota zero. A esparsidade restringe a comparação por notas; esse foi o problema que motivou a segunda versão.')
elif page == 7:
    st.header('Apenas avaliações em comum')
    st.write('Comparamos notas válidas, transformamos a proximidade em peso e selecionamos até k vizinhos que avaliaram o filme candidato.')
    st.code(inspect.getsource(metrics.get_common_ratings),language='python')
    st.warning('Na primeira execução, os cinco filmes recomendados tinham nota 5,00 e apenas 1–2 vizinhos. A nota alta sozinha não garantia suporte.')
    takeaway('A primeira versão funcionou, mas precisava de mais evidências de proximidade e de um filtro de suporte mínimo. Hamming com k=20 foi o melhor na validação inicial.')
elif page == 8:
    st.table(pd.DataFrame({'Modelo':['Tradicional','Gêneros','Perfil ampliado'],'Atributos':['Notas','Notas + médias por gênero','Notas + gêneros + idade, ocupação e sexo registrado']}))
    st.latex(r'S=\frac{\sum_{j\in D}w_jS_j}{\sum_{j\in D}w_j}')
    st.write('Pesos iniciais: notas 50, gêneros 35, demografia 15. D contém os blocos disponíveis. Sem evidência positiva de gosto, demografia sozinha não torna um usuário elegível.')
    st.latex(r'S_g=1-\frac{\operatorname{mean}(|A_g-B_g|)}{4}')
    st.caption('A métrica selecionada aplica-se às notas. Gêneros e demografia usam comparações fixas; pesos não otimizados.')
    st.markdown('**Exemplo de execução do main.py:** 505 usuários elegíveis no tradicional → 943 nos dois híbridos (perfil fictício). Não significa que todos sejam igualmente semelhantes.')
    takeaway('O perfil de gênero reduz a dependência de filmes idênticos. O filtro de suporte evita recomendações sustentadas por uma única avaliação.')
elif page == 9:
    for c,label,value in zip(st.columns(3),['Treino','Validação','Teste'],['64.000','16.000','20.000']):c.metric(label,value)
    st.write('Divisão aleatória por avaliação (semente 42). Validação: 3 modelos × 5 métricas × 6 valores de k = 90 configurações. Escolha pelo RMSE; teste com treino + validação.')
    st.latex(r'MAE=\frac{1}{n}\sum_i|r_i-\hat r_i|\quad RMSE=\sqrt{\frac{1}{n}\sum_i(r_i-\hat r_i)^2}')
    st.write('Baseline: média do filme (ou média global). Cobertura: proporção de previsões com vizinhos. O teste é de notas ocultas de usuários conhecidos.')
    st.markdown('**Avaliação de ranking:** Precision@N = acertos/N; Recall@N = acertos/total relevante. Relevante = nota real ≥ 4; N = 5 ou 10.')
    takeaway('Selecionamos a melhor configuração de cada modelo na validação. O teste mede previsão de notas; o ranking é restrito aos filmes do teste.')
elif page == 10:
    results_page()
elif page == 11:
    funcs={'Encontrar vizinhos':MovieKNN.neighbors,'Selecionar vizinhos':MovieKNN.select_neighbors,'Média ponderada':MovieKNN.weighted_average,'Perfil por gênero':HybridKNN.genre_profile,'Similaridade combinada':HybridKNN.combined_similarity,'Prever nota':MovieKNN.predict,'Ranking':MovieKNN.recommend}
    name=st.selectbox('Função real do projeto',list(funcs))
    st.code(inspect.getsource(funcs[name]),language='python')
    st.caption('Fluxo: dataset → metrics/hybrid → knn → experiments ou app.')
    takeaway('As mesmas funções são usadas no experimento e na demonstração. Refatoramos o código em funções menores para facilitar a explicação.')
elif page == 12:
    st.caption('Demonstração com a base completa; não faz parte da avaliação experimental.')
    from app import render_recommender
    render_recommender()
else:
    st.header('O que aprendemos?')
    st.write('O k-NN é simples e interpretável. A qualidade depende da representação dos usuários, da métrica, do valor de k e da disponibilidade de avaliações.')
    st.markdown('**O que os experimentos indicaram:** Hamming foi a melhor opção tradicional; o modelo com gêneros e cosseno obteve o menor RMSE. O perfil ampliado não melhorou o RMSE em relação a gêneros. As diferenças são pequenas e exploratórias.')
    st.write('Limitações: esparsidade, usuários novos, filmes raros, pesos heurísticos, avaliação de ranking restrita e diferenças pequenas entre modelos.')
    st.markdown('**Referências**\n\n- [GroupLens — MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)\n- [Cover & Hart (1967) — Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964)\n- [Sarwar et al. (2001) — Item-based collaborative filtering recommendation algorithms](https://doi.org/10.1145/371920.372071)')

if notes:
    prompts=['Apresente o problema e o objetivo.','Contraste popularidade com personalização.','Explique as três abordagens.','Diferencie atributos, classes e regressão.','Mude k e mostre a média ponderada.','Escolha métricas e altere notas; relacione fórmula, código e resultado.','Mostre NaN e a esparsidade.','Explique a limitação de filmes em comum.','Explique os blocos e pesos; não confunda atributos com classes.','Diferencie treino, validação e teste.','Interprete erros e ranking; destaque limitações.','Mostre a sequência de chamadas do código.','Simule um perfil e altere a métrica.','Retome a pergunta inicial e abra para dúvidas.']
    st.info('Roteiro: '+prompts[page])
