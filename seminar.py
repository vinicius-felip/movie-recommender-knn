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

st.set_page_config(page_title='MovieLens | Seminário de IA', layout='wide')
st.markdown('''<style>
.stApp {background:#101218;color:#f4f4f5;}
[data-testid="stMainBlockContainer"] {max-width:1440px;padding-top:1.5rem;}
h1,h2,h3 {letter-spacing:-0.035em;}
[data-testid="stMetric"] {border:1px solid #393d48;border-radius:10px;padding:14px;}
[data-testid="stSidebar"] {background:#191c24;}
</style>''', unsafe_allow_html=True)

if 'seminar_page' not in st.session_state:
    st.session_state.seminar_page = 0

def navigate(delta):
    st.session_state.seminar_page = max(0, min(len(PAGES)-1, st.session_state.seminar_page+delta))

page = st.session_state.seminar_page
with st.sidebar:
    st.caption('SEMINÁRIO DE INTELIGÊNCIA ARTIFICIAL')
    st.selectbox('Página', range(len(PAGES)), format_func=lambda i:f'{i+1:02d} · {PAGES[i]}', key='seminar_page')
    st.caption('Use F11 no navegador para tela cheia.')
    notes = st.checkbox('Mostrar roteiro de fala', False)
page = st.session_state.seminar_page
left, mid, right = st.columns([1,5,1])
left.button('← Voltar', on_click=navigate, args=(-1,), disabled=page==0)
mid.caption(f'MOVIELENS / IA   •   {page+1:02d} / {len(PAGES)}   •   {TIMES[page]} min')
right.button('Avançar →', on_click=navigate, args=(1,), disabled=page==len(PAGES)-1)
st.progress((page+1)/len(PAGES))
st.title(PAGES[page])

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
        'Euclidiana': ('euclidiana', metrics.euclidean_distance, r'd=\sqrt{\sum_i(a_i-b_i)^2}', 'Distância em linha reta; penaliza diferenças grandes.'),
        'Manhattan': ('manhattan', metrics.manhattan_distance, r'd=\sum_i|a_i-b_i|', 'Soma diferenças absolutas; depende da quantidade de notas em comum.'),
        'Hamming': ('hamming', metrics.hamming_distance, r'd=\frac{1}{n}\sum_i\mathbf{1}(a_i\ne b_i)', 'Proporção de notas diferentes; não distingue a magnitude da divergência.'),
        'Cosseno': ('cosseno', metrics.cosine_similarity, r's=\frac{A\cdot B}{\|A\|\|B\|}', 'Compara a orientação; notas positivas podem gerar similaridades altas.'),
        'Pearson': ('pearson', metrics.pearson_correlation, r'r=\frac{(A-\bar A)\cdot(B-\bar B)}{\|A-\bar A\|\|B-\bar B\|}', 'Compara padrões centralizados; é indefinida para vetores constantes.'),
    }
    chosen = st.radio('Métrica', list(options), horizontal=True)
    metric, fn, formula, description = options[chosen]
    st.write(description)
    a = np.array([5,4,2,3], dtype=float)
    cols = st.columns(4)
    b = np.array([c.slider(f'Filme {i+1} · A = {int(a[i])}',1,5,v,key=f'sem_b{i}') for i,(c,v) in enumerate(zip(cols,[4,3,2,3]))],dtype=float)
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
        calc = rf'\frac{{{ac@bc:g}}}{{\sqrt{{{ac@ac:g}}}\sqrt{{{bc@bc:g}}}}}' if np.linalg.norm(bc)>0 else r'\text{Indefinida: norma centralizada zero}'
    l,r = st.columns([1,1.2])
    with l:
        st.subheader('1 · Fórmula')
        st.latex(formula)
        st.subheader('2 · Substituição')
        st.latex(calc)
        val,weight,n = metrics.proximity(a,b,metric)
        c1,c2=st.columns(2)
        c1.metric('Resultado',f'{val:.4f}' if np.isfinite(val) else 'Indefinido')
        c2.metric('Peso do vizinho',f'{weight:.4f}')
        st.latex(r'w=1/(1+d)' if metric in ('euclidiana','manhattan','hamming') else r'w=\max(0,s)')
        st.caption('Similaridade indefinida recebe peso zero. As distâncias menores e as similaridades positivas maiores favorecem a seleção.')
    with r:
        st.subheader('3 · Código executado')
        st.code(inspect.getsource(fn),language='python')
        if metric=='pearson':
            with st.expander('Função de cosseno reutilizada'):
                st.code(inspect.getsource(metrics.cosine_similarity),language='python')
        st.caption('Fonte: src/metrics.py — sem reimplementar o cálculo na apresentação.')


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
    st.write('Sistemas de recomendação com k-Nearest Neighbors e filtragem colaborativa.')
    for c,label,value in zip(st.columns(3),['Avaliações reais','Métricas de proximidade','Configurações avaliadas'],['100.000','5','90']): c.metric(label,value)
    st.caption('Seminário de IA · apresentação em dupla · 2026')
elif page == 1:
    st.header('Muitos filmes. Pouco tempo para escolher.')
    st.write('Um ranking de popularidade é igual para todos. Nosso objetivo é prever notas personalizadas para filmes ainda não avaliados.')
    st.info('Pergunta: usuários com gostos semelhantes ajudam a prever avaliações futuras?')
elif page == 2:
    st.table(pd.DataFrame({'Abordagem':['Conteúdo','Colaborativa','Híbrida'],'Base':['Características dos filmes','Avaliações de outros usuários','Avaliações + atributos adicionais']}))
    st.write('Começamos com filtragem colaborativa baseada em usuários e evoluímos para uma representação híbrida.')
elif page == 3:
    st.header('O k-NN guarda exemplos, não ajusta uma rede de parâmetros.')
    st.write('Lazy learning: organiza as instâncias conhecidas; na consulta, encontra vizinhos e agrega suas respostas.')
    st.table(pd.DataFrame({'Classificação':['Votação','Classe discreta'],'Regressão':['Média ponderada','Nota numérica de 1 a 5']}))
    st.info('Nosso problema é regressão. Gênero e ocupação são atributos, não classes de saída.')
elif page == 4:
    st.caption('Exemplo fictício: notas para dois filmes conhecidos e um filme candidato.')
    df=pd.DataFrame({'Filme A':[5,4,1,2,5],'Filme B':[4,4,5,5,3],'Filme C':[5,4,1,2,3]},index=[1,2,3,4,5])
    k=st.slider('k',1,5,3)
    model=MovieKNN(df,'euclidiana')
    profile=pd.Series({'Filme A':5,'Filme B':4})
    neighbors=model.neighbors(profile)
    score,details=model.predict('Filme C',neighbors,k)
    l,r=st.columns(2)
    l.dataframe(df); r.dataframe(pd.DataFrame(details,columns=['Usuário','Distância','Peso','Em comum','Nota']),hide_index=True)
    st.latex(r'\hat r=\frac{\sum_v w_vr_v}{\sum_v w_v}')
    st.metric('Nota prevista',f'{score:.2f}')
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
    st.write('NaN = avaliação ausente, não nota zero.')
elif page == 7:
    st.header('Apenas avaliações em comum')
    st.write('Comparamos notas válidas, transformamos a proximidade em peso e selecionamos até k vizinhos que avaliaram o filme candidato.')
    st.code(inspect.getsource(metrics.get_common_ratings),language='python')
    st.warning('Pouca sobreposição → menos vizinhos. Filmes raros podem receber previsão sustentada por poucas notas.')
elif page == 8:
    st.table(pd.DataFrame({'Modelo':['Tradicional','Gêneros','Perfil ampliado'],'Atributos':['Notas','Notas + médias por gênero','Notas + gêneros + idade, ocupação e sexo registrado']}))
    st.latex(r'S=\frac{\sum_{j\in D}w_jS_j}{\sum_{j\in D}w_j}')
    st.write('Pesos iniciais: notas 50, gêneros 35, demografia 15. D contém os blocos disponíveis. Sem evidência positiva de gosto, demografia sozinha não torna um usuário elegível.')
    st.latex(r'S_g=1-\frac{\operatorname{mean}(|A_g-B_g|)}{4}')
    st.caption('A métrica selecionada aplica-se às notas. Gêneros e demografia usam comparações fixas; pesos não otimizados.')
elif page == 9:
    for c,label,value in zip(st.columns(3),['Treino','Validação','Teste'],['64.000','16.000','20.000']):c.metric(label,value)
    st.write('Divisão aleatória por avaliação (semente 42). Validação: 3 modelos × 5 métricas × 6 valores de k = 90 configurações. Escolha pelo RMSE; teste com treino + validação.')
    st.latex(r'MAE=\frac{1}{n}\sum_i|r_i-\hat r_i|\quad RMSE=\sqrt{\frac{1}{n}\sum_i(r_i-\hat r_i)^2}')
    st.write('Baseline: média do filme (ou média global). Cobertura: proporção de previsões com vizinhos. O teste é de notas ocultas de usuários conhecidos.')
elif page == 10:
    results_page()
elif page == 11:
    funcs={'Encontrar vizinhos':MovieKNN.neighbors,'Selecionar vizinhos':MovieKNN.select_neighbors,'Média ponderada':MovieKNN.weighted_average,'Perfil por gênero':HybridKNN.genre_profile,'Similaridade combinada':HybridKNN.combined_similarity,'Prever nota':MovieKNN.predict,'Ranking':MovieKNN.recommend}
    name=st.selectbox('Função real do projeto',list(funcs))
    st.code(inspect.getsource(funcs[name]),language='python')
    st.caption('Fluxo: dataset → metrics/hybrid → knn → experiments ou app.')
elif page == 12:
    st.caption('Demonstração com a base completa; não faz parte da avaliação experimental.')
    from app import render_recommender
    render_recommender()
else:
    st.header('O que aprendemos?')
    st.write('O k-NN é simples e interpretável. A qualidade depende da representação dos usuários, da métrica, do valor de k e da disponibilidade de avaliações.')
    st.write('Limitações: esparsidade, usuários novos, filmes raros, pesos heurísticos, avaliação de ranking restrita e diferenças pequenas entre modelos.')
    st.markdown('**Referências**\n\n- [GroupLens — MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)\n- [Cover & Hart (1967) — Nearest neighbor pattern classification](https://doi.org/10.1109/TIT.1967.1053964)\n- [Sarwar et al. (2001) — Item-based collaborative filtering recommendation algorithms](https://doi.org/10.1145/371920.372071)')

if notes:
    prompts=['Apresente o problema e o objetivo.','Contraste popularidade com personalização.','Explique as três abordagens.','Diferencie atributos, classes e regressão.','Mude k e mostre a média ponderada.','Escolha métricas e altere notas; relacione fórmula, código e resultado.','Mostre NaN e a esparsidade.','Explique a limitação de filmes em comum.','Explique os blocos e pesos; não confunda atributos com classes.','Diferencie treino, validação e teste.','Interprete erros e ranking; destaque limitações.','Mostre a sequência de chamadas do código.','Simule um perfil e altere a métrica.','Retome a pergunta inicial e abra para dúvidas.']
    st.info('Roteiro: '+prompts[page])
