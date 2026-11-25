import pandas as pd
import streamlit as st
from src.dataset import load_ratings,load_movies,build_rating_matrix,load_genres,load_users
from src.metrics import METRICS
from src.hybrid import HybridKNN, MODES

@st.cache_data
def data():
    return build_rating_matrix(load_ratings()),load_movies().set_index('movie_id').title.to_dict(),load_genres(),load_users()

def render_recommender():
    st.title('Recomendador de filmes — k-NN')
    try: matrix,titles,genres,users=data()
    except FileNotFoundError:
        st.error('Execute python download_data.py primeiro.'); st.stop()
    mode=st.sidebar.selectbox('Modelo',MODES,index=1)
    metric=st.sidebar.selectbox('Métrica das notas',METRICS,index=2)
    k=st.sidebar.slider('Máximo de vizinhos (k)',1,80,20)
    minimum=st.sidebar.slider('Mínimo de filmes em comum (bloco de notas)',2,10,2)
    support=st.sidebar.number_input('Mínimo de vizinhos por recomendação',1,k,min(5,k),key=f'support-{k}')
    weights=(.5,.35,.15)
    if mode!='tradicional':
        st.sidebar.caption('Gêneros: distância absoluta média normalizada. Pesos redistribuídos quando faltam dados.')
        weights=(st.sidebar.slider('Peso das notas',0,100,50),st.sidebar.slider('Peso dos gêneros',0,100,35),st.sidebar.slider('Peso do perfil',0,100,15) if mode=='perfil' else 0)
        if sum(weights[:2])==0: st.warning('Dê peso positivo a notas ou gêneros.'); st.stop()
    demo=None
    if mode=='perfil':
        st.subheader('Perfil fictício (opcional)')
        age=st.number_input('Idade (0 = não informar)',0,100,0)
        occupation=st.selectbox('Ocupação',['Não informar']+sorted(users.occupation.unique().tolist()))
        sex=st.selectbox('Sexo registrado',['Não informar','F','M'])
        demo={'age':age or None,'occupation':None if occupation=='Não informar' else occupation,'sex':None if sex=='Não informar' else sex}
    selected=st.multiselect('Filmes já avaliados',list(matrix.columns),format_func=lambda x:f'{titles.get(x,x)} [{x}]')
    ratings={movie:st.slider(titles.get(movie,str(movie)),1,5,3,key=f'rating-{movie}') for movie in selected}
    if st.button('Recomendar'):
        if len(ratings)<2 or (mode=='tradicional' and len(ratings)<minimum): st.warning('Avalie mais filmes.'); st.stop()
        model=HybridKNN(matrix,metric,minimum,genres,users,mode,weights)
        recs,neighbors=model.recommend(ratings,k,min_support=int(support),demographics=demo)
        st.metric('Usuários semelhantes elegíveis',len(neighbors))
        st.write('Usuários elegíveis sem filmes em comum:',sum(r[4]==0 for r in neighbors))
        if not recs: st.warning('Nenhum filme atingiu o suporte mínimo. Reduza o suporte ou avalie mais filmes.'); st.stop()
        st.dataframe(pd.DataFrame([(titles.get(m,m),round(s,3),n) for m,s,n in recs],columns=['Filme','Nota prevista','Vizinhos']),hide_index=True)
        st.subheader('Vizinhos mais próximos do perfil (ranking geral)')
        st.dataframe(pd.DataFrame([r[1:] for r in neighbors[:k]],columns=['Usuário','Proximidade','Peso','Filmes em comum']),hide_index=True)
        for movie,score,n in recs:
            with st.expander(f'Como foi prevista a nota de {titles.get(movie,movie)}?'):
                _,details=model.predict(movie,neighbors,k)
                st.dataframe(pd.DataFrame(details,columns=['Usuário','Proximidade','Peso','Filmes em comum','Nota']),hide_index=True)
    st.caption('Demonstração: base completa. Experimentos: conjuntos separados. O filtro de suporte só afeta o ranking exibido.')


if __name__ == "__main__":
    render_recommender()
