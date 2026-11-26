import pandas as pd
import streamlit as st
from src.dataset import load_ratings,load_movies,build_rating_matrix,load_genres,load_users
from src.metrics import METRICS
from src.hybrid import HybridKNN, MODES

@st.cache_data
def data():
    return build_rating_matrix(load_ratings()),load_movies().set_index('movie_id').title.to_dict(),load_genres(),load_users()

def render_recommender(presentation=False):
    if not presentation:
        st.title('Recomendador de filmes — k-NN')
    try: matrix,titles,genres,users=data()
    except FileNotFoundError:
        st.error('Execute python download_data.py primeiro.'); st.stop()
    if presentation:
        row=st.columns([1.2,1.2,1,.85])
        mode=row[0].selectbox('Modelo',MODES,index=0,key='seminar_demo_mode')
        metric=row[1].selectbox('Métrica das notas',METRICS,index=2,key='seminar_demo_metric')
        k=row[2].number_input('Vizinhos (k)',1,80,20,key='seminar_demo_k')
        with row[3],st.popover('Mais opções'):
            minimum=st.slider('Mínimo de filmes em comum',2,10,2,key='seminar_min_common')
            support=st.number_input('Suporte mínimo',1,k,min(5,k),key=f'seminar_support-{k}')
            weights=(.5,.35,.15)
            if mode!='tradicional':
                st.caption('Gêneros: distância absoluta média normalizada.')
                weights=(st.slider('Peso das notas',0,100,50,key='seminar_weight_r'),st.slider('Peso dos gêneros',0,100,35,key='seminar_weight_g'),st.slider('Peso do perfil',0,100,15,key='seminar_weight_d') if mode=='perfil' else 0)
    else:
        mode=st.sidebar.selectbox('Modelo',MODES,index=1)
        metric=st.sidebar.selectbox('Métrica das notas',METRICS,index=2)
        k=st.sidebar.slider('Máximo de vizinhos (k)',1,80,20)
        minimum=st.sidebar.slider('Mínimo de filmes em comum (bloco de notas)',2,10,2)
        support=st.sidebar.number_input('Mínimo de vizinhos por recomendação',1,k,min(5,k),key=f'support-{k}')
        weights=(.5,.35,.15)
        if mode!='tradicional':
            st.sidebar.caption('Gêneros: distância absoluta média normalizada. Pesos redistribuídos quando faltam dados.')
            weights=(st.sidebar.slider('Peso das notas',0,100,50),st.sidebar.slider('Peso dos gêneros',0,100,35),st.sidebar.slider('Peso do perfil',0,100,15) if mode=='perfil' else 0)
    if mode!='tradicional' and sum(weights[:2])==0:
        st.warning('Dê peso positivo a notas ou gêneros.')
        return
    if presentation:
        _render_presentation(matrix,titles,genres,users,mode,metric,k,minimum,support,weights)
        return
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


def _render_presentation(matrix,titles,genres,users,mode,metric,k,minimum,support,weights):
    """Organiza a demonstração em duas colunas, usando o mesmo recomendador."""
    st.caption('Avalie pelo menos dois filmes e clique em Recomendar.')
    left,right=st.columns([1,1.25],gap='medium')
    demo=None
    with left,st.container(key='seminar_panel_profile'):
        st.markdown('**1 · Perfil fictício**')
        with st.container(key='seminar_demo_inputs'):
            if mode=='perfil':
                with st.expander('Dados do perfil (opcional)',expanded=False):
                    with st.container(key='seminar_controls_profile'):
                        age_col,sex_col=st.columns(2)
                        age=age_col.number_input('Idade (0 = não informar)',0,100,0,key='seminar_demo_age')
                        sex=sex_col.selectbox('Sexo registrado',['Não informar','F','M'],key='seminar_demo_sex')
                        occupation=st.selectbox('Ocupação',['Não informar']+sorted(users.occupation.unique().tolist()),key='seminar_demo_occupation')
                    demo={'age':age or None,'occupation':None if occupation=='Não informar' else occupation,'sex':None if sex=='Não informar' else sex}
            selected=st.multiselect('Filmes já avaliados',list(matrix.columns),default=[m for m in (1,50,288) if m in matrix.columns],format_func=lambda x:f'{titles.get(x,x)} [{x}]',key='seminar_selected_movies')
            ratings={}
            with st.container(key='seminar_controls_ratings'):
                for index,movie in enumerate(selected):
                    if index%2==0:
                        columns=st.columns(2)
                    ratings[movie]=columns[index%2].slider(titles.get(movie,str(movie)),1,5,{1:5,50:4,288:2}.get(movie,3),key=f'seminar_rating-{movie}')
        requested=st.button('Recomendar',key='seminar_recommend',use_container_width=True)
        # Não exibe notas calculadas para um perfil/configuração anterior.
        signature=(mode,metric,k,minimum,int(support),tuple(weights),tuple(sorted(ratings.items())),tuple(sorted((demo or {}).items())))
        if requested:
            if len(ratings)<2 or (mode=='tradicional' and len(ratings)<minimum):
                st.warning('Avalie mais filmes.')
            else:
                model=HybridKNN(matrix,metric,minimum,genres,users,mode,weights)
                recs,neighbors=model.recommend(ratings,k,min_support=int(support),demographics=demo)
                st.session_state.seminar_recommendation={
                    'signature':signature,'recs':recs,'neighbors':neighbors,
                    'details':{movie:model.predict(movie,neighbors,k)[1] for movie,_,_ in recs},
                }
    with right,st.container(key='seminar_panel_results'):
        st.markdown('**2 · Filmes recomendados**')
        result=st.session_state.get('seminar_recommendation')
        if result is None:
            st.info('Escolha pelo menos dois filmes e atribua notas para gerar as recomendações.')
        elif result['signature']!=signature:
            st.info('Perfil ou configuração alterados. Clique em Recomendar para atualizar as previsões.')
        else:
            recs,neighbors=result['recs'],result['neighbors']
            st.caption(f'{len(neighbors)} usuários elegíveis · {sum(row[4]==0 for row in neighbors)} sem filmes em comum')
            if not recs:
                st.warning('Nenhum filme atingiu o suporte mínimo. Reduza o suporte ou avalie mais filmes.')
            else:
                st.dataframe(pd.DataFrame([(titles.get(movie,movie),round(score,3),count) for movie,score,count in recs],columns=['Filme','Nota prevista','Vizinhos']),hide_index=True,use_container_width=True,height=35*(len(recs)+1)+3)
                with st.expander('Vizinhos mais próximos do perfil (ranking geral)',expanded=False):
                    st.dataframe(pd.DataFrame([row[1:] for row in neighbors[:k]],columns=['Usuário','Proximidade','Peso','Filmes em comum']),hide_index=True,use_container_width=True)
                with st.expander('Como foi prevista a nota de um filme?',expanded=False):
                    movie=st.selectbox('Filme recomendado',[movie for movie,_,_ in recs],format_func=lambda value:titles.get(value,value))
                    st.dataframe(pd.DataFrame(result['details'][movie],columns=['Usuário','Proximidade','Peso','Filmes em comum','Nota']),hide_index=True,use_container_width=True)
    st.caption('Demonstração: base completa. Experimentos: conjuntos separados. O filtro de suporte só afeta o ranking exibido.')


if __name__ == "__main__":
    render_recommender()
