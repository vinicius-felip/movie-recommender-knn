"""Validação de 90 configurações; teste da vencedora de cada versão."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.dataset import load_ratings, build_rating_matrix
from src.metrics import METRICS
from src.hybrid import HybridKNN, MODES
from src.dataset import load_genres, load_users
KS=[5,10,20,40,60,80]
OUT=Path(__file__).resolve().parent/'results_v2'

def split(ratings):
    shuffled=ratings.sample(frac=1,random_state=42)
    return shuffled.iloc[:64000],shuffled.iloc[64000:80000],shuffled.iloc[80000:]

def evaluate(matrix, heldout, metric, ks, mode, genres, users):
    model=HybridKNN(matrix,metric,genres=genres,users=users,mode=mode)
    rows=[]
    for uid, group in heldout.groupby('user_id'):
        profile=matrix.loc[uid] if uid in matrix.index else pd.Series(index=matrix.columns,dtype=float)
        neighbors=model.neighbors(profile,exclude=uid,demographics=users.loc[uid] if uid in users.index else None)
        for r in group.itertuples():
            for k in ks:
                if r.movie_id in matrix.columns: pred,details=model.predict(r.movie_id,neighbors,k)
                else: pred,details=model.global_mean,[]
                rows.append((uid,r.movie_id,r.rating,pred,k,bool(details)))
    return pd.DataFrame(rows,columns=['user_id','movie_id','actual','predicted','k','has_neighbors'])

def summarize(df):
    e=df.predicted-df.actual
    return dict(mae=float(e.abs().mean()),rmse=float(np.sqrt((e**2).mean())),coverage=float(df.has_neighbors.mean()))

def baseline(matrix,heldout):
    means=matrix.mean(); global_mean=float(np.nanmean(matrix.to_numpy()))
    df=heldout.rename(columns={'rating':'actual'}).copy()
    df['predicted']=df.movie_id.map(means).fillna(global_mean)
    df['has_neighbors']=False
    return summarize(df)

def main():
    OUT.mkdir(exist_ok=True)
    train,val,test=split(load_ratings())
    matrix=build_rating_matrix(train)
    genres,users=load_genres(),load_users()
    scores=[]
    for mode in MODES:
        for metric in METRICS:
            print('Validando',mode,metric,flush=True)
            df=evaluate(matrix,val,metric,KS,mode,genres,users)
            for k,g in df.groupby('k'): scores.append(dict(model=mode,metric=metric,k=k,**summarize(g)))
    table=pd.DataFrame(scores).sort_values(['rmse','mae','model','metric','k'])
    table.to_csv(OUT/'validation_metrics.csv',index=False)
    winners=table.groupby('model',sort=False).head(1)
    winners.to_csv(OUT/'validation_winners.csv',index=False)
    final_matrix=build_rating_matrix(pd.concat([train,val]))
    finals=[]; predictions=[]
    for row in winners.itertuples():
        final=evaluate(final_matrix,test,row.metric,[int(row.k)],row.model,genres,users)
        final['model']=row.model; final['metric']=row.metric
        predictions.append(final)
        finals.append(dict(model=row.model,metric=row.metric,k=row.k,**summarize(final)))
    pd.concat(predictions).to_csv(OUT/'predictions.csv',index=False)
    finals.append(dict(model='baseline',**baseline(final_matrix,test)))
    pd.DataFrame(finals).to_csv(OUT/'test_metrics.csv',index=False)
    import matplotlib.pyplot as plt
    for name in ['rmse','mae']:
        for mode,gmode in table.groupby('model'):
            fig,ax=plt.subplots()
            for metric,g in gmode.groupby('metric'):
                g=g.sort_values('k'); ax.plot(g.k,g[name],marker='o',label=metric)
            ax.set(xlabel='k',ylabel=name.upper(),title=f'Validação — {mode}'); ax.legend(); fig.tight_layout(); fig.savefig(OUT/f'{name}_{mode}.png'); plt.close(fig)
    print(table.to_string(index=False)); print('Resultados finais:',OUT)
if __name__=='__main__': main()
