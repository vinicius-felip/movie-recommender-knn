from src.dataset import load_ratings,load_movies,build_rating_matrix,dataset_summary
from src.hybrid import HybridKNN, MODES
from src.dataset import load_genres, load_users, display_movie_title

def main():
    ratings,movies = load_ratings(),load_movies()
    dataset_summary(ratings,movies)
    matrix = build_rating_matrix(ratings)
    print('\nMatriz:',matrix.shape,'\n',matrix.iloc[:5,:5])
    titles = movies.set_index('movie_id').title.map(display_movie_title)
    genres,users=load_genres(),load_users()
    for mode in MODES:
        model=HybridKNN(matrix,genres=genres,users=users,mode=mode)
        recs,neighbors=model.recommend({1:5,50:4,288:2},k=20,min_support=5,demographics={'age':24,'occupation':'student'})
        print(f'\n{mode}: {len(neighbors)} usuários elegíveis')
        for movie,score,n in recs: print(f'{titles[movie]}: {score:.2f} ({n} vizinhos)')

if __name__ == '__main__': main()
