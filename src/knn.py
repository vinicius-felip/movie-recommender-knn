import numpy as np
import pandas as pd
from .metrics import proximity


class MovieKNN:
    def __init__(self, matrix, metric='cosseno', min_common=2):
        self.matrix = matrix
        self.values = matrix.to_numpy(dtype=float)
        self.metric = metric
        self.min_common = min_common
        self.means = matrix.mean()
        self.global_mean = float(np.nanmean(self.values))

    def neighbors(self, profile, exclude=None):
        """Compara o perfil com todos os usuários e ordena por peso."""
        profile = profile.reindex(self.matrix.columns).to_numpy(dtype=float)
        rows = []
        for i, user in enumerate(self.matrix.index):
            if user == exclude:
                continue
            value, weight, common = proximity(
                profile, self.values[i], self.metric, self.min_common
            )
            if weight > 0:
                rows.append((i, user, value, weight, common))
        return sorted(rows, key=lambda row: (-row[3], row[1]))

    def select_neighbors(self, movie_index, neighbors, k):
        """Seleciona até k vizinhos que avaliaram o filme desejado."""
        if k < 1:
            raise ValueError('k deve ser positivo')
        eligible = [row for row in neighbors
                    if np.isfinite(self.values[row[0], movie_index])]
        return eligible[:k]

    def weighted_average(self, movie_index, eligible):
        """Calcula a nota prevista: soma(peso * nota) / soma(pesos)."""
        numerator = sum(row[3] * self.values[row[0], movie_index]
                        for row in eligible)
        denominator = sum(row[3] for row in eligible)
        return float(numerator / denominator)

    def fallback(self, movie):
        """Sem vizinhos: média do filme ou, se ausente, média global."""
        mean = self.means.loc[movie]
        return float(mean if np.isfinite(mean) else self.global_mean)

    def predict(self, movie, neighbors, k=20):
        """Prevê uma nota e retorna os detalhes dos vizinhos usados."""
        if k < 1:
            raise ValueError('k deve ser positivo')
        j = self.matrix.columns.get_loc(movie)
        eligible = self.select_neighbors(j, neighbors, k)
        if not eligible:
            return self.fallback(movie), []
        score = self.weighted_average(j, eligible)
        details = [(r[1], r[2], r[3], r[4], self.values[r[0], j])
                   for r in eligible]
        return score, details

    def recommend(self, ratings, k=20, top=5, min_support=1, demographics=None):
        """Prevê filmes não avaliados, filtra por suporte e ordena."""
        profile = pd.Series(ratings, dtype=float).reindex(self.matrix.columns)
        if not 1 <= min_support <= k:
            raise ValueError('Suporte deve estar entre 1 e k')
        neighbors = (self.neighbors(profile) if demographics is None
                     else self.neighbors(profile, demographics=demographics))
        results = []
        for movie in self.matrix.columns[profile.isna()]:
            score, details = self.predict(movie, neighbors, k)
            if len(details) >= min_support:
                results.append((movie, score, len(details)))
        results.sort(key=lambda r: (-r[1], -r[2], r[0]))
        return results[:top], neighbors
