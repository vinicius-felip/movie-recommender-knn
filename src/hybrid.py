"""Extensão do k-NN: combina blocos de atributos, sem mudar a regressão."""
import numpy as np
import pandas as pd
from .knn import MovieKNN
from .metrics import proximity

MODES = ('tradicional', 'generos', 'perfil')


class HybridKNN(MovieKNN):
    def __init__(self, matrix, metric='hamming', min_common=2, genres=None,
                 users=None, mode='generos', weights=(.5, .35, .15)):
        super().__init__(matrix, metric, min_common)
        if mode not in MODES:
            raise ValueError('Modelo desconhecido')
        self.mode, self.users, self.weights = mode, users, weights
        if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError('Pesos inválidos')
        if mode != 'tradicional':
            if genres is None:
                raise ValueError('Gêneros necessários')
            self.genres = genres.reindex(matrix.columns).fillna(0).to_numpy(float)
            self.genre_profiles = self.genre_profile(self.values)

    def genre_profile(self, profile):
        """Média das notas conhecidas por gênero; ausência permanece NaN."""
        count = np.isfinite(profile).astype(float) @ self.genres
        sums = np.nan_to_num(profile, nan=0) @ self.genres
        return np.divide(sums, count, out=np.full_like(sums, np.nan), where=count > 0)

    @staticmethod
    def genre_similarity(a, b):
        """Compara as médias dos gêneros em comum, na escala de 0 a 1."""
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            return None
        return 1 - float(np.mean(np.abs(a[mask] - b[mask]))) / 4

    @staticmethod
    def age_similarity(a, b):
        """Diferença de idade normalizada e limitada a 60 anos."""
        return 1 - min(abs(float(a) - float(b)) / 60, 1)

    @staticmethod
    def occupation_similarity(a, b):
        return float(a == b)

    @staticmethod
    def sex_similarity(a, b):
        return float(a == b)

    @staticmethod
    def demographic_similarity(a, b):
        """Média das similaridades demográficas disponíveis."""
        functions = {
            'age': HybridKNN.age_similarity,
            'occupation': HybridKNN.occupation_similarity,
            'sex': HybridKNN.sex_similarity,
        }
        scores = []
        for key, similarity in functions.items():
            x, y = a.get(key), b.get(key)
            if pd.isna(x) or pd.isna(y) or x == '' or y == '':
                continue
            scores.append(similarity(x, y))
        return float(np.mean(scores)) if scores else None

    @staticmethod
    def combined_similarity(blocks):
        """Média ponderada dos blocos disponíveis, sem imputar ausências."""
        denominator = sum(w for w, s in blocks)
        return sum(w * s for w, s in blocks) / denominator if denominator else 0

    def neighbors(self, profile, exclude=None, demographics=None):
        """Encontra vizinhos pela similaridade de notas, gêneros e perfil."""
        if self.mode == 'tradicional':
            return super().neighbors(profile, exclude)
        p = profile.reindex(self.matrix.columns).to_numpy(float)
        gp = self.genre_profile(p)
        rows = []
        for i, uid in enumerate(self.matrix.index):
            if uid == exclude:
                continue

            # Bloco 1: avaliações em comum.
            value, wr, common = proximity(p, self.values[i], self.metric, self.min_common)
            # Bloco 2: preferências por gênero.
            wg = self.genre_similarity(gp, self.genre_profiles[i])
            blocks = []
            if common >= self.min_common and np.isfinite(value):
                blocks.append((self.weights[0], wr))
            if wg is not None:
                blocks.append((self.weights[1], wg))

            # Exige evidência de gosto, não apenas de demografia.
            if not any(w > 0 and s > 0 for w, s in blocks):
                continue

            # Bloco 3: perfil demográfico, quando disponível.
            if (self.mode == 'perfil' and demographics is not None
                    and self.users is not None and uid in self.users.index):
                wd = self.demographic_similarity(demographics, self.users.loc[uid])
                if wd is not None:
                    blocks.append((self.weights[2], wd))

            score = self.combined_similarity(blocks)
            if score > 0:
                rows.append((i, uid, score, score, common))
        return sorted(rows, key=lambda r: (-r[3], r[1]))
