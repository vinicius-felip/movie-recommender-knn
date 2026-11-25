import numpy as np


METRICS = (
    "euclidiana",
    "manhattan",
    "hamming",
    "cosseno",
    "pearson"
)


# ==========================================
# 1. DISTÂNCIA EUCLIDIANA
# ==========================================

def euclidean_distance(a, b):

    differences = a - b

    squared_differences = differences ** 2

    distance = np.sqrt(np.sum(squared_differences))

    return float(distance)


# ==========================================
# 2. DISTÂNCIA MANHATTAN
# ==========================================

def manhattan_distance(a, b):

    differences = a - b

    absolute_differences = np.abs(differences)

    distance = np.sum(absolute_differences)

    return float(distance)


# ==========================================
# 3. DISTÂNCIA DE HAMMING
# ==========================================

def hamming_distance(a, b):

    differences = a != b

    distance = np.mean(differences)

    return float(distance)


# ==========================================
# 4. SIMILARIDADE DE COSSENO
# ==========================================

def cosine_similarity(a, b):

    dot_product = np.dot(a, b)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    denominator = norm_a * norm_b

    if denominator == 0:
        return np.nan

    similarity = dot_product / denominator

    return float(np.clip(similarity, -1, 1))


# ==========================================
# 5. CORRELAÇÃO DE PEARSON
# ==========================================

def pearson_correlation(a, b):

    mean_a = np.mean(a)
    mean_b = np.mean(b)

    centered_a = a - mean_a
    centered_b = b - mean_b

    correlation = cosine_similarity(
        centered_a,
        centered_b
    )

    return correlation


# ==========================================
# 6. IDENTIFICAR AVALIAÇÕES EM COMUM
# ==========================================

def get_common_ratings(a, b):

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    common = np.isfinite(a) & np.isfinite(b)

    return a[common], b[common]


# ==========================================
# 7. CONVERTER DISTÂNCIA EM PESO
# ==========================================

def distance_to_weight(distance):

    return float(1 / (1 + distance))


# ==========================================
# 8. CONVERTER SIMILARIDADE EM PESO
# ==========================================

def similarity_to_weight(similarity):

    if not np.isfinite(similarity):
        return 0.0

    return max(0.0, similarity)


# ==========================================
# 9. SELECIONAR E EXECUTAR A MÉTRICA
# ==========================================

def proximity(a, b, metric="cosseno", min_common=2):

    if metric not in METRICS:
        raise ValueError("Métrica desconhecida")

    # Seleciona apenas avaliações em comum
    a, b = get_common_ratings(a, b)

    n = len(a)

    # Verifica a quantidade mínima de avaliações
    if n < min_common:
        return np.nan, 0.0, n

    # Calcula a métrica escolhida
    if metric == "euclidiana":

        value = euclidean_distance(a, b)
        weight = distance_to_weight(value)

    elif metric == "manhattan":

        value = manhattan_distance(a, b)
        weight = distance_to_weight(value)

    elif metric == "hamming":

        value = hamming_distance(a, b)
        weight = distance_to_weight(value)

    elif metric == "cosseno":

        value = cosine_similarity(a, b)
        weight = similarity_to_weight(value)

    elif metric == "pearson":

        value = pearson_correlation(a, b)
        weight = similarity_to_weight(value)

    return value, weight, n