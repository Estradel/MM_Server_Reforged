import numpy as np
import pytest
from numba import njit, prange

# -----------------------------------------------------------------------------
# 1. Instanciation Globale (Exécutée une seule fois au chargement du module)
# -----------------------------------------------------------------------------

# Configuration
BATCH_SIZE = 10_000  # On augmente le batch car 4x4 est très rapide à calculer
DIM = 4              # Matrices de transformation 4x4

print(f"\n--- Initialisation des données (Batch: {BATCH_SIZE}, Dim: {DIM}x{DIM}) ---")

# Génération des tableaux NUMPY
# On simule des matrices de transformation (valeurs aléatoires pour le benchmark)
GLOBAL_A_NP = np.random.rand(BATCH_SIZE, DIM, DIM)
GLOBAL_B_NP = np.random.rand(BATCH_SIZE, DIM, DIM)

# Conversion en listes PURE PYTHON (List of Lists of Lists)
# .tolist() convertit récursivement le tableau numpy en structures Python standard
GLOBAL_A_LIST = GLOBAL_A_NP.tolist()
GLOBAL_B_LIST = GLOBAL_B_NP.tolist()

# -----------------------------------------------------------------------------
# 2. Implémentation des fonctions
# -----------------------------------------------------------------------------

# A. Version Pure Python (Listes standards)
# Pas de dépendance à Numpy ici, uniquement des listes et des float
def python_list_matmul(A, B):
    # On recrée la structure de résultat avec des list comprehensions (plus rapide que append)
    result = [[[0.0] * DIM for _ in range(DIM)] for _ in range(BATCH_SIZE)]

    for n in range(BATCH_SIZE):
        for i in range(DIM):
            for j in range(DIM):
                total = 0.0
                for k in range(DIM):
                    # Accès liste standard [n][i][k]
                    total += A[n][i][k] * B[n][k][j]
                result[n][i][j] = total
    return result

@njit(fastmath=True)
def python_list_matmul_numba(A, B):
    # On recrée la structure de résultat avec des list comprehensions (plus rapide que append)
    result = [[[0.0] * DIM for _ in prange(DIM)] for _ in prange(BATCH_SIZE)]

    for n in prange(BATCH_SIZE):
        for i in prange(DIM):
            for j in prange(DIM):
                total = 0.0
                for k in prange(DIM):
                    # Accès liste standard [n][i][k]
                    total += A[n][i][k] * B[n][k][j]
                result[n][i][j] = total
    return result

# B. Version Python Basique sur Numpy (Boucles For)
# C'est souvent PIRE que les listes pures car l'accès A[n, i, k] a un overhead
# de vérification de type à chaque appel dans la boucle.
def python_numpy_loops_matmul(A, B):
    result = np.zeros((BATCH_SIZE, DIM, DIM))

    for n in range(BATCH_SIZE):
        for i in range(DIM):
            for j in range(DIM):
                total = 0.0
                for k in range(DIM):
                    total += A[n, i, k] * B[n, k, j]
                result[n, i, j] = total
    return result

# C. Version Numba (JIT)
# Compile les boucles en code machine. Extrêmement efficace pour les petites matrices (4x4).
@njit(fastmath=True)
def numba_loops_matmul(A, B):
    # Note: Numba a besoin des arguments en entrée pour compiler,
    # on ne peut pas hardcoder les globales à l'intérieur facilement sans 'globals'
    N = A.shape[0]
    D = A.shape[1]
    result = np.zeros((N, D, D))

    for n in range(N):
        for i in range(D):
            for j in range(D):
                total = 0.0
                for k in range(D):
                    total += A[n, i, k] * B[n, k, j]
                result[n, i, j] = total
    return result

# D. Version Numpy Vectorisée
# Utilise BLAS via l'opérateur @
def numpy_vectorized_matmul(A, B):
    return A @ B

# E. Version Numba appelant Numpy
@njit
def numba_numpy_matmul(A, B):
    # On récupère la taille du batch (première dimension)
    batch_size = A.shape[0]
    # On prépare le résultat avec les mêmes dimensions que A
    result = np.empty_like(A)

    # On boucle explicitement sur le batch
    # Numba va compiler cette boucle très efficacement
    for i in range(batch_size):
        # Ici, A[i] et B[i] sont des matrices 2D, donc l'opérateur @ fonctionne parfaitement
        result[i] = A[i] @ B[i]

    return result

# -----------------------------------------------------------------------------
# 3. Tests Benchmark
# -----------------------------------------------------------------------------

def test_pure_python_list(benchmark):
    benchmark(python_list_matmul, GLOBAL_A_LIST, GLOBAL_B_LIST)

def test_pure_python_list_numba(benchmark):
    # Warmup pour compiler
    python_list_matmul_numba(GLOBAL_A_LIST, GLOBAL_B_LIST)
    benchmark(python_list_matmul_numba, GLOBAL_A_LIST, GLOBAL_B_LIST)

def test_python_numpy_loops(benchmark):
    benchmark(python_numpy_loops_matmul, GLOBAL_A_NP, GLOBAL_B_NP)

def test_numba_loops(benchmark):
    # Warmup pour compiler
    numba_loops_matmul(GLOBAL_A_NP, GLOBAL_B_NP)
    benchmark(numba_loops_matmul, GLOBAL_A_NP, GLOBAL_B_NP)

def test_numpy_vectorized(benchmark):
    benchmark(numpy_vectorized_matmul, GLOBAL_A_NP, GLOBAL_B_NP)

def test_numba_numpy(benchmark):
    # Warmup pour compiler
    numba_numpy_matmul(GLOBAL_A_NP, GLOBAL_B_NP)
    benchmark(numba_numpy_matmul, GLOBAL_A_NP, GLOBAL_B_NP)