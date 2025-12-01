import numpy as np
import pytest
from numba import guvectorize, njit, prange, float64

# ---------------------------------------------------------
# 1. Version NumPy Pure (Référence)
# ---------------------------------------------------------
def lerp_vec3_numpy(v1, v2, t):
    # Création de tableaux temporaires en mémoire :
    # T1 = (1-t) * v1
    # T2 = t * v2
    # Res = T1 + T2
    return (1 - t) * v1 + t * v2

# ---------------------------------------------------------
# 2. Version @guvectorize (Flexible & Broadcasting)
# ---------------------------------------------------------
# Signature : vecteur(n), vecteur(n), scalaire -> sortie(n)
@guvectorize(['void(float64[:], float64[:], float64, float64[:])'], '(n),(n),()->(n)', target='cpu', nopython=True)
def lerp_vec3_gu(v1, v2, t, res):
    # Optimisation : On pré-calcule le facteur inversé
    t_inv = 1.0 - t

    # Pour un vec3, n vaut 3.
    # Le compilateur va probablement dérouler (unroll) cette boucle automatiquement.
    for i in range(v1.shape[0]):
        res[i] = t_inv * v1[i] + t * v2[i]

# ---------------------------------------------------------
# 3. Version @njit (Performance Maximale "Bare Metal")
# ---------------------------------------------------------
@njit(parallel=False, fastmath=True)
def lerp_vec3_jit(v1, v2, t):
    # v1.shape est (N, 3)
    n = v1.shape[0]
    m = v1.shape[1] # Devrait être 3

    # Allocation unique du résultat
    res = np.empty((n, m), dtype=np.float64)

    t_inv = 1.0 - t

    # Boucle parallèle sur les vecteurs (lignes)
    for i in prange(n):
        # Boucle sur les composantes (x, y, z)
        # Comme m=3, c'est minuscule, le CPU va vectoriser ça (SIMD)
        for k in range(m):
            res[i, k] = t_inv * v1[i, k] + t * v2[i, k]

    return res

@pytest.fixture(scope="module")
def vec3_data():
    # N = 1_000_000 # 1 million de vecteurs
    # N = 100_000 # 1 million de vecteurs
    # N = 10_000 # 1 million de vecteurs
    # N = 1_000 # 1 million de vecteurs
    N = 100 # 1 million de vecteurs
    # N = 32 # 1 million de vecteurs
    v1 = np.random.rand(N, 3)
    v2 = np.random.rand(N, 3)
    t = 0.5
    # Pré-allocation pour guvectorize
    out = np.empty_like(v1)
    return v1, v2, t, out

def test_correctness(vec3_data):
    v1, v2, t, out = vec3_data

    expected = lerp_vec3_numpy(v1, v2, t)
    jit_res = lerp_vec3_jit(v1, v2, t)
    lerp_vec3_gu(v1, v2, t, out)

    np.testing.assert_allclose(jit_res, expected, atol=1e-8)
    np.testing.assert_allclose(out, expected, atol=1e-8)

# --- Benchmarks ---

def test_bench_numpy(benchmark, vec3_data):
    v1, v2, t, _ = vec3_data
    benchmark(lerp_vec3_numpy, v1, v2, t)

def test_bench_guvectorize(benchmark, vec3_data):
    v1, v2, t, out = vec3_data
    # Warmup
    lerp_vec3_gu(v1[:10], v2[:10], t, out[:10])
    benchmark(lerp_vec3_gu, v1, v2, t, out)

def test_bench_jit(benchmark, vec3_data):
    v1, v2, t, _ = vec3_data
    # Warmup
    lerp_vec3_jit(v1[:10], v2[:10], t)
    benchmark(lerp_vec3_jit, v1, v2, t)