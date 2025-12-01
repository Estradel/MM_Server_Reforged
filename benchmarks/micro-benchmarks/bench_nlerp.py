import numpy as np
import pytest
import math
from numba import guvectorize, prange, njit, float64


# --- 1. L'implémentation Numba (Optimisée) ---
@guvectorize(['void(float64[:], float64[:], float64, float64[:])'], '(n),(n),()->(n)', target='cpu', nopython=True, fastmath=False)
def nlerp_numba_vectorize(q1, q2, t, res):
    dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3]

    sign = 1.0
    if dot < 0.0:
        sign = -1.0

    t1 = 1.0 - t

    res[0] = t1 * q1[0] * sign + t * q2[0]
    res[1] = t1 * q1[1] * sign + t * q2[1]
    res[2] = t1 * q1[2] * sign + t * q2[2]
    res[3] = t1 * q1[3] * sign + t * q2[3]

    norm = res[0] * res[0] + res[1] * res[1] + res[2] * res[2] + res[3] * res[3]
    inv_norm = 1.0 / math.sqrt(norm)

    res[0] *= inv_norm
    res[1] *= inv_norm
    res[2] *= inv_norm
    res[3] *= inv_norm

@njit(float64[:,:](float64[:,:], float64[:,:], float64),parallel=False, fastmath=False)
def nlerp_jit(q1, q2, t):
    n = q1.shape[0]
    # On alloue le tableau de résultat une seule fois
    res = np.empty((n, 4), dtype=np.float64)

    t_sub = 1.0 - t

    # Boucle parallèle sur tous les quaternions
    for i in prange(n):

        # --- A. Produit scalaire manuel ---
        dot = 0.0
        for k in range(4):
            dot += q1[i, k] * q2[i, k]

        # --- B. Choix du signe ---
        sign = 1.0
        if dot < 0.0:
            sign = -1.0

        # --- C. LERP + Calcul Norme (combinés) ---
        # On calcule la valeur ET on accumule le carré de la norme dans la même boucle
        norm_sq = 0.0
        for k in range(4):
            # Formule : ((1-t) * q1 * sign) + (t * q2)
            val = (t_sub * q1[i, k] * sign) + (t * q2[i, k])
            res[i, k] = val
            norm_sq += val * val

        # --- D. Normalisation ---
        # On évite la division par zéro avec une sécurité minimale
        if norm_sq > 0.0:
            inv_norm = 1.0 / np.sqrt(norm_sq)
            for k in range(4):
                res[i, k] *= inv_norm
        else:
            # Cas rare (vecteur nul), on remet l'identité ou 0
            for k in range(4):
                res[i, k] = 0.0

    return res

# --- 2. L'implémentation NumPy Pure (Référence de justesse) ---
def nlerp_numpy_ref(q1, q2, t):
    # Produit scalaire pour vérifier l'orientation
    dot = np.sum(q1 * q2, axis=1, keepdims=True)

    # Si le produit scalaire est négatif, on inverse q1 pour le chemin le plus court
    # Broadcasting de NumPy pour gérer le signe
    sign = np.where(dot < 0, -1.0, 1.0)
    q1_aligned = q1 * sign

    # Interpolation linéaire
    res = (1 - t) * q1_aligned + t * q2

    # Normalisation
    res /= np.linalg.norm(res, axis=1, keepdims=True)
    return res

# --- 3. Configuration des Données (Fixture) ---
@pytest.fixture(scope="module")
def data_sample():
    """Génère des données de test aléatoires."""
    # N = 100_000  # Taille de l'échantillon pour le test
    N = 31  # Taille de l'échantillon pour le test
    # Création de quaternions aléatoires
    q1 = np.random.rand(N, 4)
    q2 = np.random.rand(N, 4)

    # Normalisation pour que ce soient de vrais quaternions
    q1 /= np.linalg.norm(q1, axis=1, keepdims=True)
    q2 /= np.linalg.norm(q2, axis=1, keepdims=True)

    # Tableau de sortie pré-alloué pour Numba
    out = np.empty_like(q1)

    return q1, q2, 0.5, out

# --- 4. Test Unitaire : Justesse (Correctness) ---
# def test_nlerp_correctness(data_sample):
#     q1, q2, t, out_numba = data_sample
#
#     # Calcul via NumPy (Référence)
#     expected = nlerp_numpy_ref(q1, q2, t)
#
#     # Calcul via Numba
#     # Note: Numba écrit le résultat dans 'out_numba' in-place
#     nlerp_numba(q1, q2, t, out_numba)
#
#     # Vérification : Les résultats doivent être quasi identiques
#     # atol=1e-8 est une tolérance raisonnable pour des float64
#     np.testing.assert_allclose(out_numba, expected, atol=1e-8, err_msg="Le résultat Numba diffère de NumPy")

# --- 5. Benchmark : Performance ---
def test_nlerp_jit_benchmark(benchmark, data_sample):
    q1, q2, t, out_numba = data_sample


    nlerp_jit(q1[:10], q2[:10], t)

    benchmark(nlerp_jit, q1, q2, t)

def test_nlerp_numpy_benchmark(benchmark, data_sample):
    q1, q2, t, out_numba = data_sample

    benchmark(nlerp_numpy_ref, q1, q2, t)

def test_nlerp_numba_vectorized_benchmark(benchmark, data_sample):
    q1, q2, t, out_numba = data_sample

    # Important : On appelle la fonction une fois avant le benchmark
    # pour forcer la compilation JIT. Sinon, le benchmark mesurera
    # le temps de compilation + exécution.
    nlerp_numba_vectorize(q1[:10], q2[:10], t, out_numba[:10])

    # Lancement du benchmark
    # On passe la fonction et ses arguments
    benchmark(nlerp_numba_vectorize, q1, q2, t, out_numba)