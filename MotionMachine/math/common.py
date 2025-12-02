import numpy as np
from numba import njit, prange, float64, float32


@njit([float64[:,:](float64[:,:], float64[:,:], float64),
       float32[:,:](float32[:,:], float32[:,:], float32)],parallel=False, fastmath=False)
def lerp_jit(v1, v2, t):
    # v1.shape est (N, 3)
    n = v1.shape[0]
    m = v1.shape[1] # Devrait être 3

    # Allocation unique du résultat
    res = np.empty((n, m), dtype=v1.dtype)

    t_inv = 1.0 - t

    # Boucle parallèle sur les vecteurs (lignes)
    for i in prange(n):
        # Boucle sur les composantes (x, y, z)
        # Comme m=3, c'est minuscule, le CPU va vectoriser ça (SIMD)
        for k in range(m):
            res[i, k] = t_inv * v1[i, k] + t * v2[i, k]

    return res

@njit([float64[:,:](float64[:,:], float64[:,:], float64),
       float32[:,:](float32[:,:], float32[:,:], float32)],parallel=False, fastmath=False)
def nlerp_jit(q1, q2, t):
    n = q1.shape[0]
    # On alloue le tableau de résultat une seule fois
    res = np.empty((n, 4), dtype=q1.dtype)

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