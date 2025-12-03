import numpy as np
from numba import njit, prange, float64, float32


@njit(float32[:,:](float32[:,:], float32[:,:], float32),parallel=False, fastmath=False)
def lerp_vec3_jit(v1, v2, t):
    # v1.shape est (N, 3)
    n = v1.shape[0]
    m = v1.shape[1] # Devrait être 3

    # Allocation unique du résultat
    res = np.empty((n, m), dtype=v1.dtype)

    t_inv = 1.0 - t

    # Boucle parallèle sur les vecteurs (lignes)
    for i in prange(n):
        res[i, 0] = t_inv * v1[i, 0] + t * v2[i, 2]
        res[i, 1] = t_inv * v1[i, 1] + t * v2[i, 2]
        res[i, 2] = t_inv * v1[i, 2] + t * v2[i, 2]

    return res

@njit(float32[:,:](float32[:,:], float32[:,:], float32),fastmath=False, parallel=False)
def nlerp_quat_jit(q0, q1, t):
    """
    Interpolation NLERP optimisée pour Numba (CPU multi-cœur).
    q0, q1 : tableaux (N, 4)
    t      : float (scalaire)
    """

    nb_bone = q0.shape[0]
    result = np.empty((nb_bone, 4), dtype=q0.dtype)

    # Numba parallélise cette boucle automatiquement
    for i in prange(nb_bone):
        # 1. Calcul du produit scalaire (manuellement)
        # On déroule la boucle sur les 4 composantes pour la vitesse
        dot = (q0[i, 0] * q1[i, 0] +
               q0[i, 1] * q1[i, 1] +
               q0[i, 2] * q1[i, 2] +
               q0[i, 3] * q1[i, 3])

        # 2. Gestion du Shortest Path
        # Si dot < 0, on inverse le signe de t pour q1
        t_val = t
        if dot < 0.0:
            t_val = -t

        # 3. Interpolation Linéaire
        # result = q0 * (1-t) + q1 * t_sign
        f0 = 1.0 - t

        # On calcule les composantes interpolées temporaires
        rx = q0[i, 0] * f0 + q1[i, 0] * t_val
        ry = q0[i, 1] * f0 + q1[i, 1] * t_val
        rz = q0[i, 2] * f0 + q1[i, 2] * t_val
        rw = q0[i, 3] * f0 + q1[i, 3] * t_val

        # 4. Normalisation (manuelle)
        sq_norm = rx*rx + ry*ry + rz*rz + rw*rw

        # Inverse sqrt est souvent plus rapide qu'une division
        # Ajout d'epsilon pour sécurité
        inv_len = 1.0 / np.sqrt(sq_norm + 1e-8)

        result[i, 0] = rx * inv_len
        result[i, 1] = ry * inv_len
        result[i, 2] = rz * inv_len
        result[i, 3] = rw * inv_len

    return result