from pickle import FALSE

import numpy as np
from numba import njit, prange
from numpy._typing import NDArray

@njit(fastmath=False, parallel=False)
def fast_nlerp_vectorized(q0, q1, t):
    """
    Interpolation NLERP optimisée pour Numba (CPU multi-cœur).
    q0, q1 : tableaux (N, 4)
    t      : float (scalaire)
    """
    nb_bone = q0.shape[0]
    result = np.empty((nb_bone, 4), dtype=np.float64)

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

@njit(fastmath=False, parallel=False)
def compute_fk_single_frame(positions, rotations, scales, bone_parents, local=True):
    """
    Calcule les matrices globales pour UNE seule frame.
    """
    num_bones = positions.shape[0]

    X, Y, Z, W = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

    xx, yy, zz = X*X, Y*Y, Z*Z
    xy, xz, yz = X*Y, X*Z, Y*Z
    wx, wy, wz = W*X, W*Y, W*Z

    # Matrice Locale
    local_m = np.zeros((num_bones, 4, 4), dtype=np.float32)
    local_m[:, 3, 3] = 1.0

    # Rotation
    local_m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
    local_m[:, 0, 1] = 2.0 * (xy - wz)
    local_m[:, 0, 2] = 2.0 * (xz + wy)

    local_m[:, 1, 0] = 2.0 * (xy + wz)
    local_m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
    local_m[:, 1, 2] = 2.0 * (yz - wx)

    local_m[:, 2, 0] = 2.0 * (xz - wy)
    local_m[:, 2, 1] = 2.0 * (yz + wx)
    local_m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

    # Scale
    if scales is not None:
        local_m[:, :3, 0] *= scales[:, 0:1]
        local_m[:, :3, 1] *= scales[:, 1:2]
        local_m[:, :3, 2] *= scales[:, 2:3]

    # Translation
    local_m[:, :3, 3] = positions

    if local:
        return local_m

    # Propagation FK
    global_m = np.zeros((num_bones, 4, 4), dtype=np.float32)

    for i in range(num_bones):
        parent_idx = bone_parents[i]
        if parent_idx == -1:
            global_m[i] = local_m[i]
        else:
            global_m[i] = global_m[parent_idx] @ local_m[i]

    return global_m

@njit(parallel=False, fastmath=False)
def compute_fk_fast(positions, rotations, scales, bone_parents, local=True):
    """
    Calcule la FK complète (Local + Global) optimisée Numba.
    Alloue et retourne la matrice globale.
    """
    n = positions.shape[0]

    # 1. ALLOCATION INTERNE (Plus pratique pour l'utilisateur)
    # On crée les buffers temporaires ici.
    # On utilise float32 car c'est le standard graphique (2x plus rapide que float64).
    local_m = np.zeros((n, 4, 4), dtype=np.float32)
    global_m = np.zeros((n, 4, 4), dtype=np.float32)

    # Vérifier si scales est None en dehors de la boucle
    has_scale = scales is not None

    # --- PHASE 1 : Calcul des Matrices Locales (Parallélisé) ---
    # Cette partie est totalement indépendante pour chaque os -> Multithreading
    for i in prange(n):
        # Récupération des quaternions (x, y, z, w)
        qx, qy, qz, qw = rotations[i, 0], rotations[i, 1], rotations[i, 2], rotations[i, 3]

        # Récupération du scale (avec valeur par défaut si None)
        if has_scale:
            sx, sy, sz = scales[i, 0], scales[i, 1], scales[i, 2]
        else:
            sx, sy, sz = 1.0, 1.0, 1.0

        # Calculs intermédiaires quaternion
        xx, yy, zz = qx*qx, qy*qy, qz*qz
        xy, xz, yz = qx*qy, qx*qz, qy*qz
        wx, wy, wz = qw*qx, qw*qy, qw*qz

        # Construction de la matrice locale (Rotation * Scale fusionnés)
        # Ligne 0
        local_m[i, 0, 0] = (1.0 - 2.0 * (yy + zz)) * sx
        local_m[i, 0, 1] = (2.0 * (xy - wz)) * sy
        local_m[i, 0, 2] = (2.0 * (xz + wy)) * sz
        local_m[i, 0, 3] = positions[i, 0]

        # Ligne 1
        local_m[i, 1, 0] = (2.0 * (xy + wz)) * sx
        local_m[i, 1, 1] = (1.0 - 2.0 * (xx + zz)) * sy
        local_m[i, 1, 2] = (2.0 * (yz - wx)) * sz
        local_m[i, 1, 3] = positions[i, 1]

        # Ligne 2
        local_m[i, 2, 0] = (2.0 * (xz - wy)) * sx
        local_m[i, 2, 1] = (2.0 * (yz + wx)) * sy
        local_m[i, 2, 2] = (1.0 - 2.0 * (xx + yy)) * sz
        local_m[i, 2, 3] = positions[i, 2]

        # Ligne 3 (Homogène)
        local_m[i, 3, 3] = 1.0

    if local == True:
        return local_m

    # --- PHASE 2 : Propagation Globale (Séquentielle) ---
    # On ne peut pas paralléliser car l'os 2 dépend du résultat de l'os 1.
    # Mais Numba exécute cette boucle à la vitesse du C.
    for i in range(n):
        parent = bone_parents[i]

        if parent == -1:
            # Pas de parent (Root) : Global = Local
            global_m[i] = local_m[i]
        else:
            # Enfant : Global = Global_Parent @ Local_Enfant
            # Multiplication matricielle manuelle (unroll) pour max perf
            # sans appel de fonction externe.
            for r in range(4):
                for c in range(4):
                    acc = 0.0
                    for k in range(4):
                        # parent est déjà calculé car i > parent (généralement)
                        acc += global_m[parent, r, k] * local_m[i, k, c]
                    global_m[i, r, c] = acc

    return global_m

@njit(fastmath=False, parallel=False)
def get_pose_at_time_numba(local_positions, local_rotations, local_scales,
                           bone_parents, frame_time, time_sec, loop=True, local=True):
    """
    Version Numba de get_pose_at_time.
    """
    num_frames = local_positions.shape[0]

    frame_float = time_sec / frame_time

    if loop:
        frame_float = frame_float % num_frames
        idx0 = int(frame_float)
        idx1 = (idx0 + 1) % num_frames
    else:
        idx0 = min(int(frame_float), num_frames - 1)
        idx1 = min(idx0 + 1, num_frames - 1)

    t = frame_float - int(frame_float)
    if idx0 == idx1:
        t = 0.0

    # Interpolation
    p_interp = local_positions[idx0] * (1.0 - t) + local_positions[idx1] * t
    r_interp = fast_nlerp_vectorized(local_rotations[idx0], local_rotations[idx1], t)

    s_interp = None
    if local_scales is not None:
        s_interp = local_scales[idx0] * (1.0 - t) + local_scales[idx1] * t

    return compute_fk_fast(p_interp, r_interp, s_interp, bone_parents, local)
    # return compute_fk_single_frame(p_interp, r_interp, s_interp, bone_parents, local)

@njit(parallel=False, fastmath=False)
def interpolate_frames_numba(pos_frames, rot_frames, scl_frames, idx0, idx1, t):
    """
    Interpole positions, rotations et échelles entre deux frames.
    Retourne les tableaux interpolés (N, 3) et (N, 4).
    """
    n_bones = pos_frames.shape[1]

    # Allocation des résultats temporaires
    # float32 est vital pour la vitesse (SIMD AVX)
    p_out = np.empty((n_bones, 3), dtype=np.float32)
    r_out = np.empty((n_bones, 4), dtype=np.float32)
    s_out = np.empty((n_bones, 3), dtype=np.float32)

    # Vérifier si scl_frames est None
    has_scale = scl_frames is not None

    # --- BOUCLE PARALLÈLE ---
    for i in prange(n_bones):
        # 1. Position LERP
        # p = p0 * (1-t) + p1 * t
        p_out[i, 0] = pos_frames[idx0, i, 0] * (1-t) + pos_frames[idx1, i, 0] * t
        p_out[i, 1] = pos_frames[idx0, i, 1] * (1-t) + pos_frames[idx1, i, 1] * t
        p_out[i, 2] = pos_frames[idx0, i, 2] * (1-t) + pos_frames[idx1, i, 2] * t

        # 2. Rotation NLERP (Intégré pour performance)
        # Lecture des quaternions
        q0_x, q0_y, q0_z, q0_w = rot_frames[idx0, i]
        q1_x, q1_y, q1_z, q1_w = rot_frames[idx1, i]

        # Produit scalaire
        dot = q0_x*q1_x + q0_y*q1_y + q0_z*q1_z + q0_w*q1_w

        # Shortest Path
        sign = 1.0
        if dot < 0.0:
            sign = -1.0

        # Interpolation
        qx = q0_x * (1.0 - t) + q1_x * (t * sign)
        qy = q0_y * (1.0 - t) + q1_y * (t * sign)
        qz = q0_z * (1.0 - t) + q1_z * (t * sign)
        qw = q0_w * (1.0 - t) + q1_w * (t * sign)

        # Normalisation (Inverse Sqrt rapide)
        inv_norm = 1.0 / np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw + 1e-8)
        r_out[i, 0] = qx * inv_norm
        r_out[i, 1] = qy * inv_norm
        r_out[i, 2] = qz * inv_norm
        r_out[i, 3] = qw * inv_norm

        # 3. Scale LERP avec gestion de None
        if has_scale:
            s_out[i, 0] = scl_frames[idx0, i, 0] * (1-t) + scl_frames[idx1, i, 0] * t
            s_out[i, 1] = scl_frames[idx0, i, 1] * (1-t) + scl_frames[idx1, i, 1] * t
            s_out[i, 2] = scl_frames[idx0, i, 2] * (1-t) + scl_frames[idx1, i, 2] * t
        else:
            s_out[i, 0] = 1.0
            s_out[i, 1] = 1.0
            s_out[i, 2] = 1.0

    return p_out, r_out, s_out

@njit(parallel=False, fastmath=False)
def get_pose_at_time_fast(local_positions, local_rotations, local_scales,
                          bone_parents, frame_time, time_sec, loop=True):
    """
    Fonction principale (Orchestrateur).
    1. Calcule le temps.
    2. Interpole (Parallel).
    3. Calcule la FK (Parallel + Serial).
    """
    num_frames = local_positions.shape[0]

    # --- 1. Gestion du Temps ---
    # float division
    frame_float = time_sec / frame_time

    idx0 = 0
    idx1 = 0

    if loop:
        # Modulo optimisé pour float
        frame_float = frame_float % num_frames
        idx0 = int(frame_float)
        idx1 = (idx0 + 1) % num_frames
    else:
        # Clamp
        frame_idx = int(frame_float)
        if frame_idx >= num_frames - 1:
            idx0 = num_frames - 1
            idx1 = num_frames - 1
            frame_float = float(idx0) # t sera 0
        else:
            idx0 = frame_idx
            idx1 = idx0 + 1

    t = frame_float - int(frame_float)

    # Cas limite t très petit
    if idx0 == idx1:
        t = 0.0

    # --- 2. Interpolation (Le gros du travail) ---
    # Note : On force l'envoi de local_scales. S'il est None en Python,
    # il faut envoyer un tableau de 1.0 avant d'appeler cette fonction.
    p_now, r_now, s_now = interpolate_frames_numba(local_positions, local_rotations, local_scales, idx0, idx1, t)

    # --- 3. Calcul FK (Réutilisation de votre fonction précédente) ---
    return compute_fk_fast(p_now, r_now, s_now, bone_parents)

class SkeletonAnimation:
    """
    Classe de base contenant les données d'animation brutes et la logique mathématique.
    """
    def __init__(self):
        self.bone_names : list[str] = []
        self.bone_parents : NDArray[np.int32] = np.empty((0,), dtype=np.int32)  # np.array int32

        # --- Ajout : Stockage de la Bind Pose (Pose de repos) ---
        # Ces données définissent la forme du squelette sans animation
        self.rest_positions = None # (B, 3)
        self.rest_rotations = None # (B, 4) - Quaternions
        self.rest_scales = None    # (B, 3)

        # Données d'animation (Frames, Bones, ...)
        self.local_positions = None # (F, B, 3)
        self.local_rotations = None # (F, B, 4) - Quaternions (x, y, z, w)
        self.local_scales = None    # (F, B, 3) - Optionnel

        self.frame_time = 0.033

    @property
    def num_frames(self):
        if self.local_positions is not None:
            return self.local_positions.shape[0]
        return 0

    @property
    def duration(self):
        return max(0, self.num_frames - 1) * self.frame_time

    def get_skeleton_definition(self) -> dict:
        """
        Génère un dictionnaire contenant la structure statique du squelette.
        Idéal pour être envoyé en JSON au client lors de l'initialisation.
        """
        # Conversion des tableaux NumPy en listes pour la sérialisation JSON
        parents_list : list[int] = self.bone_parents.tolist()

        # Préparation de la bind pose (si disponible, sinon identité)
        num_bones = len(self.bone_names)

        # Valeurs par défaut si les loaders n'ont pas rempli les rest_xxx
        r_pos = self.rest_positions.tolist() if self.rest_positions is not None else [[0,0,0]] * num_bones
        r_rot = self.rest_rotations.tolist() if self.rest_rotations is not None else [[0,0,0,1]] * num_bones
        r_scl = self.rest_scales.tolist()    if self.rest_scales is not None    else [[1,1,1]] * num_bones

        return {
            "type": "SKELETON_DEF",
            "bone_names": self.bone_names,
            "parents": parents_list,
            "bind_pose": {
                "positions": r_pos,
                "rotations": r_rot,
                "scales": r_scl
            }
        }

    def get_pose_at_time_numba(self, time_sec, loop=True, local=True):
        """Wrapper Python vers la fonction Numba."""
        if self.local_positions is None:
            return None

        # return get_pose_at_time_numba(
        return get_pose_at_time_numba(
            self.local_positions,
            self.local_rotations,
            self.local_scales,
            self.bone_parents,
            self.frame_time,
            time_sec,
            loop        )

    def get_pose_at_time(self, time_sec, loop=True, local=True):
        """
        Calcule la pose globale interpolée à un instant t précis.
        Approche Temps Réel.
        """
        if self.local_positions is None:
            return None

        # 1. Calcul des indices de frames
        # On convertit le temps en "index flottant" (ex: frame 10.5)
        frame_float = time_sec / self.frame_time

        if loop:
            # Gestion boucle correcte : 10.5 avec durée 10 -> 0.5
            frame_float %= self.num_frames

            idx0 = int(frame_float)
            idx1 = (idx0 + 1) % self.num_frames
        else:
            # Clamp à la fin
            idx0 = int(frame_float)
            idx1 = min(idx0 + 1, self.num_frames - 1)
            idx0 = min(idx0, self.num_frames - 1)

        # Facteur d'interpolation (t entre 0.0 et 1.0)
        t = frame_float - int(frame_float)
        # Correction si on est exactement sur la dernière frame sans loop
        if idx0 == idx1: t = 0.0

        # 2. Interpolation des données locales
        # Position : Lerp (Linear Interpolation)
        p0 = self.local_positions[idx0]
        p1 = self.local_positions[idx1]
        p_interp = p0 * (1.0 - t) + p1 * t

        # Rotation : Nlerp (Normalized Linear Interpolation) - Rapide et stable
        r0 = self.local_rotations[idx0]
        r1 = self.local_rotations[idx1]
        r_interp = self._fast_nlerp_vectorized(r0, r1, t)

        # Scale : Lerp (si présent)
        s_interp = None
        if self.local_scales is not None:
            s0 = self.local_scales[idx0]
            s1 = self.local_scales[idx1]
            s_interp = s0 * (1.0 - t) + s1 * t

        # 3. Calcul du FK pour cette pose unique
        return self._compute_fk_single_frame(p_interp, r_interp, s_interp, local)

    def _fast_nlerp_vectorized(self, q0, q1, t):
        """
        Interpolation linéaire normalisée de quaternions (Bones, 4).
        Gère le "Shortest Path" (évite que la rotation fasse le tour complet).
        """
        # Produit scalaire pour vérifier l'orientation (Shortest Path)
        # sum(q0 * q1, axis=1)
        dot = np.sum(q0 * q1, axis=1)

        # Si le dot product est négatif, on inverse q1 pour prendre le chemin court
        # On utilise np.where pour le faire de manière vectorisée sans 'if'
        # q1_corrected = q1 * sign(dot) (approximatif)

        # Masque booléen pour les inversions
        mask = dot < 0.0
        q1_adj = q1.copy()
        q1_adj[mask] = -q1[mask]

        # Interpolation linéaire
        qt = q0 * (1.0 - t) + q1_adj * t

        # Normalisation (Essentiel pour que ce soit une rotation valide)
        # Norme Euclidienne
        norm = np.linalg.norm(qt, axis=1, keepdims=True)
        return qt / norm

    def _compute_fk_single_frame(self, positions, rotations, scales=None, local=False):
        """
        Calcule les matrices globales pour UNE seule frame.
        positions: (Bones, 3)
        rotations: (Bones, 4)
        scales: (Bones, 3) ou None
        """
        num_bones = positions.shape[0]

        # --- Quaternion à Matrice (Optimisé pour 1 frame) ---
        X, Y, Z, W = rotations[:, 0], rotations[:, 1], rotations[:, 2], rotations[:, 3]

        xx, yy, zz = X*X, Y*Y, Z*Z
        xy, xz, yz = X*Y, X*Z, Y*Z
        wx, wy, wz = W*X, W*Y, W*Z

        # Matrice Locale (Bones, 4, 4)
        local_m = np.zeros((num_bones, 4, 4), dtype=np.float32)
        local_m[:, 3, 3] = 1.0

        # Rotation
        local_m[:, 0, 0] = 1.0 - 2.0 * (yy + zz)
        local_m[:, 0, 1] = 2.0 * (xy - wz)
        local_m[:, 0, 2] = 2.0 * (xz + wy)

        local_m[:, 1, 0] = 2.0 * (xy + wz)
        local_m[:, 1, 1] = 1.0 - 2.0 * (xx + zz)
        local_m[:, 1, 2] = 2.0 * (yz - wx)

        local_m[:, 2, 0] = 2.0 * (xz - wy)
        local_m[:, 2, 1] = 2.0 * (yz + wx)
        local_m[:, 2, 2] = 1.0 - 2.0 * (xx + yy)

        # Scale
        if scales is not None:
            # Broadcasting (Bones, 1) * (Bones, 3) nécessite un reshape
            # On multiplie chaque colonne de la matrice de rot par le scale correspondant
            local_m[:, :3, 0] *= scales[:, 0:1]
            local_m[:, :3, 1] *= scales[:, 1:2]
            local_m[:, :3, 2] *= scales[:, 2:3]

        # Translation
        local_m[:, :3, 3] = positions

        if (local) :
            return local_m

        # --- Propagation FK ---
        # Note: On ne peut pas vectoriser la hiérarchie elle-même (dépendance de données),
        # mais la boucle en Python sur ~60 os est négligeable (< 0.1ms).
        global_m = np.zeros_like(local_m)

        # Utilisation directe des tableaux numpy pour vitesse max
        parents = self.bone_parents

        for i in range(num_bones):
            parent_idx = parents[i]
            if parent_idx == -1:
                global_m[i] = local_m[i]
            else:
                # Matrice Parent @ Matrice Enfant
                global_m[i] = global_m[parent_idx] @ local_m[i]

        # return local_m
        return global_m

    # Garde la version vectorisée complète (batch) pour le debug ou export
    def compute_fk_vectorized(self):
        # ... (Code existant inchangé pour compatibilité si besoin) ...
        return super().compute_fk_vectorized() if hasattr(super(), 'compute_fk_vectorized') else None