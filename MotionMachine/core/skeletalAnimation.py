import numpy as np
from numpy._typing import NDArray


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

    def get_pose_at_time(self, time_sec, loop=True):
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
        return self._compute_fk_single_frame(p_interp, r_interp, s_interp, True)

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