import numpy as np
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as R, Slerp
from scipy.interpolate import interp1d

from MotionMachine.core import SkeletonAnimation


class FastGLTF(SkeletonAnimation):
    def __init__(self, filename, target_fps=30):
        super().__init__()
        self.frame_time = 1.0 / target_fps
        self._load_gltf(filename, target_fps)

    def _load_gltf(self, filename, target_fps):
        print(f"Chargement GLTF: {filename}...")
        gltf = GLTF2().load(filename)

        if not gltf.skins:
            raise ValueError("Ce fichier GLTF ne contient pas de 'Skin' (Squelette).")

        skin = gltf.skins[0]
        joint_indices = skin.joints
        num_bones = len(joint_indices)

        # --- Hiérarchie ---
        gltf_idx_to_internal = {g_idx: i for i, g_idx in enumerate(joint_indices)}
        self.bone_parents = np.full(num_bones, -1, dtype=np.int32)
        self.bone_names = []

        rest_positions = np.zeros((num_bones, 3), dtype=np.float32)
        rest_rotations = np.zeros((num_bones, 4), dtype=np.float32)
        rest_scales = np.ones((num_bones, 3), dtype=np.float32)

        for internal_i, gltf_node_idx in enumerate(joint_indices):
            node = gltf.nodes[gltf_node_idx]
            self.bone_names.append(node.name or f"Bone_{internal_i}")

            if node.translation: rest_positions[internal_i] = node.translation
            if node.rotation: rest_rotations[internal_i] = node.rotation
            else: rest_rotations[internal_i] = [0, 0, 0, 1]
            if node.scale: rest_scales[internal_i] = node.scale

            if node.children:
                for child_gltf_idx in node.children:
                    if child_gltf_idx in gltf_idx_to_internal:
                        child_internal = gltf_idx_to_internal[child_gltf_idx]
                        self.bone_parents[child_internal] = internal_i

        # --- Animations ---
        if not gltf.animations:
            num_frames = 1
            self.local_positions = rest_positions[np.newaxis, :, :]
            self.local_rotations = rest_rotations[np.newaxis, :, :]
            self.local_scales = rest_scales[np.newaxis, :, :]
            return

        anim = gltf.animations[0]

        max_time = 0.0
        for sampler in anim.samplers:
            input_accessor = gltf.accessors[sampler.input]
            if input_accessor.max:
                max_time = max(max_time, input_accessor.max[0])

        num_frames = int(max_time * target_fps) + 1
        target_times = np.linspace(0, max_time, num_frames)

        self.local_positions = np.repeat(rest_positions[np.newaxis, :, :], num_frames, axis=0)
        self.local_rotations = np.repeat(rest_rotations[np.newaxis, :, :], num_frames, axis=0)
        self.local_scales = np.repeat(rest_scales[np.newaxis, :, :], num_frames, axis=0)

        binary_blob = gltf.binary_blob()

        def extract_data(accessor_idx):
            acc = gltf.accessors[accessor_idx]
            bv = gltf.bufferViews[acc.bufferView]
            if bv.byteOffset is None: bv.byteOffset = 0
            start = bv.byteOffset + (acc.byteOffset or 0)
            length = bv.byteLength
            dtype = np.float32
            if acc.componentType == 5123: dtype = np.ushort

            # Note: Gestion simplifiée du buffer. Idéalement gérer uri externe.
            raw_bytes = binary_blob[start : start + length]
            data = np.frombuffer(raw_bytes, dtype=dtype)

            if acc.type == "VEC3": data = data.reshape(-1, 3)
            elif acc.type == "VEC4": data = data.reshape(-1, 4)
            return data

        print(f"Resampling GLTF: {max_time:.2f}s -> {num_frames} frames...")

        for channel in anim.channels:
            if channel.target.node not in gltf_idx_to_internal:
                continue

            bone_idx = gltf_idx_to_internal[channel.target.node]
            sampler = anim.samplers[channel.sampler]

            times = extract_data(sampler.input).flatten()
            values = extract_data(sampler.output)

            path = channel.target.path
            if path == "translation":
                f = interp1d(times, values, axis=0, kind='linear', fill_value="extrapolate")
                self.local_positions[:, bone_idx, :] = f(target_times)
            elif path == "scale":
                f = interp1d(times, values, axis=0, kind='linear', fill_value="extrapolate")
                self.local_scales[:, bone_idx, :] = f(target_times)
            elif path == "rotation":
                key_rots = R.from_quat(values)
                slerp = Slerp(times, key_rots)
                interp_rots = slerp(target_times)
                self.local_rotations[:, bone_idx, :] = interp_rots.as_quat()

        print("GLTF Prêt.")