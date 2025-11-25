from typing import Any

import numpy as np
import bvhio
from SpatialTransform.lib.transform import Transform
from bvhio.lib.bvh import BvhContainer
from bvhio.lib.hierarchy import Joint
from numpy import ndarray, dtype

from MotionMachine.core import SkeletonAnimation


class FastBVH(SkeletonAnimation):

    def __init__(self, filename):
        super().__init__()
        self._load_with_bvhio(filename)

    def _load_with_bvhio(self, filename):
        print(f"Chargement BVH: {filename}...")
        try:
            bvh : BvhContainer = bvhio.readAsBvh(filename)
            root : Joint = bvhio.readAsHierarchy(filename)
        except Exception as e:
            print(f"Erreur chargement BVH: {e}")
            return

        # Aplatissement BFS
        flat_joints : list[Joint] = []
        queue = [root]
        while queue:
            joint = queue.pop(0)
            flat_joints.append(joint)
            queue.extend(joint.Children)

        self.bone_names = [j.Name for j in flat_joints]
        joint_to_idx = {j: i for i, j in enumerate(flat_joints)}

        bone_parents = []
        for j in flat_joints:
            bone_parents.append(-1 if j.Parent is None else joint_to_idx[j.Parent])
        self.bone_parents = np.array(bone_parents, dtype=np.int32)

        self.frame_times = bvh.FrameTime

        num_frames = bvh.FrameCount
        num_bones = len(flat_joints) # len(root.layout())

        # Initialisation des tableaux de Bind Pose
        self.rest_positions = np.zeros((num_bones, 3), dtype=np.float32)
        self.rest_rotations = np.zeros((num_bones, 4), dtype=np.float32)
        self.rest_scales = np.ones((num_bones, 3), dtype=np.float32)

        for i, joint in enumerate(flat_joints):
            # BVHio nous donne la RestPose (Offset par rapport au parent)
            rest_pos = joint.RestPose.Position
            rest_rot = joint.RestPose.Rotation

            self.rest_positions[i] = [rest_pos.x, rest_pos.y, rest_pos.z]
            self.rest_rotations[i] = [rest_rot.x, rest_rot.y, rest_rot.z, rest_rot.w]
            # BVH n'a généralement pas de scale, on laisse à 1.0

        self.local_positions = np.zeros((num_frames, num_bones, 3), dtype=np.float32)
        self.local_rotations = np.zeros((num_frames, num_bones, 4), dtype=np.float32)

        for i, joint in enumerate(flat_joints):
            if not joint.Keyframes: continue
            # On suppose que tous les joints ont le même nombre de frames
            max_f = min(num_frames, len(joint.Keyframes))

            for f in range(max_f):
                kf: Transform = joint.Keyframes[f][1]
                p, r = joint.RestPose.Position + kf.Position, joint.RestPose.Rotation * kf.Rotation
                self.local_positions[f, i] = [p.x, p.y, p.z]
                # Quaternion (x, y, z, w)
                self.local_rotations[f, i] = [r.x, r.y, r.z, r.w]

        print(f"BVH prêt: {num_frames} frames.")