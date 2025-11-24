import time
import numpy as np

from MotionMachine.core import SkeletonAnimation


class AnimationPlayer:
    def __init__(self, animation_data : SkeletonAnimation):
        """
        :param animation_data: Une instance de SkeletonAnimation (BVH ou GLTF).
        """
        self.anim : SkeletonAnimation = animation_data

        # PLUS DE BAKING ICI !
        # On garde juste la référence aux données brutes.
        print("Mode Temps Réel activé (Interpolation).")

        self.is_playing = False
        self.loop = True
        self.speed = 1.0

        self.current_time = 0.0
        self.duration = self.anim.duration

        self._last_update_time = time.time()

    def play(self):
        self.is_playing = True
        self._last_update_time = time.time()

    def stop(self):
        self.is_playing = False
        self.current_time = 0.0

    def pause(self):
        self.is_playing = False

    def update(self):
        """Met à jour le temps interne."""
        if not self.is_playing:
            return

        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        self.current_time += dt * self.speed

        # Gestion de la boucle
        # Note : self.duration peut être 0 si pas d'anim
        if self.duration > 0:
            if self.current_time >= self.duration:
                if self.loop:
                    self.current_time %= self.duration
                else:
                    self.current_time = self.duration
                    self.is_playing = False

    def get_current_pose_bytes(self) -> bytes:
        """
        Calcule la pose à la volée pour le temps courant.
        """
        # Appel à la nouvelle méthode temps réel
        # loop=True/False selon la config du player
        matrices = self.anim.get_pose_at_time(self.current_time, loop=self.loop)

        if matrices is None:
            return b""

        # NumPy to bytes (Zero Copy)
        return matrices.tobytes()

    def get_current_pose(self):
        """
        Calcule la pose à la volée pour le temps courant.
        """
        # Appel à la nouvelle méthode temps réel
        # loop=True/False selon la config du player
        matrices = self.anim.get_pose_at_time(self.current_time, loop=self.loop)

        if matrices is None:
            return b""

        # NumPy to bytes (Zero Copy)
        return matrices

    def get_bone_names(self) -> list[str]:
        return self.anim.bone_names