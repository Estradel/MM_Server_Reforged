"""
Benchmark de la fonction get_pose_at_time() de SkeletonAnimation.

Ce benchmark teste les performances de la fonction qui:
1. Calcule les indices de frames et le facteur d'interpolation
2. Interpole les positions (Lerp)
3. Interpole les rotations (NLerp avec quaternions)
4. Interpole les scales (si présent)
5. Calcule le Forward Kinematics (FK) pour obtenir les matrices globales

C'est LA fonction critique pour l'animation temps réel.

Pour exécuter: pytest benchmarks/benchmark_get_pose_at_time.py --benchmark-only
Pour comparer: pytest benchmarks/benchmark_get_pose_at_time.py --benchmark-compare
"""

import pytest
import sys
from pathlib import Path

# Ajouter le chemin du projet pour les imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from MotionMachine.core.fastBVH import FastBVH


@pytest.fixture(scope="module")
def animation():
    """Fixture pytest: charge l'animation une seule fois pour tous les tests."""
    bvh_path = Path(__file__).parent / "test.bvh"

    if not bvh_path.exists():
        raise FileNotFoundError(
            f"Fichier test.bvh introuvable: {bvh_path}\n"
            "Placez un fichier BVH de test dans le dossier benchmarks/"
        )

    anim = FastBVH(str(bvh_path))
    return anim


def test_benchmark_pose_at_start_local(benchmark, animation):
    """Benchmark: Pose au début de l'animation (frame 0)"""
    benchmark(animation.get_pose_at_time, 0.0, loop=True, local=True)

def test_benchmark_pose_at_start(benchmark, animation):
    """Benchmark: Pose au début de l'animation (frame 0)"""
    benchmark(animation.get_pose_at_time, 0.0, loop=True, local=False)

def test_benchmark_pose_at_start_local_numba(benchmark, animation):
    """Benchmark: Pose au début de l'animation (frame 0)"""
    benchmark(animation.get_pose_at_time_numba, 0.0, loop=True, local=True)

def test_benchmark_pose_at_start_numba(benchmark, animation):
    """Benchmark: Pose au début de l'animation (frame 0)"""
    benchmark(animation.get_pose_at_time_numba, 0.0, loop=True, local=False)


def test_benchmark_pose_at_middle(benchmark, animation):
    """Benchmark: Pose au milieu de l'animation"""
    middle_time = animation.duration / 2.0
    benchmark(animation.get_pose_at_time, middle_time, loop=True)


def test_benchmark_pose_interpolated(benchmark, animation):
    """Benchmark: Pose interpolée entre 2 frames (pire cas)"""
    # Temps pile entre 2 frames (ex: frame 10.5)
    time = (10.5 * animation.frame_time) if animation.num_frames > 11 else (animation.frame_time * 0.5)
    benchmark(animation.get_pose_at_time, time, loop=True)


def test_benchmark_pose_looping(benchmark, animation):
    """Benchmark: Pose avec boucle (teste le modulo)"""
    # Temps après la fin de l'animation (force une boucle)
    time = animation.duration + animation.frame_time * 5.5
    benchmark(animation.get_pose_at_time, time, loop=True)


def test_benchmark_pose_no_loop(benchmark, animation):
    """Benchmark: Pose sans boucle (clamp à la fin)"""
    time = animation.duration * 0.75
    benchmark(animation.get_pose_at_time, time, loop=False)


def test_benchmark_pose_rapid_succession(benchmark, animation):
    """Benchmark: Plusieurs poses successives (simule lecture temps réel)"""
    # Simule 3 appels consécutifs comme dans un player
    dt = animation.frame_time

    def rapid_succession():
        animation.get_pose_at_time(0.0, loop=True)
        animation.get_pose_at_time(dt, loop=True)
        animation.get_pose_at_time(dt * 2, loop=True)

    benchmark(rapid_succession)


