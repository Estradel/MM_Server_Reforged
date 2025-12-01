from pathlib import Path

from MotionMachine.core import FastBVH

bvh_path = Path(__file__).parent / "test.bvh"

if not bvh_path.exists():
    raise FileNotFoundError(
        f"Fichier test.bvh introuvable: {bvh_path}\n"
        "Placez un fichier BVH de test dans le dossier benchmarks/"
    )

anim = FastBVH(str(bvh_path))
anim.get_pose_at_time_numba(0.0, loop=True, local=True)