# Add grasp_err, grasp_err_ee and place_err_ee to obs/ and next_obs/ of every demo of a dataset.
#
#   ./isaaclab.sh -p tools_c/add_grasp_err_to_hdf5.py --input datasets/mimic_1000.hdf5 \
#       --output datasets/mimic_1000_ge.hdf5 --frame base [--max_demos 1000]
#
# offsets / symmetric / frames must be identical to the ObsTerms in place_toy2box_rmp_rel_env_cfg.py:
#   grasp_err (frame = --frame), grasp_err_ee (frame "ee"), place_err_ee (frame "ee").
import argparse
import importlib.util
import os

import h5py
import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEAT = os.path.join(REPO, "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/mdp/grasp_feat.py")
# load grasp_feat.py standalone (importing the package would require Isaac Sim)
spec = importlib.util.spec_from_file_location("grasp_feat", FEAT)
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
ap.add_argument("--offset", default=gf.DEFAULT_OFFSET_PATH)
ap.add_argument("--place_offset", default=gf.DEFAULT_PLACE_OFFSET_PATH)
ap.add_argument("--frame", default="base", choices=["base", "ee"])
ap.add_argument("--no_symmetric", action="store_true")
ap.add_argument("--max_demos", type=int, default=None, help="keep only the first N demos (by index)")
A = ap.parse_args()
assert os.path.abspath(A.input) != os.path.abspath(A.output), "output must differ from input"
off, poff = os.path.abspath(A.offset), os.path.abspath(A.place_offset)

with h5py.File(A.input, "r") as fi, h5py.File(A.output, "w") as fo:
    for k, v in fi.attrs.items():
        fo.attrs[k] = v
    di = fi["data"]
    do = fo.create_group("data")
    for k, v in di.attrs.items():
        do.attrs[k] = v
    demos = sorted(di, key=lambda s: int(s.split("_")[1]))
    if A.max_demos is not None:
        demos = demos[: A.max_demos]
    total = 0
    for k in demos:
        fi.copy(di[k], do, name=k)
        total += int(do[k]["actions"].shape[0])
        for grp in ("obs", "next_obs"):
            if grp not in do[k]:
                continue
            g = do[k][grp]
            t = lambda n: torch.as_tensor(g[n][:], dtype=torch.float64)  # noqa: E731
            pose = (t("eef_pos"), t("eef_quat"), t("toy_truck_positions"), t("toy_truck_orientations"))
            feats = {
                "grasp_err": gf.grasp_err_from_poses(*pose, off, not A.no_symmetric, A.frame),
                "grasp_err_ee": gf.grasp_err_from_poses(*pose, off, not A.no_symmetric, "ee"),
                "place_err_ee": gf.place_err_from_poses(t("eef_pos"), t("eef_quat"), t("box_positions"), poff, "ee"),
            }
            for name, v in feats.items():
                if name in g:
                    del g[name]
                g.create_dataset(name, data=v.numpy().astype(np.float32))
    do.attrs["total"] = total
    # filter keys (masks) are not used by the training config; drop ones that refer to removed demos
    if "mask" in fi:
        keep = set(demos)
        for mk, mv in fi["mask"].items():
            names = [n for n in mv[:] if n.decode() in keep]
            fo.create_dataset(f"mask/{mk}", data=np.array(names, dtype="S"))

print(f"wrote {len(demos)} demos ({total} steps) with grasp_err, grasp_err_ee, place_err_ee -> {A.output}")
