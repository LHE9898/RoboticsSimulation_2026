# Copy a robomimic hdf5 and add obs/gravity_ee = R_E^T (0, 0, -1) (world down in EE axes), the same function as the
# env ObsTerm (place/mdp/grasp_feat.py: gravity_ee_from_quat). CPU only.
#
#   python tools_c/add_gravity_ee.py --input datasets/mimic_500_evt_ee.hdf5 --output datasets/mimic_500_evt_ee_rel.hdf5
import argparse
import importlib.util
import shutil

import h5py
import numpy as np
import torch

spec = importlib.util.spec_from_file_location(
    "grasp_feat", "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/mdp/grasp_feat.py")
gf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gf)

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--output", required=True)
A = ap.parse_args()

shutil.copyfile(A.input, A.output)
with h5py.File(A.output, "r+") as f:
    for k in f["data"]:
        o = f["data"][k]["obs"]
        g = gf.gravity_ee_from_quat(torch.tensor(o["eef_quat"][()], dtype=torch.float64)).numpy().astype(np.float32)
        if "gravity_ee" in o:
            del o["gravity_ee"]
        o.create_dataset("gravity_ee", data=g)
    n = len(f["data"])
    g0 = f["data"]["demo_0"]["obs"]["gravity_ee"][()]
print(f"{n} demos -> {A.output} | gravity_ee demo_0 t0 {np.round(g0[0], 3)} |norm| {np.linalg.norm(g0, axis=1).min():.4f}-{np.linalg.norm(g0, axis=1).max():.4f}")
