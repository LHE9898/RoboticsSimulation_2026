# Joint-limit and body-clearance margins of the right arm in the demos, G vs G' grasps. CPU only.
#
# joint_pos in the dataset is joint_pos_rel (minus the default pose in isaaclab_assets/robots/agibot.py). Right arm joints are
# indices 5, 7, ..., 17 (checked: URDF forward kinematics of right_gripper_center reproduces eef_pos to 0.1 mm after one
# fixed transform). Limits from the RMPFlow URDF (agibot.urdf), RMPFlow joint_limit_buffers 0.01 rad.
# Body clearance: RMPFlow keeps Link5_r..Link7_r (r = 3 cm) outside a cylinder of r = 45 cm around the base_link z axis.
#
#   python tools_c/diag_joint_margin.py --urdf <agibot.urdf> --demo datasets/mimic_500_evt.hdf5
import argparse
import glob
import json
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

ap = argparse.ArgumentParser()
ap.add_argument("--urdf", required=True)
ap.add_argument("--demo", default="datasets/mimic_500_evt.hdf5")
A = ap.parse_args()

IDX = [5, 7, 9, 11, 13, 15, 17]
D0 = np.array([1.0817, -0.5907, -0.3442, 1.2819, -0.6928, -0.7, 0.0])  # agibot.py init_state right_arm_joint1..7
root = ET.parse(A.urdf).getroot()
J = {j.find("child").get("link"): j for j in root.findall("joint")}
LIM = np.array([[float(J[f"Link{i}_r"].find("limit").get(k)) for k in ("lower", "upper")] for i in range(1, 8)])


def tf(j, q=0.0):
    o = j.find("origin")
    T = np.eye(4)
    if o is not None:
        T[:3, :3] = Rot.from_euler("xyz", [float(x) for x in o.get("rpy", "0 0 0").split()]).as_matrix()
        T[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
    if j.get("type") == "revolute":
        M = np.eye(4)
        M[:3, :3] = Rot.from_rotvec(np.array([float(x) for x in j.find("axis").get("xyz").split()]) * q).as_matrix()
        T = T @ M
    return T


chain, link = [], "right_gripper_center"
while link in J and link != "base_link_r":
    chain.append(J[link])
    link = J[link].find("parent").get("link")
chain = chain[::-1]


def fk_all(q7):
    """Poses (in base_link_r) of Link5_r, Link6_r, Link7_r and right_gripper_center."""
    T, k, out = np.eye(4), 0, {}
    for j in chain:
        if j.get("type") == "revolute":
            T = T @ tf(j, q7[k])
            k += 1
        else:
            T = T @ tf(j)
        out[j.find("child").get("link")] = T.copy()
    return out


# fixed transform base_link_r -> robot base frame (frame of eef_pos), fitted once
with h5py.File(A.demo, "r") as f:
    keys = list(f["data"])
    Q, P = [], []
    for k in keys[:20]:
        o = f["data"][k]["obs"]
        Q.append(o["joint_pos"][()][::10, IDX] + D0)
        P.append(o["eef_pos"][()][::10])
Q, P = np.concatenate(Q), np.concatenate(P)
X = np.array([fk_all(q)["right_gripper_center"][:3, 3] for q in Q])
mx, mp = X.mean(0), P.mean(0)
U, _, Vt = np.linalg.svd((X - mx).T @ (P - mp))
Dd = np.eye(3)
Dd[2, 2] = np.sign(np.linalg.det(Vt.T @ U.T))
RB = Vt.T @ Dd @ U.T
TB = mp - RB @ mx
print(f"FK check: |eef_pos - FK| max {np.abs(X @ RB.T + TB - P).max() * 1000:.2f} mm")

c = json.load(open(glob.glob("source/**/grasp_offset.json", recursive=True)[0]))
RBAR = np.array(c["R_bar"])
F = -np.eye(3)
F[2, 2] = 1


def q2m(q):
    return Rot.from_quat(np.asarray(q)[:, [1, 2, 3, 0]]).as_matrix()


def yaw(q):
    return np.degrees(Rot.from_quat(np.asarray(q)[[1, 2, 3, 0]]).as_euler("xyz")[2])


rows = []
with h5py.File(A.demo, "r") as f:
    for k in keys:
        o = f["data"][k]["obs"]
        tp, tq, eq = o["toy_truck_positions"][()], o["toy_truck_orientations"][()], o["eef_quat"][()]
        z = tp[:, 2] - tp[0, 2]
        lift = np.where(z > 0.02)[0]
        if not len(lift):
            continue
        tl = int(lift[0])
        tg = max(0, tl - 5)
        Rt = q2m(tq[tg:tg + 1])[0] @ RBAR
        Ree = q2m(eq[tg:tg + 1])[0]
        gp = Rot.from_matrix((Rt @ F).T @ Ree).magnitude() < Rot.from_matrix(Rt.T @ Ree).magnitude()
        q = o["joint_pos"][()][: tl + 1, IDX] + D0
        m = np.minimum(q - LIM[:, 0], LIM[:, 1] - q)  # rad, per joint
        clr = []
        for t in range(0, tl + 1, 5):
            Ts = fk_all(q[t])
            d = [np.linalg.norm((RB @ Ts[n][:3, 3] + TB)[:2]) - 0.45 - 0.03 for n in ("Link5_r", "Link6_r", "Link7_r")]
            clr.append(min(d))
        rows.append({"gp": gp, "yaw": yaw(tq[0]), "dist": float(np.linalg.norm(tp[0, :2])), "m_min": m.min(0), "m_grasp": m[tg], "q_grasp": q[tg],
                     "clr_min": min(clr), "q_range": q.max(0) - q.min(0), "T": tl})

print(f"demos with a lift: {len(rows)}\n")
print("joint limits [deg]:", " ".join(f"j{i + 1} ±{np.degrees(LIM[i, 1]):.0f}" for i in range(7)))
for lab, sel in (("G", [r for r in rows if not r["gp"]]), ("G'", [r for r in rows if r["gp"]])):
    mm = np.degrees(np.array([r["m_min"] for r in sel]))
    mg = np.degrees(np.array([r["m_grasp"] for r in sel]))
    qg = np.degrees(np.array([r["q_grasp"] for r in sel]))
    rg = np.degrees(np.array([r["q_range"] for r in sel]))
    cl = np.array([r["clr_min"] for r in sel]) * 100
    print(f"[{lab}] n={len(sel)}  접근~잡기 길이 median {np.median([r['T'] for r in sel]):.0f} step")
    print("   접근 중 최소 여유 median [deg]: " + " ".join(f"j{i + 1} {np.median(mm[:, i]):5.0f}" for i in range(7)))
    print("   여유 < 10° 인 데모 비율:       " + " ".join(f"j{i + 1} {np.mean(mm[:, i] < 10):5.2f}" for i in range(7)))
    print("   잡을 때 관절각 median [deg]:   " + " ".join(f"j{i + 1} {np.median(qg[:, i]):5.0f}" for i in range(7)))
    print("   접근 중 관절 이동 범위 median: " + " ".join(f"j{i + 1} {np.median(rg[:, i]):5.0f}" for i in range(7)))
    print(f"   몸통 실린더까지 최소 여유 median {np.median(cl):.1f} cm (< 2 cm 비율 {np.mean(cl < 2):.2f})")

print("\n## 셀별 (트럭 수평거리 65 cm 기준)")
for lab, gsel in (("G", False), ("G'", True)):
    for dl, cond in (("가까움", lambda r: r["dist"] < 0.65), ("멂", lambda r: r["dist"] >= 0.65)):
        sel = [r for r in rows if r["gp"] == gsel and cond(r)]
        mm = np.degrees(np.array([r["m_min"] for r in sel]))
        cl = np.array([r["clr_min"] for r in sel]) * 100
        print(f"[{lab}/{dl}] n={len(sel)} | 길이 {np.median([r['T'] for r in sel]):.0f} step | 최소 여유 median j2 {np.median(mm[:, 1]):.0f}° j4 {np.median(mm[:, 3]):.0f}° j6 {np.median(mm[:, 5]):.0f}°"
              f" | j4 < 10° 비율 {np.mean(mm[:, 3] < 10):.2f} | 어느 관절이든 < 10° 비율 {np.mean(mm.min(1) < 10):.2f} | 몸통 여유 {np.median(cl):.1f} cm (< 2 cm {np.mean(cl < 2):.2f})")
