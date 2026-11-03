# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Relative errors between the end-effector and object-fixed targets.

``grasp_err`` (6D): symmetry-aware error to the grasp target frame on the toy truck.
``place_err`` (3D): position error to the place target above the box center.

Only ``json`` and ``torch`` are imported at module level so that this file can also be loaded standalone
(e.g. by ``tools_c/add_grasp_err_to_hdf5.py``) without launching Isaac Sim.

Notation (all poses expressed in the robot base frame B, quaternions in (w, x, y, z)):
    T: toy truck, E: end-effector, G: grasp target frame fixed to T (mean human grasp pose),
    G': G rotated by 180 deg about the approach axis, G*: whichever of G / G' is closer to E,
    P: place target = box center + h * z_B (mean human EE height above the box at release).
"""

import json
import os
from functools import lru_cache

import torch

DEFAULT_OFFSET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "agibot", "grasp_offset.json")
)
DEFAULT_PLACE_OFFSET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config", "agibot", "place_offset.json")
)


@lru_cache(maxsize=8)
def load_offset(path):
    with open(path) as f:
        return json.load(f)


def quat2mat(q):
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        -1,
    ).reshape(q.shape[:-1] + (3, 3))


def geo(A, B):
    """Geodesic angle between rotation matrices A and B."""
    tr = (A.transpose(-1, -2) @ B).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.acos(((tr - 1) / 2).clamp(-1, 1))


def mat2quat(R):
    """Rotation matrix -> unit quaternion (w, x, y, z), numerically stable for all angles."""
    m00, m11, m22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    # |w|, |x|, |y|, |z| candidates; the largest one is used as the pivot
    q_abs = torch.sqrt(
        torch.stack(
            [1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22], -1
        ).clamp(min=0)
    )
    r21, r12 = R[..., 2, 1], R[..., 1, 2]
    r02, r20 = R[..., 0, 2], R[..., 2, 0]
    r10, r01 = R[..., 1, 0], R[..., 0, 1]
    cand = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, r21 - r12, r02 - r20, r10 - r01], -1),
            torch.stack([r21 - r12, q_abs[..., 1] ** 2, r10 + r01, r02 + r20], -1),
            torch.stack([r02 - r20, r10 + r01, q_abs[..., 2] ** 2, r12 + r21], -1),
            torch.stack([r10 - r01, r20 + r02, r21 + r12, q_abs[..., 3] ** 2], -1),
        ],
        -2,
    ) / (2 * q_abs[..., None].clamp(min=0.1))
    idx = q_abs.argmax(-1)
    q = torch.gather(cand, -2, idx[..., None, None].expand(idx.shape + (1, 4))).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True)


def so3_log(R):
    """Rotation matrix -> rotation vector (axis * angle, angle in [0, pi])."""
    q = mat2quat(R)
    q = torch.where(q[..., :1] < 0, -q, q)  # w >= 0  ->  angle <= pi
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    th = 2 * torch.atan2(n, q[..., :1])
    # th / n -> 2 as n -> 0
    return torch.where(n > 1e-8, th / n.clamp(min=1e-8), torch.full_like(n, 2.0)) * v


def grasp_err_from_poses(ee_p, ee_q, o_p, o_q, offset_path=DEFAULT_OFFSET_PATH, symmetric=True, frame="base"):
    """Compute grasp_err = [e_p, e_R] (6D).

    Args:
        ee_p, ee_q: end-effector position / quaternion in base frame, shape (..., 3) / (..., 4).
        o_p, o_q: toy truck position / quaternion in base frame, shape (..., 3) / (..., 4).
        offset_path: grasp_offset.json produced by tools_c/compute_grasp_offset.py.
        symmetric: if True, use the closer of G and G' (180 deg about the approach axis) as the target.
        frame: "base" -> errors expressed in base axes, "ee" -> errors expressed in end-effector axes.
    """
    c = load_offset(offset_path)
    kw = dict(device=ee_p.device, dtype=ee_p.dtype)
    p_bar, R_bar = torch.tensor(c["p_bar"], **kw), torch.tensor(c["R_bar"], **kw)
    a = c["approach_axis"]
    F = -torch.eye(3, **kw)
    F[a, a] = 1  # R_a(pi)

    Ree, Ro = quat2mat(ee_q), quat2mat(o_q)
    Rt = Ro @ R_bar  # ^B R_G
    if symmetric:
        Rt2 = Rt @ F  # ^B R_G'
        Rt = torch.where((geo(Rt2, Ree) < geo(Rt, Ree))[..., None, None], Rt2, Rt)
    e_p = o_p + (Ro @ p_bar.unsqueeze(-1)).squeeze(-1) - ee_p
    if frame == "base":
        e_R = so3_log(Rt @ Ree.transpose(-1, -2))
    elif frame == "ee":
        e_p = (Ree.transpose(-1, -2) @ e_p.unsqueeze(-1)).squeeze(-1)
        e_R = so3_log(Ree.transpose(-1, -2) @ Rt)
    else:
        raise ValueError(f"Unknown frame: {frame}")
    return torch.cat([e_p, e_R], -1)


def grasp_err_obs(env, offset_path=DEFAULT_OFFSET_PATH, symmetric=True, frame="base"):
    """Observation term: grasp_err for the toy truck, computed from the same quantities as the policy obs."""
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
    from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

    tr = SceneEntityCfg("toy_truck")
    return grasp_err_from_poses(
        stack_mdp.ee_frame_pose_in_base_frame(env, return_key="pos"),
        stack_mdp.ee_frame_pose_in_base_frame(env, return_key="quat"),
        place_mdp.object_poses_in_base_frame(env, object_cfg=tr, return_key="pos"),
        place_mdp.object_poses_in_base_frame(env, object_cfg=tr, return_key="quat"),
        offset_path,
        symmetric,
        frame,
    )


def place_err_from_poses(ee_p, ee_q, box_p, offset_path=DEFAULT_PLACE_OFFSET_PATH, frame="base"):
    """Compute place_err = e_P (3D), the position error from the end-effector to the place target P.

    Args:
        ee_p, ee_q: end-effector position / quaternion in base frame, shape (..., 3) / (..., 4).
        box_p: box position in base frame, shape (..., 3).
        offset_path: place_offset.json produced by tools_c/compute_place_offset.py.
        frame: "base" -> error expressed in base axes, "ee" -> error expressed in end-effector axes.
    """
    h = load_offset(offset_path)["h"]
    e_p = box_p - ee_p
    e_p = torch.cat([e_p[..., :2], e_p[..., 2:] + h], -1)  # P = box center + h * z_B
    if frame == "base":
        return e_p
    if frame == "ee":
        return (quat2mat(ee_q).transpose(-1, -2) @ e_p.unsqueeze(-1)).squeeze(-1)
    raise ValueError(f"Unknown frame: {frame}")


def place_err_obs(env, offset_path=DEFAULT_PLACE_OFFSET_PATH, frame="base"):
    """Observation term: place_err for the box, computed from the same quantities as the policy obs."""
    from isaaclab.managers import SceneEntityCfg

    from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
    from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

    return place_err_from_poses(
        stack_mdp.ee_frame_pose_in_base_frame(env, return_key="pos"),
        stack_mdp.ee_frame_pose_in_base_frame(env, return_key="quat"),
        place_mdp.object_poses_in_base_frame(env, object_cfg=SceneEntityCfg("box"), return_key="pos"),
        offset_path,
        frame,
    )


def gravity_ee_from_quat(ee_q):
    """World down direction expressed in end-effector axes, R_E^T (0, 0, -1), shape (..., 3).

    The robot base z axis is world up (place_err_from_poses relies on the same assumption).
    """
    return -quat2mat(ee_q)[..., 2, :]


def gravity_ee_obs(env):
    """Observation term: gravity direction in end-effector axes (does not depend on the object poses)."""
    from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp

    return gravity_ee_from_quat(stack_mdp.ee_frame_pose_in_base_frame(env, return_key="quat"))
