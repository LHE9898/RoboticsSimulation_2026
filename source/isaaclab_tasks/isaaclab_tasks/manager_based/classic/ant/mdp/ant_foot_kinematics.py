# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Forward kinematics from the Ant torso to each foot tip.

The kinematic chain (parent/child links, joint frames, joint axes) and the tip point of each foot are
read once from the env_0 USD, so nothing about the Ant geometry is hard-coded. Every step the chain is
evaluated in torch from the torso pose and the joint positions:

    torso --hip joint--> *_leg --ankle joint--> *_foot --tip offset--> foot tip

On the first evaluation the FK link positions are compared against the simulator's link poses and the
error is printed, so a wrong axis sign or scale shows up immediately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

FOOT_NAMES = ("front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot")


@dataclass
class _JointLink:
    """One joint of a chain: parent link -> joint frame -> rotation about axis -> child link."""

    name: str
    joint_id: int | None  # index into asset.joint_names; None for fixed joints
    pos0: torch.Tensor  # joint frame position in the parent link frame (m)
    quat0: torch.Tensor  # joint frame orientation in the parent link frame (w, x, y, z)
    pos1: torch.Tensor  # joint frame position in the child link frame (m)
    quat1_inv: torch.Tensor  # inverse of the joint frame orientation in the child link frame
    axis: torch.Tensor | None  # unit rotation axis in the joint frame; None for fixed joints


class AntFootKinematics:
    """Torso-to-foot-tip forward kinematics for the Ant, built from the USD joint definitions."""

    def __init__(self, env, asset_name: str = "robot", foot_names: tuple[str, ...] = FOOT_NAMES):
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        self.asset_name = asset_name
        self.foot_names = list(foot_names)
        asset = env.scene[asset_name]
        device = env.device

        stage = omni.usd.get_context().get_stage()
        robot_path = re.sub(r"env_\.\*", "env_0", asset.cfg.prim_path)
        robot_prim = stage.GetPrimAtPath(robot_path)
        if not robot_prim.IsValid():
            raise RuntimeError(f"[foot_fk] Robot prim not found at '{robot_path}'.")
        xform_cache = UsdGeom.XformCache()

        # rigid links by name; joints keyed by the path of their child link
        links, joints_by_child = {}, {}
        for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                links[prim.GetName()] = prim
            elif prim.IsA(UsdPhysics.Joint):
                joint = UsdPhysics.Joint(prim)
                body0, body1 = joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets()
                if body0 and body1:  # skip the articulation root joint to the world
                    joints_by_child[body1[0]] = prim

        def link_scale(prim) -> Gf.Vec3d:
            return Gf.Transform(xform_cache.GetLocalToWorldTransform(prim)).GetScale()

        def as_tensor(values) -> torch.Tensor:
            return torch.tensor([float(v) for v in values], dtype=torch.float32, device=device)

        def quat_tensor(q) -> torch.Tensor:
            return as_tensor([q.GetReal(), *q.GetImaginary()])

        self.chains: list[list[_JointLink]] = []
        self.tip_offsets: list[torch.Tensor] = []
        root_names = set()
        for foot_name in self.foot_names:
            if foot_name not in links:
                raise RuntimeError(f"[foot_fk] Link '{foot_name}' not found under '{robot_path}'.")
            # walk from the foot up to the root link
            chain, path = [], links[foot_name].GetPath()
            while path in joints_by_child:
                prim = joints_by_child[path]
                joint = UsdPhysics.Joint(prim)
                parent_prim = stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
                child_prim = stage.GetPrimAtPath(path)
                s0, s1 = link_scale(parent_prim), link_scale(child_prim)
                p0, p1 = joint.GetLocalPos0Attr().Get(), joint.GetLocalPos1Attr().Get()
                if prim.IsA(UsdPhysics.RevoluteJoint):
                    axis_name = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                    axis = torch.zeros(3, device=device)
                    axis["XYZ".index(axis_name)] = 1.0
                    joint_id = asset.joint_names.index(prim.GetName())
                elif prim.IsA(UsdPhysics.FixedJoint):
                    axis, joint_id = None, None
                else:
                    raise RuntimeError(f"[foot_fk] Unsupported joint type at '{prim.GetPath()}'.")
                chain.insert(
                    0,
                    _JointLink(
                        name=prim.GetName(),
                        joint_id=joint_id,
                        pos0=as_tensor([p0[i] * s0[i] for i in range(3)]),
                        quat0=quat_tensor(joint.GetLocalRot0Attr().Get()),
                        pos1=as_tensor([p1[i] * s1[i] for i in range(3)]),
                        quat1_inv=math_utils.quat_inv(quat_tensor(joint.GetLocalRot1Attr().Get())),
                        axis=axis,
                    ),
                )
                path = parent_prim.GetPath()
            root_names.add(path.name)
            self.chains.append(chain)
            self.tip_offsets.append(as_tensor(self._tip_offset(links[foot_name], xform_cache)))

        if len(root_names) != 1:
            raise RuntimeError(f"[foot_fk] Feet do not share one root link: {sorted(root_names)}.")
        self.root_name = root_names.pop()
        self.root_body_id = asset.body_names.index(self.root_name)
        self.foot_body_ids = [asset.body_names.index(name) for name in self.foot_names]
        self._verified = False

        print(f"[foot_fk] joints (asset order): {asset.joint_names}")
        for foot_name, chain, tip in zip(self.foot_names, self.chains, self.tip_offsets):
            joints = " -> ".join(f"{j.name}({'fixed' if j.axis is None else 'XYZ'[int(j.axis.argmax())]})" for j in chain)
            print(f"[foot_fk] {foot_name}: {self.root_name} -> {joints} -> tip offset {[round(v, 3) for v in tip.tolist()]} m")

    @staticmethod
    def _tip_offset(body_prim, xform_cache) -> list[float]:
        """Collision point farthest from the link origin, in the unscaled link frame (m)."""
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        colliders, visuals = [], []
        prim_range = Usd.PrimRange(body_prim, Usd.TraverseInstanceProxies())
        for prim in prim_range:
            if prim != body_prim and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim_range.PruneChildren()  # another link starts here
                continue
            if prim.IsA(UsdGeom.Capsule):
                geom = UsdGeom.Capsule(prim)
                half = geom.GetHeightAttr().Get() / 2 + geom.GetRadiusAttr().Get()
                axis = "XYZ".index(geom.GetAxisAttr().Get())
                local = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
                local[0][axis], local[1][axis] = half, -half
            elif prim.IsA(UsdGeom.Sphere):
                r = UsdGeom.Sphere(prim).GetRadiusAttr().Get()
                local = [[s * r if i == j else 0.0 for j in range(3)] for i in range(3) for s in (1, -1)]
            elif prim.IsA(UsdGeom.Mesh):
                local = [list(p) for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
            else:
                continue
            geom_to_world = xform_cache.GetLocalToWorldTransform(prim)
            points = [geom_to_world.Transform(Gf.Vec3d(*p)) for p in local]
            (colliders if prim.HasAPI(UsdPhysics.CollisionAPI) else visuals).extend(points)
        points = colliders or visuals
        if not points:
            raise RuntimeError(f"[foot_fk] No collision or visual geometry found under '{body_prim.GetPath()}'.")

        body_to_world = xform_cache.GetLocalToWorldTransform(body_prim)
        origin = body_to_world.ExtractTranslation()
        world_to_body_rot = Gf.Rotation(body_to_world.RemoveScaleShear().ExtractRotationQuat()).GetInverse()
        tip = max((world_to_body_rot.TransformDir(p - origin) for p in points), key=lambda v: v.GetLength())
        return [tip[0], tip[1], tip[2]]

    def compute(self, env) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (foot tip positions, foot link positions) in the world frame, each (num_envs, num_feet, 3)."""
        asset = env.scene[self.asset_name]
        joint_pos = asset.data.joint_pos
        root_pos = asset.data.body_link_pos_w[:, self.root_body_id]
        root_quat = asset.data.body_link_quat_w[:, self.root_body_id]
        n = root_pos.shape[0]

        tips, link_positions = [], []
        for chain, tip_offset in zip(self.chains, self.tip_offsets):
            pos, quat = root_pos, root_quat
            for j in chain:
                # parent link -> joint frame
                pos = pos + math_utils.quat_apply(quat, j.pos0.expand(n, 3))
                quat = math_utils.quat_mul(quat, j.quat0.expand(n, 4))
                # rotate by the joint angle about the joint axis
                if j.axis is not None:
                    quat = math_utils.quat_mul(
                        quat, math_utils.quat_from_angle_axis(joint_pos[:, j.joint_id], j.axis.expand(n, 3))
                    )
                # joint frame -> child link
                quat = math_utils.quat_mul(quat, j.quat1_inv.expand(n, 4))
                pos = pos - math_utils.quat_apply(quat, j.pos1.expand(n, 3))
            link_positions.append(pos)
            tips.append(pos + math_utils.quat_apply(quat, tip_offset.expand(n, 3)))
        tip_pos, link_pos = torch.stack(tips, dim=1), torch.stack(link_positions, dim=1)

        if not self._verified:
            error = torch.norm(link_pos - asset.data.body_link_pos_w[:, self.foot_body_ids], dim=-1).max().item()
            status = "OK" if error < 1.0e-3 else "MISMATCH - check joint axis sign / scale"
            print(f"[foot_fk] FK vs simulator foot link position, max error = {error * 1000:.3f} mm ({status})")
            self._verified = True
        return tip_pos, link_pos


def foot_tip_height(
    env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sensor_cfg: SceneEntityCfg | None = None
) -> torch.Tensor:
    """Height of each foot tip above the ground in metres, shape (num_envs, num_feet), FOOT_NAMES order.

    Without a sensor the ground is the z = 0 plane. With a height-scanner sensor_cfg the ground level
    is the mean of its ray hits, for use on uneven terrain.
    """
    fk = getattr(env, "_ant_foot_fk", None)
    if fk is None:
        fk = AntFootKinematics(env, asset_cfg.name)
        env._ant_foot_fk = fk
    tip_pos, _ = fk.compute(env)
    ground_z = 0.0
    if sensor_cfg is not None:
        ground_z = env.scene.sensors[sensor_cfg.name].data.ray_hits_w[..., 2].mean(dim=1, keepdim=True)
    return tip_pos[..., 2] - ground_z

def foot_tip_state(env, asset_name: str = "robot") -> tuple[torch.Tensor, torch.Tensor]:
    """World position and linear velocity of each foot tip, each (num_envs, 4, 3) in FOOT_NAMES order.

    The tip offsets in the foot-link frame come from the USD collision shapes (AntFootKinematics, built once
    and cached on the env); position and velocity follow from the simulated link pose and twist.
    """
    fk = getattr(env, "_ant_foot_fk", None)
    if fk is None:
        fk = AntFootKinematics(env, asset_name)
        env._ant_foot_fk = fk
    asset = env.scene[asset_name]
    ids = fk.foot_body_ids
    offsets = torch.stack(fk.tip_offsets).unsqueeze(0).expand(asset.num_instances, -1, -1)
    r = math_utils.quat_apply(asset.data.body_link_quat_w[:, ids], offsets)  # link origin -> tip, world frame
    tip_pos = asset.data.body_link_pos_w[:, ids] + r
    tip_vel = asset.data.body_link_lin_vel_w[:, ids] + torch.cross(asset.data.body_link_ang_vel_w[:, ids], r, dim=-1)
    return tip_pos, tip_vel


def foot_tip_height_local(env, scanner_cfg: SceneEntityCfg, asset_name: str = "robot") -> torch.Tensor:
    """각 발끝 xy에 가장 가까운 scanner hit을 그 발의 지면으로 사용 -> (num_envs, 4), FOOT_NAMES 순서.

    지형을 맞히지 못한 ray(inf)는 후보에서 제외한다. 그런 hit은 xy도 inf일 수 있어 cdist가 nan을
    내므로 유한값으로 치환해 거리 계산에서 밀어낸다. 유효한 후보가 하나도 없으면 지면을 0으로 둔다.
    """
    tip_pos, _ = foot_tip_state(env, asset_name)
    hits = env.scene.sensors[scanner_cfg.name].data.ray_hits_w          # (N, R, 3)
    hit_z = hits[..., 2]
    finite = torch.isfinite(hit_z)
    xy = torch.where(finite.unsqueeze(-1), hits[..., :2], torch.full_like(hits[..., :2], 1.0e6))
    idx = torch.cdist(tip_pos[..., :2], xy).argmin(dim=-1)              # (N, 4)
    ground_z = hit_z.gather(1, idx)                                     # (N, 4)
    ground_z = torch.where(torch.isfinite(ground_z), ground_z, torch.zeros_like(ground_z))
    return tip_pos[..., 2] - ground_z