# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# =======code edit=======
"""Custom MDP terms for the Ant rough-terrain task (moved here from ant_env_cfg.py)."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

from .ant_foot_kinematics import (
    FOOT_NAMES,
    AntFootKinematics,
    foot_tip_height,
    foot_tip_height_local,
    foot_tip_state,
)


# =======code edit=======
def torso_height_above_ground(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso height above the terrain under it, shape (num_envs,).

    The ground level is the mean of the height-scanner hits; rays that missed the terrain (inf) are
    ignored. World z alone is wrong on generated terrain, whose sub-terrains (e.g. inverted pyramids)
    can sit below z = 0.
    """
    scanner = env.scene.sensors[sensor_cfg.name]
    hits_z = scanner.data.ray_hits_w[..., 2]
    finite = torch.isfinite(hits_z)
    ground_z = torch.where(finite, hits_z, 0.0).sum(dim=1) / finite.sum(dim=1).clamp(min=1)
    return scanner.data.pos_w[:, 2] - ground_z


def torso_height_obs(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Torso height above the terrain as a (num_envs, 1) observation."""
    return torso_height_above_ground(env, sensor_cfg).unsqueeze(-1)


def base_height_l2(env, target_height: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Squared error of the torso height above the terrain.

    Same as mdp.base_height_l2 with a sensor, but a ray that misses the terrain (inf) is left out of the
    ground level instead of turning the reward into -inf and the policy into NaN.
    """
    return torch.square(torso_height_above_ground(env, sensor_cfg) - target_height)


# def torso_below_minimum_height(env, minimum_height: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
#     """Terminate when the torso is closer than minimum_height to the terrain under it."""
#     return torso_height_above_ground(env, sensor_cfg) < minimum_height


# =======code edit=======
def foot_clearance_reward(
    env,
    contact_sensor_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg,
    target_height: float,
    max_air_time: float,
    # =======code edit=======
    forward_vel_ref: float = 1.0,
    # =======code edit=======
    min_support_feet: int = 2,
) -> torch.Tensor:
    """Reward swinging foot tips for rising toward target_height above the local terrain.

    The foot link origin is the ankle joint, which stays level with the torso, so the tip height comes
    from torso-to-tip forward kinematics instead. Swing is detected with the contact sensor, and the
    reward is linear up to target_height so every extra centimetre counts. Only the first max_air_time
    seconds of a swing are paid, so holding a foot in the air does not farm reward.
    """
    tip_height = foot_tip_height(env, sensor_cfg=height_sensor_cfg)  # (num_envs, 4), FOOT_NAMES order
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, contact_sensor_cfg.body_ids]
    swinging = (air_time > 0.0) & (air_time < max_air_time)
    height_score = torch.clamp(tip_height, 0.0, target_height) / target_height
    # return torch.sum(height_score * swinging, dim=1)
    # =======code edit=======
    # Scale by forward progress so stepping in place earns nothing: 0 when not moving toward +x,
    # full reward from forward_vel_ref (m/s, world x) upward.
    asset = env.scene["robot"]
    forward_scale = torch.clamp(asset.data.root_lin_vel_w[:, 0] / forward_vel_ref, 0.0, 1.0)
    # return torch.sum(height_score * swinging, dim=1) * forward_scale
    # =======code edit=======
    # Pay a raised foot only while at least min_support_feet other feet stand on the ground, so the reward
    # is for stepping over things, not for flight: lifting all four feet at once (hopping) earns nothing.
    in_contact = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] > 0.0
    supported = (in_contact.sum(dim=1) >= min_support_feet).float()
    return torch.sum(height_score * swinging, dim=1) * forward_scale * supported


# =======code edit=======
# def feet_air_time(env, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
# =======code edit=======
# max_air_time caps the paid air time, so long ballistic flights earn no more than a normal swing.
def feet_air_time(
    env,
    sensor_cfg: SceneEntityCfg,
    clamp_vxforward_height: float | None = None,
    clamp_vxbackward_height: float | None = None,
    scanner_cfg: SceneEntityCfg | None = None,
    base_sensor_cfg: SceneEntityCfg | None = None,
    reference_height: float | None = None,
) -> torch.Tensor:
    """Reward steps longer than threshold: sum of (last air time - threshold) at each touchdown.

    Same as velocity_rewards.feet_air_time without the command gate; the forward command is fixed,
    so the gate was always on.

    The per-foot weight can instead come from the foot tip height rather than the binary touchdown flag.
    Passing scanner_cfg together with either base_sensor_cfg or reference_height switches to that mode:

        weight = clamp(foot_tip_height_local, min=0) / reference_height

    so it is 0 with the tip on its local ground, 1 once the tip reaches reference_height, and linear
    (not capped) above that. reference_height defaults to the torso height above the mean terrain, i.e.
    the base-to-ground distance; pass a float to pin it to a constant. Note the ant's foot can only rise
    to roughly (torso height - 0.32 m) because the ankle stops at |30 deg|, so with the torso at 0.6 m
    the weight saturates near 0.45 unless a smaller reference_height is given.

    foot_tip_height_local returns FOOT_NAMES order while sensor_cfg.body_ids follow the contact sensor's
    own body order, so the height array is re-mapped here; sensor_cfg does not need preserve_order=True.
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    # =======code edit=======
    # if max_air_time is not None:
    #     current_air_time = torch.clamp(current_air_time, max=max_air_time)

    if scanner_cfg is None or (base_sensor_cfg is None and reference_height is None):
        # original behaviour: an impulse at each touchdown
        weight = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    else:
        # =======code edit=======
        # Continuous height weight in place of the touchdown impulse.
        height = foot_tip_height_local(env, scanner_cfg)  # (num_envs, 4), FOOT_NAMES order
        foot_index = {name: i for i, name in enumerate(FOOT_NAMES)}
        order = [foot_index[contact_sensor.body_names[b]] for b in sensor_cfg.body_ids]
        height = height[:, order]
        _, tip_vel = foot_tip_state(env)
        vx_forward = torch.clamp(tip_vel[..., 0][:, order], min=0.0)
        vx_backward = torch.clamp(tip_vel[..., 0][:, order], max=0.0)
        if clamp_vxforward_height is None:
            weight = torch.clamp(height, min=0.0) * vx_forward
        else:
            weight = torch.clamp(height, min=0.0, max=clamp_vxforward_height) * vx_forward / clamp_vxforward_height
        if clamp_vxbackward_height is None:
            weight += torch.clamp(height, min=0.0) * vx_backward
        else:
            weight += torch.clamp(height, min=0.0, max=clamp_vxbackward_height) * vx_backward / clamp_vxbackward_height

        # 전진 + 공중 -> reward, 후진 + 공중 -> penalty

    return torch.sum((current_air_time) * weight, dim=1)


# =======code edit=======
def mechanical_power(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Sum of |applied joint torque x joint velocity| (W). Valid for any action type, unlike
    power_consumption, which multiplies the raw action and so assumed action = torque."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=1)


# =======code edit=======
def feet_all_airborne(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """1 while none of the selected feet touches the ground (flight phase of a hop), else 0."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    return torch.amin(in_contact, dim=1)


# =======code edit=======
def yaw_deviation_l2(env, target_yaw: float = 0.0, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Squared torso yaw error from target_yaw (rad, world frame), wrapped to (-pi, pi].

    Penalizes the heading itself drifting away from the +x walking direction, not just the turn rate.
    """
    asset = env.scene[asset_cfg.name]
    _, _, yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.square(math_utils.wrap_to_pi(yaw - target_yaw))


# =======code edit=======
def feet_stumble(env, sensor_cfg: SceneEntityCfg, ratio: float) -> torch.Tensor:
    """Number of foot links pushing against a wall: horizontal contact force > ratio x vertical force.

    The *_foot link is the whole lower leg, so a shin caught on a box side counts too. ratio must stay
    above the friction coefficient (max 1.2 here) so ordinary push-off and slopes are not penalized;
    feet in the air carry no force and never count.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    horizontal = torch.norm(forces[..., :2], dim=-1)
    vertical = torch.abs(forces[..., 2])
    return torch.sum(horizontal > ratio * vertical, dim=1).float()


# =======code edit=======
# foot_tip_state now lives in ant_foot_kinematics.py, next to the FK it belongs to. It used to be defined
# here, but foot_tip_height_local (also in ant_foot_kinematics.py) calls it, and importing it back from
# this module would be circular. It is imported at the top of this file instead.


# =======code edit=======
def swing_obstacle_clearance(
    env,
    contact_sensor_cfg: SceneEntityCfg,
    scanner_cfg: SceneEntityCfg,
    look_ahead: float,
    half_width: float,
    margin: float,
    max_deficit: float,
    min_speed: float,
) -> torch.Tensor:
    """Height (m) by which swinging foot tips stay below the terrain just ahead of them, summed over feet.

    For each airborne foot moving horizontally faster than min_speed, the obstacle height is the highest
    foothold_scanner hit in a strip from 0.05 m behind to look_ahead in front of the tip along its direction
    of motion (+-half_width sideways). The deficit is (obstacle height + margin - tip height), clipped to
    [0, max_deficit]. Because it is measured against the terrain ahead of each foot, a wall or a step up
    demands a higher swing, a step down demands none, and flat ground only needs the margin.
    """
    tip_pos, tip_vel = foot_tip_state(env)
    contact_sensor = env.scene.sensors[contact_sensor_cfg.name]
    swinging = contact_sensor.data.current_contact_time[:, contact_sensor_cfg.body_ids] <= 0.0  # (N, 4)
    v_xy = tip_vel[..., :2]
    speed = torch.norm(v_xy, dim=-1)
    direction = v_xy / speed.clamp(min=1e-6).unsqueeze(-1)  # (N, 4, 2)

    hits = env.scene.sensors[scanner_cfg.name].data.ray_hits_w  # (N, R, 3)
    rel = hits[:, None, :, :2] - tip_pos[:, :, None, :2]  # (N, 4, R, 2)
    along = (rel * direction[:, :, None, :]).sum(-1)
    lateral = torch.abs(rel[..., 0] * direction[:, :, None, 1] - rel[..., 1] * direction[:, :, None, 0])
    hit_z = hits[:, None, :, 2].expand_as(along)
    in_strip = (along > -0.05) & (along < look_ahead) & (lateral < half_width) & torch.isfinite(hit_z)
    obstacle_z = torch.where(in_strip, hit_z, torch.full_like(hit_z, -1.0e3)).max(dim=-1).values  # (N, 4)

    deficit = torch.clamp(obstacle_z + margin - tip_pos[..., 2], 0.0, max_deficit)
    active = swinging & (speed > min_speed) & in_strip.any(dim=-1)
    return torch.sum(deficit * active, dim=1)


# =======code edit=======
def stumble_without_lift(env, sensor_cfg: SceneEntityCfg, ratio: float, lift_vel_ref: float) -> torch.Tensor:
    """Feet pushing against a wall (as in feet_stumble) weighted by how little they are lifting.

    Each stumbling foot counts 1 when its tip is not rising and less as the tip rises, reaching 0 at
    lift_vel_ref (m/s) upward. Never negative, so touching walls cannot earn anything.
    sensor_cfg must list the feet in FOOT_NAMES order (preserve_order=True) to match the tip velocities.
    """
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w[:, sensor_cfg.body_ids]
    stumbling = torch.clamp(-ratio * forces[..., 2]-forces[..., 0], 0.0, 5.0)
    _, tip_vel = foot_tip_state(env)
    not_lifting = 1.0 - torch.clamp(tip_vel[..., 2] / lift_vel_ref, -1.0, 1.0)
    return torch.sum(stumbling * not_lifting, dim=1)


# =======code edit=======
def randomize_discrete_friction(
    env, env_ids, asset_cfg: SceneEntityCfg, friction_values: tuple[float, ...]
) -> None:
    """Assign one discrete, shared friction value to every Ant collider in each environment."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    asset = env.scene[asset_cfg.name]
    values = torch.tensor(friction_values, dtype=torch.float32, device="cpu")
    value_ids = torch.randint(len(values), (len(env_ids),), device="cpu")
    friction = values[value_ids]
    material_samples = torch.stack((friction, friction, torch.zeros_like(friction)), dim=-1)

    materials = asset.root_physx_view.get_material_properties()
    materials[env_ids] = material_samples.unsqueeze(1).expand(-1, materials.shape[1], -1)
    asset.root_physx_view.set_material_properties(materials, env_ids)


# =======code edit=======
def terrain_levels_speed(
    env, env_ids, up_speed: float, down_speed: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Terrain-level curriculum from the average forward (+x) speed of the episode that just ended.

    Runs at reset, before the scene is reset, so position, episode length and the termination flags still
    describe the finished episode. A robot moves one row up (harder) if it reached the time-out without
    falling and averaged at least up_speed (m/s), and one row down (easier) if it fell or averaged less
    than down_speed. Same idea as terrain_levels_vel, which needs a velocity command this task does not use.

    Returns:
        The mean terrain level over all robots.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    distance = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    elapsed = (env.episode_length_buf[env_ids] * env.step_dt).clamp(min=env.step_dt)
    avg_speed = distance / elapsed
    fell = env.termination_manager.terminated[env_ids]
    timed_out = env.termination_manager.time_outs[env_ids]
    move_up = timed_out & ~fell & (avg_speed >= up_speed)
    move_down = (fell | (avg_speed < down_speed)) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
