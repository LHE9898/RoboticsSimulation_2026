# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import isaaclab.sim as sim_utils
# =======code edit=======
# import isaaclab.utils.math as math_utils  # only used by the terms now in ant/mdp/reward.py
# =======code edit=======
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
# =======code edit=======
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp

import math

# =======code edit=======
# import torch  # only used by the terms now in ant/mdp/reward.py
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
# =======code edit=======
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
# from .ant_foot_kinematics import FOOT_NAMES, foot_tip_height
# from .ant_foot_kinematics import FOOT_NAMES, AntFootKinematics, foot_tip_height
# =======code edit=======
# Custom MDP terms (rewards, observations, events) and foot kinematics live in ant/mdp.
import isaaclab_tasks.manager_based.classic.ant.mdp as mymdp
from .mdp import FOOT_NAMES
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

##
# Pre-defined configs
##
from isaaclab_assets.robots.ant import ANT_CFG  # isort: skip

# =======code edit=======
# Held-out terrain split: "boxes" is only used for evaluation (AntEnvCfg_PLAY); training uses the other
# ROUGH_TERRAINS_CFG sub-terrains, whose proportions the terrain generator renormalizes.
HELD_OUT_TERRAIN = "boxes"  # pyramid_stairs, pyramid_stairs_inv, boxes, random_rough, hf_pyramid_slope, hf_pyramid_slope_inv
# TRAIN_TERRAINS_CFG = ROUGH_TERRAINS_CFG.replace(
#     sub_terrains={name: cfg for name, cfg in ROUGH_TERRAINS_CFG.sub_terrains.items() if name != HELD_OUT_TERRAIN}
# )
# =======code edit=======
# curriculum=True lays the difficulty out by row (row i ~ difficulty i/num_rows) so the terrain_levels
# curriculum can move each robot between rows; with False every tile had a random difficulty in [0, 1].
TRAIN_TERRAINS_CFG = ROUGH_TERRAINS_CFG.replace(
    sub_terrains={name: cfg for name, cfg in ROUGH_TERRAINS_CFG.sub_terrains.items() if name != HELD_OUT_TERRAIN},
    curriculum=True,
)
EVAL_TERRAINS_CFG = ROUGH_TERRAINS_CFG.replace(
    sub_terrains={HELD_OUT_TERRAIN: ROUGH_TERRAINS_CFG.sub_terrains[HELD_OUT_TERRAIN]}
)

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with an ant robot."""

    # # terrain
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="average",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )
    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        # terrain_generator=ROUGH_TERRAINS_CFG,
        terrain_generator=TRAIN_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    # =======code edit=======
    # Contact reporting is required by ContactSensorCfg for air-time and collision rewards.
    robot = ANT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.spawn.activate_contact_sensors = True
    # =======code edit=======
    # Joint PD for position control (ANT_CFG has stiffness=damping=0, i.e. pure torque control).
    # The Ant weighs 0.91 kg and one ankle holds ~1 Nm when standing, while effort control reached
    # 5-15 Nm in kicks. stiffness 20 Nm/rad sags ~0.05 rad under that load; effort_limit_sim 4 Nm
    # (~4x standing torque) leaves room for climbing but caps the explosive push-offs.
    # robot.actuators = {
    #     "body": ImplicitActuatorCfg(
    #         joint_names_expr=[".*"],
    #         stiffness=20.0,
    #         damping=1.0,
    #         effort_limit_sim=4.0,
    #     ),
    # }
    # =======code edit=======
    # The PhysX drive (ImplicitActuatorCfg) does not track targets on this Ant asset: holding the default
    # pose left ~26 deg of error even with a 100 Nm limit. IdealPDActuatorCfg computes the same PD torque in
    # Isaac Lab and applies it as joint effort (the path the old effort control used): ~0.5 deg error.
    robot.actuators = {
        "body": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=20.0,
            damping=1.0,
            effort_limit=10.0,
        ),
    }
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.2, 1.2]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )
    # =======code edit=======
    # Wider scan for the policy to see where its feet land: the tips touch down 0.94-1.07 m from the
    # torso, outside the +-0.6 m height_scanner. 2.4 m x 2.4 m at 0.15 m = 17 x 17 = 289 rays.
    # height_scanner stays as-is for torso height (base_height reward/obs, foot_clearance).
    foothold_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[2.4, 2.4]),
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
        visualizer_cfg=RAY_CASTER_MARKER_CFG.replace(prim_path="/Visuals/FootholdScanner"),  # red spheres at hits
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=7.5)
    # =======code edit=======
    # Joint position targets around the default pose (rad): target = default + 0.5 * action. With
    # clip_actions=1.0 that is +-0.5 rad (~29 deg) per joint; the PD actuator turns it into torque.
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for the policy."""

        # base_height = ObsTerm(func=mdp.base_pos_z)
        # =======code edit=======
        # Torso height above the terrain, not world z, so it means the same on every sub-terrain.
        base_height = ObsTerm(func=mymdp.torso_height_obs, params={"sensor_cfg": SceneEntityCfg("height_scanner")})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        base_yaw_roll = ObsTerm(func=mdp.base_yaw_roll)
        base_angle_to_target = ObsTerm(func=mdp.base_angle_to_target, params={"target_pos": (1000.0, 0.0, 0.0)})
        base_up_proj = ObsTerm(func=mdp.base_up_proj)
        base_heading_proj = ObsTerm(func=mdp.base_heading_proj, params={"target_pos": (1000.0, 0.0, 0.0)})
        joint_pos_norm = ObsTerm(func=mdp.joint_pos_limit_normalized)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.2)
        feet_body_forces = ObsTerm(
            func=mdp.body_incoming_wrench,
            scale=0.1,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"]
                )
            },
        )

        # =======code edit=======
        # Terrain scan supplies ground-relative height; binary contacts provide support-state feedback.
        # terrain_height_scan = ObsTerm(
        #     func=mdp.height_scan,
        #     params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
        #     clip=(-1.0, 1.0),
        # )
        # =======code edit=======
        # Scan covering the foot landing area (289 values instead of 169); changes the observation size.
        terrain_height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("foothold_scanner"), "offset": 0.0},
            clip=(-1.0, 1.0),
        )
        
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # =======code edit=======
    # Stage-2 domain randomization: exact 0.1 friction increments from 0.6 through 1.2.
    discrete_friction = EventTerm(
        func=mymdp.randomize_discrete_friction,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_values": (0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {}, "velocity_range": {}},
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Reward for moving forward
    progress = RewTerm(func=mdp.progress_reward, weight=3.0, params={"target_pos": (1000.0, 0.0, 0.0)})
    # (2) Stay alive bonus
    alive = RewTerm(func=mdp.is_alive, weight=0.25)
    # (3) Reward for non-upright posture
    # upright = RewTerm(func=mdp.upright_posture_bonus, weight=0.1, params={"threshold": 0.93})
    # (4) Reward for moving in the right direction
    move_to_target = RewTerm(
        func=mdp.move_to_target_bonus, weight=0.01, params={"threshold": 0.8, "target_pos": (1000.0, 0.0, 0.0)}
    )
    # (5) Penalty for large action commands
    # action_l2 = RewTerm(func=mdp.action_l2, weight=-0.005)
    # =======code edit=======
    # Actions are now position targets, so action_l2 would only pull toward the default pose. Penalize the
    # applied joint torque instead; torque = 7.5 * action before, so -0.005 / 7.5^2 ~ -1e-4 keeps the scale.
    # joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4)
    # (6) Penalty for energy consumption
    # energy = RewTerm(func=mdp.power_consumption, weight=-0.05, params={"gear_ratio": {".*": 15.0}})
    # =======code edit=======
    # power_consumption multiplies the action by joint velocity, which assumed action = torque. Use the
    # applied torque instead; -0.05 / 7.5 keeps the old magnitude.
    energy = RewTerm(func=mymdp.mechanical_power, weight=-0.05 / 7.5)
    # (7) Penalty for reaching close to joint limits
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits_penalty_ratio, weight=-0.1, params={"threshold": 0.99, "gear_ratio": {".*": 15.0}}
    )

    # =======code edit=======
    # Ground-relative body-height objective. The foot tip can rise at most to (torso height - 0.32 m)
    # because the ankle stops at |30 deg|, so a 0.25 m foot lift needs the torso at about 0.62 m.
    # base_height = RewTerm(
    #     func=mdp.base_height_l2,
    #     weight=-10.0,
    #     params={"target_height": 0.5, "sensor_cfg": SceneEntityCfg("height_scanner")},
    # )
    base_height = RewTerm(
        func=mymdp.base_height_l2,
        weight=-10.0,
        params={"target_height": 0.6, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )

    feet_air_time = RewTerm(
        func=mymdp.feet_air_time,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["front_left_foot", "front_right_foot", "left_back_foot", "right_back_foot"],
            ),
            "clamp_vxforward_height": 0.05,
            # "clamp_vxbackward_height": 0.05,
            # =======code edit=======
            # "max_air_time": 0.2,
            # Per-foot weight from the foot tip height instead of the binary touchdown flag:
            #   0 with the tip on its local ground (nearest foothold_scanner hit),
            #   1 once the tip reaches the reference height, linear (uncapped) above that.
            # The reference defaults to the torso height above the mean terrain; pin it with
            # "reference_height": <float> instead of base_sensor_cfg. body_ids need no preserve_order,
            # feet_air_time re-maps the FOOT_NAMES-ordered heights onto the contact sensor's order.
            "scanner_cfg": SceneEntityCfg("foothold_scanner"),
            "base_sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    # Swing foot-tip height above local terrain (torso FK). A standing tip reads ~0.04 m, so 0.29 m
    # corresponds to a 0.25 m lift. preserve_order keeps contact indices in FOOT_NAMES order.
    # foot_clearance = RewTerm(
    #     func=_foot_clearance_reward,
    #     weight=1.0,
    #     params={
    #         "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES), preserve_order=True),
    #         "height_sensor_cfg": SceneEntityCfg("height_scanner"),
    #         "target_height": 0.35,
    #         "max_air_time": 0.5,
    #         # =======code edit=======
    #         "forward_vel_ref": 2.0,
    #         # =======code edit=======
    #         "min_support_feet": 2,
    #     },
    # )
    # =======code edit=======
    # (A) Swing feet must clear the terrain just ahead of them: a wall or step up demands a higher swing, a
    # step down or flat ground demands only the margin. A standing tip reads ~0.04 m above the ground, so
    # margin 0.05 asks for a ~1 cm lift on flat ground. max_deficit caps walls taller than the Ant can clear.
    swing_obstacle_clearance = RewTerm(
        func=mymdp.swing_obstacle_clearance,
        weight=-10.0,
        params={
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES), preserve_order=True),
            "scanner_cfg": SceneEntityCfg("foothold_scanner"),
            "look_ahead": 0.25,
            "half_width": 0.12,
            "margin": 0.05,
            "max_deficit": 0.15,
            "min_speed": 0.2,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # =======code edit=======
    # Penalize vertical torso velocity (squared, torso frame) to damp bouncing and hopping.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    # =======code edit=======
    # Keep the torso heading on the +x walking direction (yaw 0); 0.3 rad (~17 deg) of drift costs -0.045/step.
    yaw_deviation = RewTerm(func=mymdp.yaw_deviation_l2, weight=-2.0, params={"target_yaw": 0.0})
    # =======code edit=======
    # Penalize feet/shins hitting vertical faces (box sides, stair risers) at the moment it happens.
    # feet_stumble = RewTerm(
    #     func=_feet_stumble,
    #     weight=-0.5,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES)), "ratio": 2.0},
    # )
    # =======code edit=======
    # (B) A foot pushing a wall costs as much as feet_stumble did while it stays put, and nothing once its
    # tip rises at lift_vel_ref, so the way out of a wall is to lift the foot.
    stumble_without_lift = RewTerm(
        func=mymdp.stumble_without_lift,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES), preserve_order=True),
            "ratio": 2.0,
            "lift_vel_ref": 0.1,
        },
    )
    # =======code edit=======
    # One-off penalty when an episode ends by falling (bad_orientation / torso_contact; time-outs excluded).
    termination = RewTerm(func=mdp.is_terminated, weight=-100.0)
    # =======code edit=======
    # Penalize torso roll/pitch rate (squared, torso frame) to stop the airborne spinning that flips the
    # robot during bounces; a steady tilt on slopes/stairs has no rate and is not penalized.
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    # =======code edit=======
    # Penalize flight phases (all four feet off the ground): the torso cannot be steered in the air, so
    # speed should come from pushing against the ground. Being airborne the whole time costs 2/s.
    feet_all_airborne = RewTerm(
        func=mymdp.feet_all_airborne,
        weight=-20.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_NAMES))},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Terminate if the episode length is exceeded
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Terminate if the robot falls
    # torso_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.31})
    # =======code edit=======
    # Measured from the terrain under the torso; world z made robots on sunken sub-terrains reset every step.
    # torso_height = DoneTerm(
    #     func=_torso_below_minimum_height,
    #     params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_scanner")},
    # )
    # =======code edit=======
    # Terrain-independent fall checks: flipped or rolled over (tilt from world z beyond 50 deg), or collapsed
    # onto the belly (torso touching the ground, which a level torso would pass under a tilt-only check).
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(50.0)})
    # torso_contact = DoneTerm(
    #     func=mdp.illegal_contact,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="torso"), "threshold": 1.0},
    # )


# =======code edit=======
@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Per-robot terrain levels: at each reset a robot moves to a harder row if it lasted the whole episode
    # without falling at >= 2.5 m/s average forward speed, and to an easier row if it fell or averaged
    # < 0.3 m/s. Logged to TensorBoard as Curriculum/terrain_levels (mean level).
    terrain_levels = CurrTerm(func=mymdp.terrain_levels_speed, params={"up_speed": 1.5, "down_speed": 0.3})


@configclass
class AntEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the MuJoCo-style Ant walking environment."""

    # Scene settings
    # Contact sensors need real USD prims for every env; fabric cloning only creates env_0 in USD.
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    # =======code edit=======
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 16.0
        # simulation settings
        self.sim.dt = 1 / 120.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        # =======code edit=======
        # Keep the simulation material aligned with the unchanged plane material and tick sensors at the required rates.
        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        # =======code edit=======
        if self.scene.foothold_scanner is not None:
            self.scene.foothold_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # default friction material
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.restitution = 0.0
        


@configclass
class AntEnvCfg_PLAY(AntEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        # gen = self.scene.terrain.terrain_generator.copy()   # 학습 설정(ROUGH_TERRAINS_CFG)은 건드리지 않도록 복사
        # gen.sub_terrains = {"boxes": gen.sub_terrains["boxes"]}  # 원하는 지형만
        # =======code edit=======
        # Evaluate only on the held-out terrain. Start from EVAL_TERRAINS_CFG, not the inherited training
        # generator, which no longer contains the held-out sub-terrain.
        gen = EVAL_TERRAINS_CFG.copy()
        gen.num_rows, gen.num_cols = 30, 30                    # 작게
        gen.curriculum = False
        gen.difficulty_range = (0.8, 0.8)                    # 난이도 고정 (0=쉬움 ~ 1=어려움)
        self.scene.terrain.terrain_generator = gen
        # =======code edit=======
        # Show the foothold scan hit points as red spheres (play only; drawing 4096 x 289 markers slows training).
        self.scene.foothold_scanner.debug_vis = True
        # =======code edit=======
        # Evaluation keeps its fixed difficulty: no terrain-level changes between episodes.
        self.curriculum.terrain_levels = None
