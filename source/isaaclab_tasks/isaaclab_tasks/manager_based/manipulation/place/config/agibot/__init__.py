# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym
import importlib.util
import os
import sys

from . import agents

# Register the project robomimic algo "bc_event" (tools_c/bc_event.py) when the tasks are imported, so the stock
# scripts/imitation_learning/robomimic/play.py can load evrz/eez2 checkpoints. Shared with tools_c scripts via sys.modules.
_BC_EVENT = os.path.join(os.path.dirname(__file__), *[".."] * 8, "tools_c", "bc_event.py")
if "bc_event" not in sys.modules and os.path.exists(_BC_EVENT):
    try:
        _spec = importlib.util.spec_from_file_location("bc_event", os.path.abspath(_BC_EVENT))
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules["bc_event"] = _mod
        _spec.loader.exec_module(_mod)
    except ImportError:  # robomimic not installed in this environment
        sys.modules.pop("bc_event", None)

##
# Register Gym environments.
##

##
# Agibot Right Arm: place toy2box task, with RmpFlow
##
gym.register(
    id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.place_toy2box_rmp_rel_env_cfg:"
            "RmpFlowAgibotPlaceToy2BoxEnvCfg"
        ),
        "robomimic_bc_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_rnn_low_dim.json"
        ),
        # baseline + grasp_err (6D) observation
        "robomimic_bc_grasperr_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_rnn_low_dim_grasperr.json"
        ),
        # baseline + grasp_err in EE axes (6D)
        "robomimic_bc_ge_ee_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_rnn_low_dim_ge_ee.json"
        ),
        # baseline + grasp_err and place_err in EE axes (6D + 3D)
        "robomimic_bc_ge_ee_place_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_rnn_low_dim_ge_ee_place.json"
        ),
        # same obs, gripper as an event classifier instead of a GMM dimension (algo "bc_event", tools_c/bc_event.py)
        "robomimic_bc_event_ready_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_event_ready.json"
        ),
        "robomimic_bc_event_ready_inbox_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_event_ready_inbox.json"
        ),
    },
    disable_env_checker=True,
)

# same task, arm action in end-effector axes (mdp/ee_frame_actions.py); dataset actions via tools_c/convert_actions_ee.py
gym.register(
    id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-EEAct-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.place_toy2box_rmp_rel_env_cfg:"
            "RmpFlowAgibotPlaceToy2BoxEEActEnvCfg"
        ),
        "robomimic_bc_event_ready_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_event_ready.json"
        ),
    },
    disable_env_checker=True,
)

# same as EEAct, policy sees only relative observations (gripper_pos, grasp_err_ee, place_err_ee, gravity_ee)
gym.register(
    id="Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-EEActRel-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.place_toy2box_rmp_rel_env_cfg:"
            "RmpFlowAgibotPlaceToy2BoxEEActRelEnvCfg"
        ),
        "robomimic_bc_event_ready_rel_cfg_entry_point": (
            f"{agents.__name__}:robomimic/bc_event_ready_rel.json"
        ),
    },
    disable_env_checker=True,
)

##
# Agibot Left Arm: place upright mug task, with RmpFlow
##
gym.register(
    id="Isaac-Place-Mug-Agibot-Left-Arm-RmpFlow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.place_upright_mug_rmp_rel_env_cfg:RmpFlowAgibotPlaceUprightMugEnvCfg",
    },
    disable_env_checker=True,
)
