# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RMPFlow relative action expressed in end-effector axes.

The base RMPFlow relative action is [dp, w] in robot-base axes, applied as
    p_des = p + dp,    R_des = exp(w) @ R          (math_utils.apply_delta_pose)
Here the policy outputs [dp_E, w_E] in end-effector axes and the term converts them with the current EE rotation R_E:
    dp = R_E dp_E,     w = R_E w_E                 (exp(R_E w_E) @ R_E = R_E @ exp(w_E))
so the commanded pose is the same as before for the converted action (tools_c/convert_actions_ee.py does the inverse
offline on the dataset).
"""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab.envs.mdp.actions.rmpflow_actions_cfg import RMPFlowActionCfg
from isaaclab.envs.mdp.actions.rmpflow_task_space_actions import RMPFlowAction
from isaaclab.utils import configclass


class RMPFlowEEFrameAction(RMPFlowAction):
    def process_actions(self, actions: torch.Tensor):
        _, ee_quat = self._compute_frame_pose()
        R = math_utils.matrix_from_quat(ee_quat)
        a = actions.clone()
        a[:, 0:3] = (R @ actions[:, 0:3].unsqueeze(-1)).squeeze(-1)
        a[:, 3:6] = (R @ actions[:, 3:6].unsqueeze(-1)).squeeze(-1)
        super().process_actions(a)


@configclass
class RMPFlowEEFrameActionCfg(RMPFlowActionCfg):
    class_type: type = RMPFlowEEFrameAction

    def __post_init__(self):
        assert self.use_relative_mode, "EE-frame actions are deltas; use_relative_mode must be True"
