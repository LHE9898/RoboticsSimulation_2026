"""Robomimic algo "bc_event": BC-RNN-GMM for the 6D arm action + a gripper *event* classifier.

Gripper diagnosis (tools_c/gripper_mode_replay.py): with the gripper inside the GMM, the policy learns to copy the current
gripper state (99.6% of steps) and switches only when a low-weight mode is sampled. Here the gripper leaves the GMM:

    arm:     GMM over a[:6] (same RNN/GMM settings as bc)
    gripper: p_t = sigmoid(f([x_t, x_t - x_{t-1}])[state_t]), head 0 = "close now" while open, head 1 = "open now" while closed
             trained with positive-weighted BCE on the label key (tools_c/add_grip_event_to_hdf5.py)
    rollout: switch the commanded gripper state when p_t > tau[state] for `consecutive` steps (deterministic, no sampling)
    optional close rule (apply_close_rule): while open, close when the EE-frame grasp error has lateral |(e_x, e_y)| < xy_max
             and depth e_z < z_max for `consecutive` steps instead of using the close head (results/event_clf/close_region.md).
             The rule is for the grasp only: once the truck was lifted (> 2 cm while closed) and then released, the close
             head takes over again, otherwise the rule would re-grasp the truck right after placing it.

Importing this module registers the config ("bc_event") and the algo factory with robomimic.
"""

import json
import os
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.algo import register_algo_factory_func
from robomimic.algo.bc import BC_RNN_GMM
from robomimic.config.bc_config import BCConfig


class BCEventConfig(BCConfig):
    ALGO_NAME = "bc_event"

    def train_config(self):
        super().train_config()
        self.train.dataset_keys = ("actions", "grip_state", "grip_event_ready")

    def algo_config(self):
        super().algo_config()
        self.algo.optim_params.event.optimizer_type = "adam"
        self.algo.optim_params.event.learning_rate.initial = 1e-3
        self.algo.optim_params.event.learning_rate.decay_factor = 0.1
        self.algo.optim_params.event.learning_rate.epoch_schedule = []
        self.algo.optim_params.event.learning_rate.scheduler_type = "multistep"
        self.algo.optim_params.event.regularization.L2 = 0.0
        self.algo.event.label_key = "grip_event_ready"  # one of the keys written by add_grip_event_to_hdf5.py
        self.algo.event.hidden_dims = (256, 256)
        self.algo.event.tau = (0.5, 0.5)  # (close, open); replaced at rollout by event_tau.json (calibrate_event_tau.py)
        self.algo.event.consecutive = 3
        # (xy_max, z_max) in m; empty = close with the close head. A play checkpoint (tools_c/make_play_ckpt.py) stores the
        # calibrated tau and the rule here so the stock robomimic play.py runs the policy as evaluated; apply_close_rule overrides it
        self.algo.event.close_rule = ()
        # frame of the arm action the GMM was trained on: "base" (dataset as recorded) or "ee" (tools_c/convert_actions_ee.py).
        # Training never reads it. A play checkpoint with "ee" makes get_action rotate the arm action back to base axes with the
        # observed eef_quat, so the policy runs in the stock ...-RmpFlow-v0 task (same as ...-EEAct-v0 doing it in the env)
        self.algo.event.action_frame = "base"


class EventNet(nn.Module):
    def __init__(self, obs_dim, hidden_dims):
        super().__init__()
        layers, d = [], 2 * obs_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        self.mlp = nn.Sequential(*layers, nn.Linear(d, 2))
        # normalization of [x, dx] and per-head positive weights; buffers so they travel with the checkpoint
        self.register_buffer("mu", torch.zeros(2 * obs_dim))
        self.register_buffer("sd", torch.ones(2 * obs_dim))
        self.register_buffer("pos_weight", torch.ones(2))

    def forward(self, x, dx, state):
        """Logit of the head that applies to the current gripper state (0 open -> close event, 1 closed -> open event)."""
        z = (torch.cat([x, dx], -1) - self.mu) / self.sd
        return self.mlp(z).gather(-1, state.long().unsqueeze(-1)).squeeze(-1)


@register_algo_factory_func("bc_event")
def algo_config_to_class(algo_config):
    return BC_RNN_GMM_Event, {}


class BC_RNN_GMM_Event(BC_RNN_GMM):
    def _create_networks(self):
        full = self.ac_dim
        self.ac_dim = full - 1  # GMM over the arm only
        super()._create_networks()
        self.ac_dim = full
        self.event_keys = sorted(self.obs_shapes)
        dim = sum(int(np.prod(self.obs_shapes[k])) for k in self.event_keys)
        self.nets["event"] = EventNet(dim, self.algo_config.event.hidden_dims)
        self._load_event_stats()
        self.nets = self.nets.float().to(self.device)
        self.event_tau = list(self.algo_config.event.tau)
        rule = self.algo_config.event.get("close_rule", None)  # absent in checkpoints written before this key existed
        self.close_rule = tuple(float(v) for v in rule) if rule else None  # (xy_max, z_max) in m; apply_close_rule overrides
        self.action_frame = self.algo_config.event.get("action_frame", "base") or "base"
        assert self.action_frame in ("base", "ee"), self.action_frame
        self._reset_gripper()

    def _load_event_stats(self):
        """Normalization and positive weights from the dataset attrs (at load time the checkpoint buffers override these)."""
        path = os.path.expanduser(self.global_config.train.data or "")
        if not os.path.exists(path):
            return
        with h5py.File(path, "r") as f:
            at = f["data"].attrs
            if "grip_event_mu" not in at:
                return
            assert json.loads(at["grip_event_obs_keys"]) == self.event_keys, "event obs keys differ from the dataset stats"
            net = self.nets["event"]
            net.mu.copy_(torch.as_tensor(at["grip_event_mu"], dtype=torch.float32))
            net.sd.copy_(torch.as_tensor(at["grip_event_sd"], dtype=torch.float32))
            net.pos_weight.copy_(torch.as_tensor(at[f"{self.algo_config.event.label_key}_pos_weight"], dtype=torch.float32))

    def _flat(self, obs):
        return torch.cat([obs[k].reshape(*obs[k].shape[: obs[k].dim() - len(self.obs_shapes[k])], -1) for k in self.event_keys], -1)

    # ---------- training ----------
    def process_batch_for_training(self, batch):
        input_batch = super().process_batch_for_training(batch)
        extra = {"grip_state": batch["grip_state"][..., 0], "grip_event": batch[self.algo_config.event.label_key][..., 0]}
        input_batch.update(TensorUtils.to_float(TensorUtils.to_device(extra, self.device)))
        return input_batch

    def _forward_training(self, batch):
        dists = self.nets["policy"].forward_train(obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        assert len(dists.batch_shape) == 2  # [B, T]
        x = self._flat(batch["obs"])
        # dx needs the previous step, so the first step of each sequence has no event prediction
        logits = self.nets["event"](x[:, 1:], x[:, 1:] - x[:, :-1], batch["grip_state"][:, 1:])
        return OrderedDict(log_probs=dists.log_prob(batch["actions"][..., : self.ac_dim - 1]), event_logits=logits)

    def _compute_losses(self, predictions, batch):
        losses = super()._compute_losses(predictions, batch)
        y, s = batch["grip_event"][:, 1:], batch["grip_state"][:, 1:].long()
        w = torch.where(y > 0, self.nets["event"].pos_weight[s], torch.ones_like(y))
        losses["event_loss"] = (F.binary_cross_entropy_with_logits(predictions["event_logits"], y, reduction="none") * w).mean()
        return losses

    def _train_step(self, losses):
        info = super()._train_step(losses)
        info["event_grad_norms"] = TorchUtils.backprop_for_loss(
            net=self.nets["event"], optim=self.optimizers["event"], loss=losses["event_loss"])
        return info

    def log_info(self, info):
        log = super().log_info(info)
        log["Event_Loss"] = info["losses"]["event_loss"].item()
        return log

    # ---------- rollout ----------
    def _reset_gripper(self):
        self._closed, self._run, self._prev_x, self.last_event_p = False, 0, None, float("nan")
        self._truck_z0, self._lifted, self._released = None, False, False  # close-rule phase tracking

    def reset(self):
        super().reset()
        self._reset_gripper()

    @torch.no_grad()
    def event_prob(self, x, dx, state):
        return torch.sigmoid(self.nets["event"](x, dx, state))

    def get_action(self, obs_dict, goal_dict=None):
        arm = super().get_action(obs_dict, goal_dict=goal_dict)  # [1, 6], keeps the RNN schedule of BC_RNN
        x = self._flat(obs_dict)
        dx = torch.zeros_like(x) if self._prev_x is None else x - self._prev_x
        self._prev_x = x
        state = torch.full(x.shape[:-1], float(self._closed), device=x.device)
        p = float(self.event_prob(x, dx, state).reshape(-1)[0])
        if self.close_rule is not None:
            z = float(obs_dict["toy_truck_positions"].reshape(-1)[2])
            self._truck_z0 = z if self._truck_z0 is None else self._truck_z0
            self._lifted = self._lifted or (self._closed and z - self._truck_z0 > 0.02)
        if self.close_rule is not None and not self._closed and not self._released:
            ge = obs_dict["grasp_err_ee"].reshape(-1)[:3]  # EE frame, m (obs are not normalized: hdf5_normalize_obs false)
            hit = float(torch.linalg.norm(ge[:2])) < self.close_rule[0] and float(ge[2]) < self.close_rule[1]
        else:
            hit = p > self.event_tau[int(self._closed)]
        self._run = self._run + 1 if hit else 0
        if self._run >= self.algo_config.event.consecutive:
            self._released = self._released or (self._closed and self._lifted)
            self._closed, self._run = not self._closed, 0
        self.last_event_p = p
        grip = torch.full((*arm.shape[:-1], 1), -1.0 if self._closed else 1.0, device=arm.device, dtype=arm.dtype)
        if self.action_frame == "ee":
            arm = ee_to_base(arm, obs_dict["eef_quat"])
        return torch.cat([arm, grip], -1)


def quat_to_mat(q):
    """(w, x, y, z) -> rotation matrix, shape (..., 3, 3)."""
    q = q / torch.linalg.norm(q, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def ee_to_base(arm, eef_quat):
    """Arm action [dp_E, w_E] in EE axes -> [R dp_E, R w_E] in base axes (R = EE orientation in the base frame), as
    place/mdp/ee_frame_actions.py does inside the ...-EEAct-v0 task."""
    R = quat_to_mat(eef_quat.reshape(-1, 4).to(arm.dtype))  # [1, 3, 3]
    out = arm.clone()
    out[..., 0:3] = (R @ arm[..., 0:3].reshape(-1, 3, 1)).reshape(arm[..., 0:3].shape)
    out[..., 3:6] = (R @ arm[..., 3:6].reshape(-1, 3, 1)).reshape(arm[..., 3:6].shape)
    return out


def apply_event_tau(policy, ckpt_path):
    """Use the calibrated thresholds next to the run (<run>/event_tau.json) if this is a bc_event policy. Returns tau or None."""
    algo = getattr(policy, "policy", policy)
    if not hasattr(algo, "event_tau"):
        return None
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ckpt_path))), "event_tau.json")
    if os.path.exists(path):
        with open(path) as f:
            algo.event_tau = list(json.load(f)["tau"])
    return algo.event_tau


def apply_close_rule(policy, rule):
    """Replace the close head by the geometric rule (xy_max, z_max) in m if this is a bc_event policy. Returns the rule or None."""
    algo = getattr(policy, "policy", policy)
    if not hasattr(algo, "close_rule") or rule is None:
        return None
    algo.close_rule = tuple(float(v) for v in rule)
    return algo.close_rule
