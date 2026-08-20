"""Minimal outcome-based Go2W ground-trick environments.

There are two fused policies here.  One switches between ordinary four-wheel
locomotion and front/rear two-wheel locomotion.  The other switches among the
five wheel-support modes and follows one spin-rate scalar where that is
physically meaningful.  Neither environment contains a joint-pose reference,
phase clock, or reset into a requested two-wheel stance.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_go2w.go2w_constants import GO2W_LEG_JOINTS
from src.tasks.velocity.mdp import trick_rewards
from src.tasks.velocity.mdp.trick_commands import (
  StanceLocomotionCommandCfg,
  StanceSpinCommandCfg,
)

from .common_env_cfg import (
  LOCOMOTION_CONTACT_MASKS,
  LOCOMOTION_GRAVITY_TARGETS,
  STANCE_CONTACT_MASKS,
  STANCE_GRAVITY_TARGETS,
  make_base_go2w_trick_cfg,
)


_WHEEL_SITES = SceneEntityCfg(
  "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
)
_SUPPORT_LEG_GEOMETRY = SceneEntityCfg(
  "robot",
  body_names=("FL_hip", "FR_hip", "RL_hip", "RR_hip"),
  site_names=("FL", "FR", "RL", "RR"),
  preserve_order=True,
)


def _configure_fast_discovery(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep first-pass PPO on nominal flat physics, not broad robustness noise."""
  cfg.events.pop("encoder_bias", None)
  cfg.events["foot_friction"].params["ranges"] = (0.7, 1.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01), 1: (-0.01, 0.01), 2: (-0.01, 0.01)
  }


def _use_history(cfg: ManagerBasedRlEnvCfg, command_name: str) -> None:
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = command_name
    cfg.observations[group_name].history_length = 10


def unitree_go2w_stance_locomotion_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Normal/front/rear wheel locomotion from one ordinary four-wheel reset.

  Commands are ``[normal, front, rear, x_velocity, yaw_rate]``.  Lateral
  velocity is absent and is always penalised inside the x tracker.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  _configure_fast_discovery(cfg)
  cfg.episode_length_s = 8.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceLocomotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(8.0, 8.0),
      mode_probabilities=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
      idle_probability=0.45,
      lin_vel_x_range=(-0.20, 0.20),
      yaw_rate_range=(-0.30, 0.30),
      # A front/rear request must learn the rise; it may not spawn in a
      # handstand with a hidden supporting-leg configuration.
      initialize_stance_on_reset=False,
      debug_vis=False,
    )
  }
  _use_history(cfg, "trick")

  cfg.rewards = {
    # Body attitude and exact wheel support are the primary mode semantics.
    # The alignment score supplies a gradient from the 90-degree normal reset;
    # the powered copy only sharpens the final balanced stance.
    "mode_gravity": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "num_modes": 3,
      },
    ),
    "two_wheel_gravity_precision": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=45.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "power": 8.0,
        "num_modes": 3,
      },
    ),
    "support_wheels": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "num_modes": 3,
      },
    ),
    # These only describe visible wheel/leg geometry.  They are gated on an
    # already-near-target body state, so they cannot prescribe the rise.
    "free_wheel_clearance": RewardTermCfg(
      func=trick_rewards.mode_non_support_wheel_clearance,
      weight=15.0,
      params={
        "command_name": "trick",
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "minimum_height": 0.18,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 3.0,
        "asset_cfg": _WHEEL_SITES,
      },
    ),
    "extended_support_legs": RewardTermCfg(
      func=trick_rewards.mode_support_leg_length_min,
      weight=25.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "minimum_lengths": (0.0, 0.35, 0.35),
        "activation_lengths": (0.0, 0.16, 0.16),
        "length_power": 2.0,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "minimum_gravity_alignment": 0.90,
        "num_modes": 3,
        "asset_cfg": _SUPPORT_LEG_GEOMETRY,
      },
    ),
    "track_x": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_exp,
      weight=20.0,
      params={
        "command_name": "trick",
        "std": 0.25,
        "lateral_weight": 2.0,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "track_yaw": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_exp,
      weight=12.0,
      params={
        "command_name": "trick",
        "std": 0.35,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "joint_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS)},
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  cfg.curriculum = {}
  return cfg


def unitree_go2w_spin_stance_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Five support one-hots plus one signed spin-rate command.

  The normal one-hot with zero rate is ordinary four-wheel idle.  Its nonzero
  rate requests the dynamic two-wheel Thomas-like orbit.  Front and rear
  handstands may spin about their support direction; left/right are explicitly
  static two-wheel balances, so their rate channel is ignored by sampling.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  _configure_fast_discovery(cfg)
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      mode_probabilities=(0.20, 0.20, 0.20, 0.20, 0.20),
      spin_idle_probability=0.45,
      spin_rate_range=(1.0, 4.0),
      spin_rate_ramp_rate=4.0,
      debug_vis=False,
    )
  }
  _use_history(cfg, "trick")

  cfg.rewards = {
    "four_wheel_idle_gravity": RewardTermCfg(
      func=trick_rewards.stand_idle_gravity_exp,
      weight=18.0,
      params={"command_name": "trick", "speed_deadband": 0.20, "std": 0.35},
    ),
    "four_wheel_idle_contacts": RewardTermCfg(
      func=trick_rewards.stand_idle_contact_match,
      weight=18.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
      },
    ),
    "static_two_wheel_gravity": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "gravity_targets": STANCE_GRAVITY_TARGETS,
      },
    ),
    "static_two_wheel_contacts": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": STANCE_CONTACT_MASKS,
      },
    ),
    # The normal-mode spin is deliberately posture-free: it asks only for a
    # horizontal body and two current wheel supports, allowing the pair to
    # change over the orbit rather than hard-coding an axis or pose.
    "dynamic_spin_support": RewardTermCfg(
      func=trick_rewards.spin_dynamic_support_exp,
      weight=30.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        "horizontal_gravity_std": 0.45,
      },
    ),
    "dynamic_spin_rate": RewardTermCfg(
      func=trick_rewards.spin_dynamic_rate_exp,
      weight=25.0,
      params={"command_name": "trick", "speed_deadband": 0.20, "std": 1.0},
    ),
    "dynamic_support_cycle": RewardTermCfg(
      func=trick_rewards.SpinSupportCycle,
      weight=1.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        "horizontal_gravity_limit": 0.70,
        "min_transition_interval": 0.12,
      },
    ),
    "front_rear_spin_rate": RewardTermCfg(
      func=trick_rewards.fixed_pair_spin_rate_exp,
      weight=25.0,
      params={"command_name": "trick", "speed_deadband": 0.20, "std": 1.0},
    ),
    "spin_planar_drift": RewardTermCfg(
      func=trick_rewards.spin_planar_speed_l2,
      weight=-0.25,
      params={"command_name": "trick", "speed_deadband": 0.20},
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "joint_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-2.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS)},
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  cfg.curriculum = {}
  return cfg
