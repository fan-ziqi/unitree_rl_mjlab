"""Minimal fused PPO environment for Go2W aerial rotations.

The actor receives only the shared proprioceptive history and a five-way
one-hot.  This file intentionally defines only physical validity and a small
set of outcome rewards; it has no reference trajectory, phase clock, or
mode-specific joint posture.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_JOINTS,
)
from src.tasks.velocity.mdp import trick_rewards
from src.tasks.velocity.mdp.trick_commands import AerialRotationCommandCfg

from .common_env_cfg import (
  AERIAL_AXES,
  configure_compact_aerial_actuators,
  make_base_go2w_trick_cfg,
)


def unitree_go2w_aerial_rotation_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train one policy for front, back, left, right, and yaw aerial turns."""
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  configure_compact_aerial_actuators(cfg)
  cfg.episode_length_s = 3.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": AerialRotationCommandCfg(
      entity_name="robot",
      # Every training episode is a requested maneuver.  The all-zero command
      # remains the public idle encoding, but does not dilute aerial discovery.
      idle_probability=0.0,
      resampling_time_range=(3.0, 3.0),
      sensor_name=wheel_contact_cfg.name,
      axes=AERIAL_AXES,
      debug_vis=False,
    )
  }
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=GO2W_LEG_JOINTS,
    scale={
      r".*_hip_joint": 0.45,
      r".*_thigh_joint": 0.55,
      r".*_calf_joint": 0.55,
    },
    use_default_offset=True,
  )
  cfg.actions["joint_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=GO2W_WHEEL_JOINTS,
    scale=45.0,
    use_default_offset=True,
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"
    cfg.observations[group_name].history_length = 10

  # Body, thighs, and every other non-wheel link remain illegal through the
  # shared base configuration.  This extra terminal only caps a large,
  # collision-driven joint sweep; it does not choose a pose or a trajectory.
  cfg.terminations["leg_excursion"] = TerminationTermCfg(
    func=trick_rewards.aerial_leg_excursion_exceeded,
    params={
      "command_name": "trick",
      "max_deviation": 0.55,
      "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS),
    },
  )
  cfg.terminations["rotation_overrun"] = TerminationTermCfg(
    func=trick_rewards.aerial_rotation_overrun,
    params={
      "command_name": "trick",
      "target_angle": math.tau,
      "max_overrotation": 0.75,
      "activation_step": 0,
    },
  )

  # Result-space objectives: get airborne, turn in the requested signed
  # direction, then recover to a braked normal-wheel touchdown.  The recovery
  # signals are gated by *measured accumulated rotation*, never time or an
  # actor-side phase, so they do not prescribe a flip trajectory.
  cfg.rewards = {
    "takeoff_clearance": RewardTermCfg(
      func=trick_rewards.AerialClearanceProgress,
      weight=45.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "min_clearance": 0.28,
      },
    ),
    "rotation_progress": RewardTermCfg(
      func=trick_rewards.AerialRotationProgress,
      weight=28.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": AERIAL_AXES,
        "target_angle": math.tau,
        "clearance_start": 0.06,
        "clearance_full": 0.18,
      },
    ),
    "soft_four_wheel_landing": RewardTermCfg(
      func=trick_rewards.aerial_soft_landing_exp,
      weight=100.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "angle_std": 1.0,
        "gravity_std": 1.0,
        "axis_rate_std": 6.0,
      },
    ),
    # A strict four-wheel success is too sparse to teach the final braking
    # phase.  Once an attempt has physically earned most of a turn, reward a
    # normal-attitude, low-spin descent and then partial wheel touchdown.  A
    # premature body/leg impact is still an immediate illegal-contact failure.
    "post_turn_braked_descent": RewardTermCfg(
      func=trick_rewards.aerial_post_turn_descent,
      weight=60.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": 0.75 * math.tau,
        "max_overrotation": math.tau,
        "gravity_std": 0.60,
        "axis_rate_std": 6.0,
        "descent_speed": 1.5,
      },
    ),
    "post_turn_wheel_touchdown": RewardTermCfg(
      func=trick_rewards.aerial_post_turn_wheel_touchdown_exp,
      weight=80.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": 0.85 * math.tau,
        "max_overrotation": 0.90 * math.tau,
        "gravity_std": 0.60,
        "axis_rate_std": 6.0,
      },
    ),
    "completed_rotation": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      weight=300.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": AERIAL_AXES,
        "target_angle": math.tau,
        "landing_settle_time": 0.10,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
        "max_overrotation": 0.75,
      },
    ),
    "compact_leg_motion": RewardTermCfg(
      func=trick_rewards.aerial_airborne_joint_excursion_l2,
      weight=-20.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "free_deviation": 0.12,
        "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS),
        # Takeoff and landing need a physically discovered leg stroke.  Only
        # penalise flailing while wheel-free, where compactness is visible and
        # does not suppress recovery motion.
        "airborne_only": True,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  # There is deliberately no reward curriculum.  The target stays one full
  # turn from the first sample, and all five commands remain equally likely.
  cfg.curriculum = {}
  return cfg
