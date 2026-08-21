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

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_JOINTS,
)
from src.tasks.velocity.mdp import trick_rewards
from src.tasks.velocity.mdp.trick_commands import AerialRotationCommandCfg

from .common_env_cfg import (
  AERIAL_AXES,
  configure_compact_aerial_actuators,
  configure_default_idle_actions,
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
      # All-zero is a real public idle command: it means the ordinary
      # four-wheel default pose, not a residual flip/landing state.  Retain a
      # small but explicit share of it during training so this transition is
      # learned rather than being left undefined after a completed maneuver.
      idle_probability=0.12,
      # Five equally exposed events keep the one fused actor from spending
      # most of its capacity on the mechanically easiest pitch directions.
      # The former front/back bias left the negative side flip under-trained.
      mode_probabilities=(0.20, 0.20, 0.20, 0.20, 0.20),
      resampling_time_range=(3.0, 3.0),
      sensor_name=wheel_contact_cfg.name,
      axes=AERIAL_AXES,
      # Keep the command-completion state machine on exactly the same
      # one-turn braking window as the terminal guard and landing reward
      # below.  Otherwise a physically valid late touchdown can be rewarded
      # while the command silently rejects it as an over-rotation.
      target_angle=math.tau,
      max_overrotation=1.25,
      # Require five consecutive 50-Hz control steps: a transient wheel graze
      # must never count as a completed normal-wheel landing.
      landing_settle_time=0.10,
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
  configure_default_idle_actions(
    cfg,
    command_name="trick",
    # The all-zero event vector is the public aerial idle, not mode zero.
    idle_mode_index=None,
    stationary_command_start_index=0,
    command_deadband=0.5,
    idle_contact_sensor_name=wheel_contact_cfg.name,
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"
    cfg.observations[group_name].history_length = 10

  cfg.terminations["rotation_overrun"] = TerminationTermCfg(
    func=trick_rewards.aerial_rotation_overrun,
    params={
      "command_name": "trick",
      "target_angle": math.tau,
      # At the measured 3--7 rad/s residual rate, the former 0.75-rad
      # (43-degree) guard ended an otherwise upright maneuver before the
      # 0.10-s wheel-settle criterion could physically be observed.  This is
      # still a one-turn task: it only gives the recovery controller a
      # 72-degree braking window, and the upright recovery potential remains
      # maximized at exactly one turn.
      "max_overrotation": 1.25,
      "activation_step": 0,
    },
  )

  # Four outcome rewards only: clear the floor, turn in the commanded signed
  # direction, improve the final measured recovery, and complete a held
  # four-wheel landing.  Idle is an actuator invariant, so it receives no
  # reward; leg strokes are unconstrained except for non-wheel ground contact.
  cfg.rewards = {
    "takeoff_clearance": RewardTermCfg(
      func=trick_rewards.AerialClearanceProgress,
      weight=20.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "min_clearance": 0.28,
      },
    ),
    "rotation_progress": RewardTermCfg(
      func=trick_rewards.AerialRotationProgress,
      weight=30.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": AERIAL_AXES,
        "target_angle": math.tau,
        "clearance_start": 0.06,
        "clearance_full": 0.18,
      },
    ),
    "landing_recovery_progress": RewardTermCfg(
      func=trick_rewards.AerialLandingRecoveryProgress,
      weight=60.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "recovery_start_angle": 0.75 * math.tau,
        "target_angle": math.tau,
        "max_overrotation": 1.25,
        "descent_distance": 0.35,
        "wheel_contact_weight": 0.80,
        "max_axis_rate": 12.0,
        "max_linear_speed": 3.0,
      },
    ),
    "completed_rotation": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      weight=600.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": AERIAL_AXES,
        "target_angle": math.tau,
        "landing_settle_time": 0.10,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
        "max_overrotation": 1.25,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  # There is deliberately no reward curriculum: every command is one full
  # turn from the first sample and all five events remain equally likely.
  cfg.curriculum = {}
  return cfg
