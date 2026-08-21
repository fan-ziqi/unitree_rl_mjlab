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
  # First learn the maneuver on one nominal flat system.  Sensor bias and
  # broad mass/friction randomization are robustness work for a later stage;
  # they otherwise dilute the rare early takeoff/turn evidence.
  cfg.events.pop("encoder_bias", None)
  cfg.events["foot_friction"].params["ranges"] = (0.7, 1.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }
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
      # Completion remains an exact one-turn event even though a failed
      # attempt is allowed to use the rest of its three-second episode to
      # recover instead of being cut off mid-brake.
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

  # There is intentionally one dense result and one success event.  Earlier
  # versions paid separately for height, partial angle, and a loose recovery
  # score.  A policy could therefore make a high rigid bounce, collect all
  # useful return before landing, and never discover the complete maneuver.
  # ``AerialManeuverResultProgress`` is a single bounded potential: flight
  # contributes only when clearance and signed turn coexist; a four-wheel
  # recovery then adds the larger final value only close to the same full
  # turn.  Its speed scales are intentionally broad dense shaping; the
  # following completion event retains the strict physical acceptance test.
  # No term names a joint pose, limb timing, or demonstration trajectory.
  cfg.rewards = {
    "complete_maneuver_progress": RewardTermCfg(
      func=trick_rewards.AerialManeuverResultProgress,
      # A full turn must rank materially above the stable 0.6--0.8-turn hop
      # observed in V76.  The potential remains bounded; this only restores
      # a useful return gap between a partial flip and the requested result.
      weight=500.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        # Measured root clearance above the literal four-wheel default.  It
        # rules out a wheel-pivot/low-hop exploit but leaves the jump geometry
        # entirely to PPO.
        "target_clearance": 0.40,
        "landing_turn_start": 0.75 * math.tau,
        "recovery_linear_speed_scale": 5.0,
        "recovery_angular_speed_scale": 12.0,
      },
    ),
    "completed_rotation": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      weight=2000.0,
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
    # A ballistic over-rotation must rank below controlled recovery.  At the
    # former -50, clearance and one-turn progress made an uncontrolled flip a
    # positive-return local optimum even though it never landed.
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-150.0),
  }
  # There is deliberately no reward curriculum: every command is one full
  # turn from the first sample and all five events remain equally likely.
  cfg.curriculum = {}
  return cfg
