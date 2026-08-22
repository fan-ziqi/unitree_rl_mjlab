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
from mjlab.managers.termination_manager import TerminationTermCfg

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_JOINTS,
)
from src.tasks.velocity.mdp import trick_rewards
from src.tasks.velocity.mdp.terminations import (
  AerialEventFinished,
  AerialPostLandingRelaunch,
)
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
  cfg, wheel_contact_cfg, nonwheel_contact_cfg = make_base_go2w_trick_cfg(play)
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
      # All five one-hots stay equally represented in the same fused policy.
      # Reducing yaw samples did not improve pitch/roll discovery.
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
      # The physical controller becomes default idle at first contact.  Keep
      # the *training* event window short: the 400-ms verdict delayed the
      # already sparse pitch/roll learning signal.  Evaluation extends this
      # same action-gated window to verify passive settling, so one-hot still
      # means exactly one jump in both cases.
      post_landing_hold_time=0.10,
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
    # Aerial commands are one-shot: immediately after their first landing,
    # the specified controller state is four-wheel model-default idle.  A
    # later ballistic interval is separately rejected as a relaunch.
    default_after_first_landing=True,
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"
    cfg.observations[group_name].history_length = 10

  # A flip is deliberately reduced to its observable physical result: gain
  # wheel-free height, accumulate desired-axis radians while airborne, then
  # receive a one-shot bonus for a quiet one-turn four-wheel landing.  There
  # is no pose target, phase clock, action reference, landing potential, or
  # mode-specific limb schedule.
  cfg.rewards = {
    "airborne_clearance": RewardTermCfg(
      func=trick_rewards.aerial_airborne_clearance,
      # A small takeoff signal makes the first part of a flip discoverable,
      # but is far below a turn or completed landing.
      weight=15.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_clearance": 0.45,
      },
    ),
    "net_rotation_progress": RewardTermCfg(
      func=trick_rewards.AerialNetRotationProgress,
      # The actor receives each *net* desired-axis radian once.  Undoing a
      # partial turn and repeating it cannot accumulate reward; only lasting
      # progress toward the requested full turn is valuable.
      weight=20.0,
      params={
        "command_name": "trick",
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_angle": math.tau,
      },
    ),
    "completed_turn": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      # A successful result dwarfs a partial flight, but the binary landing
      # test itself remains exactly the command's physical completion test.
      weight=60.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": AERIAL_AXES,
        "target_angle": math.tau,
        "max_overrotation": 1.25,
        "landing_gravity_std": 0.30,
        "landing_settle_time": 0.10,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
      },
    ),
    # Body/leg contact still terminates the rollout; this moderate scalar cost
    # does not make a safe small hop preferable to discovering a real flip.
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-250.0),
  }
  # The command becomes all-zero after its first landing.  A later genuine
  # wheel-free interval is a second attempt, even if it began as a rebound,
  # and is therefore an event failure rather than a way to continue hopping.
  cfg.terminations["post_landing_relaunch"] = TerminationTermCfg(
    func=AerialPostLandingRelaunch,
    params={
      "command_name": "trick",
      "sensor_name": wheel_contact_cfg.name,
      "min_ballistic_time": 0.08,
    },
  )
  if not play:
    # A one-hot is a single event, not an instruction to remain idle for the
    # unused tail of a fixed three-second rollout.  Command clearing occurs
    # only after the first landing window and the separate result term has
    # received its public-idle settling interval.  Mark this reset as a
    # truncation so PPO bootstraps it normally rather than treating a
    # completed attempt as a fall.  The evaluator deliberately keeps the
    # world alive for its full three-second passive-settling audit.
    cfg.terminations["aerial_event_finished"] = TerminationTermCfg(
      func=AerialEventFinished,
      params={
        "command_name": "trick",
        "post_idle_settle_time_s": 0.20,
      },
      time_out=True,
    )
  # There is deliberately no reward curriculum: every command is one full
  # turn from the first sample and all five events remain equally likely.
  cfg.curriculum = {}
  return cfg
