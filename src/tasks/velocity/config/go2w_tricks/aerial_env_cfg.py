"""Minimal fused PPO environment for Go2W aerial rotations.

The actor receives only the shared proprioceptive history and a five-way
one-hot.  This file intentionally defines only physical validity and a small
set of outcome rewards; it has no reference trajectory, phase clock, or
mode-specific joint posture.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
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
      # All-zero remains a real public idle command: the action gate makes it
      # the ordinary four-wheel default pose and zero wheel speed, including
      # after a landing.  It is not an aerial skill, so do not spend rollout
      # samples on a condition whose output is already deterministic.  Every
      # training event is consequently one of the five requested flips.
      idle_probability=0.0,
      # All five one-hots remain equally represented in the fused policy.
      mode_probabilities=(0.20, 0.20, 0.20, 0.20, 0.20),
      resampling_time_range=(3.0, 3.0),
      sensor_name=wheel_contact_cfg.name,
      axes=AERIAL_AXES,
      # Completion remains an exact one-turn event even though a failed
      # attempt is allowed to use the rest of its three-second episode to
      # recover instead of being cut off mid-brake.
      target_angle=math.tau,
      # One command means one revolution.  A modest 29-degree tolerance keeps
      # float/contact noise from making discovery binary, but a 450-degree
      # rebound is no longer treated as the same endpoint.
      max_overrotation=0.50,
      # Require five consecutive 50-Hz control steps: a transient wheel graze
      # must never count as a completed normal-wheel landing.
      landing_settle_time=0.10,
      # A flip only passes when its final base orientation is effectively the
      # launch orientation, including yaw.  The continuous landing signal
      # below makes this tight endpoint learnable without a reference motion.
      landing_orientation_dot_min=0.999,
      debug_vis=False,
    )
  }
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=GO2W_LEG_JOINTS,
    scale={
      # This reaches the model's existing hip torque limit at the edge of the
      # residual range.  It neither raises that limit nor stiffens the
      # actuator, but avoids making a one-revolution jump physically
      # unreachable while the calf alone provides the launch impulse.
      r".*_hip_joint": 0.55,
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
      # A complete revolution needs appreciably more ballistic time than the
      # low, fast hops produced by the previous 15:20 height-to-radian ratio.
      # This is still the same measured wheel-free height outcome; it merely
      # gives PPO enough incentive to find the necessary launch.
      weight=100.0,
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
        "target_clearance": 0.45,
      },
    ),
    "completed_turn": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      # A full revolution that returns to quiet four-wheel default control is
      # the task result, not an optional bonus on top of a failed high hop.
      # With the former scale a 0.9-turn body collision retained most of the
      # clearance/rotation return, so PPO correctly preferred it over the
      # extremely rare recovery event.  This remains a single endpoint score,
      # with no reference pose, phase, or limb target.
      weight=300.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "axes": AERIAL_AXES,
        "target_angle": math.tau,
        # A correct one-turn, all-wheel touchdown is an informative precursor
        # to the unchanged five-frame strict settle event.  It supplies no
        # limb pose or time target.
        # Full-turn wheel touch-downs now exist reliably, but V98 still
        # accepts a visibly changed final heading.  Make this existing,
        # once-only physical result materially outweigh another partial
        # airborne radian; it remains gated by turn fraction and legal
        # four-wheel contact below.
        "soft_touchdown_reward": 20.0,
        # Grade that same once-only all-wheel touchdown by how close it is to
        # the existing strict landing velocity/attitude limits.  This is not a
        # new trajectory or phase reward: it only distinguishes a quiet
        # completed turn from a high-speed wheel graze.
        # Keep this discovery bridge broad until the policy reliably samples
        # full-turn wheel landings.  The strict completion below is still the
        # sole acceptance criterion for final orientation and quiet recovery.
        "soft_touchdown_speed_scale": 4.0,
        "landing_gravity_std": 0.30,
        "landing_orientation_dot_min": 0.999,
        # The landing must still pass the strict 0.995 completion threshold,
        # but a legal full-turn wheel touchdown receives a continuous
        # orientation-quality signal all the way from zero similarity.  The
        # former 0.50 floor made every badly aligned but otherwise informative
        # first landing exactly equivalent, so PPO had no return gradient
        # toward restoring the launch heading.
        "soft_touchdown_orientation_floor": 0.0,
        "soft_touchdown_orientation_exponent": 6.0,
        # A partial but real flight may receive a *graded* first-touchdown
        # signal, so orientation recovery is observable before a policy has
        # ever happened to achieve a perfect full turn.  Squaring the turn
        # fraction keeps a small hop far below an almost-complete flip.
        "soft_touchdown_turn_exponent": 2.0,
        "max_overrotation": 0.50,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
        # The strict bonus is paid only if the all-zero/default controller
        # keeps that same landing intact through this final physical window.
        # Ending in a changed heading is a failed flip even if the wheels
        # touched down cleanly for one frame.  Hold default four-wheel idle
        # long enough to make the full-base orientation a sustained result.
        "post_idle_settle_time": 0.60,
        # Score the same physical landing throughout that whole idle window,
        # rather than only on the first contact frame.  This supplies a direct
        # learning signal against a post-touchdown heading drift while keeping
        # the existing strict completion as the acceptance criterion.
        "post_idle_settle_shape_reward": 2.0,
      },
    ),
  }
  # Collision still ends an episode immediately.  Do not charge a terminal
  # scalar here: measured at a102/m100, that made literal idle preferable to
  # every exploratory launch before a single recovery had been sampled.  The
  # dominant completed-turn result above, together with the legal-touchdown
  # bridge, distinguishes recovery from a crash once a physical landing is
  # discovered without prescribing how the limbs should execute it.
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
    # One jump closes after its controlled landing window, then presents the
    # literal all-zero/default controller.  The short idle interval is enough
    # to test that it actually settles without spending an entire 3-s rollout
    # on a finished event.
    cfg.terminations["aerial_event_finished"] = TerminationTermCfg(
      func=AerialEventFinished,
      params={
        "command_name": "trick",
        "post_idle_settle_time_s": 0.60,
      },
      time_out=True,
    )
  # There is deliberately no reward curriculum: every command is one full
  # turn from the first sample and all five events remain equally likely.
  cfg.curriculum = {}
  return cfg
