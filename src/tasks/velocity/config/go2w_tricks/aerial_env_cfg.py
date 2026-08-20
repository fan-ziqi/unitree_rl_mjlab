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

  # Body, thighs, and every other non-wheel link remain illegal through the
  # shared base configuration.  This extra terminal only caps a large,
  # collision-driven joint sweep; it does not choose a pose or a trajectory.
  cfg.terminations["leg_excursion"] = TerminationTermCfg(
    func=trick_rewards.aerial_leg_excursion_exceeded,
    params={
      "command_name": "trick",
      # V54 used the entire 0.65-rad envelope and converted limb sweep into
      # collision-driven rotation.  A 0.55-rad measured bound still leaves a
      # substantial dynamic stroke but rejects that non-wheel landing route.
      "max_deviation": 0.55,
      "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS),
    },
  )
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

  # Result-space objectives: get airborne, turn in the requested signed
  # direction, then recover to a braked normal-wheel touchdown.  The recovery
  # potential is gated by *measured accumulated rotation*, never time or an
  # actor-side phase, so it does not prescribe a flip trajectory.
  cfg.rewards = {
    "idle_four_wheel_default_stand": RewardTermCfg(
      func=trick_rewards.aerial_idle_four_wheel_stand_exp,
      weight=120.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "gravity_std": 0.20,
        "linear_velocity_std": 0.12,
        "angular_velocity_std": 0.35,
      },
    ),
    "idle_default_joint_pos": RewardTermCfg(
      func=trick_rewards.aerial_idle_default_joint_pos_exp,
      weight=100.0,
      params={
        "command_name": "trick",
        "std": 0.10,
        "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS),
      },
    ),
    "idle_action_effort": RewardTermCfg(
      func=trick_rewards.aerial_idle_action_l2,
      # The all-zero event command means exactly the reset controller, not a
      # nearby asymmetric stance.  This applies no constraint once a maneuver
      # one-hot is active, so the aerial motion remains fully discovered by
      # PPO rather than being shaped by a joint trajectory.
      weight=-40.0,
      params={"command_name": "trick"},
    ),
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
    # A strict four-wheel success is too sparse to teach braking.  Use one
    # bounded state-potential instead of overlapping touchdown Gaussians: it
    # pays only new progress toward the real outcome (near-target rotation,
    # upright attitude, reduced momentum, descent, and wheel contact).
    "landing_recovery_progress": RewardTermCfg(
      func=trick_rewards.AerialLandingRecoveryProgress,
      weight=250.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "recovery_start_angle": 0.75 * math.tau,
        "target_angle": math.tau,
        "max_overrotation": 1.25,
        "descent_distance": 0.35,
        "wheel_contact_weight": 0.80,
        # The prior 30-rad/s scale still treated a 5-rad/s touchdown as 83%
        # recovered, although strict completion requires 1.5 rad/s.  This
        # is only an outcome scale, never a requested rotation-rate command.
        "max_axis_rate": 12.0,
        "max_linear_speed": 3.0,
      },
    ),
    # A partial wheel touchdown is not enough: the public event completes
    # only after a continuous strict normal-wheel landing.  This adds a small
    # per-step outcome credit inside exactly that state so PPO prefers holding
    # it rather than immediately falling out of a near-complete recovery.
    # It contains no target joint configuration or timed reference phase.
    "strict_landing_hold": RewardTermCfg(
      func=trick_rewards.aerial_strict_landing_hold,
      weight=80.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "max_overrotation": 1.25,
      },
    ),
    "late_axis_spin": RewardTermCfg(
      func=trick_rewards.aerial_late_axis_rate_abs,
      # V60 reaches an upright wheel landing most often in yaw, but then
      # continues through the one-turn window.  Penalize only measured
      # residual angular momentum after 90% of the requested angle, leaving
      # takeoff and the first 90% of ballistic discovery completely free.
      # This is an outcome-space braking objective, not a phase, a pose, or a
      # mode-specific target rate exposed to the policy.
      weight=-12.0,
      params={
        "command_name": "trick",
        "start_angle": 0.90 * math.tau,
        "max_angle": math.tau + 1.25,
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
        "max_overrotation": 1.25,
      },
    ),
    "compact_leg_motion": RewardTermCfg(
      func=trick_rewards.aerial_airborne_joint_excursion_l2,
      weight=-14.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        # Let a compact flip use continuous small hip/knee correction instead
        # of snapping every joint back to its default as soon as it is airborne.
        # The separate 0.55-rad measured-state envelope still rejects the
        # large collision-driven sweeps that are visually and physically wrong.
        "free_deviation": 0.20,
        "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS),
        # Takeoff and landing need a physically discovered leg stroke.  Only
        # penalise flailing while wheel-free, where compactness is visible and
        # does not suppress recovery motion.
        "airborne_only": True,
      },
    ),
    # Aerial motion needs impulse, but not a sequence of saturated target
    # reversals.  This is the only smoothness preference; it carries no pose,
    # time, or direction-dependent reference.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.06),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  # There is deliberately no reward curriculum: every command is one full
  # turn from the first sample and all five events remain equally likely.
  cfg.curriculum = {}
  return cfg
