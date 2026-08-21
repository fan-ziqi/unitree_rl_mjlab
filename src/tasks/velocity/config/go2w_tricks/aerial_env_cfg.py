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
from src.tasks.velocity.mdp.terminations import AerialPostLandingRelaunch
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
      # One event gets one confirmed flight.  After its first return to the
      # ground, hold the public one-hot only for this brief landing decision
      # window, then return it to all-zero idle even if the landing failed.
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
    # Small and medium exploratory hops still need the literal idle stabilizer
    # at first contact.  V103 showed that handing a one-third turn to an
    # untrained landing policy corrupts the already useful takeoff discovery
    # signal.  Only after three quarters of the measured full turn does PPO
    # retain action authority through first touchdown, which is exactly the
    # late impact-absorption/braking interval we need to improve.  This adds
    # no pose, reference timing, or extra observation; command closure and the
    # post-landing-relaunch termination still enforce exactly one jump.
    default_after_first_landing=True,
    default_after_first_landing_before_progress=0.75 * math.tau,
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"
    cfg.observations[group_name].history_length = 10

  # The objective has just two maneuver outcomes.  A discounted potential
  # supplies short-horizon credit only when genuine wheel-free height and the
  # commanded signed rotation improve together; it cancels when that state is
  # thrown away.  The one-shot result then pays only after the first landing
  # survives default idle.  Neither names a joint pose, timing, or reference.
  cfg.rewards = {
    "maneuver_progress": RewardTermCfg(
      func=trick_rewards.AerialManeuverResultProgress,
      # This is potential-based feedback, not an additional completed-event
      # prize.  Its score is removed on a failed touchdown, while the strict
      # first-landing result below remains the durable task objective.
      weight=400.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "target_clearance": 0.40,
        # At m499 the actor reaches 0.65--0.77 turns but carries 15--20 rad/s
        # into the final quadrant.  Start valuing an upright, decelerating
        # outcome early enough for its free PPO control to arrest that motion;
        # this is an outcome condition, not a prescribed braking phase.
        "landing_turn_start": 0.35 * math.tau,
        "recovery_linear_speed_scale": 5.0,
        # Keep a dense ranking signal through the measured 15--20 rad/s
        # region.  The strict first-landing verifier below is unchanged at
        # 1.5 rad/s, so this does not relax what counts as a completed flip.
        "recovery_angular_speed_scale": 20.0,
        "potential_discount": 0.997,
      },
    ),
    "first_landing_result": RewardTermCfg(
      func=trick_rewards.AerialFirstLandingResult,
      # A strictly completed event is worth 5x a perfect graded partial
      # landing, while any body contact loses more than a mediocre result.
      weight=700.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "max_overrotation": 1.25,
        # The dense potential above can reveal partial flight, but the durable
        # landing result must not let a repeatable low hop outrank a nearly
        # complete maneuver.
        "turn_exponent": 2.0,
        "target_clearance": 0.40,
        "landing_window_s": 0.10,
        # Verify the result under the all-zero/default idle command before
        # paying it.  This turns any immediate rebound into a failure rather
        # than a profitable second hop.
        "post_idle_settle_time_s": 0.20,
        "recovery_linear_speed_scale": 5.0,
        "recovery_angular_speed_scale": 20.0,
        "landing_gravity_error_limit": 0.30,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
        "strict_completion_bonus": 4.0,
      },
    ),
    # With dt-scaled rewards, -200 is only a -4 event cost and lets a crashing
    # partial turn outrank the explicit no-body-support validity rule.  The
    # convex landing result above makes -500 sufficient without freezing
    # exploration into a no-risk small hop.
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-500.0),
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
  # There is deliberately no reward curriculum: every command is one full
  # turn from the first sample and all five events remain equally likely.
  cfg.curriculum = {}
  return cfg
