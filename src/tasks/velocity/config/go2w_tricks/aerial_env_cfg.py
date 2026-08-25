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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_JOINTS,
)
from src.tasks.velocity.mdp import trick_curriculums, trick_rewards
from src.tasks.velocity.mdp.terminations import (
  AerialEventFinished,
  AerialIncompleteLanding,
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
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }
  cfg.episode_length_s = 3.5 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": AerialRotationCommandCfg(
      entity_name="robot",
      # All-zero remains a real public idle command: the action gate makes it
      # the ordinary four-wheel default pose and zero wheel speed, including
      # after a landing.  It is not an aerial skill, so do not spend rollout
      # samples on a condition whose output is already deterministic.  Every
      # training event is consequently one of the five requested flips.
      idle_probability=0.0,
      # The four somersault axes dominate sampling, but retain a small yaw
      # fraction as a shared takeoff-discovery bridge.  With yaw removed
      # entirely, PPO never found a first wheel-free launch in the harder
      # four branches; at equal 20% sampling yaw then monopolized updates.
      mode_probabilities=(0.2375, 0.2375, 0.2375, 0.2375, 0.05),
      resampling_time_range=(3.5, 3.5),
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
      # Keep the launch heading visually recognizable after touchdown without
      # treating a few degrees of contact noise as a failed demonstration.
      # abs(dot(q_land, q_launch)) = .985 admits about 20 degrees of total
      # rotation error while still rejecting an obviously changed heading.
      landing_orientation_dot_min=0.985,
      # Two control frames reject a contact glitch but let a genuine compact
      # wheel-free hop receive the same angle signal that later grows into a
      # full aerial turn.  The endpoint remains exactly one 2π event.
      min_ballistic_time=0.04,
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
    # A ten-frame public proprioceptive history was the smallest window that
    # already discovered front/side/yaw turns together.  The longer 20-frame
    # variant improved one yaw landing but diluted the same fixed-size MLP and
    # collapsed the other four one-hots before they discovered a turn.  Keep
    # phase inference entirely in the normal observation history; the strict
    # endpoint below still evaluates the complete launch-frame orientation.
    cfg.observations[group_name].history_length = 10

  # A flip is a legal four-wheel launch, commanded-axis spin during that
  # launch, net commanded-axis angle, and a quiet normal landing.  There is
  # deliberately no pose, timing, joint, clearance, or reference trajectory
  # reward.  A partial touchdown is a failure shaped only by missing angle.
  cfg.rewards = {
    "ballistic_launch": RewardTermCfg(
      func=trick_rewards.AerialBallisticLaunch,
      # One compact physical launch outcome: upward impulse while all four
      # wheels push, then the continuous wheel-free interval it produces.
      # A 1,600-environment physical action sweep measures 1.87 m/s and
      # 0.26 s as attainable compact-launch baselines under the native Go2W
      # limits.  Keep targets just below those measured values so PPO has a
      # dense path to a real launch instead of an unreachable saturated goal.
      weight=300.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_upward_speed": 1.75,
        "target_duration": 0.22,
      },
    ),
    "net_rotation_progress": RewardTermCfg(
      func=trick_rewards.AerialNetRotationProgress,
      # This is the direct maneuver objective: each additional legal radian
      # is valuable.  It intentionally outweighs a merely high instantaneous
      # rate, so a one-frame angular-velocity spike cannot replace a sustained
      # one-turn rotation.
      weight=180.0,
      params={
        "command_name": "trick",
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_angle": math.tau,
      },
    ),
    "landing_recovery": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      # Once the already-measured turn approaches 2π, give PPO a continuous
      # outcome route to bleed angular momentum while its full orientation
      # returns to launch.  It is deliberately one landing-quality term, not
      # a joint target or a reference landing trajectory.
      weight=250.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_angle": math.tau,
        "landing_gravity_std": 0.30,
        "landing_orientation_dot_min": 0.985,
        "max_overrotation": 0.50,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
        "post_idle_settle_time": 0.30,
        "recovery_start_fraction": 0.75,
        "recovery_angular_speed": 8.0,
      },
    ),
    # Any non-timeout terminal result pays once, in proportion to its missing
    # desired-axis angle.  It keeps every partial landing invalid without
    # making a stronger, correctly directed jump indistinguishable from a
    # zero-turn fall during PPO discovery.
    "event_failure": RewardTermCfg(
      func=trick_rewards.aerial_event_failure,
      weight=-200.0,
      params={
        "command_name": "trick",
        "target_angle": math.tau,
        # A nearly-complete turn that lands its trunk or a leg is still a
        # failure.  Angle-only cost accidentally made that failure free as
        # progress approached 2π, selecting a fast crash instead of wheel
        # recovery.  The completion term remains the only way to erase this
        # base outcome cost.
        "non_timeout_base_cost": 0.30,
      },
    ),
  }
  # Collision ends an episode immediately.  The decisive terminal term
  # distinguishes an illegal partial turn from a wheelward recovery without
  # prescribing how the limbs should execute either action.
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
        "post_idle_settle_time_s": 0.30,
        "target_angle": math.tau,
        "max_overrotation": 0.50,
      },
      time_out=True,
    )
    # A first wheel landing ends the physical aerial attempt.  If it has not
    # reached exactly one allowed turn after the same brief default-idle
    # window, make it a real failure rather than an unpenalized timeout.  The
    # existing generic termination cost then removes the observed profitable
    # 0.2--0.4-turn hop without supplying a pose, timing, or motion reference.
    cfg.terminations["aerial_incomplete_landing"] = TerminationTermCfg(
      func=AerialIncompleteLanding,
      params={
        "command_name": "trick",
        "post_idle_settle_time_s": 0.30,
        "target_angle": math.tau,
        "max_overrotation": 0.50,
      },
    )
  # This is sampling curriculum, not a reference-motion curriculum.  Every
  # emitted command is still one complete 2π event from the first sample.  A
  # small yaw fraction supplies a shared ballistic-launch discovery signal;
  # its probability is held below the four somersault branches so it cannot
  # monopolize the fused actor's PPO updates.
  cfg.curriculum = {
    "aerial_commands": CurriculumTermCfg(
      func=trick_curriculums.aerial_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            # 400 aerial PPO iterations at 48 rollout steps per environment.
            "step": 0,
            "idle_probability": 0.0,
            "mode_probabilities": (0.2375, 0.2375, 0.2375, 0.2375, 0.05),
          },
          {
            # Keep every requested one-hot present throughout discovery.
            # Removing yaw for 700 iterations makes it a stale observation
            # branch rather than a fused five-direction policy, and the fixed
            # audit showed its partial turn then cannot improve.  The small
            # 5% share prevents yaw from dominating while preserving its own
            # PPO evidence and the common launch representation.
            "step": 4_800,
            "idle_probability": 0.0,
            "mode_probabilities": (0.2375, 0.2375, 0.2375, 0.2375, 0.05),
          },
          {
            # Once all branches have the shared launch skill, balance the
            # five one-hots equally for full-turn and landing refinement.
            "step": 38_400,
            "idle_probability": 0.0,
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
          },
        ),
      },
    )
  }
  return cfg
