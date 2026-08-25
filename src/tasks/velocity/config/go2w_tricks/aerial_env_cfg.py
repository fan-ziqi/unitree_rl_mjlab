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

  # A flip has one generic launch signal, one airborne angular-momentum
  # signal, one command-specific net-angle result, and one strict terminal
  # landing.  A partial touchdown deliberately earns nothing: it was the
  # source of the yaw-only local optimum in the previous long run.
  cfg.rewards = {
    "takeoff_upward_velocity": RewardTermCfg(
      func=trick_rewards.aerial_takeoff_upward_velocity,
      # Every flip needs a real vertical launch.  Keep this below the
      # command-specific radians reward; it is only a generic discovery route
      # to the already-required wheel-free flight, not an alternate endpoint.
      weight=30.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_upward_speed": 2.0,
      },
    ),
    "airborne_clearance": RewardTermCfg(
      func=trick_rewards.aerial_airborne_clearance,
      weight=20.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_clearance": 0.45,
      },
    ),
    "axis_spin_rate": RewardTermCfg(
      func=trick_rewards.aerial_axis_spin_rate,
      # The m700 audit reaches legal 0.46--0.56 s flights, yet only
      # 0.27--0.42 turns.  Its recorded axis-rate peaks show that the model
      # can create angular momentum but cannot keep it through the flight.
      # This moderate bridge pays only sustained, signed, in-air axis speed
      # below the one-turn target; the much larger net-angle term and strict
      # landing remain the task outcome.  It adds no reference pose, phase,
      # trajectory, or actor observation.
      weight=180.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        # A 2π turn during the observed roughly half-second flight requires
        # about 13 rad/s.  The slightly higher bounded target encourages a
        # useful margin without rewarding unbounded spin.
        "target_axis_speed": 14.0,
        "target_angle": math.tau,
        "target_clearance": 0.45,
      },
    ),
    "net_rotation_progress": RewardTermCfg(
      func=trick_rewards.AerialNetRotationProgress,
      weight=300.0,
      params={
        "command_name": "trick",
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_angle": math.tau,
        "target_clearance": 0.45,
      },
    ),
    "completed_turn": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      weight=75.0,
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
      },
    ),
    # This generic temporal regularizer rejects high-frequency flailing but
    # does not choose a pose, phase, or reference action for the maneuver.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
    # Colliding the trunk or a leg is already a physical failure.  Once the
    # policy has discovered real multi-axis jumps, a decisive terminal cost
    # prevents a high-but-illegal partial turn from competing with a recoverable
    # wheel landing.  Timeouts remain excluded by the standard termination
    # reward function.
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-300.0),
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
      },
      time_out=True,
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
            # The m200 fixed-command audit showed that even a 5% yaw branch
            # can become the shared policy's early high-return shortcut:
            # yaw reached 0.79 turns while every somersault axis remained at
            # 0.24--0.32.  Keep yaw only for the first 100 iterations as a
            # generic launch bridge, then give the four genuinely aerial
            # directions an uninterrupted common-policy discovery interval.
            # This changes sampling only: it preserves the same five-element
            # actor command and never introduces a pose, phase, or reference
            # trajectory.
            "step": 4_800,
            "idle_probability": 0.0,
            "mode_probabilities": (0.25, 0.25, 0.25, 0.25, 0.0),
          },
          {
            # Reintroduce the easy fifth branch only after the four real
            # somersault directions have received a long uninterrupted
            # discovery interval.  The f32 fixed m300 audit still measured
            # only 0.25--0.33 turns on the hard modes; exposing yaw at 400
            # iterations would recreate its proven high-return shortcut
            # before any hard axis had a full turn.  At 48 control steps per
            # PPO iteration, 38,400 is iteration 800 and leaves a matching
            # long five-way interval in the fresh 1,600-iteration run.
            "step": 38_400,
            "idle_probability": 0.0,
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
          },
        ),
      },
    )
  }
  return cfg
