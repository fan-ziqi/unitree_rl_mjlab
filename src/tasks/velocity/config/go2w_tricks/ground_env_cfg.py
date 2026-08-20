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
# Reward-manager resolution stores IDs on this mutable config object.  Keep a
# separate identical selector for the root-clearance term rather than reusing
# ``_WHEEL_SITES`` after the free-wheel term has resolved it.
_ROOT_CLEARANCE_WHEEL_SITES = SceneEntityCfg(
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
      # The m399 audit showed a stable 50--55 degree lean with the desired
      # wheel pair down.  It is a useful discovery waypoint, but not the
      # requested handstand, so make the final attitude decisively preferable.
      weight=75.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "power": 8.0,
        "num_modes": 3,
      },
    ),
    # The normal one-hot is a first-class commanded stance, not a fall-back
    # idle behaviour.  Give its all-wheel attitude the same sharp terminal
    # signal as front/rear, otherwise the two handstand-only terms dominate
    # the shared actor and it learns to roll through a zero-speed normal
    # request.
    "normal_gravity_precision": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=45.0,
      params={
        "command_name": "trick",
        "modes": (0,),
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
    "normal_four_wheel_precision": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
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
      weight=70.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "minimum_lengths": (0.0, 0.35, 0.35),
        "activation_lengths": (0.0, 0.16, 0.16),
        "length_power": 2.0,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        # The old 0.90 gate lay beyond the observed low lean (about 0.89),
        # making the extension result completely invisible at its local
        # optimum.  This is a state threshold rather than a trajectory cue:
        # it starts scoring the real support legs only once the body has
        # already made most of the requested rotation.
        "minimum_gravity_alignment": 0.75,
        "num_modes": 3,
        "asset_cfg": _SUPPORT_LEG_GEOMETRY,
      },
    ),
    "two_wheel_root_clearance": RewardTermCfg(
      func=trick_rewards.mode_support_wheel_root_clearance_min_exp,
      weight=70.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        # These are measured wheel-to-root clearances for the physical Go2W
        # support geometry, not a joint target.  They reject the low diagonal
        # prop seen in evaluation while leaving every sufficiently extended
        # leg configuration equally valid.
        "minimum_clearances": (0.0, 0.30, 0.25),
        "std": 0.10,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 2.0,
        "minimum_gravity_alignment": 0.75,
        "asset_cfg": _ROOT_CLEARANCE_WHEEL_SITES,
      },
    ),
    "track_x": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_exp,
      weight=30.0,
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
      weight=20.0,
      params={
        "command_name": "trick",
        "std": 0.35,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    # The exponential trackers provide sharp precision near a correct command
    # but vanish for the uncontrolled rolling seen in the first audit.  These
    # two absolute errors keep the actor ranking less drift above more drift;
    # they still use only measured root motion and the public x/yaw command.
    "track_x_error": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_abs_error,
      weight=-30.0,
      params={
        "command_name": "trick",
        "lateral_weight": 2.0,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "track_yaw_error": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_abs_error,
      weight=-20.0,
      params={
        "command_name": "trick",
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    # Zero x/yaw is a meaningful command in every one-hot.  The ordinary
    # trackers cover it but are too weak once a two-wheel reward has been
    # found; these terms are active only for an actual zero command and only
    # after the requested gravity direction has been reached.  They constrain
    # measured root motion, never a joint position, action, or trajectory.
    "stationary_speed": RewardTermCfg(
      func=trick_rewards.stance_stationary_ground_speed_exp,
      weight=40.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "velocity_deadband": 0.04,
        "std": 0.15,
        "lateral_weight": 2.0,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "stationary_speed_error": RewardTermCfg(
      func=trick_rewards.stance_stationary_ground_speed_abs_error,
      weight=-45.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "velocity_deadband": 0.04,
        "lateral_weight": 2.0,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
        "minimum_gravity_alignment": 0.90,
      },
    ),
    "stationary_angular_speed": RewardTermCfg(
      func=trick_rewards.mode_stationary_root_ang_speed,
      # The fixed normal-mode audit still had a persistent ~0.2 rad/s yaw
      # drift.  This remains inactive for every nonzero x/yaw request.
      weight=-12.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "velocity_deadband": 0.04,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 2.0,
        "minimum_gravity_alignment": 0.90,
      },
    ),
    # The normal four-wheel command has no transition to complete, so it must
    # never trade a correctly upright attitude for a self-propelled roll when
    # x=yaw=0.  Its earlier shared stillness terms were outweighed by posture
    # credit after learning progressed.  These are deliberately normal-only:
    # front/rear remain free to build momentum until their requested two-wheel
    # gravity direction has actually been found.
    "normal_stationary_speed": RewardTermCfg(
      func=trick_rewards.stance_stationary_ground_speed_exp,
      weight=80.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "velocity_deadband": 0.04,
        "std": 0.10,
        "lateral_weight": 2.0,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "normal_stationary_speed_error": RewardTermCfg(
      func=trick_rewards.stance_stationary_ground_speed_abs_error,
      weight=-150.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "velocity_deadband": 0.04,
        "lateral_weight": 2.0,
        "num_modes": 3,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
        "minimum_gravity_alignment": 0.90,
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
      # Front/rear and right static support are already strong.  Keep one
      # fused actor but devote fresh rollouts to the unfinished normal dynamic
      # orbit and left static balance instead of letting easy modes dominate.
      mode_probabilities=(0.32, 0.16, 0.08, 0.30, 0.14),
      # Preserve some normal four-wheel idle rollouts, but let the normal
      # one-hot spend most of its samples on the unfinished dynamic orbit.
      spin_idle_probability=0.30,
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
    "static_two_wheel_gravity_precision": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=65.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "power": 8.0,
      },
    ),
    "static_two_wheel_contact_precision": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": STANCE_CONTACT_MASKS,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    # The normal-mode spin is deliberately posture-free: it asks only for a
    # horizontal body and two current wheel supports, allowing the pair to
    # change over the orbit rather than hard-coding an axis or pose.
    "dynamic_spin_support": RewardTermCfg(
      func=trick_rewards.spin_dynamic_support_exp,
      weight=100.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        # This is deliberately broader than the rate gate below.  From the
        # ordinary normal reset it gives PPO a usable outcome gradient toward
        # a horizontal two-wheel orbit; spin-rate credit remains strict, so a
        # merely tilted four-wheel robot cannot satisfy the command.
        "horizontal_gravity_std": 0.75,
      },
    ),
    "dynamic_spin_horizontal_precision": RewardTermCfg(
      func=trick_rewards.spin_dynamic_horizontal_precision_exp,
      weight=80.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        # The broad support signal above discovers the tilt; this outcome
        # term makes a horizontal trunk materially better than its observed
        # low two-wheel crouch, without choosing a spin plane or axis.
        "std": 0.28,
      },
    ),
    "dynamic_spin_rate": RewardTermCfg(
      func=trick_rewards.spin_dynamic_rate_exp,
      weight=25.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "std": 1.0,
        "horizontal_gravity_std": 0.45,
      },
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
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "std": 1.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "spin_rate_error": RewardTermCfg(
      func=trick_rewards.commanded_spin_rate_abs_error,
      weight=-15.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "dynamic_horizontal_gravity_std": 0.45,
        "fixed_gravity_power": 4.0,
      },
    ),
    "extended_support_legs": RewardTermCfg(
      func=trick_rewards.mode_support_leg_length_min,
      weight=80.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "minimum_lengths": (0.0, 0.35, 0.35, 0.35, 0.35),
        "activation_lengths": (0.0, 0.16, 0.16, 0.16, 0.16),
        "length_power": 2.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "minimum_gravity_alignment": 0.90,
        "asset_cfg": _SUPPORT_LEG_GEOMETRY,
      },
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
