"""Lean flat-ground Go2W trick environments.

These two tasks deliberately reward public command outcomes.  They do not
encode leg lengths, a contact sequence, or an action-space posture for the
two-wheel skills.  The only exception is the user's explicit normal-mode
requirement: four-wheel rolling must remain recognizably close to the Go2W
model default pose.  The spin task measures only trunk-to-support-wheel
clearance so that a visibly upright two-wheel pose cannot be replaced by a low
crouch; PPO discovers every joint coordination that realizes that geometry.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

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
  configure_default_idle_actions,
  configure_ground_support_actuators,
  make_base_go2w_trick_cfg,
)


def _support_wheels() -> SceneEntityCfg:
  """Return a fresh mutable wheel selector for one reward-manager term.

  ``SceneEntityCfg.resolve`` records resolved ids in-place.  Each reward must
  therefore own its selector; reusing one global instance works in a training
  process but can fail when a play/evaluation environment resolves it again.
  """
  return SceneEntityCfg(
    "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
  )


def _leg_joints() -> SceneEntityCfg:
  """Select actuated leg joints but intentionally exclude continuous wheels."""
  return SceneEntityCfg(
    "robot",
    joint_names=(
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
      "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ),
    preserve_order=True,
  )


def _configure_fast_discovery(cfg: ManagerBasedRlEnvCfg) -> None:
  """Keep first-pass PPO on nominal flat physics, not robustness noise."""
  cfg.events.pop("encoder_bias", None)
  cfg.events["foot_friction"].params["ranges"] = (0.7, 1.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }


def _use_history(cfg: ManagerBasedRlEnvCfg, command_name: str) -> None:
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = command_name
    cfg.observations[group_name].history_length = 10


def unitree_go2w_stance_locomotion_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """One policy for normal/front/rear x-yaw locomotion.

  Commands are ``[normal, front, rear, x_velocity, yaw_rate]``.  A normal
  zero-velocity command is hard-gated to default four-wheel idle.  The other
  two one-hots remain ordinary outcome-conditioned two-wheel commands.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  configure_ground_support_actuators(cfg)
  _configure_fast_discovery(cfg)
  cfg.episode_length_s = 8.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceLocomotionCommandCfg(
      entity_name="robot",
      # An 8-s rollout with an 8-s resample interval only ever sees one mode
      # per physical episode.  Re-sample once halfway through so normal,
      # front, and rear are actually trained as a fused switching skill rather
      # than merely evaluated as independent fixed poses.  This changes no
      # observation or target: it simply presents the existing one-hot at a
      # real, non-reset state.
      resampling_time_range=(4.0, 4.0),
      mode_probabilities=(0.30, 0.35, 0.35),
      # Normal rolling is present from update zero.  Front/rear get a mostly
      # static distribution without a hidden reset pose or a timed curriculum.
      mode_idle_probabilities=(0.35, 0.85, 0.85),
      lin_vel_x_range=(-0.20, 0.20),
      yaw_rate_range=(-0.30, 0.30),
      initialize_stance_on_reset=False,
      debug_vis=False,
    )
  }
  configure_default_idle_actions(
    cfg,
    command_name="trick",
    idle_mode_index=0,
    stationary_command_start_index=3,
    command_deadband=0.04,
    idle_contact_sensor_name=wheel_contact_cfg.name,
  )
  _use_history(cfg, "trick")

  cfg.rewards = {
    # This single dense score is deliberately additive: it supplies a useful
    # direction from four wheels toward the requested support without making
    # contact an all-or-nothing gate on the attitude signal.
    "commanded_support": RewardTermCfg(
      func=trick_rewards.mode_support_score,
      weight=8.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "num_modes": 3,
        # Normal four-wheel idle has the natural low clearance of the Go2W
        # default.  A front/rear two-wheel command instead must visibly lift
        # the trunk above its transverse support axle; without this existing
        # support-outcome geometry, V1 found a low front crouch with the right
        # two contacts but not the requested inverted stand.
        "minimum_root_clearance": (0.18, 0.40, 0.40),
        "asset_cfg": _support_wheels(),
      },
    ),
    "track_x_and_zero_lateral": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_exp,
      weight=3.0,
      params={
        "command_name": "trick",
        "std": 0.30,
        "lateral_weight": 2.0,
        # A light physical gate keeps a fallen robot from being paid for a
        # coincidental root velocity; it is not a posture target.
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 1.0,
      },
    ),
    "track_yaw": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_exp,
      # The final m999 audit retained the requested normal form and x
      # tracking, but under-tracked a normal yaw-rate command.  Raise the
      # already existing command-tracking outcome rather than introducing a
      # posture or wheel-action target.
      weight=5.0,
      params={
        "command_name": "trick",
        "std": 0.35,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 1.0,
      },
    ),
    # Four-wheel mode is not merely a legal contact mask: the requirement is
    # the familiar Go2W default silhouette.  This static joint-space distance
    # applies only to normal mode and excludes wheels, so it neither constrains
    # their rolling nor prescribes a front/rear standing trajectory.
    "normal_leg_default_pose": RewardTermCfg(
      func=trick_rewards.normal_leg_default_pose_exp,
      # In four-wheel rolling there is no physical need to bend a leg: wheel
      # velocity alone supplies x/yaw motion.  Give the visual/default-pose
      # requirement comparable scale to contact support so normal locomotion
      # cannot buy a little tracking accuracy by adopting an unrelated squat.
      # This term is identically zero for front/rear commands.
      weight=8.0,
      params={
        "command_name": "trick",
        "std": 0.20,
        "asset_cfg": _leg_joints(),
      },
    ),
    # This remains a generic temporal smoothness cost—not a free-leg pose
    # target—but now has enough scale to reject the visibly flailing airborne
    # pair once the support and velocity outcomes are already satisfied.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.04),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  cfg.curriculum = {}
  return cfg


def unitree_go2w_spin_stance_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Default four-wheel idle plus all five contact-mode commands.

  The public layout stays ``[normal, front, rear, left, right, spin_rate]``.
  Zero command is four-wheel default idle.  A nonzero normal rate requests the
  video's all-wheel in-place yaw spin after front/rear wheel axles are
  co-linear; nonzero front/rear rates request its upright local pivot.
  Left/right select static side-wheel support and deliberately ignore the
  rate channel: the reference does not support inventing a same-side spin.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  configure_ground_support_actuators(cfg)
  _configure_fast_discovery(cfg)
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      # Every one-hot remains in the policy domain.  The command term emits a
      # nonzero spin rate only for normal/front/rear; left/right are static
      # side-support examples in the same fused policy.
      # This is one fused five-direction policy, so no branch may receive
      # preferential exploration mass.  Front/rear used to dominate 56% of
      # samples and produced exactly the observed collapse: those poses made
      # progress while normal/left/right did not.  Keep every public one-hot
      # at the same 20% probability from the first PPO update.
      mode_probabilities=(0.20, 0.20, 0.20, 0.20, 0.20),
      spin_idle_probability=0.25,
      spin_rate_range=(4.0, 8.0),
      spin_rate_ramp_rate=12.0,
      debug_vis=False,
    )
  }
  configure_default_idle_actions(
    cfg,
    command_name="trick",
    # The sampler emits an all-zero vector whenever speed is absent, so this
    # gate gives every zero-speed request the literal four-wheel default.
    idle_mode_index=None,
    stationary_command_start_index=0,
    command_deadband=0.20,
    idle_contact_sensor_name=wheel_contact_cfg.name,
  )
  _use_history(cfg, "trick")

  cfg.rewards = {
    "commanded_spin_pivot": RewardTermCfg(
      func=trick_rewards.StanceSpinPivotResult,
      # One outcome term: normal tracks a four-wheel local yaw spin only once
      # its front/rear axles are co-linear; front/rear track the upright
      # collinear pivot.  Left/right use the same outcome term as static side
      # supports.  This prevents a normal command from collapsing to ordinary
      # differential steering around a floor circle without adding a second
      # policy or a motion reference.
      weight=18.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "std": 8.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        # Circle-running is not an acceptable approximation to the reference
        # pivot.  The support axle midpoint must be nearly stationary.
        "pivot_speed_limit": 0.12,
        "static_angular_velocity_scale": 0.75,
        "asset_cfg": _support_wheels(),
      },
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  cfg.curriculum = {}
  return cfg
