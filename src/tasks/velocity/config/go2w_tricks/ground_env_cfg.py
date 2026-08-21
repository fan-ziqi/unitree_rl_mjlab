"""Lean flat-ground Go2W trick environments.

These two tasks deliberately reward only public command outcomes.  They do
not encode leg lengths, a contact sequence, or an action-space posture.  The
spin task measures only trunk-to-support-wheel clearance so that a visibly
upright two-wheel pose cannot be replaced by a low crouch; PPO discovers every
joint coordination that realizes that physical support geometry.
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
      resampling_time_range=(8.0, 8.0),
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
      weight=3.0,
      params={
        "command_name": "trick",
        "std": 0.35,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 1.0,
      },
    ),
    # This remains a generic temporal smoothness cost—not a free-leg pose
    # target—but now has enough scale to reject the visibly flailing airborne
    # pair once the support and velocity outcomes are already satisfied.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  cfg.curriculum = {}
  return cfg


def unitree_go2w_spin_stance_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Default four-wheel idle plus four signed, coaxial two-wheel pivots.

  The public layout stays ``[normal, front, rear, left, right, spin_rate]``.
  Zero speed is represented by the all-zero default command; any nonzero
  front/rear/left/right request must form its named balancing axle and pivot
  in place.  This preserves one compact policy interface without asking PPO
  to maintain a separate static handstand for every one-hot.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  configure_ground_support_actuators(cfg)
  _configure_fast_discovery(cfg)
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      # ``normal`` is the all-zero default rather than a moving balancing
      # stance.  The four active axle labels are equally sampled.
      mode_probabilities=(0.0, 0.25, 0.25, 0.25, 0.25),
      # A quarter of rollouts retain literal no-speed four-wheel idle; every
      # remaining rollout is an actual signed pivot request.
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
      # The only non-idle reward: full return needs the named pair, a tall
      # body, co-linear wheel axles, rate tracking, and a local pivot centre.
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
        "asset_cfg": _support_wheels(),
      },
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  cfg.curriculum = {}
  return cfg
