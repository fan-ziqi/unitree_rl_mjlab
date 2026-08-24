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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.tasks.velocity.mdp import trick_curriculums, trick_rewards
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
      # Establish the two upright supports from the required normal four-wheel
      # reset before asking for command changes inside a rollout.
      resampling_time_range=(8.0, 8.0),
      # Normal four-wheel rolling is action-gated to the default pose, so
      # early samples concentrate on the two support forms.  The prior 50/50
      # stationary/moving split switched to short commands before the front
      # form had ever been found; keep most of these discovery samples static.
      mode_probabilities=(0.05, 0.65, 0.30),
      mode_idle_probabilities=(0.0, 0.70, 0.65),
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
    # Normal x/yaw locomotion should roll on the wheels with the same compact
    # four-leg silhouette as the untriggered Go2W, rather than spending policy
    # capacity on an unnecessary squat or leg swing.
    hold_default_position_mode_index=0,
    # Normal is explicitly the model-default leg pose even while wheels move.
    # A missing reset-time contact frame otherwise releases that invariant and
    # produces the unwanted squatting normal gait.
    hold_default_position_requires_physical_idle=False,
  )
  _use_history(cfg, "trick")

  cfg.rewards = {
    # This single dense score is deliberately additive: it supplies a useful
    # direction from four wheels toward the requested support without making
    # contact an all-or-nothing gate on the attitude signal.
    "commanded_support": RewardTermCfg(
      func=trick_rewards.mode_support_score,
      # The m400 policy genuinely finds the commanded two-wheel contacts,
      # but visual replay shows a low folded support.  Put the same measured
      # outcome ahead of velocity tracking until it is a visibly extended
      # stand; this adds neither a joint target nor a transition schedule.
      weight=50.0,
      params={
        "command_name": "trick",
        # Normal's default four-wheel support is supplied by the action gate.
        # Paying the same support return there let it dominate the harder
        # upright modes before either had found a valid two-wheel result.
        # Normal remains trained through its x/yaw commands below; this term
        # is reserved for the two outcomes that actually need discovery.
        "modes": (1, 2),
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "num_modes": 3,
        # Normal four-wheel idle has the natural low clearance of the Go2W
        # default.  A front/rear two-wheel command instead must visibly lift
        # the trunk above its transverse support axle; without this existing
        # support-outcome geometry, V1 found a low front crouch with the right
        # two contacts but not the requested inverted stand.
        # The m400 fixed-command audit measures only 0.23 m of true
        # trunk-to-support-axle clearance.  Requiring 0.62 m *and* squaring
        # it leaves the normal-reset-to-upright discovery gradient almost
        # zero.  The spin task has independently demonstrated that 0.45 m is
        # a physically reachable, visibly extended Go2W two-wheel support.
        # This remains a measured geometry outcome, not a leg pose target.
        "minimum_root_clearance": (0.18, 0.45, 0.45),
        # Keep a direct monotonic incentive from the current low support all
        # the way to that extended physical clearance.  Exact contacts and
        # body attitude remain multiplicative validity requirements below.
        "orientation_power": 4.0,
        "clearance_power": 1.0,
        "asset_cfg": _support_wheels(),
      },
    ),
    "track_x_and_zero_lateral": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_exp,
      # Support discovery is the prerequisite for useful public x control.
      # Before a legal two-wheel support exists, a velocity error mostly
      # encourages a fast four-wheel escape.  Keep x response in this single
      # policy, but make the physical support result the dominant discovery
      # return; command tracking remains active throughout the same rollout.
      weight=30.0,
      params={
        "command_name": "trick",
        "std": 0.45,
        "lateral_weight": 2.0,
        # A light physical gate keeps a fallen robot from being paid for a
        # coincidental root velocity; it is not a posture target.
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        # A front/rear command must not earn most of its velocity return by
        # simply rolling in the ordinary four-wheel attitude.  At the normal
        # pose its target-gravity alignment is 0.5; cubing the existing
        # outcome gate reduces that bypass to 0.125 while preserving a smooth
        # signal toward the requested two-wheel attitude.  No joint pose or
        # transition timing is prescribed.
        # With the strengthened x/yaw weights, a cubic validity gate still
        # lets an ordinary four-wheel front/rear request earn enough velocity
        # return to avoid standing up.  An eighth power makes that bypass
        # negligible at the reset attitude yet leaves a strong command signal
        # once the measured two-wheel orientation is genuinely established.
        "gravity_power": 8.0,
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
      },
    ),
    "track_yaw": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_exp,
      # Yaw stays in the same outcome-conditioned policy, but cannot displace
      # the still-undiscovered upright support.
      weight=30.0,
      params={
        "command_name": "trick",
        "std": 0.60,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        # Normal needs the stronger yaw signal: its default four-wheel
        # support is already valid, so no stance discovery competes with
        # turning.  Front/rear retain the previous effective scale until
        # their two-wheel form is established, preventing a normal-pose yaw
        # response from displacing the requested inverted support.
        "mode_weights": (1.0, 1.0, 1.0),
        # Apply the same mode-validity gate to yaw, otherwise rear mode can
        # collect yaw return while visibly remaining a normal wheeled robot.
        "gravity_power": 8.0,
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
      },
    ),
    # This remains a generic temporal smoothness cost—not a free-leg pose
    # target—but now has enough scale to reject the visibly flailing airborne
    # pair once the support and velocity outcomes are already satisfied.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.04),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  # This changes only which already-public commands are sampled.  The first
  # a143 audit established that roughly 200 PPO updates are enough to find a
  # legal front/rear support, while keeping the 50% static share thereafter
  # let the x response collapse back to zero.  Once support is discoverable,
  # expose more nonzero x/yaw requests without introducing a pose, phase, or
  # a separate controller.
  cfg.curriculum = {
    "locomotion_commands": CurriculumTermCfg(
      func=trick_curriculums.stance_locomotion_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            "step": 0,
            "mode_probabilities": (0.05, 0.65, 0.30),
            "mode_idle_probabilities": (0.0, 0.70, 0.65),
            "lin_vel_x_range": (-0.20, 0.20),
            "yaw_rate_range": (-0.30, 0.30),
            "resampling_time_range": (8.0, 8.0),
          },
          {
            # Add moving samples only after both supports have long-horizon
            # balance practice; the actor and its public command do not change.
            "step": 38_400,
            "mode_probabilities": (0.15, 0.45, 0.40),
            "mode_idle_probabilities": (0.0, 0.40, 0.40),
            "lin_vel_x_range": (-0.20, 0.20),
            "yaw_rate_range": (-0.30, 0.30),
            "resampling_time_range": (8.0, 8.0),
          },
          {
            # Train normal/front/rear transitions only after the individual
            # normal-reset-to-support outcomes have a full episode to settle.
            "step": 76_800,
            "mode_probabilities": (0.40, 0.30, 0.30),
            "mode_idle_probabilities": (0.0, 0.25, 0.25),
            "lin_vel_x_range": (-0.20, 0.20),
            "yaw_rate_range": (-0.30, 0.30),
            "resampling_time_range": (2.0, 3.0),
          },
        ),
      },
    )
  }
  return cfg


def unitree_go2w_spin_stance_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Default four-wheel idle plus all five contact-mode commands.

  The public layout stays ``[normal, front, rear, left, right, spin_rate]``.
  Zero command is four-wheel default idle.  A nonzero normal rate requests the
  video's two-wheel, front/rear-co-axial local pivot; front/rear one-hots
  request their named two-wheel local pivots.  The
  left/right one-hots are the physically distinct, static side supports: once
  side-on, their wheel axes are vertical, so their spin-rate input is ignored.
  All five modes still share exactly the same one-hot plus signed-rate command
  interface and one policy.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  configure_ground_support_actuators(cfg)
  _configure_fast_discovery(cfg)
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      # First discover the reference's *normal* folded four-wheel geometry
      # from the ordinary default reset.  It is the common prerequisite for
      # a high-rate yaw pivot and the later switch to the other one-hots; an
      # early mixture had only 50% normal samples and instead converged on a
      # four-wheel stepping turn.
      mode_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
      spin_idle_probability=0.0,
      # The physical layout is found before asking its first pass to sustain
      # the final maximum rate.  This is a range curriculum over the existing
      # public spin-rate channel, not an added phase or pose command.
      spin_rate_range=(1.0, 2.5),
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
      # Normal is a folded four-wheel common-axis pivot, not wheel-steering.
      weight=18.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "std": 8.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "pivot_speed_limit": 0.12,
        "upright_support_weight": 0.20,
        "side_support_weight": 0.25,
        "side_pivot_speed_limit": 0.35,
        "normal_coaxial_weight": 0.15,
        "asset_cfg": _support_wheels(),
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  cfg.curriculum = {
    "spin_commands": CurriculumTermCfg(
      func=trick_curriculums.stance_spin_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            "step": 0,
            # Fold all four rolling supports onto the reference's common
            # horizontal axle first.  Holding a single one-hot for the whole
            # 6-s episode makes the only high-return solution a stationary
            # four-wheel pivot, rather than a transient stepping yaw.
            "mode_probabilities": (1.0, 0.0, 0.0, 0.0, 0.0),
            "spin_idle_probability": 0.0,
            "spin_rate_range": (1.0, 2.5),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Once the nested four-wheel geometry is established, retain it
            # as the majority of samples while introducing the other physical
            # stances and the working spin-rate range.  This remains one
            # actor, one one-hot-plus-rate interface and ordinary default
            # resets--only the sampled command distribution changes.
            "step": 25_600,
            "mode_probabilities": (0.65, 0.10, 0.15, 0.05, 0.05),
            "spin_idle_probability": 0.0,
            "spin_rate_range": (3.0, 6.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Only after each fixed form has a full-horizon solution expose
            # the requested real transitions, including the all-zero default
            # idle command.  Mixing transitions before normal geometry was
            # valid simply trained the policy to step around the floor.
            "step": 51_200,
            "mode_probabilities": (0.25, 0.15, 0.25, 0.175, 0.175),
            "spin_idle_probability": 0.20,
            "spin_rate_range": (4.0, 8.0),
            "resampling_time_range": (2.0, 3.0),
          },
        ),
      },
    )
  }
  return cfg
