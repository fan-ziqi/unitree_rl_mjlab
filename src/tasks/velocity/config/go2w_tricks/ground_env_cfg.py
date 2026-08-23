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
      # First establish a full physical transition from the required normal
      # four-wheel reset to the commanded support.  The measured m400 policy
      # still had not reached its front support after eight seconds; changing
      # one-hots halfway through that first attempt only shortens the useful
      # outcome horizon.  A later command-switch robustness pass can reuse
      # this same fused policy, but this discovery run keeps every sampled
      # command present from the ordinary default state for the whole episode.
      resampling_time_range=(8.0, 8.0),
      # Four-wheel rolling is already supplied deterministically by the idle
      # action gate.  Keep normal x/yaw requests in this fused policy, but
      # devote most fresh-policy discovery to the two normal-reset-to-upright
      # outcomes rather than their trivial four-wheel support.
      mode_probabilities=(0.15, 0.425, 0.425),
      # Every mode needs both a stopped balance and a real x/yaw response.
      # The old 85% front/rear static share let an upright front pose emerge,
      # but supplied too few moving samples for either that pose or its rear
      # counterpart to learn the requested controls.  This is only command
      # coverage; it introduces no reset stance, target posture, or phase.
      # A 15% stationary share let normal-wheel velocity tracking outscore
      # discovery of either upright support.  Keep a substantial balance
      # subset inside the same fused command distribution; the remaining
      # front/rear samples still carry x/yaw requests.
      # Literal normal idle is a deterministic action gate, not a skill PPO
      # needs to spend samples rediscovering.  Keep static examples only for
      # the two upright forms, whose balance still benefits from them.
      # A front/rear controller needs many successful no-motion supports
      # before x/yaw reward can improve its wheel motion. Reducing this share
      # prematurely made both upright modes reset before they ever became a
      # controllable base, so retain half the samples as static balance.
      mode_idle_probabilities=(0.0, 0.50, 0.50),
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
        # The previous linear 0.45-m scale paid most of the support return to
        # a visibly crouched 0.35-m result.  This remains a measured trunk to
        # wheel-axle clearance, not a leg pose, but makes the requested
        # extended two-wheel support the substantially better outcome.
        "minimum_root_clearance": (0.18, 0.62, 0.62),
        # This remains the same measured contact/attitude/clearance result,
        # but m600 showed that a linear score accepts a 0.90-aligned, folded
        # front stance as nearly as valuable as the requested upright support.
        # Sharpening its existing physical factors gives PPO a direct outcome
        # gradient toward an actually tall, vertical wheel pair.
        "orientation_power": 4.0,
        "clearance_power": 2.0,
        # Static front/rear commands should primarily establish a legal
        # upright support.  A non-zero public x/yaw command needs that exact
        # support but cannot be allowed to make a fixed, wrong-direction roll
        # just as valuable as tracking it.  Scale this existing outcome while
        # moving; the velocity rewards below then decide the signed response.
        "moving_command_start_index": 3,
        "moving_command_scale": 0.25,
        "asset_cfg": _support_wheels(),
      },
    ),
    "track_x_and_zero_lateral": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_exp,
      # Support discovery is already established by the measured wheel
      # contact/attitude outcome above.  Give the public x command enough
      # importance that a fixed forward roll cannot outscore command response.
      # Before a legal two-wheel support exists, a velocity error mostly
      # encourages a fast four-wheel escape.  Keep x response in this single
      # policy, but make the physical support result the dominant discovery
      # return; command tracking remains active throughout the same rollout.
      weight=10.0,
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
      },
    ),
    "track_yaw": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_exp,
      # Fixed-command audits showed normal x control and default posture were
      # recovered while yaw-only commands stayed near zero rate.  Make this
      # existing public-command outcome large enough to compete with the
      # always-available support score; it still contains no posture, wheel
      # target, phase, or transition prescription.
      weight=20.0,
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
  co-linear; front/rear one-hots request their two-wheel local pivots.  The
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
      # The balanced run preserved normal/front but stopped improving the
      # physically harder rear and side forms once its scalar exploration
      # collapsed. Give those forms a modest majority while retaining over a
      # third of samples for the established normal/front pivots.
      mode_probabilities=(0.20, 0.15, 0.30, 0.20, 0.15),
      spin_idle_probability=0.0,
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
      # One outcome term: normal/front/rear track their local yaw pivots only
      # with physically valid horizontal wheel axles; left/right are still
      # two-wheel side supports because their vertical wheel axles cannot
      # produce that pivot.  This prevents the normal or side modes from
      # collapsing to ordinary circle-driving without a second policy or a
      # motion reference.
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
        # Front/rear must establish the same physical two-wheel support
        # before a high-rate pivot is possible.  Keep that discovery signal
        # inside the existing outcome; most of its value still requires rate,
        # axle geometry, and a stationary support centre.
        "upright_support_weight": 0.20,
        "asset_cfg": _support_wheels(),
      },
    ),
    # A generic temporal regularizer keeps the non-supporting legs from
    # accumulating high-frequency exploratory flailing once a local pivot is
    # found.  It supplies no joint posture, contact sequence, or reference
    # trajectory, and remains small compared with the measured pivot result.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.02),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  cfg.curriculum = {}
  return cfg
