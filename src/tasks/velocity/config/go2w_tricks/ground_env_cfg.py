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
from mjlab.managers.termination_manager import TerminationTermCfg

from src.tasks.velocity.mdp import terminations, trick_curriculums, trick_rewards
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
  # Discover the clean continuous-contact pivot on nominal flat contact
  # first.  The former friction range mixed an avoidable slipping/bouncing
  # disturbance into the core geometry task.
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
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
  # A legal front/rear support must lift the trunk over one wheel pair.  Keep
  # it free of a prescribed pose, but expose the model's usable hip workspace
  # so that the measured clearance is physically reachable from four-wheel
  # reset without changing torque authority.
  cfg.actions["joint_pos"].scale[r".*_hip_joint"] = 0.90
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceLocomotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      mode_probabilities=(0.0, 0.5, 0.5),
      mode_idle_probabilities=(0.0, 1.0, 1.0),
      lin_vel_x_range=(0.0, 0.0),
      yaw_rate_range=(0.0, 0.0),
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
      weight=100.0,
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
        # Four wheels touching is the reset, not a partial front/rear
        # support.  Any non-commanded wheel contact must remove the support
        # fraction, otherwise the policy can improve the attitude score while
        # retaining the low ordinary gait that the task explicitly rejects.
        "extra_contact_discount": 1.0,
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
        # Even the squared gate left the m200 policy orbiting the low
        # four-wheel form: changing its wheel contacts and rotating its trunk
        # must initially happen together, so their product was still too
        # sparse.  Use the measured attitude itself as the continuous route;
        # exact contacts, clearance, and non-wheel collision termination are
        # unchanged.  This is not a joint pose or transition trajectory.
        "orientation_power": 1.0,
        # From normal four-wheel reset the desired front/rear attitude is
        # only half aligned, while the target wheel pair and tall clearance
        # cannot improve until the body has first begun to tip.  Reserve part
        # of this same measured support outcome for that physical attitude
        # progress; only the simultaneous correct contacts and clearance can
        # reach its final full score.
        "orientation_progress_floor": 0.50,
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
  cfg.curriculum = {
    "locomotion_commands": CurriculumTermCfg(
      func=trick_curriculums.stance_locomotion_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            "step": 0,
            "mode_probabilities": (0.0, 0.5, 0.5),
            "mode_idle_probabilities": (0.0, 1.0, 1.0),
            "lin_vel_x_range": (0.0, 0.0),
            "yaw_rate_range": (0.0, 0.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            "step": 96_000,
            "mode_probabilities": (0.20, 0.40, 0.40),
            "mode_idle_probabilities": (0.25, 0.55, 0.55),
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            "step": 256_000,
            "mode_probabilities": (0.34, 0.33, 0.33),
            "mode_idle_probabilities": (0.10, 0.25, 0.25),
            "lin_vel_x_range": (-0.20, 0.20),
            "yaw_rate_range": (-0.30, 0.30),
            "resampling_time_range": (6.0, 6.0),
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
  # The reference's normal pivot moves the wheel centres from a four-corner
  # footprint onto one transverse line.  A modest extension beyond the shared
  # ±0.55-rad residual makes that geometry reachable; the 0.90-rad trial
  # instead taught wheel liftoff, so retain a ground-contact-safe 0.70-rad
  # envelope.  This increases no torque limit and supplies no desired pose.
  cfg.actions["joint_pos"].scale[r".*_hip_joint"] = 0.70
  # A ±40-rad/s wheel command was sufficient for the visibly slow prototype
  # but capped the faster common-axis pivot seen in the reference.  Motor
  # torque authority remains the model's physical limit.
  cfg.actions["joint_vel"].scale = 80.0
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      mode_probabilities=(1.0, 0.0, 0.0, 0.0, 0.0),
      spin_idle_probability=0.0,
      spin_rate_range=(2.0, 5.0),
      spin_rate_ramp_rate=36.0,
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
        # A 22-rad/s tolerance made a 4-rad/s request score highly even at
        # zero body yaw.  The separate geometry bridge below remains available
        # for discovery; this narrower outcome tolerance makes actual rotation
        # necessary once that layout exists.
        "std": 3.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "pivot_speed_limit": 0.08,
        "upright_support_weight": 0.20,
        "side_support_weight": 0.25,
        "side_pivot_speed_limit": 0.35,
        # Keep a small geometry bridge, but make a stationary common-axis
        # form materially inferior to one that actually tracks z-rate.  The
        # previous 40% bridge converged to low-speed deformation instead of
        # the reference pivot; this still supplies a measured discovery path
        # without becoming a substitute for rotation.
        "normal_formation_weight": 0.45,
        # Keep the signed world-z rate valuable through a direct one-hot
        # change.  The strict final tracking score remains present; this
        # dense measured-rate component merely makes acceleration toward it
        # discoverable without adding a pose or transition target.
        "rate_progress_weight": 0.75,
        "asset_cfg": _support_wheels(),
      },
    ),
    # Common-axis formation is a coordinated but smooth movement over many
    # control frames.  The former temporal cost was comparable to the entire
    # zero-action formation signal, so it selected immobility before PPO could
    # measure any useful geometric improvement.  Retain a light regularizer
    # without turning a visibly fluid pivot into a frozen default pose.
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.005),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  # The reference's final normal pivot keeps every wheel in continuous
  # contact, but folding there from default four-wheel idle may transiently
  # unload one wheel.  Applying this hard validity check from reset killed
  # precisely those formation attempts before the co-axial/local-centre
  # reward could distinguish them from a travelling floor circle.  Enable it
  # only once the curriculum has already exposed the dynamic front/rear
  # pivots; it still guards the final high-rate policy against hopping.
  cfg.terminations["normal_spin_support_lost"] = TerminationTermCfg(
    func=terminations.normal_spin_support_lost,
    params={
      "command_name": "trick",
      "sensor_name": wheel_contact_cfg.name,
      "speed_deadband": 0.20,
      "grace_period_s": 1.5,
      "enable_after_steps": 256_000,
    },
  )
  cfg.curriculum = {
    "spin_commands": CurriculumTermCfg(
      func=trick_curriculums.stance_spin_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            "step": 0,
            # Find the normal four-wheel common-axis pivot from default idle.
            "mode_probabilities": (1.0, 0.0, 0.0, 0.0, 0.0),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "spin_rate_range": (2.0, 5.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # First discover front/rear as legal, still two-wheel supports
            # using their existing zero-rate command value.  The second
            # segment of an episode can still be dynamic, so this also gives
            # PPO normal-to-stance transition examples without an artificial
            # pose or phase target.
            "step": 102_400,
            "mode_probabilities": (0.50, 0.25, 0.25, 0.0, 0.0),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.75,
            "spin_rate_range": (2.0, 5.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Remove most static examples and require slow signed pivots once
            # those two supports have a discovery route.
            "step": 179_200,
            "mode_probabilities": (0.40, 0.30, 0.30, 0.0, 0.0),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.40,
            "spin_rate_range": (3.0, 6.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Final direct normal/front/rear switching keeps one signed world
            # z-rate throughout; only the separate left/right modes remain
            # static.
            "step": 256_000,
            "mode_probabilities": (0.35, 0.25, 0.25, 0.075, 0.075),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "spin_rate_range": (4.0, 8.0),
            "resampling_time_range": (6.0, 6.0),
          },
        ),
      },
    )
  }
  return cfg
