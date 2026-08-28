"""Lean flat-ground Go2W trick environments.

These two tasks deliberately reward public command outcomes.  They do not
encode leg lengths, a contact sequence, or an action-space posture for the
two-wheel skills.  The only exception is the user's explicit normal-mode
requirement: four-wheel rolling must remain recognizably close to the Go2W
model default pose.  With a nonzero normal spin command, PPO discovers the
compact four-wheel common-axle geometry from measured wheel contacts, axle
alignment, and local-centre motion; no joint pose or reference trajectory is
provided.
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
      direct_switch_probability=0.0,
      # Rear is the harder support, but keep enough front samples for both
      # one-hots to share a single static-discovery policy.
      mode_probabilities=(0.0, 0.25, 0.75),
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
    "commanded_support_attitude": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment_rise,
      # At four-wheel reset, the target two-wheel gravity alignment is 0.5.
      # This term is exactly zero there and rises only as the base rotates in
      # the requested direction.  It supplies a persistent pose-result route
      # to the later contact/clearance outcome, without a leg target or a
      # prescribed get-up motion.
      weight=100.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "num_modes": 3,
        "power": 1.0,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    # This single dense score is deliberately additive: it supplies a useful
    # direction from four wheels toward the requested support without making
    # contact an all-or-nothing gate on the attitude signal.
    "commanded_support": RewardTermCfg(
      func=trick_rewards.mode_support_score,
      # This is the only held-stance endpoint: target gravity direction and
      # exactly the named wheel pair.  Earlier versions separately rewarded
      # absolute base height, free-wheel height, and leg span.  Those are
      # correlated diagnostics, not independent task goals; together they
      # created a profitable low slant before the body became upright.
      weight=120.0,
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
        "mode_weights": (3.0, 1.0, 1.0),
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
        "mode_weights": (2.5, 1.0, 1.0),
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
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.01),
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
            # Start balanced static discovery in the same fused actor, while
            # retaining a small default-normal replay share.  The
            # all-zero normal idle is still supplied by the action gate;
            # this one-hot share merely keeps its observation branch present
            # before normal x/yaw commands enter.
            "mode_probabilities": (0.0, 0.50, 0.50),
            "mode_idle_probabilities": (1.0, 1.0, 1.0),
            "direct_switch_probability": 0.0,
            "lin_vel_x_range": (0.0, 0.0),
            "yaw_rate_range": (0.0, 0.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Static audits now reach both pitch supports by update 400.  Start
            # a small x/yaw command range at that measured milestone so the
            # fused actor learns locomotion without waiting for an arbitrary
            # long static-only block.
            "step": 19_200,
            "mode_probabilities": (0.30, 0.35, 0.35),
            "mode_idle_probabilities": (0.30, 0.35, 0.35),
            "direct_switch_probability": 0.15,
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Expand to the requested walking/turning range after one short
            # low-speed locomotion block, while retaining all three modes.
            "step": 38_400,
            "mode_probabilities": (0.30, 0.35, 0.35),
            "mode_idle_probabilities": (0.10, 0.25, 0.25),
            "direct_switch_probability": 0.30,
            "lin_vel_x_range": (-0.20, 0.20),
            "yaw_rate_range": (-0.30, 0.30),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Only after the extended static block introduce normal <->
            # front/rear one-hot transitions, still at zero x/yaw.
            "step": 76_800,
            "mode_probabilities": (0.30, 0.30, 0.40),
            "mode_idle_probabilities": (1.0, 1.0, 1.0),
            "direct_switch_probability": 0.25,
            "lin_vel_x_range": (0.0, 0.0),
            "yaw_rate_range": (0.0, 0.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Start conservative x/yaw replay after static support and direct
            # transitions have both been trained in the same policy.
            "step": 96_000,
            "mode_probabilities": (0.30, 0.25, 0.45),
            "mode_idle_probabilities": (0.10, 0.25, 0.25),
            "direct_switch_probability": 0.35,
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Reach the requested command range after the low-range stage.
            "step": 115_200,
            "mode_probabilities": (0.30, 0.25, 0.45),
            "mode_idle_probabilities": (0.10, 0.25, 0.25),
            "direct_switch_probability": 0.50,
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
  video's compact, level, four-wheel common-axle local pivot.  Front/rear
  one-hots request their named pivots; left/right one-hots are physically
  distinct held side supports and ignore spin rate.  All five modes still
  share exactly the same one-hot-plus-rate command interface and one policy.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  configure_ground_support_actuators(cfg)
  _configure_fast_discovery(cfg)
  # A nonzero normal command must reshape the ordinary four-wheel rectangle
  # into a compact common axle while all wheels remain grounded.  A modest
  # extension beyond the shared ±0.55-rad residual makes that geometry
  # reachable without adding a prescribed joint pose or torque authority.
  # A two-wheel stand has to move through the same physical hip workspace as
  # the front/rear stance-locomotion task.  Its 0.90-rad residual remains
  # inside the model's +/-1.05-rad abduction range; the compact-pivot outcome
  # below, not a smaller unexplained action envelope, is what prevents a
  # splayed final form.
  cfg.actions["joint_pos"].scale[r".*_hip_joint"] = 0.90
  # Discovery needs the wheels to stay planted while legs find the common
  # axle.  With an 80-rad/s residual range, the bounded exploratory policy
  # drove the four-wheel footprint across the plane before it could improve
  # its geometry.  This restores the model's proven 40-rad/s working range;
  # it changes neither motor torque limit nor the later requested body rate.
  cfg.actions["joint_vel"].scale = 40.0
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      mode_probabilities=(0.50, 0.25, 0.25, 0.0, 0.0),
      spin_idle_probability=0.0,
      upright_static_probability=0.0,
      direct_switch_probability=0.0,
      spin_rate_range=(0.5, 2.0),
      spin_rate_ramp_rate=36.0,
      debug_vis=False,
    )
  }
  configure_default_idle_actions(
    cfg,
    command_name="trick",
    # The literal all-zero command and normal@zero-rate are default four-wheel
    # idle.  In contrast, a named one-hot at zero rate is the required static
    # two-wheel support, so it must release policy action authority.
    idle_mode_index=0,
    stationary_command_start_index=5,
    command_deadband=0.20,
    idle_contact_sensor_name=wheel_contact_cfg.name,
    zero_mode_vector_is_idle=True,
  )
  _use_history(cfg, "trick")

  cfg.rewards = {
    "commanded_spin_pivot": RewardTermCfg(
      func=trick_rewards.StanceSpinPivotResult,
      # Normal is a compact level four-wheel pivot, not wheel-steering.  At
      # weight 18, the -50 collision boundary made leaving ordinary idle a
      # bad exploration trade before PPO ever sampled a better geometry.
      # This matches the practical scale of the proven support objective;
      # no target or action authority changes.
      weight=80.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        # The previous 3-rad/s tolerance still gave a 0.5--2 rad/s command a
        # high score at literal zero yaw speed, so PPO settled into four-wheel
        # stillness.  This keeps zero speed reward-free for every active spin
        # request while retaining a continuous signed-rate progress route.
        "std": 0.50,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        # The previous 0.06 m/s gate suppressed nearly all signed-rate
        # gradient before either direction could spin.  This is a discovery
        # scale, not the acceptance threshold: the evaluator still requires
        # a visibly local support midpoint.
        "pivot_speed_limit": 0.12,
        # The common axle must live below a recognizably level Go2W trunk.
        # Without this measured root-to-wheel clearance, the new all-wheel
        # line reward has a low crouching local optimum: it can pack wheels
        # together but looks like a crawling chassis.  This names no leg
        # angle—the policy remains free to find its own extended support.
        "normal_min_root_clearance": 0.35,
        "upright_support_weight": 0.35,
        # The early common-axle geometry bridge is only for discovering the
        # four-wheel packing from its ordinary rectangular reset.  It fades
        # after discovery so the same geometry must carry the signed rate at
        # a stationary all-wheel centroid.
        "normal_final_geometry_weight": 0.04,
        # A spin rollout contains 48 control steps.  Keep the bridge through
        # the first 600 PPO updates, then make the normal branch earn its
        # reward from an actual local pivot while the named forms are learned.
        "normal_geometry_decay_start_steps": 28_800,
        "normal_geometry_decay_steps": 28_800,
        # Keep the signed world-z rate valuable through a direct one-hot
        # change.  The strict final tracking score remains present; this
        # dense measured-rate component merely makes acceleration toward it
        # discoverable without adding a pose or transition target.
        "rate_progress_weight": 0.50,
        # The separate held-support term below is the sole zero-rate named
        # outcome, so two incompatible products cannot compete for it.
        "static_support_weight": 0.0,
        "asset_cfg": _support_wheels(),
      },
    ),
    "named_static_support": RewardTermCfg(
      func=trick_rewards.mode_support_score,
      # A named zero-rate one-hot is a static two-wheel outcome.  Reuse the
      # direct contact/attitude/clearance result that gives the locomotion
      # task a route out of four-wheel idle; it contains no leg pose or
      # transition reference.
      weight=80.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "num_modes": 5,
        "extra_contact_discount": 1.0,
        "soft_support_height": 0.11,
        "soft_support_height_std": 0.07,
        "soft_support_pair_height_std": 0.06,
        # A low slant has both commanded wheels but is not the requested
        # stand.  Use the same measured attitude pressure as the successful
        # front/rear stance task, and require a visibly extended trunk above
        # its named support pair.  These are outcome geometry, never a joint
        # target.
        # The pitch-pair support is physically established around 0.30 m of
        # root-to-axle clearance; requiring 0.45 m made the named front/rear
        # result unreachable even after the correct wheel pair was found.
        "minimum_root_clearance": (0.18, 0.30, 0.30, 0.35, 0.35),
        "orientation_power": 3.0,
        "clearance_power": 1.0,
        # The rate channel alone distinguishes a held support from a pivot.
        "stationary_command_index": 5,
        "static_command_start_index": 5,
        "command_deadband": 0.20,
        "static_angular_velocity_scale": 0.80,
        "static_linear_velocity_scale": 0.12,
        "static_stillness_floor": 0.0,
        # A zero-rate named one-hot must begin preferring low angular speed
        # once it has a meaningful partial support.  Waiting for the former
        # near-final thresholds let the policy spin/rock through the contact
        # bridge forever, because the existing stillness result was never
        # activated.  This leaves the reset and initial lift unconstrained;
        # it only changes the outcome ranking after partial physical rise.
        # f160 m1800 reaches about 0.58 attitude alignment with commanded
        # contacts, then keeps rocking because the prior 0.60/0.40/0.35 gate
        # never activates its measured stillness result.  Begin damping at a
        # genuine partial two-wheel support, as in stance locomotion; the
        # full contact/attitude/height outcome remains the only maximum.
        "static_settling_alignment_threshold": 0.50,
        "static_settling_support_threshold": 0.20,
        "static_settling_clearance_threshold": 0.20,
        # This measured attitude-rate bridge is active only away from the
        # outcome, so it cannot reward a permanent fling.
        "attitude_progress_weight": 0.12,
        "attitude_progress_rate_scale": 4.0,
        "asset_cfg": _support_wheels(),
      },
    ),
    "named_static_attitude": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment_rise,
      # Named two-wheel pivots need a state-level route from ordinary four
      # wheels to their commanded gravity direction before their exact
      # wheel-pair/height score can become non-zero.  Normal is deliberately
      # excluded: it remains a level four-wheel common-axle problem.
      weight=30.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "num_modes": 5,
        "power": 1.5,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "named_pitch_support_height": RewardTermCfg(
      func=trick_rewards.mode_root_height_exp,
      # A selected front or rear wheel pair can touch while the trunk remains
      # in the observed low slant.  The fixed-command stance audit establishes
      # a roughly 0.39-m root height as the reachable extended pitch-pair
      # outcome.  It is a whole-body endpoint, not a leg target or trajectory.
      weight=15.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "num_modes": 5,
        "target_height": 0.39,
        "scale": 5.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "orientation_power": 2.0,
        "extra_contact_discount": 1.0,
        "asset_cfg": _support_wheels(),
      },
    ),
    "named_non_support_wheel_clearance": RewardTermCfg(
      func=trick_rewards.mode_non_support_wheel_clearance,
      # The named static two-wheel commands face the same sparse contact
      # barrier as the locomotion supports.  Share the exact wheel-clearance
      # outcome so all one-hots retain one actor and one physical objective,
      # with no leg pose, reference frame, or transition schedule added.
      weight=18.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "contact_masks": STANCE_CONTACT_MASKS,
        "target_height": 0.40,
        "minimum_height": 0.10,
        "num_modes": 5,
        "asset_cfg": _support_wheels(),
      },
    ),
    "named_roll_support_height": RewardTermCfg(
      func=trick_rewards.mode_root_height_exp,
      # A left/right pair places the short base dimension vertically, so its
      # physically extended wheel-pair height is lower than the pitch pair.
      # Reward the state result only; PPO still selects all joint geometry.
      weight=12.0,
      params={
        "command_name": "trick",
        "modes": (3, 4),
        "num_modes": 5,
        "target_height": 0.45,
        "scale": 5.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "orientation_power": 2.0,
        "extra_contact_discount": 1.0,
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
  cfg.curriculum = {
    "spin_commands": CurriculumTermCfg(
      func=trick_curriculums.stance_spin_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            "step": 0,
            # Learn a genuine rotating pivot from the outset.  Isolating
            # normal at 0.5--1 rad/s for hundreds of iterations produced a
            # policy that had never experienced the requested fast command.
            "mode_probabilities": (0.34, 0.28, 0.28, 0.05, 0.05),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.20,
            "direct_switch_probability": 0.0,
            "spin_rate_range": (4.0, 8.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Add both static side supports and a higher-rate fused pivot
            # after one short discovery block, rather than serially spending
            # thousands of updates on five independently staged forms.
            "step": 28_800,
            "mode_probabilities": (0.40, 0.25, 0.25, 0.05, 0.05),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.15,
            "direct_switch_probability": 0.10,
            "spin_rate_range": (7.0, 11.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Spend the majority of the run on the actual delivery setting:
            # fast normal/front/rear pivoting plus continuous one-hot changes.
            "step": 57_600,
            "mode_probabilities": (0.36, 0.27, 0.27, 0.05, 0.05),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.10,
            "direct_switch_probability": 0.30,
            "spin_rate_range": (10.0, 15.0),
            "resampling_time_range": (6.0, 6.0),
          },
        ),
      },
    )
  }
  return cfg
