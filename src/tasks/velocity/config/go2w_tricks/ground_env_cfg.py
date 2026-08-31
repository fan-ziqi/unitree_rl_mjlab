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

from src.assets.robots.unitree_go2w.go2w_constants import GO2W_LEG_JOINTS
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
  # Keep the locomotion branch on the compact, proven outcome scale.  Large
  # residual/action overrides make the front support overshoot into a low
  # folded basin; the base actuator limits already provide enough authority.
  cfg.episode_length_s = 8.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceLocomotionCommandCfg(
      entity_name="robot",
      # Four seconds exposes a real transition from the ordinary reset while
      # still allowing two command changes in the eight-second rollout.
      resampling_time_range=(4.0, 4.0),
      mode_probabilities=(0.20, 0.40, 0.40),
      mode_idle_probabilities=(0.25, 0.15, 0.15),
      direct_switch_probability=0.0,
      lin_vel_x_range=(-0.20, 0.20),
      yaw_rate_range=(-0.30, 0.30),
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
  )
  _use_history(cfg, "trick")

  cfg.rewards = {
    # This single dense score is deliberately additive: it supplies a useful
    # direction from four wheels toward the requested support without making
    # contact an all-or-nothing gate on the attitude signal.
    "commanded_support": RewardTermCfg(
      func=trick_rewards.mode_support_score,
      weight=12.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "num_modes": 3,
        "extra_contact_discount": 1.0,
        "minimum_root_clearance": (0.18, 0.45, 0.45),
        "asset_cfg": _support_wheels(),
      },
    ),
    "track_x_and_zero_lateral": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_exp,
      weight=5.0,
      params={
        "command_name": "trick",
        "std": 0.45,
        "lateral_weight": 2.0,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 3.0,
      },
    ),
    "track_yaw": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_exp,
      weight=6.0,
      params={
        "command_name": "trick",
        "std": 0.60,
        "gravity_targets": LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 3.0,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.04),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-50.0),
  }
  # Keep the complete command distribution active from update zero.  The
  # action/command gate still starts every event from the ordinary reset, but
  # no staged command range can starve the front branch before locomotion is
  # actually learned.
  cfg.curriculum = {}
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
      # The right-side mirror is mechanically easier in this model and was
      # monopolising the shared actor.  Oversample the harder left support
      # while retaining both side one-hots in the same policy.
      mode_probabilities=(0.20, 0.10, 0.10, 0.45, 0.15),
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
        # Keep local-centre quality in the outcome, but avoid making the
        # early front/rear rotation gradient almost zero while the support
        # axle is still settling.
        "pivot_speed_limit": 0.20,
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
        # The previous 0.04 late-stage floor was too weak to preserve the
        # common-axle form: the policy learned a low, travelling pivot while
        # still receiving a healthy signed-rate return.  Keep geometry a
        # material part of the final outcome so normal cannot collapse after
        # the discovery bridge fades.
        "normal_final_geometry_weight": 0.30,
        # A spin rollout contains 48 control steps.  Keep the bridge through
        # the first 600 PPO updates, then make the normal branch earn its
        # reward from an actual local pivot while the named forms are learned.
        "normal_geometry_decay_start_steps": 28_800,
        "normal_geometry_decay_steps": 28_800,
        # Keep the signed world-z rate valuable through a direct one-hot
        # change.  The strict final tracking score remains present; this
        # dense measured-rate component merely makes acceleration toward it
        # discoverable without adding a pose or transition target.
        "rate_progress_weight": 0.75,
        # Keep the dynamic result focused on commanded-rate pivots.  Held
        # zero-rate named supports use the separate measured endpoint below;
        # combining both products here diluted the side-roll gradient.
        "static_support_weight": 0.0,
        "asset_cfg": _support_wheels(),
      },
    ),
    "named_static_support": RewardTermCfg(
      func=trick_rewards.mode_support_score,
      # A zero-rate side one-hot is a real two-wheel balance outcome.  Restore
      # the direct contact/attitude/clearance route that previously produced
      # reliable left/right supports; it contains no joint target or phase.
      weight=100.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "num_modes": 5,
        "extra_contact_discount": 1.0,
        "orientation_power": 1.0,
        # AS2W's lateral two-wheel form is tall and wheel-supported, not a
        # body-on-floor roll.  This is a measured trunk-to-selected-wheel
        # clearance outcome; it does not prescribe a joint pose.
        # The front/rear wheel stands naturally leave 0.30 m or more below
        # the trunk, but a lateral roll places the selected wheel pair about
        # 0.19 m below the root in this model.  Requiring the front/rear
        # clearance for side supports made the final static-centre gate
        # unreachable, so keep the physically reachable side envelope while
        # retaining the tall non-wheel clearance outcome below.
        # m1000 reached the correct side contacts but left the trunk only
        # 0.16--0.19 m above the selected wheels, visibly lower than AS2W.
        # Raise the measured side envelope modestly; this remains a result
        # constraint, not a prescribed joint configuration.
        # The m600--m900 side checkpoints satisfy the selected contacts but
        # leave the trunk only ~0.20--0.23 m above the support wheels.  AS2W's
        # side stand has a visibly taller, wheel-supported body.  Raise only
        # this measured side envelope modestly; the policy remains free to
        # discover the joint configuration and can still use the existing
        # contact/attitude bridge during the approach.
        # Side support is a 90-degree body roll.  Its attainable root-to-wheel
        # z separation is set by the model's lateral hip offset (~0.15--0.23
        # m), so retain the reachable side envelope instead of asking PPO for
        # an impossible tall target.
        # The previous 0.26-m side floor was learned literally: the fixed
        # audit reached ~0.26 m and the video showed a low, folded support.
        # Raise only this measured trunk-to-wheel outcome to the reachable
        # 0.30-m envelope so the side form has to stand clear of the ground.
        "minimum_root_clearance": (0.30, 0.30, 0.30, 0.30, 0.30),
        # Contact bits are binary and provide no signal while the side pair
        # is approaching the floor.  These measured wheel-centre terms give
        # the side modes a continuous route to the AS2W support line without
        # prescribing any joint angles or transition trajectory.
        "soft_support_height": 0.086,
        "soft_support_height_std": 0.050,
        "soft_support_pair_height_std": 0.040,
        "stationary_command_index": 5,
        "static_command_start_index": 5,
        "command_deadband": 0.20,
        "static_angular_velocity_scale": 0.80,
        "static_linear_velocity_scale": 0.12,
        # The AS2W side-support pair must remain a fixed ground pivot.  Root
        # linear speed alone misses a rotating/drifting support axle, so use
        # the measured centre velocity of the commanded wheel pair once the
        # support has actually formed.
        # 0.08 made the rational stillness score numerically flat once a
        # support pair drifted at ~1 m/s, so PPO received almost no gradient
        # for braking the side stance.  A broader measured scale preserves a
        # strong preference for a fixed pivot while keeping that gradient
        # usable throughout the approach.
        # The right-side branch still settles with a support-centre drift of
        # roughly 0.9 m/s.  A wider rational scale keeps a usable derivative
        # in that regime, so PPO can learn to brake the selected pair instead
        # of receiving an effectively flat near-zero score.
        # The evaluator's local-pivot condition is center_speed < 0.08 m/s.
        # A 0.50-m/s rational scale made the reward almost flat at the
        # observed 0.10--0.16 m/s side drift, so the shared actor could keep
        # a visually correct roll while never learning to brake the selected
        # pair.  Tighten only this measured centre-speed scale; it does not
        # prescribe a pose or alter the front/rear dynamic-rate objective.
        "static_support_center_speed_scale": 0.20,
        "static_stillness_floor": 0.0,
        "static_settling_alignment_threshold": 0.50,
        "static_settling_support_threshold": 0.20,
        "static_settling_clearance_threshold": 0.15,
        "attitude_progress_weight": 0.12,
        "attitude_progress_rate_scale": 4.0,
        # Reuse the existing hip-to-wheel extension outcome so a low folded
        # side support is not the easiest way to satisfy contact and attitude.
        "support_leg_length_target": 0.30,
        "asset_cfg": _support_wheels(),
      },
    ),
    "named_side_attitude": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment_rise,
      # Side support must first roll away from the ordinary four-wheel reset.
      # This direct measured attitude progress prevents contact-only PPO from
      # settling into a flat or belly-low local optimum.
      weight=80.0,
      params={
        "command_name": "trick",
        "modes": (3, 4),
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "num_modes": 5,
        "power": 1.0,
        "asset_cfg": SceneEntityCfg("robot"),
      },
    ),
    "side_non_support_clearance": RewardTermCfg(
      func=trick_rewards.mode_non_support_wheel_clearance,
      # Side commands are static two-wheel supports.  Gravity alignment alone
      # lets the policy roll toward the requested side and then settle with
      # one or both uncommanded wheels still carrying the body.  This measured
      # wheel-centre clearance supplies the missing continuous outcome signal:
      # the opposite pair must actually rise above the floor while the named
      # pair remains selected by the support score.  It names no joint angle,
      # pose, or transition trajectory.
      weight=80.0,
      params={
        "command_name": "trick",
        "modes": (3, 4),
        "contact_masks": STANCE_CONTACT_MASKS,
        "target_height": 0.30,
        "minimum_height": 0.10,
        "num_modes": 5,
        "asset_cfg": _support_wheels(),
      },
    ),
    "side_static_angular_velocity": RewardTermCfg(
      func=trick_rewards.mode_static_angular_velocity_exp,
      # Left/right one-hots are held two-wheel supports.  Reward the measured
      # end state of a quiet body once the side attitude is formed; this stops
      # the policy from satisfying the side contacts while continuing to roll
      # around them.  No pose or timing target is introduced.
      weight=120.0,
      params={
        "command_name": "trick",
        "modes": (3, 4),
        "angular_velocity_scale": 0.60,
        "num_modes": 5,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "alignment_power": 4.0,
      },
    ),
    "normal_default_leg_pose": RewardTermCfg(
      func=trick_rewards.normal_leg_default_pose_exp,
      # Normal one-hot spinning must retain the ordinary Go2W four-leg
      # silhouette; the only allowed change is the small geometry adjustment
      # needed to bring wheel axes together.  This measured result is active
      # for normal (including nonzero-rate) commands only and names no
      # trajectory or target pose for the other four modes.
      weight=25.0,
      params={
        "command_name": "trick",
        "std": 0.55,
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=GO2W_LEG_JOINTS, preserve_order=True
        ),
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
            # First form a continuous-contact four-wheel pivot at a rate the
            # reset can physically reach.  The other one-hots are visible,
            # but cannot drown out normal discovery.
            # Keep all five branches represented while preserving the
            # balanced normal/front/rear discovery mix used by the best
            # two-sided checkpoint.  The later side-heavy trial improved one
            # mirror but destabilized the other.
            # The right-side mirror is mechanically easier in this model;
            # give the harder left-side one-hot more rollout mass so PPO sees
            # enough braking/support examples to remove that bias.  All five
            # modes remain represented in the same actor.
            "mode_probabilities": (0.20, 0.10, 0.10, 0.45, 0.15),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 0.0,
            "spin_rate_range": (0.5, 1.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            "step": 28_800,
            # Once the harder left mirror has a support basin, restore equal
            # side replay so the shared actor does not forget the right form.
            "mode_probabilities": (0.20, 0.15, 0.15, 0.25, 0.25),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.30,
            "direct_switch_probability": 0.0,
            "spin_rate_range": (0.5, 2.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            "step": 57_600,
            # Once the side supports have a basin, restore all five modes so
            # the shared actor retains the front/rear dynamic pivots instead
            # of collapsing to the easier static side forms.
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.30,
            "direct_switch_probability": 0.25,
            "spin_rate_range": (2.0, 5.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            "step": 86_400,
            # Side supports were under-sampled at the final stage and their
            # zero-rate one-hots never became reliable.  Give both static
            # forms enough rollouts without removing the normal/front/rear
            # dynamic branches.
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 0.60,
            "spin_rate_range": (5.0, 8.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Leave the final 600 updates for the high-rate, continuous
            # delivery setting and direct mode changes.
            "step": 115_200,
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 1.0,
            "spin_rate_range": (10.0, 15.0),
            "resampling_time_range": (6.0, 6.0),
          },
        ),
      },
    )
  }
  return cfg
