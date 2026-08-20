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
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots.unitree_go2w.go2w_constants import GO2W_LEG_JOINTS
from src.tasks.velocity.mdp import trick_rewards
from src.tasks.velocity.mdp import trick_curriculums
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
# Reward-manager resolution writes numeric IDs onto selector objects.  Every
# reward term therefore needs its own otherwise-identical selector; sharing
# one across terms works in a training build by accident but fails when a
# play/evaluation environment resolves the same site names again.
_STATIC_FREE_WHEEL_SITES = SceneEntityCfg(
  "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
)
# Reward-manager resolution stores IDs on this mutable config object.  Keep a
# separate identical selector for the root-clearance term rather than reusing
# ``_WHEEL_SITES`` after the free-wheel term has resolved it.
_ROOT_CLEARANCE_WHEEL_SITES = SceneEntityCfg(
  "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
)
_FIXED_PIVOT_WHEEL_SITES = SceneEntityCfg(
  "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
)
_DYNAMIC_PIVOT_WHEEL_SITES = SceneEntityCfg(
  "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
)
_DYNAMIC_CLEARANCE_WHEEL_SITES = SceneEntityCfg(
  "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
)
_DYNAMIC_FREE_WHEEL_SITES = SceneEntityCfg(
  "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
)
_SUPPORT_LEG_GEOMETRY = SceneEntityCfg(
  "robot",
  body_names=("FL_hip", "FR_hip", "RL_hip", "RR_hip"),
  site_names=("FL", "FR", "RL", "RR"),
  preserve_order=True,
)
_NORMAL_LEG_JOINTS = SceneEntityCfg(
  "robot", joint_names=GO2W_LEG_JOINTS, preserve_order=True
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
      # V8 established the rear rise but its fused actor still under-sampled
      # front support and let the normal idle branch drift.  Preserve rear
      # exposure while putting the next zero-start discovery batch on normal
      # and front; this remains one mode-conditioned policy.
      mode_probabilities=(0.35, 0.40, 0.25),
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
      # V9 reaches a legal but roughly 45-degree lean.  The dense alignment
      # term gets it off the four-wheel reset; make this *existing* sharp
      # terminal outcome decisive enough that a true vertical support pair
      # beats that local optimum.  It still names no joint configuration.
      weight=160.0,
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
      weight=100.0,
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
      # V10 remained upright but often carried only a diagonal subset of its
      # wheels.  A normal command is not complete until all four physical
      # wheels share support; raise the existing exact-contact outcome rather
      # than adding a posture target.
      weight=200.0,
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
      # V11 made this geometry term dominant and front could no longer reach
      # its vertical support at all.  Preserve the normal/rear extension
      # outcome, while asking front for a visible *incremental* extension
      # instead of an unreachable 0.35 m stroke.
      weight=100.0,
      params={
        "command_name": "trick",
        # Normal zero command must be a properly supported four-wheel stand,
        # not the low crouch seen in V9.  The same measured hip-to-wheel
        # outcome applies to all supports; this does not select joint angles.
        "modes": (0, 1, 2),
        "contact_masks": LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "minimum_lengths": (0.32, 0.20, 0.35),
        # A 0.16 m activation threshold made a visibly folded 0.13 m front
        # support receive exactly zero extension gradient.  Start from zero:
        # this still specifies only hip-to-wheel length, not any joint pose.
        "activation_lengths": (0.0, 0.0, 0.0),
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
      # V11's normal command tracked x but ignored a simultaneous yaw=0.2
      # request.  The target is a public command outcome, so strengthen the
      # existing tracker for every mode rather than adding a normal-only pose
      # rule.
      weight=50.0,
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
      weight=-50.0,
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
        # Once the stance is mostly found, prolonged rolling is no longer a
        # useful way to discover the last part of the rise.
        "minimum_gravity_alignment": 0.80,
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
        "minimum_gravity_alignment": 0.80,
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
    # The correct normal zero-command reset is already a physical four-wheel
    # support state.  Penalise only residual controller effort in that exact
    # command case, so the actor cannot improve a shared representation by
    # continually issuing a non-zero action that lifts one wheel.  This is
    # neither a joint target nor a prescribed normal pose, and it is inactive
    # during every two-wheel rise and every requested x/yaw motion.
    "normal_stationary_action_effort": RewardTermCfg(
      func=trick_rewards.stance_stationary_action_l2,
      weight=-5.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "velocity_deadband": 0.04,
        "num_modes": 3,
      },
    ),
    "normal_default_leg_geometry": RewardTermCfg(
      func=trick_rewards.mode_default_joint_pos_excess_exp,
      # Wheel driving needs only small suspension-like leg motion.  Keep the
      # ordinary four-wheel one-hot in the model's init/default posture rather
      # than letting it borrow a distant trick posture from front/rear modes.
      # This applies to moving commands too, but has a 0.12-rad free band.
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "num_modes": 3,
        "free_deviation": 0.12,
        "std": 0.10,
        "asset_cfg": _NORMAL_LEG_JOINTS,
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
  rate requests a dynamic tall front-*or*-rear two-wheel local pivot.  The
  AS2-W reference shows its high-speed contact turn on a laterally separated
  front/rear wheel pair whose midpoint stays fixed while the body turns about
  world-down.  Left/right remain static two-wheel supports in this environment.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  _configure_fast_discovery(cfg)
  cfg.episode_length_s = 6.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 6.0),
      # V9's uniformly sampled but over-constrained policy preserved only the
      # two easy side balances.  Keep a single fused actor while giving the
      # still-undiscovered front branch additional fresh rollouts; rear and
      # both sides remain present in every batch.
      # The reference spends about 0.76 s for its clearest contact-pivot
      # turn, i.e. roughly 8 rad/s.  Sample the front/rear skills frequently
      # enough that the one shared policy sees that speed rather than treating
      # the two easy static side balances as the main task.
      mode_probabilities=(0.30, 0.28, 0.22, 0.10, 0.10),
      spin_idle_probability=0.25,
      spin_rate_range=(5.0, 9.0),
      spin_rate_ramp_rate=12.0,
      debug_vis=False,
    )
  }
  _use_history(cfg, "trick")

  cfg.rewards = {
    "four_wheel_idle_gravity": RewardTermCfg(
      func=trick_rewards.stand_idle_gravity_exp,
      weight=60.0,
      params={"command_name": "trick", "speed_deadband": 0.20, "std": 0.35},
    ),
    "four_wheel_idle_contacts": RewardTermCfg(
      func=trick_rewards.stand_idle_contact_match,
      weight=80.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
      },
    ),
    "four_wheel_idle_default_joint_pos": RewardTermCfg(
      func=trick_rewards.stand_idle_default_joint_pos_exp,
      # A normal one-hot with zero rate is the untriggered public state.  It
      # must hold the model's actual four-wheel default pose, rather than a
      # lingering two-wheel trick configuration from a prior command.
      weight=80.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "std": 0.10,
        "asset_cfg": _NORMAL_LEG_JOINTS,
      },
    ),
    "four_wheel_idle_stillness": RewardTermCfg(
      func=trick_rewards.stand_idle_stillness_exp,
      weight=50.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "linear_velocity_std": 0.12,
        "angular_velocity_std": 0.35,
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
      # V9's weight of 100 made the formerly viable front/rear recovery
      # collapse into contact failures.  Retain a sharper terminal preference
      # than V8, without overpowering the transition and support signals.
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
    # These two result-space signals make the four-wheel-to-two-wheel rise
    # discoverable.  They name only which wheels must become free and the
    # minimum trunk-to-support clearance; neither is a joint target or a
    # timed stand-up trajectory.
    "static_free_wheel_clearance": RewardTermCfg(
      func=trick_rewards.mode_non_support_wheel_clearance,
      weight=45.0,
      params={
        "command_name": "trick",
        "contact_masks": STANCE_CONTACT_MASKS,
        "minimum_height": 0.22,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "gravity_power": 1.0,
        "asset_cfg": _STATIC_FREE_WHEEL_SITES,
      },
    ),
    "static_support_clearance": RewardTermCfg(
      func=trick_rewards.mode_support_wheel_root_clearance_min_exp,
      weight=35.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "contact_masks": STANCE_CONTACT_MASKS,
        "minimum_clearances": (0.0, 0.38, 0.38, 0.38, 0.38),
        "std": 0.14,
        "asset_cfg": _ROOT_CLEARANCE_WHEEL_SITES,
      },
    ),
    # A normal nonzero rate is the AS2-W-style local contact pivot.  It can
    # settle on either transverse high pair; neither command nor observation
    # tells the actor which one to choose.  These are measured final outcomes,
    # not a joint target, a contact sequence, or a reference trajectory.
    "dynamic_tall_pair_support": RewardTermCfg(
      func=trick_rewards.dynamic_tall_pair_support_exp,
      weight=100.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
      },
    ),
    "dynamic_tall_pair_clearance": RewardTermCfg(
      func=trick_rewards.dynamic_tall_pair_support_clearance_exp,
      weight=35.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        "minimum_clearance": 0.38,
        "std": 0.14,
        "asset_cfg": _DYNAMIC_CLEARANCE_WHEEL_SITES,
      },
    ),
    "dynamic_tall_pair_free_wheel_clearance": RewardTermCfg(
      func=trick_rewards.dynamic_tall_pair_free_wheel_clearance_exp,
      weight=45.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        "minimum_height": 0.22,
        "asset_cfg": _DYNAMIC_FREE_WHEEL_SITES,
      },
    ),
    "dynamic_tall_pair_rate": RewardTermCfg(
      func=trick_rewards.dynamic_tall_pair_spin_rate_exp,
      weight=45.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        "std": 1.0,
      },
    ),
    "dynamic_tall_pair_rate_error": RewardTermCfg(
      func=trick_rewards.dynamic_tall_pair_spin_rate_abs_error,
      weight=-30.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
      },
    ),
    "dynamic_tall_pair_turn_center_stillness": RewardTermCfg(
      func=trick_rewards.DynamicTallPairSupportCenterStillness,
      # Root motion is legitimate while the trunk turns around the wheel axle.
      # The support midpoint, rather than the root, is the local-pivot test.
      weight=110.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        "speed_std": 0.25,
        "asset_cfg": _DYNAMIC_PIVOT_WHEEL_SITES,
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
    "fixed_pair_turn_center_stillness": RewardTermCfg(
      func=trick_rewards.FixedPairSupportCenterStillness,
      # A two-wheel handstand may let the trunk move around its support axle,
      # so root velocity alone is wrong.  The actual two-wheel midpoint must
      # stay in place: this rules out a bicycle-like drive across the floor
      # without prescribing how hips, thighs, or wheels coordinate.
      weight=110.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "sensor_name": wheel_contact_cfg.name,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "speed_std": 0.25,
        "gravity_power": 4.0,
        "asset_cfg": _FIXED_PIVOT_WHEEL_SITES,
      },
    ),
    "spin_rate_error": RewardTermCfg(
      func=trick_rewards.commanded_spin_rate_abs_error,
      weight=-30.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.20,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "fixed_gravity_power": 4.0,
      },
    ),
    "extended_support_legs": RewardTermCfg(
      func=trick_rewards.mode_support_leg_length_min,
      weight=60.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "contact_masks": STANCE_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        "minimum_lengths": (0.0, 0.35, 0.35, 0.35, 0.35),
        "activation_lengths": (0.0, 0.16, 0.16, 0.16, 0.16),
        "length_power": 2.0,
        "gravity_targets": STANCE_GRAVITY_TARGETS,
        "minimum_gravity_alignment": 0.85,
        "asset_cfg": _SUPPORT_LEG_GEOMETRY,
      },
    ),
    # The normal one-hot at zero spin starts in a valid four-wheel support.
    # Prefer its zero residual controller over needless leg actuation, but
    # leave every actual spin request (including dynamic normal) unrestricted.
    "normal_idle_action_effort": RewardTermCfg(
      func=trick_rewards.stance_stationary_action_l2,
      weight=-15.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "velocity_deadband": 0.20,
        "num_modes": 5,
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
  # This is a command-distribution curriculum, not a trajectory curriculum.
  # First the *same actor* learns all four static wheel stands from default
  # four-wheel resets.  It then sees a gentle signed rate and finally the
  # AS2-W-speed range.  The command itself is unchanged throughout:
  # five-way one-hot plus one scalar rate.
  cfg.curriculum = {
    "spin_commands": CurriculumTermCfg(
      func=trick_curriculums.stance_spin_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            "step": 0,
            "mode_probabilities": (0.12, 0.24, 0.24, 0.20, 0.20),
            "spin_idle_probability": 1.0,
            "spin_rate_range": (2.0, 4.0),
          },
          {
            "step": 100_000_000,
            "mode_probabilities": (0.18, 0.25, 0.25, 0.16, 0.16),
            "spin_idle_probability": 0.65,
            "spin_rate_range": (2.0, 4.0),
          },
          {
            "step": 220_000_000,
            "mode_probabilities": (0.22, 0.24, 0.24, 0.15, 0.15),
            "spin_idle_probability": 0.40,
            "spin_rate_range": (4.0, 6.0),
          },
          {
            "step": 320_000_000,
            "mode_probabilities": (0.24, 0.23, 0.23, 0.15, 0.15),
            "spin_idle_probability": 0.25,
            "spin_rate_range": (5.0, 9.0),
          },
        ),
      },
    )
  }
  return cfg
