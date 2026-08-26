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
        # Both one-hots can retain a shallow slanted support at roughly 0.89
        # orientation alignment.  A sixth power keeps the reset bridge but
        # makes that visibly incomplete outcome materially worse than a true
        # upright support, without introducing any joint-pose target.
        "orientation_power": 6.0,
        # From normal four-wheel reset the desired front/rear attitude is
        # only half aligned, while the target wheel pair and tall clearance
        # cannot improve until the body has first begun to tip.  Reserve part
        # of this same measured support outcome for that physical attitude
        # progress; only the simultaneous correct contacts and clearance can
        # reach its final full score.
        "orientation_progress_floor": 0.50,
        # The front/rear stances use identical outcome measurements, but the
        # front support discovers first and otherwise consumes the shared
        # actor's PPO advantage.  Weight the harder rear *same outcome* so
        # its one-hot receives comparable policy pressure without a separate
        # network, a pose target, or a reference transition.
        "mode_weights": (0.0, 1.0, 2.5),
        "clearance_power": 1.0,
        # A zero x/yaw request is a static two-wheel stand.  The m200 fixed
        # audit found that without this existing outcome's stillness factor,
        # rear could collect a large support reward while translating at
        # 0.54 m/s and yawing at 1.02 rad/s.  These measured root speeds are
        # applied only when both public command components are zero; moving
        # x/yaw requests later in the same fused policy are unaffected.
        "static_command_start_index": 3,
        "command_deadband": 0.04,
        "static_angular_velocity_scale": 0.55,
        "static_linear_velocity_scale": 0.15,
        # The m600 replay reaches a tall rear support but repeatedly throws
        # itself sideways.  Once the measured support is substantially there,
        # static one-hots must rank a truly quiet wheel balance above that
        # high-speed transient; this does not apply to requested x/yaw motion.
        "static_stillness_floor": 0.0,
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
            # Start rear-biased static discovery in the same fused actor,
            # while retaining a small default-normal replay share.  The
            # all-zero normal idle is still supplied by the action gate;
            # this one-hot share merely keeps its observation branch present
            # before normal x/yaw commands enter.
            "mode_probabilities": (0.10, 0.25, 0.65),
            "mode_idle_probabilities": (1.0, 1.0, 1.0),
            "lin_vel_x_range": (0.0, 0.0),
            "yaw_rate_range": (0.0, 0.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Continue rear-biased static discovery through update 1,200.
            # The m600 audit can touch a rear support but cannot hold it: its
            # commanded contact rate is only 11%.  Asking for locomotion at
            # m750 would turn that incomplete balance into a rolling escape.
            # Keep the same fused actor and direct command interface, but
            # make a quiet static support the prerequisite for x/yaw motion.
            "step": 25_600,
            "mode_probabilities": (0.10, 0.35, 0.55),
            "mode_idle_probabilities": (1.0, 1.0, 1.0),
            "lin_vel_x_range": (0.0, 0.0),
            "yaw_rate_range": (0.0, 0.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Introduce conservative x/yaw requests after 1,200 updates of
            # static support discovery.  The command still switches directly
            # between all three modes; the reduced range only lets the same
            # policy discover rolling without overwriting the new supports.
            "step": 76_800,
            "mode_probabilities": (0.30, 0.25, 0.45),
            "mode_idle_probabilities": (0.10, 0.25, 0.25),
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Reach the requested command range at update 1,800, leaving
            # 1,200 full-range updates after the static forms are actually
            # stable, rather than consuming that capacity on a moving fall.
            "step": 115_200,
            "mode_probabilities": (0.30, 0.25, 0.45),
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
  video's compact, level, four-wheel common-axle local pivot.  Front/rear
  one-hots request their named pivots; left/right one-hots are the physically
  distinct side two-wheel pivots.  All active one-hots retain the signed
  spin-rate input.
  All five modes still share exactly the same one-hot plus signed-rate command
  interface and one policy.
  """
  cfg, wheel_contact_cfg, _ = make_base_go2w_trick_cfg(play)
  configure_ground_support_actuators(cfg)
  _configure_fast_discovery(cfg)
  # A nonzero normal command must reshape the ordinary four-wheel rectangle
  # into a compact common axle while all wheels remain grounded.  A modest
  # extension beyond the shared ±0.55-rad residual makes that geometry
  # reachable without adding a prescribed joint pose or torque authority.
  cfg.actions["joint_pos"].scale[r".*_hip_joint"] = 0.70
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
    # Every zero-rate request, including a normal one-hot supplied by an
    # external caller, uses the literal four-wheel default controller.
    idle_mode_index=None,
    stationary_command_start_index=5,
    command_deadband=0.20,
    idle_contact_sensor_name=wheel_contact_cfg.name,
  )
  _use_history(cfg, "trick")

  cfg.rewards = {
    "commanded_spin_pivot": RewardTermCfg(
      func=trick_rewards.StanceSpinPivotResult,
      # Normal is a compact level four-wheel pivot, not wheel-steering.
      weight=18.0,
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
        "upright_support_weight": 0.20,
        # The early common-axle geometry bridge is only for discovering the
        # four-wheel packing from its ordinary rectangular reset.  It fades
        # after discovery so the same geometry must carry the signed rate at
        # a stationary all-wheel centroid.
        "normal_final_geometry_weight": 0.04,
        # A spin rollout contains 64 control steps.  Keep the bridge through
        # the first 600 PPO updates, then make the normal branch earn its
        # reward from an actual local pivot while the named forms are learned.
        "normal_geometry_decay_start_steps": 38_400,
        "normal_geometry_decay_steps": 38_400,
        # Keep the signed world-z rate valuable through a direct one-hot
        # change.  The strict final tracking score remains present; this
        # dense measured-rate component merely makes acceleration toward it
        # discoverable without adding a pose or transition target.
        "rate_progress_weight": 0.50,
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
            # First discover a compact, level four-wheel common axle from the
            # ordinary reset.  At 64 control steps/update, this lasts 600
            # updates—not 1,200.  The policy still needs a normal discovery
            # period, but a 3,000-update run must spend most of its time on
            # the required five-command fused behaviour.
            "mode_probabilities": (1.0, 0.0, 0.0, 0.0, 0.0),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 0.0,
            "spin_rate_range": (0.5, 1.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Introduce all four named pivots once the normal bridge has had
            # 600 updates.  Keep their first rate deliberately low and do not
            # yet ask the actor to solve a support change in the same sample.
            "step": 38_400,
            "mode_probabilities": (0.50, 0.15, 0.15, 0.10, 0.10),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 0.0,
            "spin_rate_range": (0.5, 2.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Give the named supports about 350 updates at low rate, then
            # begin sparse direct changes.  Previously this stage began only
            # at update 2,000, leaving too little training after the five
            # command interface was ever exercised.
            "step": 60_800,
            "mode_probabilities": (0.45, 0.20, 0.20, 0.075, 0.075),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 0.10,
            "spin_rate_range": (2.0, 5.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Keep a substantial normal replay share while the low-rate named
            # supports are accelerated.  Make the next request overlap in
            # speed and keep switches rare until each pair can survive on its
            # own.
            "step": 86_400,
            "mode_probabilities": (0.45, 0.20, 0.20, 0.075, 0.075),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 0.15,
            "spin_rate_range": (3.0, 6.0),
            "resampling_time_range": (6.0, 6.0),
          },
          {
            # Put the fused high-rate/direct-switch distribution in place by
            # update 1,700.  It therefore receives roughly 1,300 updates in
            # this zero-start run instead of only the final 200.
            # Reference-rate 10--15 rad/s remains a subsequent zero-start
            # long run after all five supports are validated.
            "step": 108_800,
            "mode_probabilities": (0.40, 0.22, 0.22, 0.08, 0.08),
            "spin_idle_probability": 0.0,
            "upright_static_probability": 0.0,
            "direct_switch_probability": 0.25,
            "spin_rate_range": (5.0, 8.0),
            "resampling_time_range": (6.0, 6.0),
          },
        ),
      },
    )
  }
  return cfg
