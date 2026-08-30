"""Minimal fused PPO environment for Go2W aerial rotations.

The actor receives only the shared proprioceptive history and a five-way
one-hot.  This file intentionally defines only physical validity and a small
set of outcome rewards; it has no reference trajectory, phase clock, or
mode-specific joint posture.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_JOINTS,
)
from src.tasks.velocity.mdp import trick_curriculums, trick_rewards
from src.tasks.velocity.mdp.terminations import (
  AerialEventFinished,
  AerialIncompleteLanding,
  AerialPostLandingRelaunch,
  illegal_contact_after_grace,
)
from src.tasks.velocity.mdp.trick_commands import AerialRotationCommandCfg

from .common_env_cfg import (
  AERIAL_AXES,
  configure_compact_aerial_actuators,
  configure_default_idle_actions,
  make_base_go2w_trick_cfg,
)

# Centres of the rigid links that must stay above the wheels in the final
# wheel-first part of a legal aerial.  These names are a collision/clearance
# envelope, not joint targets and not a reference pose.
WHEEL_FIRST_ENVELOPE_BODIES = (
  "base_link",
  "FL_hip", "FL_thigh", "FL_calf",
  "FR_hip", "FR_thigh", "FR_calf",
  "RL_hip", "RL_thigh", "RL_calf",
  "RR_hip", "RR_thigh", "RR_calf",
)


def _aerial_wheels() -> SceneEntityCfg:
  """Return an independently resolvable selector for the four wheel sites."""
  return SceneEntityCfg(
    "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
  )


def unitree_go2w_aerial_rotation_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Train one policy for front, back, left, right, and yaw aerial turns."""
  cfg, wheel_contact_cfg, nonwheel_contact_cfg = make_base_go2w_trick_cfg(play)
  configure_compact_aerial_actuators(cfg)
  # First learn the maneuver on one nominal flat system.  Sensor bias and
  # broad mass/friction randomization are robustness work for a later stage;
  # they otherwise dilute the rare early takeoff/turn evidence.
  cfg.events.pop("encoder_bias", None)
  cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }
  cfg.episode_length_s = 3.5 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": AerialRotationCommandCfg(
      entity_name="robot",
      # All-zero remains a real public idle command: the action gate makes it
      # the ordinary four-wheel default pose and zero wheel speed, including
      # after a landing.  It is not an aerial skill, so do not spend rollout
      # samples on a condition whose output is already deterministic.  Every
      # training event is consequently one of the five requested flips.
      idle_probability=0.0,
      # Every one-hot must stay represented from the very first reset.  This
      # is still one fused actor; it only prevents a lucky easy branch from
      # monopolizing PPO's first updates.
      mode_probabilities=(0.2, 0.2, 0.2, 0.2, 0.2),
      resampling_time_range=(3.5, 3.5),
      sensor_name=wheel_contact_cfg.name,
      axes=AERIAL_AXES,
      # Completion remains an exact one-turn event even though a failed
      # attempt is allowed to use the rest of its three-second episode to
      # recover instead of being cut off mid-brake.
      target_angle=math.tau,
      # One command means one revolution.  A modest 29-degree tolerance keeps
      # float/contact noise from making discovery binary, but a 450-degree
      # rebound is no longer treated as the same endpoint.
      max_overrotation=0.50,
      # Require five consecutive 50-Hz control steps: a transient wheel graze
      # must never count as a completed normal-wheel landing.
      landing_settle_time=0.10,
      # Keep the launch heading visually recognizable after touchdown without
      # treating a few degrees of contact noise as a failed demonstration.
      # abs(dot(q_land, q_launch)) = .985 admits about 20 degrees of total
      # rotation error while still rejecting an obviously changed heading.
      landing_orientation_dot_min=0.985,
      # Two control frames reject a contact glitch but let a genuine compact
      # wheel-free hop receive the same angle signal that later grows into a
      # full aerial turn.  The endpoint remains exactly one 2π event.
      min_ballistic_time=0.04,
      # A wheel touch closes the flight immediately, but the actor retains
      # this same one-hot for a short contact-braking interval before public
      # idle/default-pose settlement begins.  Rebound remains a termination.
      landing_control_time=0.40,
      debug_vis=False,
    )
  }
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=GO2W_LEG_JOINTS,
    scale={
      # This reaches the model's existing hip torque limit at the edge of the
      # residual range.  It neither raises that limit nor stiffens the
      # actuator, but avoids making a one-revolution jump physically
      # unreachable while the calf alone provides the launch impulse.
      # The m1000 audit reached the commanded turn but expanded the wheel
      # package to 0.49--0.59 m and sent the legs as far as 2.3 rad from
      # default.  Limit the two joints that create the large lateral swing;
      # leave the calf impulse free to make the higher, compact launch.
      # 0.35 reduced the m600 peak leg deviation, but also reduced every
      # commanded direction below a full turn.  Keep a wider comparison seed
      # alongside the middle-envelope run so PPO can trade signed rotation
      # authority against the compact-flight outcome without changing the
      # task interface.
      # The wide .55 envelope consistently produces full turns by throwing
      # the leg package out to 2+ rad, then striking the ground before the
      # wheels.  Tighten only the two swing-producing joints; the calf keeps
      # its native launch authority so this remains a physically reachable
      # compact tuck rather than a frozen jump.
      r".*_hip_joint": 0.40,
      r".*_thigh_joint": 0.40,
      # Keep hip/thigh motion compliant while exposing more of the native
      # calf torque range for the single launch impulse.  This is an action
      # envelope, not a prescribed tuck or jump trajectory.
      # The current five-way policy consistently reaches the commanded turn
      # but back/yaw strike the floor before the wheel package can recover.
      # Expose the native calf actuator's remaining bounded position range so
      # PPO can discover a stronger single launch impulse; this is not a
      # torque-limit change and does not prescribe a jump pose.
      r".*_calf_joint": 1.35,
    },
    use_default_offset=True,
  )
  cfg.actions["joint_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=GO2W_WHEEL_JOINTS,
    scale=45.0,
    use_default_offset=True,
  )
  configure_default_idle_actions(
    cfg,
    command_name="trick",
    # The all-zero event vector is the public aerial idle, not mode zero.
    idle_mode_index=None,
    stationary_command_start_index=0,
    command_deadband=0.5,
    idle_contact_sensor_name=wheel_contact_cfg.name,
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"
    # A ten-frame public proprioceptive history was the smallest window that
    # already discovered front/side/yaw turns together.  The longer 20-frame
    # variant improved one yaw landing but diluted the same fixed-size MLP and
    # collapsed the other four one-hots before they discovered a turn.  Keep
    # phase inference entirely in the normal observation history; the strict
    # endpoint below still evaluates the complete launch-frame orientation.
    cfg.observations[group_name].history_length = 10

  # A flip is a legal four-wheel launch, commanded-axis spin during that
  # launch, net commanded-axis angle, and a quiet normal landing.  There is
  # deliberately no pose, timing, joint, or reference-trajectory reward.  A
  # late-flight wheel-first envelope makes the limbs tuck/recover
  # toward a real wheel landing without ever giving them a pose target.  A
  # partial touchdown is a failure shaped only by missing angle.
  cfg.rewards = {
    "ballistic_launch": RewardTermCfg(
      func=trick_rewards.AerialBallisticLaunch,
      # The first half of a flip is simply a powerful legal take-off.  The
      # native torque-limited model can reach this target in a physical action
      # sweep; retaining it prevents a low hop from becoming a local optimum
      # before the slower back/left branches ever spin.
      # A complete maneuver needs enough flight time to accelerate, turn, and
      # then brake before the wheels return.  The former 1.25 m/s, 0.12 s
      # launch cap saturated on the short low hop visible in the m600 audit,
      # leaving no reward distinction between that crash and a real aerial.
      # These remain measured vertical impulse and wheel-free time, not a
      # take-off pose or a prescribed timing trace.
      # The compact branch already reaches a useful body-axis turn, but its
      # measured wheel-free interval is only about 0.58 s and the trunk still
      # reaches the plane before the wheel package.  Strengthen this single
      # launch outcome so PPO values a higher, longer hop; it remains a
      # measured impulse/duration result rather than a pose or timing trace.
      weight=10.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        # The m1000 failures reached only 0.19--0.27 m and roughly 0.4 s of
        # flight: enough to crash through a turn but not to brake and recover
        # a wheel package.  Ask for a longer measured ballistic result, not a
        # takeoff pose or timing trace.
        # The m500 audit now shows the learned body-axis turns reaching the
        # requested angle but striking the floor after a low 0.18--0.30 m
        # hop.  Ask for a higher measured launch impulse so the wheel package
        # has physical clearance to complete its rotation and recover.
        "target_upward_speed": 2.30,
        "target_duration": 0.50,
      },
    ),
    "net_rotation_progress": RewardTermCfg(
      func=trick_rewards.AerialNetRotationProgress,
      # During the single legal flight interval this is the completed fraction
      # of the commanded one-turn angle.  It asks for sustained rotation
      # without a separate rate target that a one-frame spike could game, and
      # it is deliberately bounded so a partial crash cannot numerically
      # dominate the landing result.
      # A compact jump is not sufficient: the fixed-command audits show that
      # the prior 6-point turn term was numerically drowned by the roughly
      # 20-point tuck return.  Raise only this existing normalized net-angle
      # outcome so the fused actor keeps rotating while it searches for the
      # same wheel-first landing; no rate, phase, or pose target is added.
      weight=20.0,
      params={
        "command_name": "trick",
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_angle": math.tau,
      },
    ),
    "tuck_then_wheel_landing": RewardTermCfg(
      func=trick_rewards.AerialTuckThenWheelLanding,
      # The video has a compact airborne package followed by a wheel-first
      # landing package.  The earlier reward requested wheel-lowest geometry
      # from 10% of the turn onward, which encouraged the opposite: legs
      # extended and flailing during the whole flip.  This single geometric
      # outcome rewards compact wheels/legs through the middle of the actual
      # accumulated turn, then wheel-lowest clearance only near landing.
      # The m2500 audit has exposed the actual remaining failure mode: front
      # can make almost one turn, but the trunk reaches the floor first, and
      # the other three body-axis commands never discover their own compact
      # recovery.  Raise this *single physical clearance outcome* above the
      # partial-turn signal so PPO has an earlier, continuous reason to fold
      # the wheel package below every link.  It remains neither a joint pose
      # nor a timing/reference trajectory; the strict four-wheel endpoint
      # below is still the only completion.
      # At m300 the smooth distance bridge was present but too small compared
      # with the one-turn progress return: the policy still reached 1.2 turns
      # with a 0.5--0.6 m wheel package before striking the body.  Raise this
      # measured compactness outcome so folding the wheel/leg package is worth
      # more than simply spinning faster; it still contains no joint target.
      # The first high-launch trial (weight 80, target 0.24 m) produced a
      # taller hop but also drove large limb excursions before a full turn.
      # Keep tuck stronger than the original while restoring room for PPO to
      # discover the compact, fast body-axis rotation.
      # The m500 audit still leaves the wheel package around 0.56 m from the
      # trunk, while the configured compact outcome is 0.27 m.  Make that
      # existing measured distance meaningfully discriminative without
      # introducing another pose or phase signal.
      weight=80.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "body_names": WHEEL_FIRST_ENVELOPE_BODIES,
        "target_angle": math.tau,
        # These are actual accumulated-turn fractions, not a clock or a
        # reference pose.  The policy itself determines when it reaches them.
        "tuck_start_turn_fraction": 0.04,
        "tuck_ramp_end_turn_fraction": 0.16,
        "tuck_end_turn_fraction": 0.64,
        # The default wheel-root mean is about 0.364 m, while AS2W's airborne
        # package is visibly tighter.  Move the measured compactness optimum
        # modestly inward and strengthen this existing outcome term so a wide
        # body strike is no longer as profitable as folding the wheel package.
        "tuck_target_wheel_root_distance": 0.27,
        "tuck_max_wheel_root_distance": 0.40,
        "landing_start_turn_fraction": 0.64,
        # Keep the wheel-first result dense even when an exploratory leg is
        # still below the wheel plane; the identical clearance score reaches
        # one only when every non-wheel link is safely above it.
        "minimum_clearance_for_progress": -0.30,
        "target_clearance": 0.12,
        "asset_cfg": _aerial_wheels(),
      },
    ),
    "landing_recovery": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      # A final whole-body orientation-return signal distinguishes a clean 2π
      # turn from one that has the right commanded-axis integral but unwanted
      # off-axis tumble.  It contains no joint, timing, or rate reference;
      # the strict four-wheel endpoint is still paid by the same term.
      # The late-flight recovery measurement is the only dense signal for
      # braking and returning the wheel package before touchdown.  It must be
      # strong enough to compete with launch/turn progress, while the larger
      # one-off completion multiplier below still makes a quiet four-wheel
      # result the best event outcome.
      # A high dense root-orientation return made a fast near-2π body strike
      # profitable even after its terminal failure was made negative.  Keep
      # this as a small late-flight diagnostic, but reserve the meaningful
      # landing return for the actual quiet four-wheel completion below.
      # Keep this whole-frame return diagnostic modest during discovery.  A
      # stronger dense term made the first 600 iterations prefer fast body
      # strikes over the already-demonstrated wheel-first landing pathway.
      # The physical endpoint bonus below remains the dominant landing goal.
      # A m300 audit still overshot to roughly 1.1--1.25 turns.  Strengthen
      # the existing whole-body orientation/quietness return so PPO learns to
      # brake at one turn instead of trading the endpoint for excess rate.
      weight=8.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "nonwheel_sensor_name": nonwheel_contact_cfg.name,
        "target_angle": math.tau,
        "landing_gravity_std": 0.30,
        "landing_orientation_dot_min": 0.985,
        "max_overrotation": 0.50,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
        # A complete turn that reaches its launch-frame orientation while
        # still spinning quickly has no physical path to a quiet wheel
        # touchdown.  Blend the existing late-flight orientation-return
        # score into a measured whole-body angular-speed score only over the
        # final forty percent.  This rewards braking an actual nearly-complete
        # aerial, not a prescribed phase, joint pose, or commanded rate.
        "late_flight_brake_start_turn_fraction": 0.60,
        "late_flight_brake_angular_speed_std": 14.0,
        # A legal four-wheel touchdown with residual full-frame error is a
        # physical discovery step.  It receives at most one third of the
        # strict endpoint credit below, so it guides final alignment without
        # redefining a wrong-heading landing as success.
        "partial_landing_bonus": 8.0,
        # Preserve the 48-point one-off completion outcome (2 * 24 = 48).
        "completion_bonus": 24.0,
        "post_idle_settle_time": 0.30,
      },
    ),
    # Any non-timeout terminal result pays once, in proportion to its missing
    # desired-axis angle.  It keeps every partial landing invalid without
    # making a stronger, correctly directed jump indistinguishable from a
    # zero-turn fall during PPO discovery.
    "event_failure": RewardTermCfg(
      func=trick_rewards.aerial_event_failure,
      # Aerial f147 showed that -2500 made this sparse terminal cost dominate
      # every available recovery gradient and explode the critic loss.  This
      # remains stricter than the old prototype, but a legal late-flight
      # recovery can now improve its return before the first perfect landing
      # is sampled.
      # At the final curriculum stage an illegal contact must erase the
      # dense takeoff/turn discovery return.  Otherwise a nearly-complete
      # high-speed trunk strike remains more profitable than attempting the
      # rare quiet wheel landing.  This is still an outcome-only cost: it
      # names neither a joint configuration nor a flight trajectory.
      weight=-30.0,
      params={
        "command_name": "trick",
        "target_angle": math.tau,
        # Preserve the original early absolute loss while the final terminal
        # scale is stronger, so a tentative legal lift remains discoverable.
        # The f170 m800 audit establishes that the previous 0.0625 factor
        # makes a visibly illegal half-turn profitable: launch, turn, and
        # wheel-envelope gains sum to roughly +3.8 while its terminal loss is
        # only -0.2.  Keep enough early credit for a real takeoff, but make a
        # no-turn crash lose to a compact ballistic attempt from update zero.
        "early_missing_angle_cost": 0.20,
        "final_missing_angle_cost": 1.0,
        # A nearly-complete turn that lands its trunk or a leg is still a
        # failure.  Angle-only cost accidentally made that failure free as
        # progress approached 2π, selecting a fast crash instead of wheel
        # recovery.  The completion term remains the only way to erase this
        # base outcome cost.
        "early_non_timeout_base_cost": 0.0,
        "final_non_timeout_base_cost": 1.0,
        # At the desired 2π angle, distinguish a rapid body/leg impact from
        # an otherwise quiet invalid touchdown.  This supplies a continuous
        # terminal braking preference using only the measured root speed; it
        # is not a reference angular-rate command or a motion trajectory.
        "terminal_angular_speed_scale": 8.0,
        "minimum_motion_failure_fraction": 0.25,
        # The m1000 fixed-command audit is the useful window: four modes have
        # wheel-first landings, while turning the terminal cost fully on at
        # update 1,200 collapses those behaviours before the shared actor has
        # solved the hard back direction.  Delay and lengthen this same
        # outcome-only validity ramp so PPO can retain the discovered landing
        # basin while the difficult signed axis receives more samples.
        "base_cost_ramp_start_steps": 48_000,
        "base_cost_ramp_steps": 72_000,
      },
    ),
  }
  # Give the policy a short physical launch window before rejecting a
  # non-wheel scrape.  With immediate history-based termination, the
  # exploratory crouch/impulse was cut off on its first transient contact and
  # every direction learned a ballistic body strike instead of a tucked jump.
  # The grace is only for takeoff; sustained trunk/leg support remains an
  # illegal outcome and is still terminated once the window expires.
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=illegal_contact_after_grace,
    params={
      "sensor_name": nonwheel_contact_cfg.name,
      "force_threshold": 10.0,
      # A full wheel-first turn needs roughly 0.4--0.5 s of flight in the
      # current model.  The stronger launch experiment needs a little more
      # physical room to complete the wheel-first touchdown; body support is
      # still an illegal outcome once this short discovery window expires.
      "grace_period_s": 0.75,
    },
  )
  # Collision ends an episode immediately.  The decisive terminal term
  # distinguishes an illegal partial turn from a wheelward recovery without
  # prescribing how the limbs should execute either action.
  # The command becomes all-zero after its first landing.  A later genuine
  # wheel-free interval is a second attempt, even if it began as a rebound,
  # and is therefore an event failure rather than a way to continue hopping.
  cfg.terminations["post_landing_relaunch"] = TerminationTermCfg(
    func=AerialPostLandingRelaunch,
    params={
      "command_name": "trick",
      "sensor_name": wheel_contact_cfg.name,
      # A wheel-first landing can unload the suspension for a few control
      # frames before all four wheels settle.  The one-shot event is already
      # closed at first contact, so allow this measured transient to damp
      # instead of labelling it a second jump at 80 ms.
      "min_ballistic_time": 0.30,
      # A first individual tyre graze can rebound during a real landing.
      # Arm the second-jump detector only after a brief all-wheel hold, so
      # that rebound is available to the same one-shot contact controller.
      # Give the default idle controller time to absorb a passive suspension
      # bounce before arming the second-flight detector.  The one-hot itself
      # still ends at the first four-wheel touchdown.
      "arming_settle_time": 0.50,
    },
  )
  if not play:
    # One jump closes after its controlled landing window, then presents the
    # literal all-zero/default controller.  The short idle interval is enough
    # to test that it actually settles without spending an entire 3-s rollout
    # on a finished event.
    cfg.terminations["aerial_event_finished"] = TerminationTermCfg(
      func=AerialEventFinished,
      params={
        "command_name": "trick",
        "post_idle_settle_time_s": 0.30,
        "target_angle": math.tau,
        "max_overrotation": 0.50,
      },
      time_out=True,
    )
    # A first wheel landing ends the physical aerial attempt.  If it has not
    # reached exactly one allowed turn after the same brief default-idle
    # window, make it a real failure rather than an unpenalized timeout.  The
    # existing generic termination cost then removes the observed profitable
    # 0.2--0.4-turn hop without supplying a pose, timing, or motion reference.
    cfg.terminations["aerial_incomplete_landing"] = TerminationTermCfg(
      func=AerialIncompleteLanding,
      params={
        "command_name": "trick",
        "post_idle_settle_time_s": 0.30,
        "target_angle": math.tau,
        "max_overrotation": 0.50,
      },
    )
  # This is sampling curriculum, not a reference-motion curriculum.  Every
  # emitted command is still one complete 2π event from the first sample.
  # Each one-hot shares this one actor.  Yaw is mechanically easier than the
  # body-axis flips, so retain a small yaw fraction during early discovery
  # instead of letting it dominate the shared policy.  Crucially, this is
  # never zero: an all-zero early yaw branch made a five-way checkpoint
  # incapable of responding to its yaw command.  ``common_step_counter`` is
  # measured in policy control steps (48 per PPO update for this task), not
  # PPO iterations.
  cfg.curriculum = {
    "aerial_commands": CurriculumTermCfg(
      func=trick_curriculums.aerial_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            # Train each signed body axis as a pair from the first update.
            # The previous run learned the mechanically easy right branch but
            # repeatedly lost back/yaw.  Keep those hard signed axes
            # over-sampled throughout discovery while retaining every one-hot
            # in the same fused actor; this changes only command frequency.
            "step": 0,
            "idle_probability": 0.0,
            # Keep every commanded axis equally visible to this single fused
            # actor.  The earlier hard-branch weighting produced a reliable
            # front flip but weak fixed-command generalization elsewhere.
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
          },
          {
            "step": 19_200,
            "idle_probability": 0.0,
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
          },
          {
            "step": 38_400,
            "idle_probability": 0.0,
            "mode_probabilities": (0.20, 0.20, 0.20, 0.20, 0.20),
          },
          {
            # Keep every one-hot active while the shared launch/tuck solution
            # is refined.  The final stages below give the hard back direction
            # extra samples without splitting the policy.
            "step": 57_600,
            "idle_probability": 0.0,
            "mode_probabilities": (0.15, 0.40, 0.15, 0.15, 0.15),
          },
          {
            "step": 76_800,
            "idle_probability": 0.0,
            "mode_probabilities": (0.15, 0.40, 0.15, 0.15, 0.15),
          },
          {
            "step": 96_000,
            "idle_probability": 0.0,
            # Keep the difficult back branch overrepresented in the final
            # fused policy; front/side/yaw remain present for retention.
            "mode_probabilities": (0.15, 0.40, 0.15, 0.15, 0.15),
          },
        ),
      },
    )
  }
  return cfg
