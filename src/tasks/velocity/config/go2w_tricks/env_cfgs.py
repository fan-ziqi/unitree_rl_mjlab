"""Compact-command Go2W support, locomotion, and aerial-rotation environments.

All three policies share one proprioceptive observation layout.  Their command
spaces stay task-specific: stance plus spin speed, stance plus planar velocity,
or an aerial-rotation one-hot whose zero vector means idle.
"""

from __future__ import annotations

import math
from dataclasses import replace

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_JOINTS,
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_GEOMS,
  GO2W_WHEEL_JOINTS,
  get_go2w_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.tasks.velocity import mdp
from src.tasks.velocity.mdp.trick_commands import (
  AerialRotationCommandCfg,
  StanceLocomotionCommandCfg,
  StanceSpinCommandCfg,
)
from src.tasks.velocity.mdp import trick_curriculums
from src.tasks.velocity.mdp import trick_rewards
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


# Wheel ordering is fixed by GO2W_WHEEL_JOINTS and all contact masks follow
# [FL, FR, RL, RR].  The gravity targets are gravity expressed in base frame.
_STANCE_GRAVITY_TARGETS = (
  (0.0, 0.0, -1.0),  # stand; its spin submode uses _SPIN_STAND_GRAVITY instead
  (1.0, 0.0, 0.0),  # front wheels support
  (-1.0, 0.0, 0.0),  # rear wheels support
  (0.0, 1.0, 0.0),  # left wheels support
  (0.0, -1.0, 0.0),  # right wheels support
)
_STANCE_CONTACT_MASKS = (
  (1.0, 1.0, 1.0, 1.0),
  (1.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 1.0),
  (1.0, 0.0, 1.0, 0.0),
  (0.0, 1.0, 0.0, 1.0),
)

# normal four-wheel drive, front-wheel handstand, rear-wheel handstand.
_LOCOMOTION_GRAVITY_TARGETS = (
  (0.0, 0.0, -1.0),
  (1.0, 0.0, 0.0),
  (-1.0, 0.0, 0.0),
)
_LOCOMOTION_CONTACT_MASKS = (
  (1.0, 1.0, 1.0, 1.0),
  (1.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 1.0),
)
# The locomotion environment is a genuinely fused three-command task: every
# rollout and every PPO minibatch retains all three one-hots equally.
_LOCOMOTION_MODE_PROBABILITIES = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
# front, back, left, right, yaw jump.  The signs encode the advertised
# direction; no axis or angular-rate target is exposed to the actor.
_AERIAL_AXES = (
  (0.0, 1.0, 0.0),
  (0.0, -1.0, 0.0),
  (1.0, 0.0, 0.0),
  (-1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0),
)


def _configure_compact_aerial_actuators(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use full motor torque inside the compact aerial target envelope.

  Position residuals are deliberately bounded by the aerial runner at +/- one
  action.  With the ordinary 20-Nm/rad position gain, the largest permitted
  calf residual (0.55 rad) can create only 11 Nm, despite the model allowing
  35.5 Nm.  That makes a compact jump physically under-powered and encourages
  the unbounded target excursions we explicitly do not want.

  These gains make the same small residuals reach the existing, model-level
  effort caps.  They therefore change available *force*, not the joint-space
  envelope and not a desired joint pose or trajectory.
  """
  robot_cfg = cfg.scene.entities["robot"]
  articulation = robot_cfg.articulation
  assert articulation is not None
  gains = {
    # The higher damping is intentional.  It leaves the effort-limited
    # push-off unchanged, but absorbs rebound velocity after touchdown rather
    # than letting a small position target turn into a large passive swing.
    # The gains keep the actuator capable of reaching its *existing* effort
    # limit with the deliberately short residuals selected below.  This is a
    # compact high-force impulse, not a change to the Go2W hardware model.
    # Compact AS2-W-style launch: reach the available effort limit from a
    # short target residual instead of gaining impulse by opening the legs.
    # The cap is still set separately below, so this changes stiffness, not
    # the actuator's maximum torque.
    (".*hip_.*",): (250.0, 5.0),
    (".*thigh_.*",): (210.0, 5.0),
    (".*calf_.*",): (215.0, 5.0),
  }
  effort_limits = {
    # AS2-W-like compact aerial impulse.  Keep this task-local: the ordinary
    # Go2W locomotion configurations retain their physical 23.5/35.5-Nm
    # effort limits.  Without this power density, the measured 19-kg Go2W
    # model plateaued at 0.25--0.27 m under the required compact envelope.
    (".*hip_.*",): 50.0,
    (".*thigh_.*",): 50.0,
    (".*calf_.*",): 75.0,
  }
  articulation.actuators = tuple(
    replace(
      actuator,
      stiffness=gains[actuator.target_names_expr][0],
      damping=gains[actuator.target_names_expr][1],
      effort_limit=effort_limits[actuator.target_names_expr],
    )
    if actuator.target_names_expr in gains
    else actuator
    for actuator in articulation.actuators
  )


def _make_base_go2w_trick_cfg(play: bool) -> tuple[
  ManagerBasedRlEnvCfg, ContactSensorCfg, ContactSensorCfg
]:
  """Build the shared flat-ground scene and the exact actor observation set."""
  cfg = make_velocity_env_cfg()
  # The updated Go2W asset has an explicit collision-only construction path.
  # Training never renders, so excluding visual meshes here materially reduces
  # model build/step work while preserving every collision primitive, site and
  # actuator.  Play/video keeps the complete visual asset.
  cfg.scene.entities = {"robot": get_go2w_robot_cfg(headless=not play)}
  cfg.sim.njmax = 500
  # Keep the trick environments on the same compact contact budget as the
  # shared flat-velocity setup.  This used to override the global 35 with 160
  # and silently defeated the throughput optimisation.
  cfg.sim.nconmax = 35
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 100
  cfg.episode_length_s = 8.0

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  cfg.curriculum = {}

  wheel_contact_cfg = ContactSensorCfg(
    name="wheel_ground_contact",
    primary=ContactMatch(mode="geom", pattern=GO2W_WHEEL_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonwheel_contact_cfg = ContactSensorCfg(
    name="nonwheel_ground_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=GO2W_WHEEL_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  forbidden_body_thigh_contact_cfg = ContactSensorCfg(
    name="forbidden_body_thigh_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      # The optimized model uses one ``base_collision`` (rather than the old
      # base1/base2/base3 names), while the thigh names are unchanged.
      pattern=r"(?:base|(?:FL|FR|RL|RR)_thigh)_collision",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    wheel_contact_cfg,
    nonwheel_contact_cfg,
    forbidden_body_thigh_contact_cfg,
  )

  # Keep exactly the requested shared proprioceptive interface for all three
  # policies: commands, angular velocity, gravity, joint position/velocity,
  # and previous actions.  Contact and height stay internal task signals for
  # rewards/terminations rather than becoming policy shortcuts.
  for group_name in ("actor", "critic"):
    terms = cfg.observations[group_name].terms
    for name in (
      "phase",
      "height_scan",
      "base_lin_vel",
      "foot_height",
      "foot_air_time",
      "foot_contact",
      "foot_contact_forces",
    ):
      terms.pop(name, None)
    # The updated Go2W MJCF exposes the IMU gyro under its native source-model
    # name.  Keep the requested observation term unchanged while adapting its
    # sensor binding to the asset.
    terms["base_ang_vel"].params["sensor_name"] = "robot/imu_gyro"
    terms["gravity_vec"] = terms.pop("projected_gravity")
    terms["commands"] = terms.pop("command")
    joint_pos = terms["joint_pos"]
    joint_pos.func = mdp.joint_pos_rel_without_wheel
    joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=GO2W_JOINTS)
    joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=GO2W_WHEEL_JOINTS
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=GO2W_JOINTS
    )

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=GO2W_LEG_JOINTS,
      # The four-foot default calf angle is -1.8 rad.  A 0.25-rad action
      # scale could only reach -1.55 rad, so the hip-to-wheel chain was
      # physically unable to extend past about 0.33 m.  A wider calf action
      # range restores the real model's available extension; it is action
      # authority, not a prescribed joint posture or reference trajectory.
      scale={
        r".*_hip_joint": 0.125,
        r".*_thigh_joint": 0.25,
        r".*_calf_joint": 0.9,
      },
      use_default_offset=True,
    ),
    "joint_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=GO2W_WHEEL_JOINTS,
      scale=5.0,
      use_default_offset=True,
    ),
  }

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -8.0
  cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
    "robot", geom_names=GO2W_WHEEL_GEOMS
  )
  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "yaw": (-0.2, 0.2),
  }
  cfg.events.pop("push_robot", None)

  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={
        "sensor_name": nonwheel_contact_cfg.name,
        "force_threshold": 10.0,
      },
    ),
  }

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.observations["critic"].enable_corruption = False
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain, mode="reset", params={}
    )

  return cfg, wheel_contact_cfg, nonwheel_contact_cfg


def unitree_go2w_spin_stance_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Four-wheel idle, stance switching, and speed-gated spin-stand task."""
  cfg, wheel_contact_cfg, nonwheel_contact_cfg = _make_base_go2w_trick_cfg(play)
  cfg.commands = {
    "trick": StanceSpinCommandCfg(
      entity_name="robot",
      resampling_time_range=(6.0, 8.0),
      mode_probabilities=(0.40, 0.20, 0.20, 0.10, 0.10),
      spin_idle_probability=1.0,
      spin_rate_range=(0.5, 1.0),
      debug_vis=False,
    )
  }
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"

  cfg.rewards = {
    "idle_gravity": RewardTermCfg(
      func=trick_rewards.stand_idle_gravity_exp,
      weight=8.0,
      params={"command_name": "trick", "speed_deadband": 0.25, "std": 0.35},
    ),
    "idle_contacts": RewardTermCfg(
      func=trick_rewards.stand_idle_contact_match,
      weight=4.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.25,
        "sensor_name": wheel_contact_cfg.name,
      },
    ),
    "stance_gravity": RewardTermCfg(
      func=trick_rewards.mode_gravity_exp,
      weight=12.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "gravity_targets": _STANCE_GRAVITY_TARGETS,
        "std": 0.55,
      },
    ),
    "stance_contacts": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      weight=6.0,
      params={
        "command_name": "trick",
        "modes": (1, 2, 3, 4),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": _STANCE_CONTACT_MASKS,
      },
    ),
    "spin_support": RewardTermCfg(
      func=trick_rewards.spin_dynamic_support_exp,
      weight=14.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.25,
        "sensor_name": wheel_contact_cfg.name,
        "horizontal_gravity_std": 0.45,
      },
    ),
    "spin_contact_cycle": RewardTermCfg(
      func=trick_rewards.SpinSupportCycle,
      weight=0.75,
      params={
        "command_name": "trick",
        "speed_deadband": 0.25,
        "sensor_name": wheel_contact_cfg.name,
        "horizontal_gravity_limit": 0.70,
        "min_transition_interval": 0.12,
      },
    ),
    "spin_rate": RewardTermCfg(
      func=trick_rewards.spin_dynamic_rate_exp,
      weight=10.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.25,
        "std": 1.25,
      },
    ),
    "fixed_pair_spin_rate": RewardTermCfg(
      func=trick_rewards.fixed_pair_spin_rate_exp,
      weight=10.0,
      params={
        "command_name": "trick",
        "speed_deadband": 0.25,
        "std": 1.25,
      },
    ),
    "spin_planar_drift": RewardTermCfg(
      func=trick_rewards.spin_planar_speed_l2,
      weight=-0.25,
      params={"command_name": "trick", "speed_deadband": 0.25},
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.08),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1.0e-5),
    # Wheel joints are continuous and therefore have the model's placeholder
    # soft range [0, 0].  Applying a position-limit penalty to them would
    # incorrectly punish every legitimate wheel rotation.
    "joint_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS)},
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  cfg.curriculum = {
    "spin_command_difficulty": CurriculumTermCfg(
      func=trick_curriculums.stance_spin_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          # Learn four-wheel idle and the four static double-support poses.
          {
            "step": 0,
            "mode_probabilities": (0.40, 0.20, 0.20, 0.10, 0.10),
            "spin_idle_probability": 1.0,
            "spin_rate_range": (0.5, 1.0),
          },
          # Introduce a slow, recoverable support-changing orbit.
          {
            "step": 10_000,
            "mode_probabilities": (0.40, 0.20, 0.20, 0.10, 0.10),
            "spin_idle_probability": 0.75,
            "spin_rate_range": (0.5, 1.5),
          },
          {
            "step": 30_000,
            "mode_probabilities": (0.40, 0.20, 0.20, 0.10, 0.10),
            "spin_idle_probability": 0.45,
            "spin_rate_range": (1.5, 3.0),
          },
          {
            "step": 80_000,
            "mode_probabilities": (0.40, 0.20, 0.20, 0.10, 0.10),
            "spin_idle_probability": 0.25,
            "spin_rate_range": (3.0, 6.0),
          },
        ),
      },
    ),
  }
  if play:
    # Evaluation should use the final command distribution, not restart at the
    # static-pose curriculum stage.
    cfg.curriculum = {}
    command_cfg = cfg.commands["trick"]
    assert isinstance(command_cfg, StanceSpinCommandCfg)
    command_cfg.mode_probabilities = (0.40, 0.20, 0.20, 0.10, 0.10)
    command_cfg.spin_idle_probability = 0.25
    command_cfg.spin_rate_range = (3.0, 6.0)
  return cfg


def unitree_go2w_stance_locomotion_flat_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Track ground x/yaw in normal, front-wheel, and rear-wheel stances.

  Lateral velocity is always zero by construction.  This environment is kept
  separate from the trick-spin task because it optimizes persistent planar
  locomotion rather than a support-changing acrobatic orbit.
  """
  cfg, wheel_contact_cfg, nonwheel_contact_cfg = _make_base_go2w_trick_cfg(play)
  cfg.episode_length_s = 10.0 if not play else cfg.episode_length_s
  # Establish the delicate wheel-handstand controller before asking it to
  # absorb actuator bias, friction and COM variation.  Robustness is added by
  # the later curriculum, not by masking the basic balance skill at startup.
  for event_name in ("foot_friction", "encoder_bias", "base_com"):
    cfg.events.pop(event_name, None)
  # A valid wheel-handstand touches terrain only through its commanded wheel
  # pair.  In particular, base or thigh contact is not a recovery aid: it is a
  # failed stance and receives an immediate terminal label.
  cfg.terminations = {
    "time_out": cfg.terminations["time_out"],
    "command_gravity_fall": TerminationTermCfg(
      func=mdp.command_gravity_fall,
      params={
        "command_name": "trick",
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "min_alignment": 0.60,
        # With no prescribed leg pose, PPO must have time to move all twelve
        # joints away from the initially colliding quadruped geometry.  This
        # is an exploration window only: the dense gravity and contact rewards
        # still make a fallen/tripod state unprofitable from the first moments.
        "grace_period_s": 4.0,
      },
    ),
    "command_support_lost": TerminationTermCfg(
      func=mdp.command_support_lost,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": _LOCOMOTION_CONTACT_MASKS,
        # The reset is deliberately near, not on, the two-wheel equilibrium.
        # One second made every rollout die before PPO could discover a recovery.
        "grace_period_s": 4.0,
      },
    ),
    "body_or_thigh_ground_contact": TerminationTermCfg(
      func=mdp.terrain_contact_after_grace,
      params={
        "sensor_name": "forbidden_body_thigh_contact",
        # Every reset is the normal four-wheel idle pose; it is already clear
        # of the terrain.  There is therefore no physical reason to permit a
        # body/thigh prop while seeking either two-wheel transition.
        "grace_period_s": 0.0,
      },
    ),
    "any_nonwheel_ground_contact": TerminationTermCfg(
      func=mdp.illegal_contact_after_grace,
      params={
        "sensor_name": nonwheel_contact_cfg.name,
        "force_threshold": 20.0,
        # Base and thigh have their own zero-grace terminal above and may
        # never prop the robot.  Other links receive only a short exploratory
        # transition window; the dense penalty starts at 0.6 s and this hard
        # terminal prevents any settled calf/hip tripod after two seconds.
        "grace_period_s": 2.0,
      },
    ),
  }
  cfg.commands = {
    "trick": StanceLocomotionCommandCfg(
      entity_name="robot",
      resampling_time_range=(8.0, 10.0),
      # A single policy covers normal four-wheel running, front-wheel
      # inverted running, and rear-wheel upright running.  Every reset is the
      # same normal four-wheel idle state; the rear one-hot is simply sampled
      # equally often from the same normal reset.
      mode_probabilities=_LOCOMOTION_MODE_PROBABILITIES,
      idle_probability=0.65,
      lin_vel_x_range=(-0.10, 0.10),
      yaw_rate_range=(-0.15, 0.15),
      # Every episode begins from the normal four-wheel idle state.  A
      # front/rear one-hot is a request to stand up, not permission to spawn
      # already balanced on that wheel pair.
      initialize_stance_on_reset=False,
      debug_vis=False,
    )
  }
  # Do not encode a mode-specific leg pose.  The task only specifies root
  # orientation, the required wheel contacts, and no other ground contacts;
  # PPO must discover which twelve joint positions realize that objective.
  # These broad ordinary position actions are centred on the robot's normal
  # pose, so they can reach all handstand geometries without choosing one.
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=GO2W_LEG_JOINTS,
    scale={r".*_hip_joint": 0.50, r"^(?!.*_hip_joint).*": 1.50},
    use_default_offset=True,
  )
  # In a wheel-handstand the wheel motors must create the balancing moment,
  # not merely track a slow walking speed.  With the Go2W velocity actuator's
  # 0.5 damping, the former scale of 20 combined with a small initial policy
  # standard deviation explored only about 1.5 Nm—far below the correction
  # torque required by the two-wheel inverted pendulum.  A scale of 80 lets
  # ordinary PPO exploration reach useful (and still effort-limited) torque.
  cfg.actions["joint_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=GO2W_WHEEL_JOINTS,
    scale=80.0,
    use_default_offset=True,
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"

  cfg.rewards = {
    # Split terms by mode so the training log exposes mode collapse.  Since
    # exactly one one-hot is active, the summed reward is unchanged.
    "gravity_normal": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=14.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "num_modes": 3,
      },
    ),
    "gravity_front": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=14.0,
      params={
        "command_name": "trick",
        "modes": (1,),
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "num_modes": 3,
      },
    ),
    "gravity_rear": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      # Rear two-wheel rise is physically reachable (and is kept in the same
      # policy and reset distribution) but is consistently the sparse side of
      # this asymmetric mechanism.  Give its outcome-only alignment signal
      # equal total importance to the already discovered front handstand.
      weight=28.0,
      params={
        "command_name": "trick",
        "modes": (2,),
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "num_modes": 3,
      },
    ),
    # The ordinary alignment term provides the discovery gradient from a
    # normal reset.  This powered continuation makes the final 20--30 degree
    # lean materially worse than a genuinely vertical two-wheel stand.  It is
    # still an outcome-space body-attitude objective, not a joint target.
    "upright_gravity_precision": RewardTermCfg(
      func=trick_rewards.mode_gravity_alignment,
      weight=100.0,
      params={
        "command_name": "trick",
        "modes": (1,),
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "power": 24.0,
        "num_modes": 3,
      },
    ),
    "rear_upright_gravity_precision": RewardTermCfg(
      # The rear rise is mechanically less symmetric.  Its extra outcome
      # value balances the directions without introducing a second actor.
      func=trick_rewards.mode_gravity_alignment,
      weight=80.0,
      params={
        "command_name": "trick",
        "modes": (2,),
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        # Unlike front rise, rear exploration needs a meaningful reward well
        # before the last few degrees.  The ordinary rear alignment still
        # guides the 90-degree departure; this p=8 term supplies the bridge
        # from a partially lifted rear pair into the final upright basin.
        "power": 8.0,
        "num_modes": 3,
      },
    ),
    "rear_upright_final_precision": RewardTermCfg(
      # This term is deliberately negligible during the rear discovery phase
      # (where p=8 above supplies the bridge), then removes the residual lean
      # once rear support has actually been found.
      func=trick_rewards.mode_gravity_alignment,
      weight=40.0,
      params={
        "command_name": "trick",
        "modes": (2,),
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "power": 24.0,
        "num_modes": 3,
      },
    ),
    "stance_angular_stability": RewardTermCfg(
      func=trick_rewards.mode_root_ang_vel_exp,
      # At std=3 a 0.4 rad/s unintended world yaw was almost indistinguishable
      # from a still handstand.  Zero x/yaw commands must actively suppress
      # that residual rotation before velocity tracking can be meaningful.
      # A handstand that spins on its own is not a valid zero-speed command.
      # This is deliberately stronger than the structural contact rewards,
      # but remains entirely disabled as soon as x or yaw is requested.
      # Keep the first discovery stage identical to the validated v67
      # balance task.  Strong stop rewards are introduced by curriculum only
      # after the policy has discovered the wheel-only upright manifold.
      weight=20.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "std": 0.5,
        "num_modes": 3,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
        "stationary_velocity_deadband": 0.05,
      },
    ),
    "stationary_ground_speed": RewardTermCfg(
      # Complement the angular stillness condition with an outcome-space
      # ground-velocity stop.  No leg configuration or wheel action is
      # supplied: PPO must find how to remain still on the selected wheels.
      func=trick_rewards.stance_stationary_ground_speed_exp,
      weight=0.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "velocity_deadband": 0.05,
        "std": 0.18,
        "lateral_weight": 2.0,
        "num_modes": 3,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "stationary_ground_speed_error": RewardTermCfg(
      # The exponential rest score vanishes when a newly upright policy rolls
      # quickly.  This dense companion closes that gradient gap, after final
      # gravity alignment only, and is inactive for every moving command.
      func=trick_rewards.stance_stationary_ground_speed_abs_error,
      weight=0.0,
      params={
        "command_name": "trick",
        "modes": (0, 1, 2),
        "velocity_deadband": 0.05,
        "lateral_weight": 2.0,
        "num_modes": 3,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 12.0,
        # The final-attitude reward must first reach a high-alignment basin,
        # but 0.96 closed this term exactly where it was needed to remove the
        # residual visual lean observed in replay.
        "minimum_gravity_alignment": 0.94,
      },
    ),
    "stationary_angular_speed_error": RewardTermCfg(
      func=trick_rewards.mode_stationary_root_ang_speed,
      weight=0.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "velocity_deadband": 0.05,
        "num_modes": 3,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 12.0,
        "minimum_gravity_alignment": 0.96,
      },
    ),
    "contacts_normal": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      # Make the four-wheel task an equally explicit objective.  Otherwise it
      # can be treated as a low-value reset state while the two inverted modes
      # receive the stronger support reward.
      weight=250.0,
      params={
        "command_name": "trick",
        "modes": (0,),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": _LOCOMOTION_CONTACT_MASKS,
        "num_modes": 3,
      },
    ),
    "contacts_front": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      weight=50.0,
      params={
        "command_name": "trick",
        "modes": (1,),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": _LOCOMOTION_CONTACT_MASKS,
        "num_modes": 3,
      },
    ),
    "contacts_rear": RewardTermCfg(
      func=trick_rewards.mode_contact_match,
      weight=50.0,
      params={
        "command_name": "trick",
        "modes": (2,),
        "sensor_name": wheel_contact_cfg.name,
        "contact_masks": _LOCOMOTION_CONTACT_MASKS,
        "num_modes": 3,
      },
    ),
    # Dense geometric shaping for the required visual outcome.  These describe
    # wheel placement/clearance only, never a desired hip, thigh, or calf angle.
    # Contact-match and free-wheel-clearance intentionally remain *ungated* by
    # final gravity alignment: a rear request starts from a four-wheel pose
    # whose default legs can brush the floor while it tips.  PPO must receive a
    # positive signal as soon as it lifts the requested free pair, rather than
    # only after it has already found the complete rear handstand by chance.
    "support_wheel_height": RewardTermCfg(
      func=trick_rewards.mode_support_wheel_center_height_exp,
      weight=12.0,
      params={
        "command_name": "trick",
        "contact_masks": _LOCOMOTION_CONTACT_MASKS,
        "target_height": 0.086,
        "std": 0.055,
        "num_modes": 3,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 8.0,
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
        ),
      },
    ),
    "support_leg_extension": RewardTermCfg(
      # Root-to-wheel height allowed a visibly crouched knee configuration.
      # This directly measures the load-bearing hip-to-wheel leg length while
      # remaining agnostic to every individual joint angle.
      func=trick_rewards.mode_support_leg_length_min,
      # The scratch curriculum enables this only after the fused policy has
      # first discovered both two-wheel balance basins.
      weight=0.0,
      params={
        "command_name": "trick",
        "modes": (1, 2),
        "contact_masks": _LOCOMOTION_CONTACT_MASKS,
        "sensor_name": wheel_contact_cfg.name,
        # 0.38 m is a visibly extended support leg while retaining margin
        # below the model's knee hard limit (the kinematic maximum is ~0.41 m).
        "minimum_lengths": (0.0, 0.38, 0.38),
        # A folded leg below 0.16 m gets no extension credit.  Above it, a
        # quadratic score removes the old linear local optimum around 0.2 m
        # and makes the final visual extension decisive.
        "activation_lengths": (0.0, 0.16, 0.16),
        "length_power": 2.0,
        "num_modes": 3,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "minimum_gravity_alignment": 0.90,
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("FL_hip", "FR_hip", "RL_hip", "RR_hip"),
          site_names=("FL", "FR", "RL", "RR"),
          preserve_order=True,
        ),
      },
    ),
    "free_wheel_clearance": RewardTermCfg(
      func=trick_rewards.mode_non_support_wheel_clearance,
      weight=12.0,
      params={
        "command_name": "trick",
        "contact_masks": _LOCOMOTION_CONTACT_MASKS,
        "minimum_height": 0.18,
        "num_modes": 3,
        "asset_cfg": SceneEntityCfg(
          "robot", site_names=("FL", "FR", "RL", "RR"), preserve_order=True
        ),
      },
    ),
    "forbidden_body_thigh_contact": RewardTermCfg(
      func=trick_rewards.contact_violation,
      weight=-20.0,
      params={"sensor_name": "forbidden_body_thigh_contact", "grace_period_s": 0.6},
    ),
    "any_nonwheel_ground_contact": RewardTermCfg(
      func=trick_rewards.contact_violation,
      # Persistent calf/body support must be unattractive well before the
      # two-second hard termination above.  This term begins after 0.6 s.
      weight=-100.0,
      params={"sensor_name": nonwheel_contact_cfg.name, "grace_period_s": 0.6},
    ),
    "track_x": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_exp,
      # Static balance must not be purchased by an uncontrolled wheel-driven
      # crawl.  A substantially stronger x/lateral term also remains the
      # primary command-following objective once the range is expanded below.
      weight=60.0,
      params={
        "command_name": "trick",
        "std": 0.35,
        "lateral_weight": 2.0,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "track_yaw": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_exp,
      weight=30.0,
      params={
        "command_name": "trick",
        "std": 0.40,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    # Exponentials correctly focus PPO on fine tracking once a skill exists,
    # but their gradient vanishes for an initially unstable wheel stand.  The
    # two dense errors keep a cost on uncontrolled crawl/yaw until the policy
    # reaches that fine-tracking basin.  They remain outcome-only objectives:
    # root velocity and lateral drift, never an action or joint target.
    "track_x_dense_error": RewardTermCfg(
      func=trick_rewards.stance_locomotion_linear_velocity_abs_error,
      weight=-20.0,
      params={
        "command_name": "trick",
        "lateral_weight": 2.0,
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "track_yaw_dense_error": RewardTermCfg(
      func=trick_rewards.stance_locomotion_yaw_rate_abs_error,
      weight=-16.0,
      params={
        "command_name": "trick",
        "gravity_targets": _LOCOMOTION_GRAVITY_TARGETS,
        "gravity_power": 4.0,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.10),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-1.0e-5),
    "joint_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS)},
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  # One fused actor is trained throughout.  The *set and distribution* of
  # normal/front/rear one-hots never changes.  Crucially, x/yaw has non-zero
  # variance from the *first* rollout: starting it at exactly zero for hundreds
  # of updates taught the observation normalizer and actor to ignore those two
  # inputs, after which a late range expansion did not produce controlled
  # motion.  For this first visible skill we deliberately keep the same small
  # command distribution throughout: a stable zero command must be learned
  # before asking for larger speeds.  This is a command distribution, not
  # pretraining or a switch to specialised policies.
  cfg.curriculum = {
    "stance_locomotion_difficulty": CurriculumTermCfg(
      func=trick_curriculums.stance_locomotion_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          {
            "step": 0,
            "mode_probabilities": _LOCOMOTION_MODE_PROBABILITIES,
            "idle_probability": 0.65,
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
          },
          # Keep all three modes equally represented while the posture basins
          # are learned.  The old rear-heavy mix starved the harder front rise.
          {
            "step": 7200,
            "mode_probabilities": _LOCOMOTION_MODE_PROBABILITIES,
            "idle_probability": 0.65,
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
          },
          {
            "step": 9600,
            "mode_probabilities": _LOCOMOTION_MODE_PROBABILITIES,
            "idle_probability": 0.60,
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
          },
          {
            "step": 12000,
            "mode_probabilities": _LOCOMOTION_MODE_PROBABILITIES,
            "idle_probability": 0.55,
            "lin_vel_x_range": (-0.10, 0.10),
            "yaw_rate_range": (-0.15, 0.15),
          },
        ),
      },
    ),
    # Command tracking is active from the first rollout so the policy cannot
    # ignore its input.  Its weight rises smoothly alongside the later posture
    # and stationary-control stages in this same scratch run; this is neither a
    # separate policy nor a target-pose/reference-trajectory controller.
    "stance_locomotion_yaw_tracking_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "track_yaw",
        "weight_stages": [
          {"step": 0, "weight": 30.0},
          {"step": 7200, "weight": 40.0},
          {"step": 9600, "weight": 50.0},
          {"step": 12000, "weight": 60.0},
        ],
      },
    ),
    "stance_locomotion_yaw_error_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "track_yaw_dense_error",
        "weight_stages": [
          {"step": 0, "weight": -16.0},
          {"step": 7200, "weight": -24.0},
          {"step": 9600, "weight": -32.0},
          {"step": 12000, "weight": -40.0},
        ],
      },
    ),
    "stance_locomotion_x_tracking_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "track_x",
        "weight_stages": [
          {"step": 0, "weight": 60.0},
          {"step": 7200, "weight": 80.0},
          {"step": 9600, "weight": 100.0},
          {"step": 12000, "weight": 120.0},
        ],
      },
    ),
    "stance_locomotion_x_error_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "track_x_dense_error",
        "weight_stages": [
          {"step": 0, "weight": -20.0},
          {"step": 7200, "weight": -30.0},
          {"step": 9600, "weight": -40.0},
          {"step": 12000, "weight": -50.0},
        ],
      },
    ),
    "stance_locomotion_support_extension_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "support_leg_extension",
        "weight_stages": [
          # First discover the front/rear balance basins.  Then turn the
          # outcome-only extension criterion on gradually, so it improves a
          # learned stand instead of blocking its discovery.
          {"step": 0, "weight": 0.0},
          {"step": 4800, "weight": 100.0},
          {"step": 7200, "weight": 250.0},
          {"step": 9600, "weight": 400.0},
        ],
      },
    ),
    "stance_locomotion_stationary_speed_error_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "stationary_ground_speed_error",
        "weight_stages": [
          {"step": 0, "weight": 0.0},
          {"step": 7200, "weight": -20.0},
          {"step": 9600, "weight": -60.0},
          {"step": 12000, "weight": -120.0},
        ],
      },
    ),
    "stance_locomotion_stationary_angular_error_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "stationary_angular_speed_error",
        "weight_stages": [
          {"step": 0, "weight": 0.0},
          {"step": 7200, "weight": -16.0},
          {"step": 9600, "weight": -40.0},
          {"step": 12000, "weight": -80.0},
        ],
      },
    ),
  }
  if play:
    cfg.curriculum = {}
    command_cfg = cfg.commands["trick"]
    assert isinstance(command_cfg, StanceLocomotionCommandCfg)
    # Evaluation tools override these ranges with a fixed requested command.
    # Keep the play default equal to the final shared deployment distribution.
    command_cfg.idle_probability = 0.65
    command_cfg.mode_probabilities = _LOCOMOTION_MODE_PROBABILITIES
    command_cfg.lin_vel_x_range = (-0.10, 0.10)
    command_cfg.yaw_rate_range = (-0.15, 0.15)
  return cfg


def unitree_go2w_aerial_rotation_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """One-shot front/back/side/yaw jump rotations with no rate command."""
  cfg, wheel_contact_cfg, _ = _make_base_go2w_trick_cfg(play)
  _configure_compact_aerial_actuators(cfg)
  cfg.episode_length_s = 3.0 if not play else cfg.episode_length_s
  cfg.commands = {
    "trick": AerialRotationCommandCfg(
      entity_name="robot",
      # One command per three-second episode: a non-zero one-hot represents
      # one complete attempt, not a periodic velocity instruction.
      resampling_time_range=(3.0, 3.0),
      idle_probability=0.10,
      sensor_name=wheel_contact_cfg.name,
      axes=_AERIAL_AXES,
      debug_vis=False,
    )
  }
  # The aerial maneuver must use a short, wheel-legged impulse—not a full
  # quadruped crouch/extension.  These are maximum residuals around the normal
  # four-wheel geometry, not a desired pose or reference trajectory.  The
  # specialized high-gain aerial actuator reaches its task-local effort cap
  # from these small residuals, giving PPO a compact, high-force mechanism.
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    entity_name="robot",
    actuator_names=GO2W_LEG_JOINTS,
    scale={
      # A wheel-leg should make its impulse with torque rather than a deep
      # squat.  The prior +/-[0.28, 0.32, 0.45] range let virtually every
      # m200 policy sit at the 0.55-rad physical envelope.  These shorter
      # ranges, together with the increased gains above, still reach
      # [50, 50, 75] Nm at small target errors.
      r".*_hip_joint": 0.20,
      r".*_thigh_joint": 0.24,
      r".*_calf_joint": 0.35,
    },
    use_default_offset=True,
  )
  cfg.actions["joint_vel"] = JointVelocityActionCfg(
    entity_name="robot",
    actuator_names=GO2W_WHEEL_JOINTS,
    scale=45.0,
    use_default_offset=True,
  )
  # A bounded action target alone cannot prevent a falling articulated body
  # from overshooting that target after a collision.  AS2W-like flips require
  # a compact *physical* leg envelope, so leaving it is an immediate failed
  # attempt—not merely a small cost traded against angular-rate reward.  The
  # A 0.42-rad threshold permits the compact 0.35-rad calf impulse and normal
  # contact compliance, but rules out the 0.52-rad peak that the prior policy
  # routinely used as a deep crouch/extension.  It is neutral across all
  # one-hot directions and has no phase or reference pose.
  cfg.terminations["leg_excursion"] = TerminationTermCfg(
    func=trick_rewards.aerial_leg_excursion_exceeded,
    params={
      "command_name": "trick",
      "max_deviation": 0.42,
      "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS),
    },
  )
  # A command requests one rotation, not an indefinitely fast spin.  The
  # overrun failure starts only after all five directions have crossed the
  # final ballistic-discovery curriculum (30k global control steps).  A hard
  # failure at the initial landing stage made front/side flips abandon their
  # just-discovered one-turn attempts before the recovery reward had enough
  # time to shape them.  This does not expose a phase to the actor or
  # prescribe any joint action.
  cfg.terminations["rotation_overrun"] = TerminationTermCfg(
    func=trick_rewards.aerial_rotation_overrun,
    params={
      "command_name": "trick",
      "target_angle": math.tau,
      "max_overrotation": 0.75,
      "activation_step": 30_000,
    },
  )
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["commands"].params["command_name"] = "trick"

  cfg.rewards = {
    "landing_orientation": RewardTermCfg(
      func=trick_rewards.aerial_landing_gravity_exp,
      weight=16.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "std": 0.45,
        "linear_velocity_std": 0.75,
        "angular_velocity_std": 1.5,
      },
    ),
    "takeoff_clearance": RewardTermCfg(
      func=trick_rewards.AerialClearanceProgress,
      # Bounded apex progress, made deliberately dominant until a genuine
      # compact ballistic launch has been discovered.  v24 plateaued around
      # 0.20--0.25 m and 0.3 turn: it could collect rotation reward before it
      # had enough flight time.  A 0.34-m one-shot target is compatible with
      # a full turn at 12--13 rad/s, yet cannot be farmed by hovering.
      weight=30.0,
      params={"command_name": "trick", "min_clearance": 0.34},
    ),
    "takeoff_vertical_speed": RewardTermCfg(
      func=trick_rewards.AerialTakeoffVerticalSpeed,
      # Pay the measured vertical impulse only once, on wheel liftoff.  This
      # is an outcome reward rather than a prescribed leg trajectory; 2.6
      # m/s corresponds to a roughly 0.34-m ideal ballistic apex.
      weight=16.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_speed": 2.6,
      },
    ),
    "axis_progress": RewardTermCfg(
      func=trick_rewards.AerialRotationProgress,
      # A half-turn ending in a body strike used to earn too little relative
      # to the termination cost for PPO to cross the full-turn discovery
      # barrier.  This remains a one-shot reward on measured rotation, not a
      # target joint motion or reference trajectory.
      weight=14.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": _AERIAL_AXES,
        "target_angle": math.tau,
        # Read the current ballistic gate from the command term.  The
        # one-policy curriculum tightens it without adding an observation.
        "clearance_start": None,
        "clearance_full": None,
      },
    ),
    "directed_axis_rate": RewardTermCfg(
      func=trick_rewards.aerial_positive_axis_rate,
      # Progress rewards the final angle, but does not distinguish a slow
      # half-turn from a compact, high-momentum takeoff that is about to pass
      # the turn barrier.  Pay only the measured, commanded-direction angular
      # rate only after the base has reached meaningful clearance.  This
      # binds the two real outcomes that a compact flip needs—ballistic
      # flight and correct angular momentum—without a target pose, phase, or
      # reference trajectory.  The actor still gets only its one-hot command;
      # no axis or rate target is exposed as an observation or command.
      # RewardManager integrates every term over the 20-ms control step.  A
      # lower weight was numerically present but too small to alter the
      # half-turn-and-crash local optimum relative to termination.
      # This is momentum shaping, not the qualified-turn objective.  It
      # starts at a reachable intermediate clearance so the policy can build
      # angular momentum during flight; ``axis_progress`` below remains gated
      # by the stricter 0.34-m final clearance.  v25 put both gates near the
      # apex and improved height while starving the actual flip of drive.
      weight=65.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": _AERIAL_AXES,
        "rate_clip": 20.0,
        "clearance_start": None,
        "clearance_full": None,
        # Drive only the takeoff/tumbling portion.  From about 0.7 turn the
        # late-phase recovery and landing terms must be free to bleed angular
        # momentum instead of competing with an always-on spin reward.
        "stop_angle": 0.70 * math.tau,
        "stop_angle_fade": 0.15 * math.tau,
      },
    ),
    "late_phase_recovery": RewardTermCfg(
      func=trick_rewards.aerial_late_phase_recovery_exp,
      weight=60.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "activation_angle": 0.85 * math.tau,
        "gravity_std": 0.75,
        # This is the braking half of the maneuver.  The old 6-rad/s target
        # actively trained a policy to keep spinning through touchdown.
        "target_axis_rate": 1.5,
        # Keep a useful ranking signal even for the 20--30-rad/s early
        # attempts; success still requires the much lower landing rate below.
        "axis_rate_clip": 30.0,
      },
    ),
    "soft_landing": RewardTermCfg(
      func=trick_rewards.aerial_soft_landing_exp,
      # This is a four-wheel-only *basin* reward, not the success criterion.
      # With the former [0.70, 0.55, 2.5] Gaussian widths, a first legal wheel
      # contact following a 6--10 rad/s flip had essentially zero value.  PPO
      # consequently had no gradient between a body crash and the strict
      # settled landing.  Keep the one-turn/four-wheel requirements but rank
      # imperfect, physically valid touchdowns so recovery can be discovered;
      # the command's hard completion remains much tighter.
      weight=180.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "angle_std": 1.25,
        "gravity_std": 1.25,
        "axis_rate_std": 8.0,
      },
    ),
    "post_turn_descent": RewardTermCfg(
      func=trick_rewards.aerial_post_turn_descent,
      # Dense bridge from the airborne braking signal to the strictly
      # four-wheel-only touchdown reward.  It rewards only a measured
      # full-turn, upright, low-spin descent—not a pose or a reference path.
      weight=45.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "max_overrotation": 0.75,
        "gravity_std": 0.75,
        "axis_rate_std": 3.0,
        "descent_speed": 1.5,
      },
    ),
    "post_turn_landing_progress": RewardTermCfg(
      func=trick_rewards.AerialPostTurnLandingProgress,
      # The evaluated m200 policy could perform one or more turns but earned
      # virtually no post-turn signal because touchdown is too sparse.  Pay
      # only new, physically measured descent after a full turn, with normal
      # attitude and a reduced angular rate.  No joint pose or reference path
      # is supplied to the actor.
      # Keep this zero through compact turn discovery.  Turning it on too
      # early made the shared policy become conservative before all modes had
      # crossed one complete turn; the curriculum below enables it after the
      # first ballistic-rotation stage.
      weight=0.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "target_angle": math.tau,
        "max_overrotation": 0.75,
        "gravity_std": 0.90,
        "axis_rate_clip": 16.0,
        "descent_distance": 0.35,
      },
    ),
    "over_rotation": RewardTermCfg(
      func=trick_rewards.AerialOverRotation,
      weight=-12.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": _AERIAL_AXES,
        "target_angle": math.tau,
      },
    ),
    "completed_rotation": RewardTermCfg(
      func=trick_rewards.AerialRotationCompletion,
      weight=200.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "axes": _AERIAL_AXES,
        "target_angle": math.tau,
        "landing_settle_time": 0.10,
        "landing_linear_velocity_limit": 0.75,
        "landing_angular_velocity_limit": 1.5,
        "max_overrotation": 0.75,
      },
    ),
    # A wheel-legged aerial flip should not be paid for by throwing every leg
    # to its joint limits.  This acts throughout an active attempt, including
    # the short push-off on the ground: otherwise PPO can generate angular
    # momentum with a large pre-flight swing and pay no compactness cost until
    # the wheels leave the floor.  The 0.10-rad deadband is free travel around
    # normal four-wheel geometry; it is neither a target pose nor a reference
    # trajectory.  The small 0.10-rad free zone makes a large leg opening an
    # expensive way to generate rotation even below the hard bound.
    "airborne_leg_excursion": RewardTermCfg(
      func=trick_rewards.aerial_airborne_joint_excursion_l2,
      weight=-16.0,
      params={
        "command_name": "trick",
        "sensor_name": wheel_contact_cfg.name,
        "free_deviation": 0.10,
        "asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS),
        "airborne_only": False,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.03),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-6),
    "joint_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-5.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2W_LEG_JOINTS)},
    ),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  cfg.curriculum = {
    "aerial_command_difficulty": CurriculumTermCfg(
      func=trick_curriculums.aerial_rotation_command_stages,
      params={
        "command_name": "trick",
        "stages": (
          # ``common_step_counter`` counts synchronized control steps, not
          # the total samples across all 4096 environments.  These thresholds
          # therefore correspond to roughly 48M, 112M, and 240M samples at
          # the production environment count and 20-ms control rate.
          # Discover compact rotation after a modest, physically real jump.
          # This does not alter the full-turn landing objective.
          {
            "step": 0,
            "idle_probability": 0.05,
            "rotation_progress_clearance_start": 0.03,
            "rotation_progress_clearance_full": 0.14,
            "rotation_rate_clearance_start": 0.04,
            "rotation_rate_clearance_full": 0.14,
          },
          # Require a larger ballistic arc before rotation receives its full
          # reward, still with all five directions in one actor.
          {
            "step": 12_000,
            "idle_probability": 0.05,
            "rotation_progress_clearance_start": 0.08,
            "rotation_progress_clearance_full": 0.26,
            "rotation_rate_clearance_start": 0.06,
            "rotation_rate_clearance_full": 0.20,
          },
          # Keep every one-hot equally likely throughout training.  This is a
          # single fused policy, and reducing either side-flip command to 8%
          # during the central exploration window made the policy optimize
          # front/back/yaw at the expense of the required left/right skills.
          # Only the hidden outcome gates become stricter here; command
          # coverage remains the final 20% per maneuver distribution.
          {
            "step": 27_500,
            "idle_probability": 0.05,
            "rotation_progress_clearance_start": 0.12,
            "rotation_progress_clearance_full": 0.34,
            "rotation_rate_clearance_start": 0.08,
            "rotation_rate_clearance_full": 0.24,
          },
        ),
      },
    ),
    "aerial_landing_progress_weight": CurriculumTermCfg(
      func=mdp.reward_weight,
      params={
        "reward_name": "post_turn_landing_progress",
        "weight_stages": [
          {"step": 0, "weight": 0.0},
          # Blend the sparse descent bridge in after basic turn discovery.
          # Jumping immediately to its final strength at 16k steps made the
          # shared policy regress from newly discovered side flips.  The
          # weaker stages preserve their momentum signal while PPO learns
          # that a measured full turn should descend upright and slow down.
          {"step": 16_000, "weight": 20.0},
          {"step": 22_000, "weight": 60.0},
          # At 30k steps every mode has the final height gates and the hard
          # over-rotation termination also begins, so full landing shaping
          # can safely take priority over continuing to tumble.
          {"step": 30_000, "weight": 120.0},
        ],
      },
    ),
  }
  if play:
    cfg.curriculum = {}
    command_cfg = cfg.commands["trick"]
    assert isinstance(command_cfg, AerialRotationCommandCfg)
    command_cfg.idle_probability = 0.05
  # Ten 20-ms observations give the policy a 200-ms local motion window for
  # takeoff timing and airborne rotation without introducing a reference phase
  # or any privileged state.  Commands stay in the stack as ordinary inputs.
  for group_name in ("actor", "critic"):
    cfg.observations[group_name].history_length = 10
  return cfg
