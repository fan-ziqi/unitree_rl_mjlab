"""Small shared scene builder for the three Go2W trick environments.

The policy interface is deliberately identical in every task: command,
angular velocity, gravity vector, joint position/velocity, and last action.
Contacts and geometry remain private outcome signals for reward and validity.
"""

from __future__ import annotations

from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg, JointVelocityActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_JOINTS,
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_GEOMS,
  GO2W_WHEEL_JOINTS,
  get_go2w_robot_cfg,
)
from src.tasks.velocity import mdp
from src.tasks.velocity.mdp.actions import (
  DefaultIdleGatedJointPositionActionCfg,
  DefaultIdleGatedJointVelocityActionCfg,
)
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Wheel order is [FL, FR, RL, RR].  Gravity targets are expressed in base
# coordinates; they are task semantics, never actor observations.
STANCE_GRAVITY_TARGETS = (
  (0.0, 0.0, -1.0),
  (1.0, 0.0, 0.0),
  (-1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, -1.0, 0.0),
)
STANCE_CONTACT_MASKS = (
  (1.0, 1.0, 1.0, 1.0),
  (1.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 1.0),
  (1.0, 0.0, 1.0, 0.0),
  (0.0, 1.0, 0.0, 1.0),
)
LOCOMOTION_GRAVITY_TARGETS = STANCE_GRAVITY_TARGETS[:3]
LOCOMOTION_CONTACT_MASKS = STANCE_CONTACT_MASKS[:3]
AERIAL_AXES = (
  (0.0, 1.0, 0.0),
  (0.0, -1.0, 0.0),
  (1.0, 0.0, 0.0),
  (-1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0),
)


def configure_compact_aerial_actuators(cfg: ManagerBasedRlEnvCfg) -> None:
  """Use the Go2W model's physical leg torque range with compliant position control."""
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  gains = {
    # Hip/thigh targets stay compliant.  Aerial telemetry showed the calf
    # command was limited to 50 * 0.55 = 27.5 Nm, far below its modelled
    # 45.43-Nm actuator range, so it could only produce a low 1.3--1.5 m/s
    # hop.  Raising *only* the calf proportional gain lets an ordinary bounded
    # residual reach the real torque cap; it does not inflate the model's
    # physical torque authority or prescribe a leg motion.
    (".*hip_.*",): (45.0, 2.0),
    (".*thigh_.*",): (45.0, 2.0),
    (".*calf_.*",): (80.0, 2.0),
  }
  effort_limits = {
    # These are the actual actuator control ranges in ``go2w.xml``.  The
    # preceding 90/90/95 override made every learned maneuver unrealistically
    # impulsive and visually rigid.
    (".*hip_.*",): 23.7,
    (".*thigh_.*",): 23.7,
    (".*calf_.*",): 45.43,
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


def configure_ground_support_actuators(cfg: ManagerBasedRlEnvCfg) -> None:
  """Give a zero position residual enough physical authority to hold default.

  The imported Go2W model's generic P=20 leg controller lets gravity fold its
  nominal four-wheel reset even when the action is exactly zero.  Ground
  support tricks require that public idle state to be physically valid before
  PPO can choose to preserve it.  These gains apply only to the ground spin
  and stance-locomotion tasks; aerial control retains its deliberately more
  compliant specialised actuator configuration.
  """
  articulation = cfg.scene.entities["robot"].articulation
  assert articulation is not None
  ground_gains = {
    # P=50 keeps every zero-action four-wheel reset standing in the 128-env,
    # three-second physics sweep.  It also leaves a much larger useful
    # proportional region than P=80 before the model's native torque limits
    # are reached, so a contact pivot need not look like a rigid strut.
    (".*hip_.*",): (50.0, 2.0),
    (".*thigh_.*",): (50.0, 2.0),
    (".*calf_.*",): (50.0, 2.0),
  }
  articulation.actuators = tuple(
    replace(
      actuator,
      stiffness=ground_gains[actuator.target_names_expr][0],
      damping=ground_gains[actuator.target_names_expr][1],
    )
    if actuator.target_names_expr in ground_gains
    else actuator
    for actuator in articulation.actuators
  )


def configure_default_idle_actions(
  cfg: ManagerBasedRlEnvCfg,
  *,
  command_name: str,
  idle_mode_index: int | None,
  stationary_command_start_index: int,
  command_deadband: float,
  idle_contact_sensor_name: str,
  hold_default_position_mode_index: int | None = None,
) -> None:
  """Make a public idle command the model's literal default controller.

  The gate is a command-interface invariant: default pose plus zero wheel
  speed when no skill is requested *and the robot is already upright on all
  four wheels*.  A normal command following a two-wheel or aerial skill first
  leaves PPO action authority to perform a controlled return; after touchdown
  the literal default controller engages.  It supplies no two-wheel or aerial
  posture to PPO.
  """
  joint_pos = cfg.actions["joint_pos"]
  joint_vel = cfg.actions["joint_vel"]
  assert isinstance(joint_pos, JointPositionActionCfg)
  assert isinstance(joint_vel, JointVelocityActionCfg)
  common = {
    "command_name": command_name,
    "idle_mode_index": idle_mode_index,
    "stationary_command_start_index": stationary_command_start_index,
    "command_deadband": command_deadband,
    "idle_contact_sensor_name": idle_contact_sensor_name,
  }
  cfg.actions["joint_pos"] = DefaultIdleGatedJointPositionActionCfg(
    entity_name=joint_pos.entity_name,
    actuator_names=joint_pos.actuator_names,
    scale=joint_pos.scale,
    offset=joint_pos.offset,
    preserve_order=joint_pos.preserve_order,
    clip=joint_pos.clip,
    use_default_offset=joint_pos.use_default_offset,
    hold_default_position_mode_index=hold_default_position_mode_index,
    **common,
  )
  cfg.actions["joint_vel"] = DefaultIdleGatedJointVelocityActionCfg(
    entity_name=joint_vel.entity_name,
    actuator_names=joint_vel.actuator_names,
    scale=joint_vel.scale,
    offset=joint_vel.offset,
    preserve_order=joint_vel.preserve_order,
    clip=joint_vel.clip,
    use_default_offset=joint_vel.use_default_offset,
    **common,
  )


def make_base_go2w_trick_cfg(
  play: bool,
) -> tuple[ManagerBasedRlEnvCfg, ContactSensorCfg, ContactSensorCfg]:
  """Build flat ground, generic observations, and wheel-only support safety."""
  cfg = make_velocity_env_cfg()
  cfg.scene.entities = {"robot": get_go2w_robot_cfg(headless=not play)}
  cfg.sim.njmax = 500
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
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    wheel_contact_cfg,
    nonwheel_contact_cfg,
  )

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
    terms["base_ang_vel"].params["sensor_name"] = "robot/imu_gyro"
    terms["gravity_vec"] = terms.pop("projected_gravity")
    terms["commands"] = terms.pop("command")
    terms["joint_pos"].func = mdp.joint_pos_rel_without_wheel
    terms["joint_pos"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=GO2W_JOINTS
    )
    terms["joint_pos"].params["wheel_asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=GO2W_WHEEL_JOINTS
    )
    terms["joint_vel"].params["asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=GO2W_JOINTS
    )

  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=GO2W_LEG_JOINTS,
      scale={
        # A 0.35-rad abduction residual froze the contact geometry close to
        # the four-wheel reset: front/rear supports settled into a low slant
        # and normal spin could not draw its wheel axles together.  The model
        # itself permits ±1.05 rad; expose a still-bounded 0.55-rad working
        # envelope so PPO can discover the support geometry without changing
        # torque limits, a pose target, or the public command interface.
        r".*_hip_joint": 0.55,
        r".*_thigh_joint": 0.85,
        r".*_calf_joint": 0.85,
      },
      use_default_offset=True,
    ),
    "joint_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=GO2W_WHEEL_JOINTS,
      scale=40.0,
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
    "x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-0.2, 0.2)
  }
  cfg.events.pop("push_robot", None)
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    # A thigh, calf, hip, or trunk is never a valid support.  This is a
    # physical validity rule, not a pose target or reference trajectory.
    "illegal_contact": TerminationTermCfg(
      func=mdp.illegal_contact,
      params={"sensor_name": nonwheel_contact_cfg.name, "force_threshold": 10.0},
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
