"""Unitree Go2W upright walking environment configuration."""

from dataclasses import replace

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots.unitree_go2w.go2w_constants import (
  GO2W_JOINTS,
  GO2W_LEG_JOINTS,
  GO2W_WHEEL_GEOMS,
  GO2W_WHEEL_JOINTS,
  get_go2w_robot_cfg,
)
from src.tasks.velocity import mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Entity-local Go2W joint order is FL leg, FL wheel, FR leg, FR wheel, ... .
# Explicit IDs avoid mjlab 1.2 re-resolving a regex differently for play-mode
# configurations, while retaining the intended FL-then-FR target ordering.
_FRONT_LEG_JOINT_IDS = [0, 1, 2, 4, 5, 6]
_FRONT_HIP_JOINT_IDS = [0, 4]
_REAR_LEG_JOINT_IDS = [8, 9, 10, 12, 13, 14]
_ALL_LEG_JOINT_IDS = _FRONT_LEG_JOINT_IDS + _REAR_LEG_JOINT_IDS
# The upright task is a rear-wheel biped.  The former front wheels stay in the
# model for contact and visual fidelity, but are deliberately omitted from the
# policy action interface so they can never propel the robot.
_REAR_WHEEL_JOINTS = ("RL_wheel_joint", "RR_wheel_joint")
# Only these two tyres are legal ground supports after the robot has reared.
# FL/FR are passive hanging-arm wheels, not support wheels.
_REAR_WHEEL_GEOMS = GO2W_WHEEL_GEOMS[2:]
# In the reared frame the former front legs are arms.  Keep both hip joints
# neutral so the two front wheels stay parallel to the trunk instead of
# splaying outward.  With the reference-mapped rear support below, forward
# kinematics places these wheel centres at about 44 cm: beside and below the
# raised trunk, not held out or used as extra supports.
_FRONT_LEG_HANGING_POSE = (0.0, 1.75, -1.30, 0.0, 1.75, -1.30)
# The reference pose provides the initial geometry, then a static MuJoCo COM
# check makes the small Go2W-specific correction below.  At this pose the COM
# projection is 1.2 mm from the RL/RR axle (instead of 4.1 cm in front of it
# with 2.25 rad thighs), so the post-stand wheel controller starts from an
# actual two-wheel balance geometry rather than a seated lean.
_REAR_LEG_SUPPORT_POSE = (0.0, 2.425, -1.75, 0.0, 2.425, -1.75)
_UPRIGHT_GRAVITY = (-1.0, 0.0, 0.0)
# One criterion defines a qualified final stance across the task and evaluator.
_QUALIFIED_UPRIGHT_ERROR = 0.12


def unitree_go2w_upright_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a flat-ground task that resets Go2W on four wheels.

  The only posture objective is the final reared orientation.  There is no
  get-up trajectory, phase reward, or fallen-state recovery: a policy must
  discover a smooth transition from the ordinary four-foot reset pose itself.
  """
  cfg = make_velocity_env_cfg()
  # Training does not render, so build the collision-only model.  Play mode
  # deliberately retains the visual meshes for video and visual inspection.
  cfg.scene.entities = {"robot": get_go2w_robot_cfg(headless=not play)}
  # This is a task-level controller mapping, not a static-model edit.  The
  # user-specified upright reference uses P gains 40/1; the static Go2W asset
  # uses 20/0.5 for ordinary quadruped locomotion.  Reaching a rear-wheel
  # stance from four wheels needs the reference controller authority, while
  # keeping the XML geometry, collision shapes, masses, and effort limits
  # unchanged.  The wheel velocity actuator is intentionally untouched.
  robot_cfg = cfg.scene.entities["robot"]
  assert robot_cfg.articulation is not None
  robot_cfg.articulation.actuators = tuple(
    replace(actuator, stiffness=40.0, damping=1.0)
    if isinstance(actuator, BuiltinPositionActuatorCfg)
    else actuator
    for actuator in robot_cfg.articulation.actuators
  )
  cfg.sim.njmax = 500
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.mujoco.ccd_iterations = 100

  # Flat terrain only: the reared task does not need terrain scan observations.
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  cfg.scene.sensors = tuple(
    sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)
  # The base velocity curriculum mutates command ranges at runtime.  It would
  # reintroduce infeasible lateral/yaw targets after this configuration fixes
  # them to zero for rear-wheel upright motion.
  cfg.curriculum.pop("command_vel", None)

  wheel_contact_cfg = ContactSensorCfg(
    name="wheel_ground_contact",
    primary=ContactMatch(mode="geom", pattern=GO2W_WHEEL_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  front_wheel_contact_cfg = ContactSensorCfg(
    name="front_wheel_ground_contact",
    primary=ContactMatch(
      mode="geom", pattern=GO2W_WHEEL_GEOMS[:2], entity="robot"
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=4,
  )
  # This is the reference upright task's ``penalised_contact_indices``:
  # every collision body except a tyre.  The reset has all four tyres on the
  # floor and is therefore valid, whereas a calf, thigh, hip, or trunk on the
  # floor is never a valid part of the stand-up motion.
  non_wheel_contact_cfg = ContactSensorCfg(
    name="non_wheel_ground_contact",
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
  # The user-specified Go2 upright reference terminates on a trunk contact,
  # while thigh/calf contacts are a dense cost during the exploratory rise.
  # A separate final-support sensor below then rejects every contact except the
  # two rear tyres once upright, so transient exploration never becomes an
  # invalid propped final stance.
  forbidden_contact_cfg = ContactSensorCfg(
    name="forbidden_ground_contact",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=_REAR_WHEEL_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  base_ground_contact_cfg = ContactSensorCfg(
    name="base_ground_contact",
    primary=ContactMatch(mode="geom", pattern="base_collision", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    wheel_contact_cfg,
    front_wheel_contact_cfg,
    non_wheel_contact_cfg,
    forbidden_contact_cfg,
    base_ground_contact_cfg,
  )

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.0
  cfg.viewer.elevation = -8.0

  # Go2W policy observation layout: 16 joint positions (wheel positions masked),
  # 16 joint velocities, 16 previous actions, commands, phase, angular velocity,
  # and projected gravity.  The critic additionally sees base linear velocity and
  # wheel-contact state through the base velocity task layout.
  for group_name in ("actor", "critic"):
    # The static Go2W MJCF provides its IMU under the source model's native
    # sensor names.  Keep that model untouched and adapt only this task's
    # observations to those sensors.
    cfg.observations[group_name].terms["base_ang_vel"].params["sensor_name"] = "robot/imu_gyro"
    joint_pos = cfg.observations[group_name].terms["joint_pos"]
    joint_pos.func = mdp.joint_pos_rel_without_wheel
    joint_pos.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=GO2W_JOINTS)
    joint_pos.params["wheel_asset_cfg"] = SceneEntityCfg(
      "robot", joint_names=GO2W_WHEEL_JOINTS
    )
    joint_vel = cfg.observations[group_name].terms["joint_vel"]
    joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=GO2W_JOINTS)

  cfg.observations["critic"].terms["base_lin_vel"].params["sensor_name"] = "robot/frame_vel"

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"] = SceneEntityCfg(
    "robot", site_names=("FL", "FR", "RL", "RR")
  )
  cfg.observations["critic"].terms["foot_air_time"].params["sensor_name"] = wheel_contact_cfg.name
  cfg.observations["critic"].terms["foot_contact"].params["sensor_name"] = wheel_contact_cfg.name
  cfg.observations["critic"].terms["foot_contact_forces"].params["sensor_name"] = wheel_contact_cfg.name

  # Position control for the legs and velocity control for the two *rear* wheel
  # motors.  The static Go2W model remains unchanged; selecting only these two
  # actuators here is a task constraint, which prevents the hanging front arms
  # from being used to drive while retaining their wheel contact geometry.
  cfg.actions = {
    "joint_pos": envs_mdp.JointPositionActionCfg(
      entity_name="robot",
      actuator_names=GO2W_LEG_JOINTS,
      # Direct reference-style residual mapping.  Start from the source Go2
      # task's conservative 0.125/0.25-rad gains; the broad deployment action
      # range below covers the whole final posture without a temporal filter.
      # Smoothness is rewarded rather than imposed in the action interface.
      scale={
        r".*_hip_joint": 0.125,
        r".*_thigh_joint": 0.25,
        r".*_calf_joint": 0.25,
      },
      use_default_offset=True,
    ),
    "joint_vel": mdp.UprightGatedJointVelocityActionCfg(
      entity_name="robot",
      actuator_names=_REAR_WHEEL_JOINTS,
      # Direct velocity mapping: ordinary unit-variance exploration starts
      # near the reference task's gentle wheel target, while deliberate large
      # residuals remain available after front-wheel lift-off.
      scale=5.0,
      use_default_offset=True,
      front_wheel_sensor_name=front_wheel_contact_cfg.name,
      front_release_force_threshold=1.0,
    ),
  }

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  # A rear-wheel upright robot is non-holonomic.  Keep the usual three-value
  # command interface, but train the initial task on feasible longitudinal
  # motion only; lateral and yaw ranges can be introduced in a later curriculum.
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (0.0, 0.0)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None

  cfg.events["foot_friction"].params["asset_cfg"] = SceneEntityCfg(
    "robot", geom_names=GO2W_WHEEL_GEOMS
  )
  cfg.events["base_com"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )
  # Learn the difficult four-foot-to-two-wheel manoeuvre under a narrow but
  # nonzero physical variation first.  The inherited ±5 cm COM shift and
  # 0.3--1.6 tyre-friction range repeatedly interrupted the transition before
  # it could reach the final-pose reward.  This remains domain randomized and
  # does not alter the reset pose or prescribe a stand-up trajectory.
  cfg.events["foot_friction"].params["ranges"] = (0.7, 1.0)
  cfg.events["base_com"].params["ranges"] = {
    0: (-0.01, 0.01),
    1: (-0.01, 0.01),
    2: (-0.01, 0.01),
  }
  cfg.events["reset_base"].params["pose_range"] = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "yaw": (-0.2, 0.2),
  }
  cfg.events.pop("push_robot", None)

  # Replace quadruped-gait rewards with final-pose upright and wheeled-motion
  # objectives.  No time-dependent reward tells the policy how to stand up.
  cfg.rewards.pop("body_orientation_l2", None)
  cfg.rewards.pop("pose", None)
  cfg.rewards.pop("foot_gait", None)
  cfg.rewards.pop("foot_clearance", None)
  cfg.rewards.pop("foot_slip", None)
  cfg.rewards.pop("soft_landing", None)
  cfg.rewards.pop("stand_still", None)
  # The source Go2W MJCF has no root-angular-momentum sensor.  This generic
  # quadruped term is not needed for the upright objective; angular velocity,
  # joint velocity, and action-rate penalties below still regularize motion.
  cfg.rewards.pop("angular_momentum", None)
  cfg.rewards["track_linear_velocity"].func = mdp.track_upright_linear_velocity
  # Direct mapping of the reference task's tracking term.  It is ineligible
  # until the static upright test passes, therefore it cannot teach four-wheel
  # driving at reset.
  cfg.rewards["track_linear_velocity"].weight = 2.5
  cfg.rewards["track_linear_velocity"].params["std"] = 0.5
  cfg.rewards["track_linear_velocity"].params["target_gravity"] = _UPRIGHT_GRAVITY
  cfg.rewards["track_linear_velocity"].params["upright_gate_error"] = _QUALIFIED_UPRIGHT_ERROR
  cfg.rewards["track_angular_velocity"].func = mdp.track_upright_angular_velocity
  cfg.rewards["track_angular_velocity"].weight = 2.5
  cfg.rewards["track_angular_velocity"].params["std"] = 0.7
  cfg.rewards["track_angular_velocity"].params["target_gravity"] = _UPRIGHT_GRAVITY
  cfg.rewards["track_angular_velocity"].params["upright_gate_error"] = _QUALIFIED_UPRIGHT_ERROR
  # A strong positive final-state attitude objective.  The old low-weight L2
  # cost is algebraically a weak alignment preference but left the policy in a
  # front-wheel-supported local optimum before it had sampled lift-off.  This
  # is still only the same target gravity direction—no phase, time signal, or
  # joint-space get-up reference is introduced.
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp.target_projected_gravity_alignment,
    weight=12.0,
    params={"target_gravity": _UPRIGHT_GRAVITY},
  )
  cfg.rewards.pop("upright_precise", None)
  cfg.rewards["alive"] = RewardTermCfg(func=envs_mdp.is_alive, weight=1.0)

  front_leg_cfg = SceneEntityCfg("robot", joint_ids=_FRONT_LEG_JOINT_IDS)
  all_leg_cfg = SceneEntityCfg("robot", joint_ids=_ALL_LEG_JOINT_IDS)
  final_leg_pose = _FRONT_LEG_HANGING_POSE + _REAR_LEG_SUPPORT_POSE
  # ``default_pos`` from the reference task, adapted to Go2W's final static
  # morphology: relaxed arms with neutral hips (parallel front wheels) and
  # naturally bent rear wheel-support legs.  This is one state-only term.
  cfg.rewards["desired_leg_pose"] = RewardTermCfg(
    func=mdp.joint_position_l1,
    weight=-0.1,
    params={"target_joint_pos": final_leg_pose, "asset_cfg": all_leg_cfg},
  )
  cfg.rewards["front_wheels_air"] = RewardTermCfg(
    func=mdp.all_contacting_geoms_in_air,
    # A front wheel on the floor is the observed sitting local optimum, not
    # the requested rear-wheel stance.  This remains a state-only support
    # objective and has no elapsed-time or transition reference.
    weight=2.0,
    params={"sensor_name": front_wheel_contact_cfg.name, "force_threshold": 1.0},
  )
  # Static final-height objective for the measured Go2W geometry.  A 0.5 m
  # width gave the low four-wheel configuration over half of the final reward,
  # which the model-200 evaluation confirmed as a local optimum.  This still
  # contains no time, phase, or intermediate-pose cue; it simply distinguishes
  # the required hanging-wheel height from the reset height.
  cfg.rewards["front_wheel_height"] = RewardTermCfg(
    func=mdp.site_height_exp,
    # Make the requested hanging-wheel height materially better than a front
    # wheel still parked on the terrain.  The former broad bell gave the
    # observed low sitting configuration a sizeable final-state reward.
    weight=5.0,
    params={
      # The natural hanging-arm pose is 11.5 cm below the upright root.  The
      # COM-aligned rear support puts that root at 56.1 cm, so the free
      # front wheels sit around 44 cm high—beside and below the trunk, rather
      # than propping it on the floor.
      "target_height": 0.446,
      # The reference task's very sharp terminal-height bell is almost zero
      # while a Go2W still has its front wheels on the floor.  That admitted a
      # stationary seated local optimum in headless evaluation.  This remains
      # the same single final world-height target, but gives a usable static
      # gradient all the way from the four-wheel reset to the lifted arms.
      "std": 0.35,
      "asset_cfg": SceneEntityCfg("robot", site_names=("FL", "FR")),
    },
  )
  cfg.rewards["front_wheel_clearance"] = RewardTermCfg(
    # A rear-wheel upright has its front wheels clear of the terrain.  This
    # direct static support criterion preserves a useful gradient from the
    # normal four-wheel reset without encoding an ordered lifting motion.
    func=mdp.site_height_at_least,
    weight=8.0,
    params={
      "minimum_height": 0.41,
      "asset_cfg": SceneEntityCfg("robot", site_names=("FL", "FR")),
    },
  )
  # Same static final-pose preference as the reference's
  # ``default_pos_reward``.  It activates only after the objective posture is
  # reached, so it cannot prescribe a separate standing-up trajectory.
  cfg.rewards["front_legs_hanging_final"] = RewardTermCfg(
    func=mdp.upright_joint_position_exp,
    # Once the robot has actually reached the outcome-space upright manifold,
    # make the relaxed side-hanging arms a meaningful final-pose objective.
    # The gate keeps it entirely out of the four-foot-to-two-wheel transition.
    weight=4.0,
    params={
      "std": 1.5,
      "target_joint_pos": _FRONT_LEG_HANGING_POSE,
      "target_gravity": _UPRIGHT_GRAVITY,
      "upright_gate_error": _QUALIFIED_UPRIGHT_ERROR,
      "asset_cfg": front_leg_cfg,
    },
  )
  # Direct mapping of the reference task's continuous base-height term.  The
  # high reference-mapped target distinguishes the reared stance from the
  # four-wheel reset, so this can remain an all-time final-state objective
  # without adding a stand-up phase or trajectory reward.
  cfg.rewards["upright_base_height"] = RewardTermCfg(
    func=mdp.root_height_l1_exp,
    weight=1.5,
    params={
      "target_height": 0.561,
      "scale": 5.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.rewards["forbidden_ground_collision"] = RewardTermCfg(
    # Literal mapping of the reference Go2 upright task's collision term:
    # penalise every non-foot/non-wheel terrain contact throughout the rollout.
    # FL/FR are deliberately excluded here because four-wheel reset is valid;
    # their final lift-off is handled by the dedicated clearance reward and
    # strict final-support terminal below.  This prevents the policy from
    # parking just outside the upright gate with a calf, thigh, hip, or body
    # quietly propping it up on the terrain.
    func=mdp.terrain_contact_count,
    # Match the reference scale exactly.  Its collision signal is a shaping
    # cost, not a terminal wall: a much larger value made the policy collapse
    # into repeated base-contact terminations before it could discover a clean
    # two-wheel rise.
    weight=-2.0,
    params={
      "sensor_name": non_wheel_contact_cfg.name,
      "force_threshold": 0.1,
    },
  )
  # Smoothness is enforced by the reference action-rate and joint-acceleration
  # costs below.  Do not add a joint-velocity cost: it discouraged the required
  # finite-speed rise and produced a low four-wheel local optimum.
  cfg.rewards["joint_acc_l2"].weight = -2.5e-4
  cfg.rewards["leg_excess_velocity"] = RewardTermCfg(
    # Deployment-compatible transition smoothing: only velocity above this
    # physical threshold is penalized, so a finite-speed rear-wheel rise is
    # possible without filtering or rewriting the policy action.
    func=mdp.joint_velocity_above_l2,
    weight=-0.01,
    params={"threshold": 4.0, "asset_cfg": all_leg_cfg},
  )
  # Continuous wheel joints have no meaningful position range.  Restrict the
  # inherited joint-limit regularizer to the twelve articulated leg joints.
  cfg.rewards["joint_pos_limits"].weight = -2.0
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=GO2W_LEG_JOINTS
  )
  # Direct reference mapping.  Failure still resets immediately; no terminal
  # bonus/penalty changes the static posture objective.
  cfg.rewards["is_terminated"].weight = 0.0
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=GO2W_LEG_JOINTS
  )
  cfg.rewards["action_rate_l2"].weight = -0.10
  cfg.rewards["body_ang_vel"].weight = -0.05
  cfg.rewards["body_ang_vel"].params["asset_cfg"] = SceneEntityCfg(
    "robot", body_names=("base_link",)
  )

  # Follow the user-specified Go2 upright reference: a trunk touch is a fall,
  # while leg contacts are penalised during the transition.  The final-support
  # validity condition rejects every contact other than the rear tyres once
  # upright.  The standard orientation termination is removed because reset is
  # an ordinary horizontal four-wheel quadruped.
  cfg.terminations.pop("fell_over", None)
  cfg.terminations.pop("illegal_contact", None)
  cfg.terminations["base_ground_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": base_ground_contact_cfg.name, "force_threshold": 0.5},
  )
  cfg.terminations["leaned_forbidden_ground_contact"] = TerminationTermCfg(
    func=mdp.upright_illegal_contact,
    params={
      "sensor_name": forbidden_contact_cfg.name,
      "target_gravity": _UPRIGHT_GRAVITY,
      # Once the base has made meaningful progress toward the final upright
      # attitude, FL/FR contact is a propped sitting posture rather than a
      # valid four-wheel reset.  A purely state-based gate preserves the
      # original reset and does not prescribe how quickly to stand.
      "upright_gate_error": 0.80,
      "force_threshold": 0.5,
    },
  )

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

  return cfg
