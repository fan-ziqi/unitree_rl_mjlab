"""Unitree Go2W upright walking environment configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
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
from src.tasks.velocity.mdp.velocity_command import UniformVelocityCommandCfg
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

# Entity-local Go2W joint order is FL leg, FL wheel, FR leg, FR wheel, ... .
# The smoothness costs cover precisely the twelve articulated leg joints.
_ALL_LEG_JOINT_IDS = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
# The upright task is a rear-wheel biped.  The former front wheels stay in the
# model for contact and visual fidelity, but are deliberately omitted from the
# policy action interface so they can never propel the robot.
_REAR_WHEEL_JOINTS = ("RL_wheel_joint", "RR_wheel_joint")
# Only these two tyres are legal ground supports after the robot has reared.
# FL/FR are passive hanging-arm wheels, not support wheels.
_REAR_WHEEL_GEOMS = GO2W_WHEEL_GEOMS[2:]
_UPRIGHT_GRAVITY = (-1.0, 0.0, 0.0)
# One criterion defines a qualified final stance across the task and evaluator.
_QUALIFIED_UPRIGHT_ERROR = 0.12
# Wheel-centre locations for a naturally bent front arm hanging down the body.
# They keep the wheels at the body's longitudinal mid-plane (not stretched out
# ahead of it).  These are task-space coordinates in base_link, not a
# joint-angle template.
_NATURAL_FRONT_WHEEL_POSITIONS_B = (
  (-0.11467, 0.14200, 0.00000),
  (-0.11467, -0.14200, 0.00000),
)


def unitree_go2w_upright_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create a flat-ground task that resets Go2W on four wheels.

  The only posture objective is the final reared orientation.  There is no
  get-up trajectory, phase reward, or fallen-state recovery: a policy must
  discover a smooth transition from the ordinary four-foot reset pose itself.
  """
  cfg = make_velocity_env_cfg()
  # The stock velocity command sampler has only continuous samples, making an
  # exact in-place turn effectively absent.  This task-local command class
  # adds explicit linear-only and yaw-only samples; it does not alter any
  # other robot task.
  cfg.commands["twist"] = UniformVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(3.0, 8.0),
    heading_command=False,
    rel_standing_envs=0.50,
    rel_yaw_only_envs=0.0,
    rel_linear_only_envs=0.0,
    yaw_upright_gate_error=_QUALIFIED_UPRIGHT_ERROR,
    yaw_minimum_root_height=0.48,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(-1.0, 1.0),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.5, 0.5),
      heading=None,
    ),
  )
  # Training does not render, so build the collision-only model.  Play mode
  # deliberately retains the visual meshes for video and visual inspection.
  cfg.scene.entities = {"robot": get_go2w_robot_cfg(headless=not play)}
  # Preserve the Go2W asset's native actuator gains.  Standing smoothness is
  # a policy objective (action-rate / joint-motion costs), not a hidden change
  # to the controller that will differ from the deployed robot.
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
      # Keep this asset's native P20/D0.5 gains, while using the verified direct
      # residual ranges needed to reach a two-wheel stance from the four-wheel
      # default.  This changes neither model gains nor action timing.
      scale={
        r"^(FL|FR)_hip_joint$": 0.25,
        r"^(FL|FR)_thigh_joint$": 0.80,
        r"^(FL|FR)_calf_joint$": 0.40,
        r"^(RL|RR)_hip_joint$": 0.25,
        r"^(RL|RR)_thigh_joint$": 0.50,
        r"^(RL|RR)_calf_joint$": 0.30,
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
  # Keep the command envelope centred on the demonstration speeds.  Narrower
  # values make accurate differential turning learnable before introducing a
  # high-speed driving curriculum.
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  # A rear-wheel upright robot is non-holonomic.  Train feasible longitudinal
  # and yaw commands directly through the two rear-wheel velocity motors;
  # lateral commands remain unavailable.  Uniform sampling includes pure
  # turns as well as forward/reverse turns for the requested driving demo.
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-0.5, 0.5)
  twist_cmd.heading_command = False
  twist_cmd.ranges.heading = None
  # Command mix: static balance, pure fore/aft driving, pure differential
  # turns, and combined arcs all occur explicitly.  Pure turns are required
  # for the requested left/right sections of the demonstration video.
  twist_cmd.rel_standing_envs = 0.50
  twist_cmd.rel_yaw_only_envs = 0.0
  twist_cmd.rel_linear_only_envs = 0.0

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
  # Once the final stance is attained, wheel speed must be a first-class
  # objective: otherwise the policy can use a constant wheel drift to balance
  # while receiving much larger posture rewards.
  cfg.rewards["track_linear_velocity"].weight = 10.0
  cfg.rewards["track_linear_velocity"].params["std"] = 0.4
  cfg.rewards["track_linear_velocity"].params["target_gravity"] = _UPRIGHT_GRAVITY
  cfg.rewards["track_linear_velocity"].params["upright_gate_error"] = _QUALIFIED_UPRIGHT_ERROR
  # A merely vertical but low body is still converting from four to two wheels;
  # it cannot receive any driving incentive until the final body-height region.
  cfg.rewards["track_linear_velocity"].params["minimum_root_height"] = 0.48
  cfg.rewards["track_angular_velocity"].func = mdp.track_upright_angular_velocity
  cfg.rewards["track_angular_velocity"].weight = 10.0
  cfg.rewards["track_angular_velocity"].params["std"] = 0.7
  cfg.rewards["track_angular_velocity"].params["target_gravity"] = _UPRIGHT_GRAVITY
  cfg.rewards["track_angular_velocity"].params["upright_gate_error"] = _QUALIFIED_UPRIGHT_ERROR
  cfg.rewards["track_angular_velocity"].params["minimum_root_height"] = 0.48
  # One continuous, all-time final-attitude objective.  It is the standard
  # positive exponential form: ordinary four-wheel reset receives little
  # reward and the requested body-frame gravity direction receives one.  This
  # avoids turning the entire pre-stand exploration region into a negative
  # return, while still specifying no phase or prescribed standing trajectory.
  cfg.rewards["upright"] = RewardTermCfg(
    func=mdp.target_projected_gravity_exp,
    # Together with the single body-height reward below, this is the complete
    # posture objective: no front/rear joint position, wheel-height, or
    # contact-lift reward is used to prescribe how the robot must get up.
    weight=5.0,
    params={"target_gravity": _UPRIGHT_GRAVITY, "std": 1.0},
  )
  cfg.rewards.pop("upright_precision", None)
  cfg.rewards.pop("upright_precise", None)
  # The supplied upright reference has no alive bonus.  Adding one makes the
  # original four-wheel pose a high-survival local optimum, even though it is
  # continuously wrong for both posture objectives.
  cfg.rewards.pop("alive", None)

  all_leg_cfg = SceneEntityCfg("robot", joint_ids=_ALL_LEG_JOINT_IDS)
  # RobotLab's standard hip-deviation regularizer.  The Go2W hip joints are
  # the lateral swing DoFs, so this one all-time cost stops both the two raised
  # front arms and the two rear support legs from splaying apart.  It keeps the
  # pitch joints unconstrained; the robot must still learn the stand-up from
  # the ordinary four-wheel reset rather than follow a joint-space trajectory.
  cfg.rewards["joint_deviation_hip_l1"] = RewardTermCfg(
    func=mdp.joint_deviation_l1,
    # RobotLab uses -0.2 in its ordinary quadruped velocity task.  Here the
    # final front-wheel geometry term is deliberately much stronger (-40), so
    # scale this identical all-hip regularizer to retain the user's no-splay
    # requirement in the completed rear-wheel stance.
    weight=-4.0,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=r".*_hip_joint")},
  )
  # The sole geometric posture objective.  The plain-Go2 reference uses
  # 0.47 m, but forward kinematics of this Go2W asset's rear-wheel support
  # gives 0.561 m.  Using the asset-correct height keeps the 34 cm low
  # front-wheel prop distinctly below the requested stance.  Natural arm hang
  # and wheel alignment must emerge from dynamics/default pose, not a hidden
  # joint-space template.
  cfg.rewards["upright_base_height"] = RewardTermCfg(
    func=mdp.root_height_l1_exp,
    # A low front-wheel-supported lean remains roughly 18 cm below the target
    # and otherwise earns a sizeable broad exponential reward.  Give this
    # same single body-height objective enough weight to prefer the requested
    # 0.561 m rear-wheel stance; no extra leg or wheel target is introduced.
    weight=5.0,
    params={
      "target_height": 0.561,
      "scale": 5.0,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  # This is a final support-state criterion, not a leg target or a timed
  # stand-up cue: the requested robot must have both former front wheels off
  # the terrain before it is a rear-wheel biped.  The pure body-only objective
  # otherwise converges to a low front-wheel-supported lean.
  cfg.rewards["front_wheels_air"] = RewardTermCfg(
    func=mdp.all_contacting_geoms_in_air,
    weight=2.0,
    params={"sensor_name": front_wheel_contact_cfg.name, "force_threshold": 1.0},
  )
  # The same geometric signal used by the Go2W handstand reference, expressed
  # for this asset's wheel-centre sites.  It supplies a continuous gradient
  # toward the required free front wheels; it does not prescribe a joint pose.
  cfg.rewards["front_wheel_height"] = RewardTermCfg(
    func=mdp.site_height_exp,
    weight=5.0,
    params={
      "target_height": 0.446,
      "std": 0.35,
      "asset_cfg": SceneEntityCfg("robot", site_names=("FL", "FR")),
    },
  )
  # Penalize excess height on exactly the same two wheel centres only after
  # the rear-wheel stand exists.  This selects the natural side-hanging wheel
  # level without obstructing discovery or specifying a single joint angle.
  cfg.rewards["front_wheel_height_error"] = RewardTermCfg(
    func=mdp.upright_site_height_l1,
    weight=-15.0,
    params={
      "target_height": 0.446,
      "target_gravity": _UPRIGHT_GRAVITY,
      "upright_gate_error": _QUALIFIED_UPRIGHT_ERROR,
      "minimum_root_height": 0.48,
      "asset_cfg": SceneEntityCfg("robot", site_names=("FL", "FR")),
    },
  )
  # Height alone admits an asymmetric folded-arm pose.  Constrain the two
  # *wheel centres* relative to the trunk in the final stance, so the passive
  # front wheels are level with the body sides and the arms hang naturally.
  # This has no joint-position target and no influence on the rise itself.
  cfg.rewards["front_wheel_side_alignment"] = RewardTermCfg(
    func=mdp.upright_site_position_l1,
    # Once the stand is physically stable this must dominate the remaining
    # front-leg redundancy: height-only supervision can still leave a wheel
    # several centimetres forward of the trunk.  The gate keeps this from
    # prescribing the rise itself.
    weight=-40.0,
    params={
      "target_positions_b": _NATURAL_FRONT_WHEEL_POSITIONS_B,
      "target_gravity": _UPRIGHT_GRAVITY,
      "upright_gate_error": _QUALIFIED_UPRIGHT_ERROR,
      "minimum_root_height": 0.48,
      "asset_cfg": SceneEntityCfg("robot", site_names=("FL", "FR")),
    },
  )
  # The two rear wheels are the sole ground-support pair.  Their centres must
  # lie on one lateral body-frame axle (same x and z, free y separation), so
  # the robot cannot stand with its rear legs scissored fore/aft.  Like the
  # front-wheel geometry terms this only selects the completed configuration;
  # it neither imposes a joint target nor prescribes the rise.
  cfg.rewards["rear_wheel_axis_alignment"] = RewardTermCfg(
    func=mdp.upright_wheel_axis_alignment_l1,
    weight=-20.0,
    params={
      "target_gravity": _UPRIGHT_GRAVITY,
      # Begin shaping only in the last near-vertical portion of the rise.  A
      # 0.12 final-only gate left the deterministic policy just outside the
      # upright manifold, where this essential geometry could not improve.
      # This is still state-gated (not timed or phased) and retains the strict
      # 0.12 criterion in the evaluator.
      "upright_gate_error": 0.25,
      "minimum_root_height": 0.48,
      "asset_cfg": SceneEntityCfg("robot", site_names=("RL", "RR")),
    },
  )
  # A bounded, continuous clearance score makes the required off-ground
  # support geometry reachable from the four-wheel reset.  It is still only a
  # final spatial condition on the two passive front wheel centres.
  cfg.rewards["front_wheel_clearance"] = RewardTermCfg(
    func=mdp.site_height_at_least,
    weight=8.0,
    params={
      "minimum_height": 0.41,
      "asset_cfg": SceneEntityCfg("robot", site_names=("FL", "FR")),
    },
  )
  cfg.rewards["forbidden_ground_collision"] = RewardTermCfg(
    # Literal mapping of the reference Go2 upright task's collision term:
    # penalise every non-foot/non-wheel terrain contact throughout the rollout.
    # FL/FR are deliberately excluded here because four-wheel reset is valid;
    # their final lift-off follows from the hanging-arm geometry and strict
    # final-support terminal below.  This prevents the policy from
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
  # Smoothness is enforced on the direct deployable policy output and its
  # resulting joint motion.  These costs neither filter nor delay an action.
  # Keep the discovery signal sufficiently dense, then use the reference
  # task's direct action-change cost to avoid an impulsive rise.
  cfg.rewards["joint_acc_l2"].weight = -2.5e-5
  cfg.rewards["leg_excess_velocity"] = RewardTermCfg(
    # Deployment-compatible transition smoothing: only velocity above this
    # physical threshold is penalized, so a finite-speed rear-wheel rise is
    # possible without filtering or rewriting the policy action.
    func=mdp.joint_velocity_above_l2,
    weight=-0.01,
    params={"threshold": 6.0, "asset_cfg": all_leg_cfg},
  )
  # Continuous wheel joints have no meaningful position range.  Restrict the
  # inherited joint-limit regularizer to the twelve articulated leg joints.
  cfg.rewards["joint_pos_limits"].weight = -2.0
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=GO2W_LEG_JOINTS
  )
  # Keep the base task's standard fall penalty.  This is a safety/failure
  # signal, not a posture target: without it a policy can improve return by
  # immediately ending an episode to avoid the continuous height/orientation
  # error instead of learning a stand-up transition.
  cfg.rewards["is_terminated"].weight = -200.0
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=GO2W_LEG_JOINTS
  )
  # Penalise target changes directly.  This is evaluated on successive policy
  # actions and does not filter, delay, or rewrite the action sent to the PD
  # controller.
  cfg.rewards["action_rate_l2"].weight = -0.055
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
      # The four-wheel reset remains valid, but a body that is already more
      # than half-way to the requested upright attitude must not settle back
      # onto its front wheels.  This removes the observed low front-wheel
      # prop without supplying a trajectory or an additional reward.
      "upright_gate_error": 0.60,
      "minimum_root_height": 0.25,
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
