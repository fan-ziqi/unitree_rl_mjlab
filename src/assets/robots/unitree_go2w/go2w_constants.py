"""Static, collision-optimized Unitree Go2W configuration."""

from copy import deepcopy
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, BuiltinVelocityActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH


# Derived directly from rl_sar_zoo's correctly oriented Go2W MJCF.  Its visual
# bodies are intact; its collision meshes/cylinders were replaced in this static
# file with 9 boxes, 4 spheres, and the 4 indispensable wheel cylinders.
GO2W_XML: Path = (
  SRC_PATH
  / "assets"
  / "robots"
  / "unitree_go2w"
  / "xmls"
  / "go2w.xml"
)
assert GO2W_XML.exists()

GO2W_LEG_JOINTS = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)
GO2W_WHEEL_JOINTS = (
  "FL_wheel_joint",
  "FR_wheel_joint",
  "RL_wheel_joint",
  "RR_wheel_joint",
)
GO2W_JOINTS = GO2W_LEG_JOINTS + GO2W_WHEEL_JOINTS
GO2W_WHEEL_GEOMS = tuple(f"{name[:2]}_wheel_collision" for name in GO2W_WHEEL_JOINTS)


def get_assets(meshdir: str) -> dict[str, bytes]:
  """Embed all final-model meshes when passing the spec to MuJoCo Warp."""
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2W_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  """Load the final static Go2W model without runtime mesh edits."""
  spec = mujoco.MjSpec.from_file(str(GO2W_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


def get_headless_spec() -> mujoco.MjSpec:
  """Remove only visual meshes; retain every optimized collision primitive."""
  spec = get_spec()
  for geom in tuple(spec.geoms):
    if geom.contype == 0:
      spec.delete(geom)
  return spec


GO2W_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=23.5,
  armature=0.01,
)
GO2W_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=23.5,
  armature=0.01,
)
GO2W_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=20.0,
  damping=0.5,
  effort_limit=35.5,
  armature=0.02,
)
GO2W_ACTUATOR_WHEEL = BuiltinVelocityActuatorCfg(
  target_names_expr=GO2W_WHEEL_JOINTS,
  damping=0.5,
  effort_limit=23.5,
  armature=0.01,
)

# Four-wheel reset; the upright posture is only a learned final objective.
GO2W_FOUR_FOOT_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.365),
  joint_pos={
    ".*hip_joint": 0.0,
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*_wheel_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

_WHEEL_REGEX = r"^[FR][LR]_wheel_collision$"
GO2W_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(r".*_collision",),
  condim={_WHEEL_REGEX: 3, r".*_collision": 1},
  priority={_WHEEL_REGEX: 1},
  friction={_WHEEL_REGEX: (0.8,)},
  solimp={_WHEEL_REGEX: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

GO2W_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2W_ACTUATOR_HIP,
    GO2W_ACTUATOR_THIGH,
    GO2W_ACTUATOR_CALF,
    GO2W_ACTUATOR_WHEEL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


class _Go2WEntity(Entity):
  """Compatibility shim for unbounded velocity-controlled wheel joints."""

  def _add_actuators(self) -> None:
    super()._add_actuators()
    for actuator in self.spec.actuators:
      if actuator.target in GO2W_WHEEL_JOINTS:
        actuator.inheritrange = 0.0
        actuator.ctrllimited = False


class _Go2WEntityCfg(EntityCfg):
  def build(self) -> Entity:
    return _Go2WEntity(self)


def get_go2w_robot_cfg(*, headless: bool = False) -> EntityCfg:
  """Return the final static model, with visuals only for play/video.

  Articulation configs are mutable dataclasses.  Give every task its own
  P20/D0.5 actuator instances so a specialised task cannot accidentally
  overwrite the native gains of the upright task during package registration.
  """
  return _Go2WEntityCfg(
    init_state=GO2W_FOUR_FOOT_INIT_STATE,
    collisions=(GO2W_FULL_COLLISION,),
    spec_fn=get_headless_spec if headless else get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=deepcopy(
        (
          GO2W_ACTUATOR_HIP,
          GO2W_ACTUATOR_THIGH,
          GO2W_ACTUATOR_CALF,
          GO2W_ACTUATOR_WHEEL,
        )
      ),
      soft_joint_pos_limit_factor=GO2W_ARTICULATION.soft_joint_pos_limit_factor,
    ),
  )
