from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def target_projected_gravity_l2(
  env: ManagerBasedRlEnv,
  target_gravity: tuple[float, float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Squared error to a desired gravity direction expressed in the base frame.

  For Go2W rearing onto its rear wheels, the desired direction is ``(-1, 0, 0)``:
  gravity points along the robot's negative forward axis when its trunk is
  vertical.  This is a final-pose reward, not a phase-dependent stand-up cue.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(
    target_gravity,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  return torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)


def target_projected_gravity_alignment(
  env: ManagerBasedRlEnv,
  target_gravity: tuple[float, float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Positive, phase-free alignment with one desired body-frame gravity vector.

  The ordinary four-wheel reset is orthogonal to Go2W's reared target, hence
  scores zero; exact rear-wheel upright scores one.  This is a pure final
  attitude objective and deliberately contains neither a time signal nor a
  joint-space stand-up reference.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(
    target_gravity,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  return torch.clamp_min(
    torch.sum(asset.data.projected_gravity_b * target, dim=1), 0.0
  )


def target_projected_gravity_alignment_power(
  env: ManagerBasedRlEnv,
  target_gravity: tuple[float, float, float],
  power: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sharpen one static target-gravity alignment near exact upright."""
  if power <= 1.0:
    raise ValueError("power must be greater than one.")
  alignment = target_projected_gravity_alignment(
    env, target_gravity=target_gravity, asset_cfg=asset_cfg
  )
  return torch.pow(alignment, power)


def joint_position_l1(
  env: ManagerBasedRlEnv,
  target_joint_pos: tuple[float, ...],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """L1 distance to one static articulated-joint pose.

  This is the direct counterpart of the reference Go2 upright task's
  ``default_pos`` term.  It depends only on the current configuration, never
  on rollout time or a stand-up phase.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.joint_ids, slice):
    raise TypeError("joint_position_l1 requires an explicit joint subset.")
  if len(target_joint_pos) != len(asset_cfg.joint_ids):
    raise ValueError(
      "target_joint_pos length must match the selected joint count: "
      f"{len(target_joint_pos)} != {len(asset_cfg.joint_ids)}"
    )
  target = torch.tensor(
    target_joint_pos, dtype=asset.data.joint_pos.dtype, device=env.device
  )
  return torch.sum(
    torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - target), dim=1
  )


def joint_deviation_l1(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """L1 distance from the asset's default joint positions.

  This is the same regularizer used by RobotLab's
  ``create_joint_deviation_l1_rewterm``.  Selecting the four Go2W hip joints
  constrains lateral leg splay on both the raised front arms and the rear-wheel
  support legs, while leaving the pitch joints free to discover the stand-up.
  """
  asset: Entity = env.scene[asset_cfg.name]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)


def joint_velocity_above_l2(
  env: ManagerBasedRlEnv,
  threshold: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize only joint speed above a deployable physical threshold.

  This is intentionally not an action filter: the policy always sends its
  direct target to the actuator.  It is a smoothness cost that permits the
  finite joint motion required for a stand-up transition while suppressing
  impulsive joint motion.
  """
  if threshold < 0.0:
    raise ValueError("threshold must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  excess = torch.relu(torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - threshold)
  return torch.sum(torch.square(excess), dim=1)


def root_height_l1(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Absolute root-height error to one static final-state target."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.abs(asset.data.root_link_pos_w[:, 2] - target_height)


def site_height_l1(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Mean absolute height error for selected sites.

  Unlike a narrow exponential bonus, this retains a useful state-only
  preference when a wheel is still far below its final hanging height.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("site_height_l1 requires an explicit site subset.")
  return torch.abs(
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - target_height
  ).mean(dim=1)


def target_projected_gravity_exp(
  env: ManagerBasedRlEnv,
  std: float,
  target_gravity: tuple[float, float, float],
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Bounded final-pose reward for matching a desired gravity direction.

  Unlike a negative orientation cost, a bounded positive reward cannot make a
  premature termination preferable to remaining in the initial four-wheel
  pose.  It still supplies no trajectory, phase, or intermediate get-up cue.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(
    target_gravity,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  error = torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)
  return torch.exp(-error / std**2)


def target_projected_gravity_exp_above_baseline(
  env: ManagerBasedRlEnv,
  std: float,
  target_gravity: tuple[float, float, float],
  baseline_error: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward final upright alignment relative to the four-foot reset baseline.

  This is still only a final-attitude objective: it depends solely on the
  current projected gravity and has no phase, time, reference trajectory, or
  intermediate pose target.  Subtracting the known reset alignment prevents a
  quadruped from collecting positive upright reward merely by remaining in its
  initial horizontal posture.  The result is normalized so exact target
  alignment remains one.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(
    target_gravity,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  error = torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)
  raw = torch.exp(-error / std**2)
  baseline = torch.exp(
    torch.tensor(-baseline_error / std**2, dtype=raw.dtype, device=raw.device)
  )
  return torch.clamp_min((raw - baseline) / (1.0 - baseline), 0.0)


def joint_position_exp(
  env: ManagerBasedRlEnv,
  std: float,
  target_joint_pos: tuple[float, ...],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Continuously reward a selected final static joint geometry.

  This is intentionally independent of time and body attitude.  It supplies a
  dense geometric gradient toward a final morphology (for example relaxed
  hanging arms) without encoding an intermediate get-up sequence.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.joint_ids, slice):
    raise TypeError("joint_position_exp requires an explicit joint subset.")
  if len(target_joint_pos) != len(asset_cfg.joint_ids):
    raise ValueError(
      "target_joint_pos length must match the selected joint count: "
      f"{len(target_joint_pos)} != {len(asset_cfg.joint_ids)}"
    )
  target = torch.tensor(
    target_joint_pos, dtype=asset.data.joint_pos.dtype, device=env.device
  )
  joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - target
  error = torch.sum(torch.square(joint_error), dim=1)
  return torch.exp(-error / std**2)


def upright_joint_position_exp(
  env: ManagerBasedRlEnv,
  std: float,
  target_joint_pos: tuple[float, ...],
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward a final joint pose only once the body is already upright.

  The binary orientation gate intentionally keeps this term out of the
  four-wheel-to-upright transition.  It can therefore specify a calm final arm
  posture without encoding a get-up trajectory.  The selected joints must be
  specified in the model's deterministic order so ``target_joint_pos`` has a
  stable correspondence.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.joint_ids, slice):
    raise TypeError("upright_joint_position_exp requires an explicit joint subset.")
  if len(target_joint_pos) != len(asset_cfg.joint_ids):
    raise ValueError(
      "target_joint_pos must contain one value for each selected joint "
      f"({len(target_joint_pos)} != {len(asset_cfg.joint_ids)})."
    )

  target_pose = torch.tensor(
    target_joint_pos, dtype=asset.data.joint_pos.dtype, device=env.device
  )
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  upright_gate = (gravity_error < upright_gate_error).to(asset.data.joint_pos.dtype)
  pose_error = torch.sum(
    torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pose), dim=1
  )
  return upright_gate * torch.exp(-pose_error / std**2)


def upright_joint_position_l2(
  env: ManagerBasedRlEnv,
  target_joint_pos: tuple[float, ...],
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return final-pose squared error only after the body is upright.

  This unbounded cost complements the bounded exponential pose reward when a
  small joint subset has a hard geometric constraint.  In particular, it can
  keep a pair of wheels parallel with the trunk instead of accepting a wide
  toe-out posture that happens to balance well.  The orientation gate makes
  it a final-pose constraint, not a get-up trajectory reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.joint_ids, slice):
    raise TypeError("upright_joint_position_l2 requires an explicit joint subset.")
  if len(target_joint_pos) != len(asset_cfg.joint_ids):
    raise ValueError(
      "target_joint_pos must contain one value for each selected joint "
      f"({len(target_joint_pos)} != {len(asset_cfg.joint_ids)})."
    )

  target_pose = torch.tensor(
    target_joint_pos, dtype=asset.data.joint_pos.dtype, device=env.device
  )
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  upright_gate = (gravity_error < upright_gate_error).to(asset.data.joint_pos.dtype)
  pose_error = torch.sum(
    torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pose), dim=1
  )
  return upright_gate * pose_error


def upright_joint_velocity_l2(
  env: ManagerBasedRlEnv,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize selected-joint motion after reaching the upright final pose."""
  asset: Entity = env.scene[asset_cfg.name]
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  upright_gate = (gravity_error < upright_gate_error).to(asset.data.joint_vel.dtype)
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return upright_gate * torch.sum(torch.square(joint_vel), dim=1)


def terrain_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 0.5,
) -> torch.Tensor:
  """Count forbidden terrain-contacting collision bodies.

  This is the same dense collision signal used by the reference Go2 upright
  task: a body is charged whenever its contact force exceeds a small threshold.
  We read the substep force history when available, because MuJoCo Warp's
  contact ``found`` count can be zero for a persistent resting contact while
  its force is correctly reported.  Wheel geometry is excluded when the sensor
  is configured, so this applies only to forbidden support bodies.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # [B, contact, history, xyz] -> [B, contact]
    force = torch.linalg.vector_norm(data.force_history, dim=-1)
    in_contact = torch.any(force > force_threshold, dim=-1)
  else:
    assert data.force is not None
    in_contact = torch.linalg.vector_norm(data.force, dim=-1) > force_threshold
  return in_contact.to(torch.float32).sum(dim=1)


def all_contacting_geoms_in_air(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Return one only when every geometry observed by a contact sensor is airborne.

  This is intentionally the same binary signal as the reference task's
  ``handstand_feet_on_air`` reward.  For Go2W the sensor contains FL/FR only:
  those wheels are valid at the quadruped reset but must leave the floor in the
  final rear-wheel stance.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force_mag = torch.norm(data.force_history, dim=-1)
    in_contact = (force_mag > force_threshold).any(dim=-1)
  else:
    assert data.found is not None
    in_contact = (data.found.reshape(env.num_envs, data.found.shape[1], -1) > 0).any(dim=-1)
  return (~in_contact).all(dim=1).to(torch.float32)


def site_height_l1_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  scale: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Exponentially reward a static site-height target using an L1 error.

  Matches the reference upright task's ``exp(-sum(abs(height-target))*scale)``
  shape.  It is a final morphology signal, not a timed standing trajectory.
  """
  if scale <= 0.0:
    raise ValueError("scale must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("site_height_l1_exp requires an explicit site subset.")
  height_error = torch.sum(
    torch.abs(asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - target_height), dim=1
  )
  return torch.exp(-height_error * scale)


def site_height_at_least(
  env: ManagerBasedRlEnv,
  minimum_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Static, bounded clearance score for selected sites.

  The score grows linearly from ground level and saturates once the requested
  final clearance is reached.  It specifies a support-valid final geometry,
  never a timed lift or a joint-space transition.
  """
  if minimum_height <= 0.0:
    raise ValueError("minimum_height must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("site_height_at_least requires an explicit site subset.")
  return torch.clamp(
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2] / minimum_height,
    min=0.0,
    max=1.0,
  ).mean(dim=1)


def root_height_l1_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  scale: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward one static root height with the reference task's L1 exponential.

  This is the direct Go2W counterpart of ``_reward_base_height`` in the Go2
  upright task: ``exp(-abs(root_z - target) * scale)``.  It has no attitude,
  phase, or time gate.  For the high rear-wheel stance, unlike the old seated
  target, this gives a useful final-state gradient from a four-wheel reset.
  """
  if scale <= 0.0:
    raise ValueError("scale must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  return torch.exp(-torch.abs(asset.data.root_link_pos_w[:, 2] - target_height) * scale)


def upright_terrain_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  force_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Charge forbidden support only near the final upright attitude.

  FL/FR wheel contact is physically necessary at the four-foot reset.  This
  final-state gate therefore leaves the discovery of the rise unconstrained,
  then applies the same force-based support rule that the terminal condition
  uses once the robot is close enough to be considered upright.
  """
  if upright_gate_error <= 0.0:
    raise ValueError("upright_gate_error must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(
    target_gravity,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target), dim=1
  )
  upright_gate = (gravity_error < upright_gate_error).to(torch.float32)
  return upright_gate * terrain_contact_count(
    env, sensor_name=sensor_name, force_threshold=force_threshold
  )


def upright_root_height_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the final upright trunk height above the flat floor.

  This is deliberately gated by the same final-attitude condition as the pose
  rewards.  It specifies where a completed two-wheel stance should settle, not
  how to get there from the initial four-wheel pose.
  """
  asset: Entity = env.scene[asset_cfg.name]
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  height_error = torch.square(asset.data.root_link_pos_w[:, 2] - target_height)
  reward = torch.exp(-height_error / std**2)
  return (gravity_error < upright_gate_error).to(reward.dtype) * reward


def upright_site_height_l1(
  env: ManagerBasedRlEnv,
  target_height: float,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  minimum_root_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Measure a site-height error only in the completed upright stance.

  This is a static final-geometry constraint.  It deliberately does not shape
  the four-wheel-to-upright transition, so a natural hanging-wheel preference
  cannot prevent the policy from first discovering a valid rear-wheel stand.
  """
  if upright_gate_error <= 0.0:
    raise ValueError("upright_gate_error must be positive.")
  if minimum_root_height < 0.0:
    raise ValueError("minimum_root_height must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("upright_site_height_l1 requires an explicit site subset.")
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  final_stance = (gravity_error < upright_gate_error) & (
    asset.data.root_link_pos_w[:, 2] >= minimum_root_height
  )
  error = torch.abs(
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2] - target_height
  ).mean(dim=1)
  return final_stance.to(error.dtype) * error


def upright_site_position_l1(
  env: ManagerBasedRlEnv,
  target_positions_b: tuple[tuple[float, float, float], ...],
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  minimum_root_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Measure final site positions relative to the trunk, without joint targets.

  This is intentionally a task-space condition on the visible front wheel
  centres, active only after a rear-wheel stance has been reached.  It selects
  wheels hanging beside the trunk rather than an arbitrary folded front-leg
  configuration, while leaving the four-wheel-to-upright motion unconstrained.
  """
  if upright_gate_error <= 0.0:
    raise ValueError("upright_gate_error must be positive.")
  if minimum_root_height < 0.0:
    raise ValueError("minimum_root_height must be non-negative.")
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("upright_site_position_l1 requires an explicit site subset.")
  if len(target_positions_b) != len(asset_cfg.site_ids):
    raise ValueError("target_positions_b must have one position per selected site.")

  asset: Entity = env.scene[asset_cfg.name]
  target_up = torch.tensor(
    target_gravity,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  final_stance = (gravity_error < upright_gate_error) & (
    asset.data.root_link_pos_w[:, 2] >= minimum_root_height
  )

  wheel_offset_w = (
    asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w[:, None]
  )
  root_quat_w = asset.data.root_link_quat_w[:, None].expand(
    -1, wheel_offset_w.shape[1], -1
  )
  wheel_pos_b = quat_apply_inverse(root_quat_w, wheel_offset_w)
  target_pos_b = torch.tensor(
    target_positions_b, dtype=wheel_pos_b.dtype, device=env.device
  )
  error = torch.abs(wheel_pos_b - target_pos_b).mean(dim=(1, 2))
  return final_stance.to(error.dtype) * error


def upright_wheel_axis_alignment_l1(
  env: ManagerBasedRlEnv,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  minimum_root_height: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Align a left/right wheel pair on one lateral axle in the final stance.

  The pair may differ only along the body's y axis.  Their body-frame x and z
  positions must match, which prevents the rear support wheels from forming a
  fore/aft scissor and keeps their rolling axes collinear.  This is a final
  wheel-centre geometry condition, not a joint-angle or get-up-trajectory
  target.
  """
  if upright_gate_error <= 0.0:
    raise ValueError("upright_gate_error must be positive.")
  if minimum_root_height < 0.0:
    raise ValueError("minimum_root_height must be non-negative.")
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 2:
    raise ValueError("upright_wheel_axis_alignment_l1 requires exactly two sites.")

  asset: Entity = env.scene[asset_cfg.name]
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  final_stance = (gravity_error < upright_gate_error) & (
    asset.data.root_link_pos_w[:, 2] >= minimum_root_height
  )
  wheel_offset_w = (
    asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w[:, None]
  )
  root_quat_w = asset.data.root_link_quat_w[:, None].expand(-1, 2, -1)
  wheel_pos_b = quat_apply_inverse(root_quat_w, wheel_offset_w)
  error = torch.sum(torch.abs(wheel_pos_b[:, 0, (0, 2)] - wheel_pos_b[:, 1, (0, 2)]), dim=1)
  return final_stance.to(error.dtype) * error


def site_height_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward selected sites approaching their final upright height.

  For Go2W this keeps the front wheel centres well above the ground while the
  rear wheels provide support.  It deliberately has no attitude or time gate:
  like the reference task's handstand-foot-height term, it gives a continuous
  geometric gradient from the ordinary four-wheel reset toward the same final
  upright morphology without encoding a get-up trajectory.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("site_height_exp requires an explicit site subset.")
  site_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  height_error = torch.mean(torch.square(site_height - target_height), dim=1)
  return torch.exp(-height_error / std**2)


def upright_site_height_exp(
  env: ManagerBasedRlEnv,
  target_height: float,
  std: float,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward a selected site height only in the completed upright stance."""
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("upright_site_height_exp requires an explicit site subset.")
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  site_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  height_error = torch.mean(torch.square(site_height - target_height), dim=1)
  reward = torch.exp(-height_error / std**2)
  return (gravity_error < upright_gate_error).to(reward.dtype) * reward


def upright_site_clearance_l2(
  env: ManagerBasedRlEnv,
  min_height: float,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return squared final-stance penalty for sites below a clearance height.

  Unlike an attraction to an average height, this one-sided cost explicitly
  rules out using the selected wheels as extra ground contacts.  It is gated
  by the final upright attitude, so it is not an instruction for the initial
  four-wheel-to-two-wheel transition.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise TypeError("upright_site_clearance_l2 requires an explicit site subset.")
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  site_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  deficit = torch.relu(min_height - site_height)
  clearance_error = torch.mean(torch.square(deficit), dim=1)
  return (gravity_error < upright_gate_error).to(clearance_error.dtype) * clearance_error


def _upright_heading_frame(asset: Entity) -> tuple[torch.Tensor, torch.Tensor]:
  """Return forward/right ground-plane axes for a reared Go2W.

  When reared, the robot's local z-axis is the wheel rolling direction while
  local x points upward.  Project z onto the ground plane to obtain a stable
  command frame.  Before the robot has risen this projection is nearly zero;
  use world x as a benign fallback until it acquires a heading.
  """
  q = asset.data.root_link_quat_w
  body_z_w = quat_apply(q, torch.tensor([0.0, 0.0, 1.0], device=q.device).expand(q.shape[0], -1))
  forward_xy = body_z_w[:, :2]
  norm = torch.linalg.vector_norm(forward_xy, dim=1, keepdim=True)
  fallback = torch.tensor([1.0, 0.0], device=q.device, dtype=q.dtype).expand_as(forward_xy)
  forward_xy = torch.where(norm > 1e-4, forward_xy / norm.clamp_min(1e-4), fallback)
  right_xy = torch.stack((-forward_xy[:, 1], forward_xy[:, 0]), dim=1)
  return forward_xy, right_xy


def track_upright_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  target_gravity: tuple[float, float, float] = (-1.0, 0.0, 0.0),
  upright_gate_error: float = 0.20,
  minimum_root_height: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track planar commands only after reaching the reared Go2W attitude.

  A four-wheel or low reared pose must never earn locomotion reward: it first
  has to solve the generic final upright geometry.  The gate is an eligibility
  condition for commanded motion, rather than an intermediate get-up reward.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  forward_xy, right_xy = _upright_heading_frame(asset)
  velocity_xy = asset.data.root_link_lin_vel_w[:, :2]
  actual = torch.stack(
    (
      torch.sum(velocity_xy * forward_xy, dim=1),
      torch.sum(velocity_xy * right_xy, dim=1),
    ),
    dim=1,
  )
  velocity_reward = torch.exp(
    -torch.sum(torch.square(command[:, :2] - actual), dim=1) / std**2
  )
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  final_stance = (gravity_error < upright_gate_error) & (
    asset.data.root_link_pos_w[:, 2] >= minimum_root_height
  )
  return final_stance.to(velocity_reward.dtype) * velocity_reward


def track_upright_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  target_gravity: tuple[float, float, float] = (-1.0, 0.0, 0.0),
  upright_gate_error: float = 0.20,
  minimum_root_height: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track yaw only after reaching the reared Go2W attitude."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  error = command[:, 2] - asset.data.root_link_ang_vel_w[:, 2]
  yaw_reward = torch.exp(-torch.square(error) / std**2)
  target_up = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target_up), dim=1
  )
  final_stance = (gravity_error < upright_gate_error) & (
    asset.data.root_link_pos_w[:, 2] >= minimum_root_height
  )
  return final_stance.to(yaw_reward.dtype) * yaw_reward


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward
