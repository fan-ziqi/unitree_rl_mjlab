"""Mode-conditioned rewards for the compact Go2W trick commands.

The command tensors intentionally contain no target axes, target gravity, or
contact masks.  Those are task semantics and live here, rather than leaking
extra targets into the actor observation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def _mode_mask(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  require_active: bool = True,
  num_modes: int = 5,
) -> tuple[torch.Tensor, torch.Tensor]:
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :num_modes], dim=1)
  mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for selected_mode in modes:
    mask |= mode == selected_mode
  if require_active:
    mask &= torch.sum(command[:, :num_modes], dim=1) > 0.5
  return mask, mode


def _wheel_contacts(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1)


def _two_wheel_pair_index(contacts: torch.Tensor) -> torch.Tensor:
  """Map each exact two-wheel support mask to one of six pair IDs.

  The four canonical adjacent pairs are front, rear, left, and right.  The two
  diagonal pairs are retained for dynamic spin because a fast recovery can
  briefly use them; the static stance modes still use their explicit masks.
  """
  weights = torch.tensor((1, 2, 4, 8), dtype=torch.long, device=contacts.device)
  mask = torch.sum(contacts.long() * weights, dim=1)
  pair_masks = torch.tensor((3, 12, 5, 10, 9, 6), device=contacts.device)
  pair_index = torch.full_like(mask, -1)
  for index, pair_mask in enumerate(pair_masks):
    pair_index[mask == pair_mask] = index
  return pair_index


def mode_gravity_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  gravity_targets: tuple[tuple[float, float, float], ...],
  std: float,
  num_modes: int = 5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Match the gravity target attached internally to a stance one-hot."""
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  targets = torch.tensor(
    gravity_targets,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  target = targets[mode]
  error = torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)
  return active.to(error.dtype) * torch.exp(-error / std**2)


def mode_gravity_alignment(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  gravity_targets: tuple[tuple[float, float, float], ...],
  num_modes: int = 5,
  power: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense stance-orientation reward with useful gradient from a flat reset.

  A Gaussian target is nearly zero when the robot starts ninety degrees away
  from a handstand.  The normalized dot product remains differentiable there,
  which lets PPO discover the commanded tipping direction without a reference
  pose, reset pose, or external assistance.  ``power`` can make a separate
  final-attitude term sharply prefer vertical over a residual lean, while
  retaining a nonzero local gradient.
  """
  if power <= 0.0:
    raise ValueError("power must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  targets = torch.tensor(
    gravity_targets,
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  gravity = asset.data.projected_gravity_b
  gravity = gravity / torch.linalg.vector_norm(
    gravity, dim=1, keepdim=True
  ).clamp_min(1.0e-6)
  alignment = torch.clamp(0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0)
  return active.to(alignment.dtype) * alignment.pow(power)


def mode_root_ang_vel_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  std: float,
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  stationary_velocity_deadband: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a quiet commanded stance, optionally only near its final attitude.

  This is a state-stability signal only: it neither specifies a wheel action
  nor a desired turn direction, so balance coordination remains learned by
  PPO from the one-hot command and proprioception.
  """
  if std <= 0.0:
    raise ValueError("std must be positive.")
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  if stationary_velocity_deadband is not None and stationary_velocity_deadband < 0.0:
    raise ValueError("stationary_velocity_deadband must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  angular_speed_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_b), dim=1)
  score = torch.exp(-angular_speed_sq / std**2)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(
      gravity_targets, dtype=score.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  # A commanded yaw rate is legitimate locomotion, not an instability.  Keep
  # the quiet-body bonus only for the zero x/yaw part of the shared command
  # space; velocity tracking supplies the relevant signal for moving samples.
  if stationary_velocity_deadband is not None:
    command = _command(env, command_name)
    stationary = torch.linalg.vector_norm(command[:, num_modes:], dim=1) <= stationary_velocity_deadband
    score = score * stationary.to(score.dtype)
  return active.to(score.dtype) * score


def mode_contact_match(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  sensor_name: str,
  contact_masks: tuple[tuple[float, float, float, float], ...],
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward selected support contacts, optionally only near target attitude."""
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  contacts = _wheel_contacts(env, sensor_name).float()
  masks = torch.tensor(contact_masks, dtype=contacts.dtype, device=env.device)
  target = masks[mode]
  # A per-wheel Hamming score gives an all-airborne state 0.5 reward for a
  # two-wheel target: it matches both *non-support* wheels while completely
  # missing both supports.  That creates a strong crouching local optimum.
  # Require every desired support to carry contact, while still rejecting
  # extra contacts from the non-support pair.  This is dense in the number of
  # desired contacts and has zero reward for an airborne robot.
  target_count = target.sum(dim=1).clamp_min(1.0)
  non_target_count = (1.0 - target).sum(dim=1).clamp_min(1.0)
  desired_fraction = (contacts * target).sum(dim=1) / target_count
  extra_fraction = (contacts * (1.0 - target)).sum(dim=1) / non_target_count
  score = desired_fraction * (1.0 - extra_fraction)
  if gravity_targets is not None and gravity_power > 0.0:
    asset: Entity = env.scene[asset_cfg.name]
    targets = torch.tensor(
      gravity_targets, dtype=contacts.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0)
    score = score * alignment.pow(gravity_power)
  return active.to(contacts.dtype) * score


def mode_support_wheel_center_height_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  contact_masks: tuple[tuple[float, float, float, float], ...],
  target_height: float,
  std: float,
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Place the commanded support wheel centres at their physical radius.

  This describes contact geometry, not a joint configuration: every leg pose
  that puts the selected wheels on the floor receives the same objective.
  """
  if target_height < 0.0 or std <= 0.0:
    raise ValueError("target_height must be non-negative and std must be positive.")
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise ValueError("mode_support_wheel_center_height_exp needs explicit wheel sites.")
  active, mode = _mode_mask(env, command_name, tuple(range(num_modes)), num_modes=num_modes)
  masks = torch.tensor(contact_masks, dtype=asset.data.site_pos_w.dtype, device=env.device)
  support = masks[mode]
  height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  error = ((height - target_height).square() * support).sum(dim=1) / support.sum(dim=1).clamp_min(1.0)
  score = torch.exp(-error / std**2)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(
      gravity_targets, dtype=score.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  return active.to(score.dtype) * score


def mode_support_wheel_root_clearance_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  target_clearance: float,
  std: float,
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward an extended support pair through wheel-to-base geometry.

  ``target_clearance`` is the vertical separation between the base root and
  the mean centre height of the commanded support wheels.  It is deliberately
  an outcome-space constraint: any hip/thigh/calf configuration with an
  adequately extended support leg receives the same score.  This avoids
  turning the requested long two-wheel stance into a hidden joint-pose
  reference.
  """
  if target_clearance <= 0.0 or std <= 0.0:
    raise ValueError("target_clearance and std must be positive.")
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise ValueError("mode_support_wheel_root_clearance_exp needs explicit wheel sites.")
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  masks = torch.tensor(contact_masks, dtype=asset.data.site_pos_w.dtype, device=env.device)
  support = masks[mode]
  support_height = (
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2] * support
  ).sum(dim=1) / support.sum(dim=1).clamp_min(1.0)
  clearance = asset.data.root_link_pos_w[:, 2] - support_height
  score = torch.exp(-torch.square(clearance - target_clearance) / std**2)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(gravity_targets, dtype=score.dtype, device=env.device)
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  return active.to(score.dtype) * score


def mode_support_wheel_root_clearance_min_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  minimum_clearances: tuple[float, ...],
  std: float,
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  minimum_gravity_alignment: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a *minimum* visible support-leg extension.

  Unlike an equality target, this gives full credit to every configuration
  above the requested clearance.  It therefore asks PPO to stop folding the
  support legs, without imposing a particular wheel-to-body geometry or
  pulling a physically asymmetric front/rear stance toward one common height.
  """
  if len(minimum_clearances) != num_modes:
    raise ValueError("minimum_clearances must provide one value per command mode.")
  if any(clearance < 0.0 for clearance in minimum_clearances) or std <= 0.0:
    raise ValueError("minimum_clearances must be non-negative and std positive.")
  if gravity_power < 0.0 or not 0.0 <= minimum_gravity_alignment < 1.0:
    raise ValueError("gravity_power must be non-negative; minimum_gravity_alignment in [0, 1).")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise ValueError("mode_support_wheel_root_clearance_min_exp needs explicit wheel sites.")
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  masks = torch.tensor(contact_masks, dtype=asset.data.site_pos_w.dtype, device=env.device)
  support = masks[mode]
  support_height = (
    asset.data.site_pos_w[:, asset_cfg.site_ids, 2] * support
  ).sum(dim=1) / support.sum(dim=1).clamp_min(1.0)
  clearance = asset.data.root_link_pos_w[:, 2] - support_height
  minima = torch.tensor(minimum_clearances, dtype=clearance.dtype, device=env.device)
  deficit = torch.clamp(minima[mode] - clearance, min=0.0)
  score = torch.exp(-torch.square(deficit) / std**2)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(gravity_targets, dtype=score.dtype, device=env.device)
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    if minimum_gravity_alignment > 0.0:
      alignment = alignment * (alignment >= minimum_gravity_alignment).to(alignment.dtype)
    score = score * alignment.pow(gravity_power)
  return active.to(score.dtype) * score


def mode_support_leg_length_min(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  sensor_name: str,
  minimum_lengths: tuple[float, ...],
  gravity_targets: tuple[tuple[float, float, float], ...],
  minimum_gravity_alignment: float,
  num_modes: int = 5,
  activation_lengths: tuple[float, ...] | None = None,
  length_power: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Require a visibly extended *supporting* leg without a joint target.

  Base-to-wheel height is not a leg-length measure: a vertical trunk can be
  high while both knee links are folded.  This term instead measures the
  world-space distance between each wheel centre and its corresponding hip.
  The *shorter* commanded support leg determines the score, so one extended
  leg cannot hide a crouched partner.  ``activation_lengths`` optionally
  removes the low-value crouched region from the signal, and ``length_power``
  can make the final extension progressively more valuable.  Thus it
  specifies the visible physical outcome (two long load-bearing legs), not a
  thigh/calf angle or a trajectory.
  """
  if len(minimum_lengths) != num_modes:
    raise ValueError("minimum_lengths must provide one value per command mode.")
  if any(length < 0.0 for length in minimum_lengths):
    raise ValueError("minimum_lengths must be non-negative.")
  if activation_lengths is not None:
    if len(activation_lengths) != num_modes:
      raise ValueError("activation_lengths must provide one value per command mode.")
    if any(length < 0.0 for length in activation_lengths):
      raise ValueError("activation_lengths must be non-negative.")
    if any(start > minimum for start, minimum in zip(activation_lengths, minimum_lengths)):
      raise ValueError("activation_lengths cannot exceed minimum_lengths.")
  if length_power <= 0.0:
    raise ValueError("length_power must be positive.")
  if not 0.0 < minimum_gravity_alignment < 1.0:
    raise ValueError("minimum_gravity_alignment must be in (0, 1).")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice) or isinstance(asset_cfg.body_ids, slice):
    raise ValueError("mode_support_leg_length_min needs explicit wheel sites and hip bodies.")
  if len(asset_cfg.site_ids) != len(asset_cfg.body_ids):
    raise ValueError("wheel sites and hip bodies must have matching order and length.")

  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  masks = torch.tensor(contact_masks, dtype=asset.data.site_pos_w.dtype, device=env.device)
  support = masks[mode]
  wheel_pos = asset.data.site_pos_w[:, asset_cfg.site_ids]
  hip_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
  leg_length = torch.linalg.vector_norm(wheel_pos - hip_pos, dim=-1)
  minimum = torch.tensor(minimum_lengths, dtype=leg_length.dtype, device=env.device)[mode]
  if activation_lengths is None:
    activation = torch.zeros_like(minimum)
  else:
    activation = torch.tensor(
      activation_lengths, dtype=leg_length.dtype, device=env.device
    )[mode]
  length_score = torch.clamp(
    (leg_length - activation.unsqueeze(1))
    / (minimum - activation).unsqueeze(1).clamp_min(1.0e-6),
    0.0,
    1.0,
  ).pow(length_power)

  # A long airborne leg is not a support leg.  A partial-contact factor keeps
  # the contact transition dense, while the minimum requires *both* intended
  # supporting legs to extend.
  contacts = _wheel_contacts(env, sensor_name).to(length_score.dtype)
  contact_fraction = (contacts * support).sum(dim=1) / support.sum(dim=1).clamp_min(1.0)
  non_support_score = torch.ones_like(length_score)
  supported_length_score = torch.where(support.bool(), length_score, non_support_score)
  score = supported_length_score.amin(dim=1) * contact_fraction

  targets = torch.tensor(gravity_targets, dtype=score.dtype, device=env.device)
  gravity = asset.data.projected_gravity_b
  gravity = gravity / torch.linalg.vector_norm(gravity, dim=1, keepdim=True).clamp_min(1.0e-6)
  alignment = torch.clamp(0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0)
  upright = alignment >= minimum_gravity_alignment
  return active.to(score.dtype) * upright.to(score.dtype) * score


def mode_non_support_wheel_clearance(
  env: "ManagerBasedRlEnv",
  command_name: str,
  contact_masks: tuple[tuple[float, float, float, float], ...],
  minimum_height: float,
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward lifting wheels that are not selected as support wheels.

  The term is a dense visual/physical clearance criterion and intentionally
  contains no leg-angle or action target.
  """
  if minimum_height <= 0.0:
    raise ValueError("minimum_height must be positive.")
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice):
    raise ValueError("mode_non_support_wheel_clearance needs explicit wheel sites.")
  active, mode = _mode_mask(env, command_name, tuple(range(num_modes)), num_modes=num_modes)
  masks = torch.tensor(contact_masks, dtype=asset.data.site_pos_w.dtype, device=env.device)
  free = 1.0 - masks[mode]
  height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  clearance = torch.clamp(height / minimum_height, min=0.0, max=1.0)
  score = (clearance * free).sum(dim=1) / free.sum(dim=1).clamp_min(1.0)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(
      gravity_targets, dtype=score.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  return active.to(height.dtype) * score


def stationary_mode_default_joint_pos_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  num_modes: int,
  velocity_deadband: float,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Hold the known stable leg geometry for a zero-speed support stance.

  This is deliberately a *final-pose* regularizer, not a demonstration: it
  compares only the leg configuration with the robot's nominal joint offsets,
  applies only while the x/yaw command is zero, and says nothing about the
  intervening motion.  Those nominal offsets are the configuration in which
  the reset geometry has two wheels exactly on the floor.  Without this term,
  early PPO exploration reliably unfolds the legs and abandons that manifold.
  """
  if std <= 0.0 or velocity_deadband < 0.0:
    raise ValueError("std must be positive and velocity_deadband non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  active, _ = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  stationary = torch.linalg.vector_norm(command[:, num_modes:], dim=1) <= velocity_deadband
  if isinstance(asset_cfg.joint_ids, slice):
    raise ValueError("stationary_mode_default_joint_pos_exp requires explicit joints.")
  joint_ids = asset_cfg.joint_ids
  error = torch.sum(
    torch.square(
      asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    ),
    dim=1,
  )
  return active.to(error.dtype) * stationary.to(error.dtype) * torch.exp(-error / std**2)


def stationary_mode_joint_pos_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  joint_targets: tuple[tuple[float, ...], ...],
  num_modes: int,
  velocity_deadband: float,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Keep each static two-wheel stance in its wheel-only final geometry.

  ``joint_targets`` supplies one complete leg target for every one-hot mode.
  It is deliberately gated to zero x/yaw commands: it defines a clean rest
  pose, while the moving skill remains free to discover its own coordination.
  """
  if std <= 0.0 or velocity_deadband < 0.0:
    raise ValueError("std must be positive and velocity_deadband non-negative.")
  if len(joint_targets) != num_modes:
    raise ValueError("joint_targets must contain one target per mode.")
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  if isinstance(asset_cfg.joint_ids, slice):
    raise ValueError("stationary_mode_joint_pos_exp requires explicit joints.")
  target = torch.tensor(
    joint_targets, dtype=asset.data.joint_pos.dtype, device=env.device
  )[mode]
  if target.shape[1] != len(asset_cfg.joint_ids):
    raise ValueError("Each joint target must match asset_cfg.joint_ids.")
  command = _command(env, command_name)
  stationary = torch.linalg.vector_norm(command[:, num_modes:], dim=1) <= velocity_deadband
  error = torch.sum(
    torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - target), dim=1
  )
  return active.to(error.dtype) * stationary.to(error.dtype) * torch.exp(-error / std**2)


def contact_violation(
  env: "ManagerBasedRlEnv", sensor_name: str, grace_period_s: float = 0.0
) -> torch.Tensor:
  """Return one when a deliberately forbidden robot geometry touches terrain."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  contact = (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1).any(dim=1)
  grace_steps = int(round(grace_period_s / env.step_dt))
  return contact.float() * (env.episode_length_buf >= grace_steps).float()


def stand_idle_mask(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
) -> torch.Tensor:
  """Return 1 only for the stand one-hot with zero speed: four-wheel idle."""
  command = _command(env, command_name)
  return (
    (command[:, 0] > 0.5) & (torch.abs(command[:, 5]) <= speed_deadband)
  ).float()


def stand_idle_gravity_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=asset.data.projected_gravity_b.dtype)
  error = torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)
  return stand_idle_mask(env, command_name, speed_deadband) * torch.exp(-error / std**2)


def stand_idle_contact_match(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  sensor_name: str,
) -> torch.Tensor:
  contacts = _wheel_contacts(env, sensor_name).float()
  return stand_idle_mask(env, command_name, speed_deadband) * contacts.mean(dim=1)


def stand_idle_default_joint_pos_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Hold the zero-rate normal command at the model's four-wheel reset pose."""
  if std <= 0.0:
    raise ValueError("std must be positive.")
  if isinstance(asset_cfg.joint_ids, slice):
    raise ValueError("stand_idle_default_joint_pos_exp needs explicit joints.")
  asset: Entity = env.scene[asset_cfg.name]
  delta = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  error = torch.mean(torch.square(delta), dim=1)
  return stand_idle_mask(env, command_name, speed_deadband) * torch.exp(
    -error / std**2
  )


def stand_idle_stillness_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  linear_velocity_std: float,
  angular_velocity_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Make an untriggered normal command a standstill, not a coasting pose."""
  if linear_velocity_std <= 0.0 or angular_velocity_std <= 0.0:
    raise ValueError("idle velocity scales must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  linear_speed_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=1)
  angular_speed_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_w), dim=1)
  return stand_idle_mask(env, command_name, speed_deadband) * torch.exp(
    -linear_speed_sq / linear_velocity_std**2
    -angular_speed_sq / angular_velocity_std**2
  )


def spin_stand_mask(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
) -> torch.Tensor:
  command = _command(env, command_name)
  return ((command[:, 0] > 0.5) & (torch.abs(command[:, 5]) > speed_deadband)).float()


def fixed_pair_spin_mask(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
) -> torch.Tensor:
  """Return a spinning front/rear two-wheel handstand command.

  The 50-fps AS2-W reference contains its rapid in-place contact pivot on a
  laterally separated *front or rear* pair.  A same-side pair is not asked to
  spin: those commands remain static support postures.
  """
  command = _command(env, command_name)
  return (
    ((command[:, 1] > 0.5)
    | (command[:, 2] > 0.5))
    & (torch.abs(command[:, 5]) > speed_deadband)
  ).float()


def spin_stand_gravity_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  target_gravity: tuple[float, float, float],
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  target = torch.tensor(target_gravity, device=env.device, dtype=asset.data.projected_gravity_b.dtype)
  error = torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)
  return spin_stand_mask(env, command_name, speed_deadband) * torch.exp(-error / std**2)


def spin_stand_contact_match(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  sensor_name: str,
  contact_mask: tuple[float, float, float, float],
) -> torch.Tensor:
  contacts = _wheel_contacts(env, sensor_name).float()
  target = torch.tensor(contact_mask, dtype=contacts.dtype, device=env.device)
  reward = torch.mean(contacts * target + (1.0 - contacts) * (1.0 - target), dim=1)
  return spin_stand_mask(env, command_name, speed_deadband) * reward


def spin_stand_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  target_gravity: tuple[float, float, float],
  posture_error_limit: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track spin about the *current* down direction once in spin posture."""
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  down = asset.data.projected_gravity_b
  down = down / torch.linalg.vector_norm(down, dim=1, keepdim=True).clamp_min(1e-6)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * down, dim=1)
  rate_reward = torch.exp(-torch.square(command[:, 5] - actual_rate) / std**2)
  target = torch.tensor(target_gravity, device=env.device, dtype=down.dtype)
  posture_error = torch.sum(torch.square(down - target), dim=1)
  posture_ready = posture_error < posture_error_limit
  return spin_stand_mask(env, command_name, speed_deadband) * posture_ready.to(rate_reward.dtype) * rate_reward


def spin_dynamic_support_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  sensor_name: str,
  horizontal_gravity_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Softly reward the Thomas-like posture and a two-wheel contact count.

  ``stand + spin_rate`` denotes a dynamic support orbit.  It must not be
  reduced to a permanently reared rear-wheel pose: the valid contact pair can
  change as the policy moves through the orbit.  The two components are added,
  rather than multiplied, so PPO can improve either one from the normal reset.
  """
  asset: Entity = env.scene[asset_cfg.name]
  contacts = _wheel_contacts(env, sensor_name)
  contact_count = torch.sum(contacts.float(), dim=1)
  contact_score = torch.exp(-torch.square(contact_count - 2.0) / 1.0**2)
  gravity_z = asset.data.projected_gravity_b[:, 2]
  horizontal = torch.exp(-torch.square(gravity_z) / horizontal_gravity_std**2)
  return spin_stand_mask(env, command_name, speed_deadband) * 0.5 * (
    contact_score + horizontal
  )


def spin_dynamic_horizontal_precision_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Sharpen the horizontal final state of a dynamic two-wheel orbit.

  The companion support reward deliberately has a broad horizontal score so
  that a policy starting on four wheels can discover a tilt.  Once it finds a
  low two-wheel crouch, that broad score alone admits a local optimum.  This
  term is the final-state counterpart: it asks only for the measured body
  gravity to be horizontal, with no selected horizontal direction, contact
  pair, joint pose, action, or clock.
  """
  if std <= 0.0:
    raise ValueError("std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  gravity = asset.data.projected_gravity_b
  gravity = gravity / torch.linalg.vector_norm(
    gravity, dim=1, keepdim=True
  ).clamp_min(1.0e-6)
  score = torch.exp(-torch.square(gravity[:, 2]) / std**2)
  return spin_stand_mask(env, command_name, speed_deadband) * score


def spin_dynamic_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  std: float,
  horizontal_gravity_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track dynamic-spin rate only once the body has tipped horizontal.

  A normal four-wheel robot can cheaply spin about world up while remaining
  upright.  That is not the commanded Thomas-like dynamic support, so letting
  it collect rate reward before the body is horizontal creates a persistent
  upright-yaw local optimum.  The horizontal factor is a smooth outcome-space
  gate; it supplies no joint target or prescribed route into the orbit.
  """
  if horizontal_gravity_std <= 0.0:
    raise ValueError("horizontal_gravity_std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  down = asset.data.projected_gravity_b
  down = down / torch.linalg.vector_norm(down, dim=1, keepdim=True).clamp_min(1e-6)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * down, dim=1)
  rate_reward = torch.exp(-torch.square(command[:, 5] - actual_rate) / std**2)
  horizontal = torch.exp(-torch.square(down[:, 2]) / horizontal_gravity_std**2)
  return spin_stand_mask(env, command_name, speed_deadband) * horizontal * rate_reward


def fixed_pair_spin_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  std: float,
  gravity_targets: tuple[tuple[float, float, float], ...],
  gravity_power: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track a fixed-pair spin after its requested two-wheel attitude is found."""
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  down = asset.data.projected_gravity_b
  down = down / torch.linalg.vector_norm(down, dim=1, keepdim=True).clamp_min(1e-6)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * down, dim=1)
  rate_reward = torch.exp(-torch.square(command[:, 5] - actual_rate) / std**2)
  mode = torch.argmax(command[:, :len(gravity_targets)], dim=1)
  targets = torch.tensor(gravity_targets, dtype=down.dtype, device=env.device)
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(down * targets[mode], dim=1)), 0.0, 1.0
  )
  return (
    fixed_pair_spin_mask(env, command_name, speed_deadband)
    * alignment.pow(gravity_power)
    * rate_reward
  )


def commanded_spin_rate_abs_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  gravity_targets: tuple[tuple[float, float, float], ...],
  dynamic_horizontal_gravity_std: float,
  fixed_gravity_power: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Non-saturating rate error for normal and front/rear spin commands.

  The Gaussian spin score is useful for precise control but almost flat when
  an untrained policy misses a fast requested rate.  This companion ranks
  smaller measured error above larger error without exposing an axis: the
  world-down direction is read from the robot's own gravity vector.
  """
  if speed_deadband < 0.0 or dynamic_horizontal_gravity_std <= 0.0 or fixed_gravity_power < 0.0:
    raise ValueError("invalid spin-rate posture-gate parameters.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :5], dim=1)
  moving = (mode <= 2) & (torch.abs(command[:, 5]) > speed_deadband)
  down = asset.data.projected_gravity_b
  down = down / torch.linalg.vector_norm(down, dim=1, keepdim=True).clamp_min(1.0e-6)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * down, dim=1)
  horizontal = torch.exp(
    -torch.square(down[:, 2]) / dynamic_horizontal_gravity_std**2
  )
  targets = torch.tensor(gravity_targets, dtype=down.dtype, device=env.device)
  fixed_alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(down * targets[mode], dim=1)), 0.0, 1.0
  )
  posture_gate = torch.where(
    mode == 0, horizontal, fixed_alignment.pow(fixed_gravity_power)
  )
  return moving.to(actual_rate.dtype) * posture_gate * torch.abs(command[:, 5] - actual_rate)


class FixedPairSupportCenterStillness:
  """Keep the active front/rear support axle at a local turn centre.

  In the reference contact pivot the trunk and free legs move substantially,
  so root velocity is the wrong proxy.  The midpoint of the two actual
  supporting wheels is the meaningful signal: it should remain local while
  the robot turns about gravity.  This is an outcome constraint, not a joint
  target or a reference trajectory.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.previous_center = torch.zeros(env.num_envs, 2, device=env.device)
    self.initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.previous_center[env_ids] = 0.0
    self.initialized[env_ids] = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    speed_deadband: float,
    sensor_name: str,
    gravity_targets: tuple[tuple[float, float, float], ...],
    speed_std: float,
    gravity_power: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    if speed_std <= 0.0 or gravity_power < 0.0:
      raise ValueError("Invalid fixed-pair support-centre parameters.")
    asset: Entity = env.scene[asset_cfg.name]
    if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
      raise ValueError("FixedPairSupportCenterStillness needs FL/FR/RL/RR sites.")

    command = _command(env, command_name)
    mode = torch.argmax(command[:, :5], dim=1)
    pair_index = torch.clamp(mode - 1, 0, 1)
    pair_indices = torch.tensor(((0, 1), (2, 3)), device=env.device)
    wheel_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
    batch = torch.arange(env.num_envs, device=env.device)
    center = 0.5 * (
      wheel_pos[batch, pair_indices[pair_index, 0]]
      + wheel_pos[batch, pair_indices[pair_index, 1]]
    )
    initialized = self.initialized.clone()
    center_speed = torch.linalg.vector_norm(
      (center - self.previous_center) / env.step_dt, dim=1
    )
    self.previous_center.copy_(center)
    self.initialized[:] = True

    active = fixed_pair_spin_mask(env, command_name, speed_deadband)
    contacts = _wheel_contacts(env, sensor_name).float()
    contact_masks = torch.tensor(
      ((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)),
      dtype=contacts.dtype,
      device=env.device,
    )
    desired_contacts = contact_masks[pair_index]
    contact_score = torch.mean(
      contacts * desired_contacts + (1.0 - contacts) * (1.0 - desired_contacts), dim=1
    )
    gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
    targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    stillness = torch.exp(-torch.square(center_speed) / speed_std**2)
    return (
      active
      * initialized.to(stillness.dtype)
      * contact_score
      * alignment.pow(gravity_power)
      * stillness
    )


def mode_default_joint_pos_excess_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  num_modes: int,
  free_deviation: float,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Keep a selected mode close to the robot's nominal leg configuration.

  This is a final-state envelope around the model's real initial/default leg
  angles, not a motion reference.  The free band preserves wheel locomotion;
  only unusually large normal-mode posture drift loses reward.
  """
  if free_deviation < 0.0 or std <= 0.0:
    raise ValueError("free_deviation must be non-negative and std positive.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.joint_ids, slice):
    raise ValueError("mode_default_joint_pos_excess_exp needs explicit joints.")
  active, _ = _mode_mask(env, command_name, modes, num_modes=num_modes)
  delta = torch.abs(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  excess = torch.clamp_min(delta - free_deviation, 0.0)
  error = torch.mean(torch.square(excess), dim=1)
  return active.to(error.dtype) * torch.exp(-error / std**2)


def spin_planar_speed_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Limit translation of the Thomas-like orbit without forcing zero motion."""
  asset: Entity = env.scene[asset_cfg.name]
  planar_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
  excess = torch.clamp_min(planar_speed - 0.5, 0.0)
  return spin_stand_mask(env, command_name, speed_deadband) * torch.square(excess)


class SpinSupportCycle:
  """One-time reward for a settled two-wheel support-pair transition.

  This encourages the contact sequence needed by a Thomas-like orbit while a
  minimum dwell time prevents reward farming by one-step contact chatter.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.last_pair = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.time_since_change = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.last_pair[env_ids] = -1
    self.time_since_change[env_ids] = 0.0
    self.previous_active[env_ids] = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    speed_deadband: float,
    sensor_name: str,
    horizontal_gravity_limit: float = 0.70,
    min_transition_interval: float = 0.12,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    active = spin_stand_mask(env, command_name, speed_deadband) > 0.5
    contacts = _wheel_contacts(env, sensor_name)
    pair = _two_wheel_pair_index(contacts)
    eligible = active & (pair >= 0) & (
      torch.abs(asset.data.projected_gravity_b[:, 2]) < horizontal_gravity_limit
    )

    reset = (env.episode_length_buf == 0) | (~active) | (~self.previous_active)
    self.last_pair[reset] = -1
    self.time_since_change[reset] = 0.0

    self.time_since_change = torch.where(
      active,
      self.time_since_change + env.step_dt,
      torch.zeros_like(self.time_since_change),
    )
    changed = (
      eligible
      & (self.last_pair >= 0)
      & (pair != self.last_pair)
      & (self.time_since_change >= min_transition_interval)
    )
    self.last_pair = torch.where(eligible, pair, self.last_pair)
    self.time_since_change[changed] = 0.0
    self.previous_active = active
    # RewardManager scales every term by dt.  Dividing an instantaneous event
    # here makes each valid transition worth exactly its configured weight.
    return changed.float() / env.step_dt


def _stance_locomotion_axes(
  asset: Entity, mode: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return a consistent ground-forward axis for normal/front/rear poses."""
  quat = asset.data.root_link_quat_w
  body_x = quat_apply(
    quat,
    torch.tensor((1.0, 0.0, 0.0), dtype=quat.dtype, device=quat.device).expand(
      quat.shape[0], -1
    ),
  )[:, :2]
  body_z = quat_apply(
    quat,
    torch.tensor((0.0, 0.0, 1.0), dtype=quat.dtype, device=quat.device).expand(
      quat.shape[0], -1
    ),
  )[:, :2]
  forward = torch.where(
    (mode == 0).unsqueeze(1),
    body_x,
    torch.where((mode == 1).unsqueeze(1), body_z, -body_z),
  )
  norm = torch.linalg.vector_norm(forward, dim=1, keepdim=True)
  fallback = torch.tensor((1.0, 0.0), dtype=forward.dtype, device=forward.device).expand_as(
    forward
  )
  forward = torch.where(norm > 1.0e-4, forward / norm.clamp_min(1.0e-4), fallback)
  right = torch.stack((-forward[:, 1], forward[:, 0]), dim=1)
  return forward, right


def stance_locomotion_linear_velocity_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  lateral_weight: float = 2.0,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track ground-forward x while explicitly holding lateral velocity at zero.

  An optional target-attitude gate prevents a stationary four-wheel reset from
  satisfying a zero-speed handstand command before it has actually stood up.
  """
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :3], dim=1)
  forward, right = _stance_locomotion_axes(asset, mode)
  velocity_xy = asset.data.root_link_lin_vel_w[:, :2]
  actual_x = torch.sum(velocity_xy * forward, dim=1)
  actual_y = torch.sum(velocity_xy * right, dim=1)
  error = torch.square(command[:, 3] - actual_x) + lateral_weight * torch.square(
    actual_y
  )
  score = torch.exp(-error / std**2)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(
      gravity_targets, dtype=score.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  return score


def stance_locomotion_yaw_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track conventional world-up yaw in all three stance modes."""
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  error = command[:, 4] - asset.data.root_link_ang_vel_w[:, 2]
  score = torch.exp(-torch.square(error) / std**2)
  if gravity_targets is not None and gravity_power > 0.0:
    mode = torch.argmax(command[:, :len(gravity_targets)], dim=1)
    targets = torch.tensor(
      gravity_targets, dtype=score.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  return score


def stance_locomotion_linear_velocity_abs_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  lateral_weight: float = 2.0,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return a dense ground-plane velocity error for the fused stance task.

  The exponential tracking score is intentionally sharp near a correctly
  controlled velocity, but becomes effectively flat after a two-wheel stance
  has acquired a large unintended roll.  This companion term is used with a
  negative reward weight, so it still supplies a learning signal in that
  regime.  It specifies only the requested root velocity and zero lateral
  drift; it does not encode a leg pose, wheel action, or reference trajectory.
  """
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :3], dim=1)
  forward, right = _stance_locomotion_axes(asset, mode)
  velocity_xy = asset.data.root_link_lin_vel_w[:, :2]
  actual_x = torch.sum(velocity_xy * forward, dim=1)
  actual_y = torch.sum(velocity_xy * right, dim=1)
  error = torch.abs(command[:, 3] - actual_x) + lateral_weight * torch.abs(actual_y)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(
      gravity_targets, dtype=error.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    error = error * alignment.pow(gravity_power)
  return error


def stance_stationary_ground_speed_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  velocity_deadband: float,
  std: float,
  lateral_weight: float = 2.0,
  num_modes: int = 3,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a true zero-command stop in each requested stance.

  The generic velocity tracker also covers zero commands, but two-wheel
  contact and attitude rewards can otherwise make a self-propelled stable
  wheel stand locally attractive.  This term is active *only* when both
  external speed commands are zero, and measures only ground-plane root
  velocity.  It contains neither a joint target nor an action target.
  """
  if velocity_deadband < 0.0 or std <= 0.0 or lateral_weight < 0.0:
    raise ValueError("velocity_deadband and lateral_weight must be non-negative; std positive.")
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  stationary = torch.linalg.vector_norm(command[:, num_modes:], dim=1) <= velocity_deadband
  forward, right = _stance_locomotion_axes(asset, mode)
  velocity_xy = asset.data.root_link_lin_vel_w[:, :2]
  forward_speed = torch.sum(velocity_xy * forward, dim=1)
  lateral_speed = torch.sum(velocity_xy * right, dim=1)
  score = torch.exp(
    -(torch.square(forward_speed) + lateral_weight * torch.square(lateral_speed))
    / std**2
  )
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(gravity_targets, dtype=score.dtype, device=env.device)
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  return active.to(score.dtype) * stationary.to(score.dtype) * score


def stance_stationary_ground_speed_abs_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  velocity_deadband: float,
  lateral_weight: float = 2.0,
  num_modes: int = 3,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  minimum_gravity_alignment: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense zero-command ground-speed error for a stance locomotion policy.

  This is the non-saturating companion to
  :func:`stance_stationary_ground_speed_exp`: a self-propelled handstand is
  penalized in proportion to its speed even when it is far from rest.  It is
  strictly inactive for non-zero x/yaw requests and before final-attitude
  alignment, so it does not prescribe the stand-up trajectory.
  """
  if velocity_deadband < 0.0 or lateral_weight < 0.0:
    raise ValueError("velocity_deadband and lateral_weight must be non-negative.")
  if gravity_power < 0.0 or not 0.0 <= minimum_gravity_alignment < 1.0:
    raise ValueError("gravity_power must be non-negative; minimum_gravity_alignment in [0, 1).")
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  stationary = torch.linalg.vector_norm(command[:, num_modes:], dim=1) <= velocity_deadband
  forward, right = _stance_locomotion_axes(asset, mode)
  velocity_xy = asset.data.root_link_lin_vel_w[:, :2]
  forward_speed = torch.sum(velocity_xy * forward, dim=1)
  lateral_speed = torch.sum(velocity_xy * right, dim=1)
  error = torch.abs(forward_speed) + lateral_weight * torch.abs(lateral_speed)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(gravity_targets, dtype=error.dtype, device=env.device)
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    if minimum_gravity_alignment > 0.0:
      alignment = alignment * (alignment >= minimum_gravity_alignment).to(alignment.dtype)
    error = error * alignment.pow(gravity_power)
  return active.to(error.dtype) * stationary.to(error.dtype) * error


def stance_stationary_action_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  velocity_deadband: float,
  num_modes: int = 3,
) -> torch.Tensor:
  """Measure control effort only for a genuinely stationary stance request.

  This has no position target: it merely lets a normal four-wheel idle state
  prefer the controller's zero residual instead of repeatedly issuing a
  constant non-zero action that lifts a wheel from an already valid reset.
  Moving x/yaw requests and the two-wheel rise remain entirely unaffected.
  """
  if velocity_deadband < 0.0:
    raise ValueError("velocity_deadband must be non-negative.")
  active, _ = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  stationary = (
    torch.linalg.vector_norm(command[:, num_modes:], dim=1) <= velocity_deadband
  )
  effort = torch.sum(torch.square(env.action_manager.action), dim=1)
  return active.to(effort.dtype) * stationary.to(effort.dtype) * effort


def mode_stationary_root_ang_speed(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  velocity_deadband: float,
  num_modes: int = 3,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  minimum_gravity_alignment: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Dense angular-speed error for a zero x/yaw stance request."""
  if (
    velocity_deadband < 0.0
    or gravity_power < 0.0
    or not 0.0 <= minimum_gravity_alignment < 1.0
  ):
    raise ValueError("velocity_deadband/gravity_power must be non-negative; minimum_gravity_alignment in [0, 1).")
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  stationary = torch.linalg.vector_norm(command[:, num_modes:], dim=1) <= velocity_deadband
  error = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(gravity_targets, dtype=error.dtype, device=env.device)
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    if minimum_gravity_alignment > 0.0:
      alignment = alignment * (alignment >= minimum_gravity_alignment).to(alignment.dtype)
    error = error * alignment.pow(gravity_power)
  return active.to(error.dtype) * stationary.to(error.dtype) * error


def stance_locomotion_yaw_rate_abs_error(
  env: "ManagerBasedRlEnv",
  command_name: str,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Return dense conventional-yaw error after the requested stance is upright."""
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  error = torch.abs(command[:, 4] - asset.data.root_link_ang_vel_w[:, 2])
  if gravity_targets is not None and gravity_power > 0.0:
    mode = torch.argmax(command[:, :len(gravity_targets)], dim=1)
    targets = torch.tensor(
      gravity_targets, dtype=error.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    error = error * alignment.pow(gravity_power)
  return error


def mode_root_height_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  height_targets: tuple[float, ...],
  std: float,
  num_modes: int,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a broad mode-specific root height without prescribing a trajectory."""
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :num_modes], dim=1)
  targets = torch.tensor(
    height_targets, dtype=asset.data.root_link_pos_w.dtype, device=env.device
  )
  score = torch.exp(
    -torch.square(asset.data.root_link_pos_w[:, 2] - targets[mode]) / std**2
  )
  if gravity_targets is not None and gravity_power > 0.0:
    gravity_targets_tensor = torch.tensor(
      gravity_targets, dtype=score.dtype, device=env.device
    )
    gravity = asset.data.projected_gravity_b
    gravity = gravity / torch.linalg.vector_norm(
      gravity, dim=1, keepdim=True
    ).clamp_min(1.0e-6)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * gravity_targets_tensor[mode], dim=1)), 0.0, 1.0
    )
    score = score * alignment.pow(gravity_power)
  return score


def aerial_active(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  return (torch.sum(_command(env, command_name)[:, :5], dim=1) > 0.5).float()


def aerial_idle_four_wheel_stand_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  gravity_std: float,
  linear_velocity_std: float,
  angular_velocity_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Score the all-zero aerial command only at a quiet normal four-wheel stand.

  The aerial one-hot is an event.  Its all-zero encoding is therefore a real
  public command, not an unspecified gap after an attempt.  This term makes
  that gap recover to the exact ordinary support state without contributing to
  any active flip or prescribing a trajectory for one.
  """
  if min(gravity_std, linear_velocity_std, angular_velocity_std) <= 0.0:
    raise ValueError("idle stand scales must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  idle = aerial_active(env, command_name) <= 0.5
  target = torch.tensor(
    (0.0, 0.0, -1.0), dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target), dim=1
  )
  linear_speed_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=1)
  angular_speed_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_w), dim=1)
  wheel_contact = _wheel_contacts(env, sensor_name).float().mean(dim=1)
  return idle.to(wheel_contact.dtype) * wheel_contact * torch.exp(
    -gravity_error / gravity_std**2
    -linear_speed_sq / linear_velocity_std**2
    -angular_speed_sq / angular_velocity_std**2
  )


def aerial_idle_default_joint_pos_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Return an untriggered aerial policy to the physical default leg pose."""
  if std <= 0.0:
    raise ValueError("std must be positive.")
  if isinstance(asset_cfg.joint_ids, slice):
    raise ValueError("aerial_idle_default_joint_pos_exp needs explicit joints.")
  asset: Entity = env.scene[asset_cfg.name]
  delta = (
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  error = torch.mean(torch.square(delta), dim=1)
  idle = aerial_active(env, command_name) <= 0.5
  return idle.to(error.dtype) * torch.exp(-error / std**2)


def _advance_qualified_aerial_rotation(
  *,
  env: "ManagerBasedRlEnv",
  active: torch.Tensor,
  contacts: torch.Tensor,
  axis_rate: torch.Tensor,
  has_grounded: torch.Tensor,
  airborne_time: torch.Tensor,
  current_flight_qualified: torch.Tensor,
  flight_rotation: torch.Tensor,
  min_ballistic_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Measure a whole continuous ballistic interval without crediting glitches.

  A short wheel-contact dropout must not be allowed to masquerade as an aerial
  turn.  Conversely, once an interval has survived the physical duration
  check, its initial measured angular displacement belongs to the same real
  flight and must not be discarded.  This helper returns the once-committed
  signed increment together with the updated per-flight state.
  """
  if min_ballistic_time <= 0.0:
    raise ValueError("min_ballistic_time must be positive.")
  airborne = ~torch.any(contacts, dim=1)
  has_grounded = has_grounded | (active & torch.all(contacts, dim=1))
  flight_step = active & has_grounded & airborne
  airborne_time = torch.where(
    flight_step,
    airborne_time + env.step_dt,
    torch.zeros_like(airborne_time),
  )
  raw_delta = flight_step.to(axis_rate.dtype) * axis_rate * env.step_dt
  flight_rotation = torch.where(
    flight_step,
    flight_rotation + raw_delta,
    torch.zeros_like(flight_rotation),
  )
  newly_qualified = (
    flight_step
    & (~current_flight_qualified)
    & (airborne_time >= min_ballistic_time)
  )
  current_flight_qualified = torch.where(
    flight_step,
    current_flight_qualified | (airborne_time >= min_ballistic_time),
    torch.zeros_like(current_flight_qualified),
  )
  rotation_increment = torch.where(
    newly_qualified,
    flight_rotation,
    torch.where(
      flight_step & current_flight_qualified,
      raw_delta,
      torch.zeros_like(axis_rate),
    ),
  )
  return (
    has_grounded,
    airborne_time,
    current_flight_qualified,
    flight_rotation,
    rotation_increment,
  )


def aerial_base_height_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  target_height: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  error = torch.square(asset.data.root_link_pos_w[:, 2] - target_height)
  return aerial_active(env, command_name) * torch.exp(-error / std**2)


def aerial_base_clearance(
  env: "ManagerBasedRlEnv",
  command_name: str,
  min_clearance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward vertical clearance above the model's nominal root height.

  This is scale-aware for the actual MuJoCo model, unlike a fixed world-z
  target copied from another robot or another reset pose.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  clearance = asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]
  return aerial_active(env, command_name) * torch.clamp(
    clearance / min_clearance, min=0.0, max=1.0
  )


class AerialClearanceProgress:
  """Reward new takeoff clearance once, never continued time in the air.

  A dense height reward is harmful for a one-shot aerial skill: after a jump,
  it pays the policy every control step for postponing the landing.  This term
  instead pays only each increase in the best normalized root clearance, so
  its full episode return is bounded by one regardless of hang time.  It does
  not encode a joint pose, a flight phase, or a demonstration trajectory.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.best_clearance = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.best_clearance[env_ids] = 0.0
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    min_clearance: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if min_clearance <= 0.0:
      raise ValueError("min_clearance must be positive.")
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & ((~self.previous_active) | (mode != self.previous_mode) | reset)
    clear = new_skill | reset | (~active)
    self.best_clearance[clear] = 0.0

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    # Height alone can be manufactured by pivoting around one or two wheels.
    # Credit clearance only while *all* wheels are clear of terrain.  This is
    # the earliest dense signal for a jump, without choosing how it launches.
    airborne = ~torch.any(_wheel_contacts(env, sensor_name), dim=1)
    normalized_clearance = torch.clamp(
      (asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]) / min_clearance,
      min=0.0,
      max=1.0,
    )
    old_best = self.best_clearance.clone()
    self.best_clearance = torch.maximum(
      self.best_clearance, normalized_clearance * active * airborne
    )
    progress = self.best_clearance - old_best
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    # Preserve a one-off configured return under RewardManager's dt scaling.
    return progress / env.step_dt


class AerialTakeoffVerticalSpeed:
  """Reward a single compact ballistic launch, never continued airtime.

  The clearance-progress term measures the eventual apex.  This companion
  term pays only at the first loss of all wheel contacts, using the measured
  world-frame upward base velocity at that instant.  It therefore favors the
  physical quantity needed for a full aerial turn—launch impulse—without
  specifying a joint posture, a phase clock, or a reference trajectory.
  Dividing by ``step_dt`` makes the configured reward a one-off return under
  RewardManager's normal time integration.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.has_grounded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.was_airborne[env_ids] = False
    self.has_grounded[env_ids] = False
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    target_speed: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if target_speed <= 0.0:
      raise ValueError("target_speed must be positive.")
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & ((~self.previous_active) | (mode != self.previous_mode) | reset)
    clear = new_skill | reset | (~active)
    self.was_airborne[clear] = False
    self.has_grounded[clear] = False

    contacts = _wheel_contacts(env, sensor_name)
    airborne = ~torch.any(contacts, dim=1)
    self.has_grounded |= active & torch.all(contacts, dim=1)
    liftoff = active & self.has_grounded & airborne & (~self.was_airborne)
    vertical_speed = torch.clamp(asset.data.root_link_lin_vel_w[:, 2], min=0.0)
    reward = liftoff.float() * torch.clamp(vertical_speed / target_speed, max=1.0)

    self.was_airborne = torch.where(
      active,
      self.was_airborne | (self.has_grounded & airborne),
      torch.zeros_like(airborne),
    )
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    return reward / env.step_dt


def aerial_airborne(
  env: "ManagerBasedRlEnv", command_name: str, sensor_name: str
) -> torch.Tensor:
  contacts = _wheel_contacts(env, sensor_name)
  return aerial_active(env, command_name) * (~torch.any(contacts, dim=1)).float()


def aerial_airborne_joint_excursion_l2(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  free_deviation: float,
  asset_cfg: SceneEntityCfg,
  airborne_only: bool = True,
) -> torch.Tensor:
  """Penalize unnecessary leg opening in an active aerial attempt.

  This is a mechanical compactness regularizer, not a reference motion.  It
  has no phase, mode-dependent pose, or desired trajectory.  Every joint can
  move freely inside ``free_deviation`` from the nominal wheel-standing
  geometry.  When ``airborne_only`` is false, the same excess cost is also
  charged during push-off and recovery.  That prevents a policy from using a
  large ground-side swing to create angular momentum and then becoming compact
  only after liftoff.
  """
  if free_deviation < 0.0:
    raise ValueError("free_deviation must be non-negative.")
  if isinstance(asset_cfg.joint_ids, slice):
    raise ValueError("aerial_airborne_joint_excursion_l2 requires explicit joints.")
  asset: Entity = env.scene[asset_cfg.name]
  deviation = torch.abs(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  excess = torch.relu(deviation - free_deviation)
  airborne = ~torch.any(_wheel_contacts(env, sensor_name), dim=1)
  phase_gate = airborne.float() if airborne_only else 1.0
  return (
    aerial_active(env, command_name)
    * phase_gate
    * torch.sum(torch.square(excess), dim=1)
  )


def aerial_leg_excursion_exceeded(
  env: "ManagerBasedRlEnv",
  command_name: str,
  max_deviation: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Fail an active attempt whose physical leg excursion is no longer compact.

  Position-action clipping limits requested targets but cannot constrain an
  unpowered leg once a bad rollout hits the ground.  This validity condition
  operates on measured joint state instead, so a policy cannot trade a large
  collision-driven swing for rotation reward.  It does not select a pose,
  a phase, or a direction-specific trajectory.
  """
  if max_deviation <= 0.0:
    raise ValueError("max_deviation must be positive.")
  if isinstance(asset_cfg.joint_ids, slice):
    raise ValueError("aerial_leg_excursion_exceeded requires explicit joints.")
  asset: Entity = env.scene[asset_cfg.name]
  deviation = torch.abs(
    asset.data.joint_pos[:, asset_cfg.joint_ids]
    - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  )
  return aerial_active(env, command_name).bool() & torch.any(
    deviation > max_deviation, dim=1
  )


def aerial_rotation_overrun(
  env: "ManagerBasedRlEnv",
  command_name: str,
  target_angle: float,
  max_overrotation: float,
  activation_step: int = 0,
) -> torch.Tensor:
  """End an attempt that has spun irrecoverably past its one-turn target.

  The command term integrates *measured* rotation around the launch axis.  A
  continued multi-turn spin is not a valid substitute for a one-turn flip
  followed by a wheel landing, so it is a task failure instead of an
  indefinitely timed-out rollout.  This selects no joint pose, phase, or
  reference trajectory.
  """
  if target_angle <= 0.0 or max_overrotation <= 0.0 or activation_step < 0:
    raise ValueError("turn limits must be positive and activation_step non-negative.")
  # Full-turn discovery needs broad exploration before a precise overrun
  # failure becomes useful.  This is a training curriculum measured in global
  # environment steps; it never appears in the actor observation.
  if env.common_step_counter < activation_step:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  command_term = env.command_manager.get_term(command_name)
  progress = getattr(
    command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
  )
  return aerial_active(env, command_name).bool() & (
    progress > target_angle + max_overrotation
  )


def aerial_axis_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  axes: tuple[tuple[float, float, float], ...],
  target_rate: float,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the internally selected aerial rotation direction, not a command rate."""
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  active = aerial_active(env, command_name)
  mode = torch.argmax(command[:, :5], dim=1)
  axes_tensor = torch.tensor(axes, dtype=asset.data.root_link_ang_vel_b.dtype, device=env.device)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * axes_tensor[mode], dim=1)
  return active * torch.exp(-torch.square(actual_rate - target_rate) / std**2)


def aerial_positive_axis_rate(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  axes: tuple[tuple[float, float, float], ...],
  rate_clip: float,
  clearance_start: float | None = None,
  clearance_full: float | None = None,
  stop_angle: float | None = None,
  stop_angle_fade: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward directed aerial rotation without prescribing one exact rate.

  ``stop_angle`` lets the takeoff-drive term release authority before the
  landing phase.  It is measured accumulated body rotation, not a desired
  pose or a command supplied to the actor.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :5], dim=1)
  axes_tensor = torch.tensor(axes, dtype=asset.data.root_link_ang_vel_b.dtype, device=env.device)
  axis_rate = torch.sum(asset.data.root_link_ang_vel_b * axes_tensor[mode], dim=1)
  airborne = ~torch.any(_wheel_contacts(env, sensor_name), dim=1)
  result = aerial_active(env, command_name) * airborne.float() * torch.clamp(
    axis_rate / rate_clip, min=0.0, max=1.0
  )
  if (clearance_start is None) != (clearance_full is None):
    raise ValueError("clearance_start and clearance_full must be specified together.")
  if clearance_start is None:
    command_term = env.command_manager.get_term(command_name)
    clearance_start = command_term.cfg.rotation_rate_clearance_start
    clearance_full = command_term.cfg.rotation_rate_clearance_full
  assert clearance_full is not None
  if clearance_start < 0.0 or clearance_full <= clearance_start:
    raise ValueError("Rotation-rate clearance limits must be an ordered range.")
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  clearance = asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]
  clearance_gate = torch.clamp(
    (clearance - clearance_start) / (clearance_full - clearance_start),
    min=0.0,
    max=1.0,
  )
  result = result * clearance_gate
  if stop_angle is not None:
    if stop_angle <= 0.0:
      raise ValueError("stop_angle must be positive when provided.")
    command_term = env.command_manager.get_term(command_name)
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    if stop_angle_fade < 0.0:
      raise ValueError("stop_angle_fade must be non-negative.")
    if stop_angle_fade == 0.0:
      drive_gate = (progress < stop_angle).float()
    else:
      # ``stop_angle`` is the point at which momentum drive begins to hand
      # over to recovery—not the end of its fade.  Keeping the full rate
      # signal through this point and fading across the *following* interval
      # lets a compact front/side flip cross its one-turn barrier before the
      # braking terms take over.  The prior expression faded over the
      # preceding interval, so a nominal 0.70-turn hand-off had already
      # reduced drive at 0.55 turn and made those modes stall below one turn.
      drive_gate = torch.clamp(
        (stop_angle + stop_angle_fade - progress) / stop_angle_fade,
        min=0.0,
        max=1.0,
      )
    result = result * drive_gate
  return result


def aerial_landing_gravity_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  std: float,
  linear_velocity_std: float = 0.75,
  angular_velocity_std: float = 1.5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a calm normal attitude only while idle or after landing."""
  asset: Entity = env.scene[asset_cfg.name]
  active = aerial_active(env, command_name) > 0.5
  command_term = env.command_manager.get_term(command_name)
  was_airborne = getattr(command_term, "was_airborne", torch.zeros_like(active))
  landed = torch.all(_wheel_contacts(env, sensor_name), dim=1)
  gate = (~active) | (was_airborne & landed)
  target = torch.tensor(
    (0.0, 0.0, -1.0),
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  error = torch.sum(torch.square(asset.data.projected_gravity_b - target), dim=1)
  linear_speed_sq = torch.sum(torch.square(asset.data.root_link_lin_vel_w), dim=1)
  angular_speed_sq = torch.sum(torch.square(asset.data.root_link_ang_vel_w), dim=1)
  return gate.float() * torch.exp(
    -error / std**2
    - linear_speed_sq / linear_velocity_std**2
    - angular_speed_sq / angular_velocity_std**2
  )


def aerial_late_phase_recovery_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  target_angle: float,
  activation_angle: float,
  gravity_std: float,
  target_axis_rate: float,
  axis_rate_clip: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a physically recoverable final part of an airborne turn.

  The phase comes from the *measured* accumulated angle around the axis fixed
  at launch.  It is not a reference trajectory: until roughly the last fifth
  of a completed turn this term is exactly zero.  Once there, it encourages
  the robot to return upright and bleed angular speed before wheel contact.
  The braking component is deliberately linear rather than a narrow Gaussian:
  at the 20--30 rad/s rates produced by early exploration, a Gaussian provides
  numerically zero learning signal, whereas this still ranks a smaller actual
  angular rate above a larger one.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  active = aerial_active(env, command_name) > 0.5
  contacts = _wheel_contacts(env, sensor_name)
  airborne = ~torch.any(contacts, dim=1)
  progress = getattr(command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device))
  launch_axis_w = getattr(
    command_term,
    "_launch_axis_w",
    torch.zeros(env.num_envs, 3, device=env.device),
  )
  phase = torch.clamp(
    (progress - activation_angle) / (target_angle - activation_angle), min=0.0, max=1.0
  )
  normal_gravity = torch.tensor(
    (0.0, 0.0, -1.0),
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
  )
  axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
  if axis_rate_clip <= 0.0:
    raise ValueError("axis_rate_clip must be positive.")
  rate_score = torch.clamp(
    1.0 - torch.abs(axis_rate - target_axis_rate) / axis_rate_clip,
    min=0.0,
    max=1.0,
  )
  recovery = torch.exp(-gravity_error / gravity_std**2) * rate_score
  return active.float() * airborne.float() * phase * recovery


def aerial_soft_landing_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  target_angle: float,
  angle_std: float,
  gravity_std: float,
  axis_rate_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Score an actual near-target, four-wheel touchdown before hard success.

  This is deliberately an outcome reward, rather than a pose reference.  It
  fires only after the robot has really been airborne and then has all four
  wheels on the ground.  A soft score supplies the missing gradient between a
  crash and the strict zero-speed, settled hard-completion condition.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  active = aerial_active(env, command_name) > 0.5
  was_airborne = getattr(command_term, "was_airborne", torch.zeros_like(active))
  contacts = _wheel_contacts(env, sensor_name)
  four_wheel_landing = torch.all(contacts, dim=1)
  progress = getattr(command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device))
  launch_axis_w = getattr(
    command_term,
    "_launch_axis_w",
    torch.zeros(env.num_envs, 3, device=env.device),
  )
  angle_error = torch.square(progress - target_angle)
  normal_gravity = torch.tensor(
    (0.0, 0.0, -1.0),
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
  )
  axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
  return (
    active.float()
    * was_airborne.float()
    * four_wheel_landing.float()
    * torch.exp(
      -angle_error / angle_std**2
      -gravity_error / gravity_std**2
      -torch.square(axis_rate) / axis_rate_std**2
    )
  )


def aerial_post_turn_wheel_touchdown_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  target_angle: float,
  max_overrotation: float,
  gravity_std: float,
  axis_rate_std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward legal *partial* wheel touchdown after a braked full turn.

  A four-wheel-only landing score is necessarily discontinuous: a rollout
  whose first one or two wheels arrive correctly receives the same zero as a
  body crash.  This term ranks that physically meaningful transition without
  prescribing a leg pose, contact order, or time-indexed reference.  The
  existing strict command completion remains the sole success condition.
  """
  if (
    target_angle <= 0.0
    or max_overrotation <= 0.0
    or gravity_std <= 0.0
    or axis_rate_std <= 0.0
  ):
    raise ValueError("turn, tolerance, and standard-deviation parameters must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  active = aerial_active(env, command_name) > 0.5
  was_airborne = getattr(command_term, "was_airborne", torch.zeros_like(active))
  progress = getattr(command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device))
  launch_axis_w = getattr(
    command_term,
    "_launch_axis_w",
    torch.zeros(env.num_envs, 3, device=env.device),
  )
  normal_gravity = torch.tensor(
    (0.0, 0.0, -1.0),
    dtype=asset.data.projected_gravity_b.dtype,
    device=env.device,
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
  )
  axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
  turn_window = (
    (progress >= target_angle) & (progress <= target_angle + max_overrotation)
  ).float()
  wheel_fraction = _wheel_contacts(env, sensor_name).float().mean(dim=1)
  return (
    active.float()
    * was_airborne.float()
    * turn_window
    * wheel_fraction
    * torch.exp(
      -gravity_error / gravity_std**2
      -torch.square(axis_rate) / axis_rate_std**2
    )
  )


def aerial_post_turn_descent(
  env: "ManagerBasedRlEnv",
  command_name: str,
  sensor_name: str,
  target_angle: float,
  max_overrotation: float,
  gravity_std: float,
  axis_rate_std: float,
  descent_speed: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a braked, upright descent only after a measured full turn.

  This bridges airborne recovery and the existing four-wheel-only landing
  terms.  It is a result condition (full measured rotation, upright attitude,
  low spin, and downward base velocity), never a joint target or trajectory.
  """
  if target_angle <= 0.0 or max_overrotation <= 0.0 or descent_speed <= 0.0:
    raise ValueError("turn and descent limits must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  active = aerial_active(env, command_name) > 0.5
  airborne = ~torch.any(_wheel_contacts(env, sensor_name), dim=1)
  progress = getattr(command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device))
  launch_axis_w = getattr(command_term, "_launch_axis_w", torch.zeros(env.num_envs, 3, device=env.device))
  normal_gravity = torch.tensor((0.0, 0.0, -1.0), dtype=asset.data.projected_gravity_b.dtype, device=env.device)
  gravity_error = torch.sum(torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1)
  axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
  turn_gate = ((progress >= target_angle) & (progress <= target_angle + max_overrotation)).float()
  upright_and_braked = torch.exp(
    -gravity_error / gravity_std**2 - torch.square(axis_rate) / axis_rate_std**2
  )
  downward = torch.clamp(-asset.data.root_link_lin_vel_w[:, 2] / descent_speed, min=0.0, max=1.0)
  return active.float() * airborne.float() * turn_gate * upright_and_braked * downward


def aerial_late_axis_rate_abs(
  env: "ManagerBasedRlEnv",
  command_name: str,
  start_angle: float,
  max_angle: float,
  sensor_name: str | None = None,
  require_wheel_contact: bool = False,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Measure residual spin after a near-complete aerial attempt.

  The directed-progress term should be free to discover whatever launch rate
  completes the first 95 percent of a turn.  At that point, however, continued
  angular momentum is precisely what prevents a wheel-first recovery.  This
  term is intentionally expressed in the command term's measured accumulated
  angle and measured root angular velocity; it exposes neither quantity to the
  actor and contains no timed reference or joint target.  A caller can
  additionally require a physical wheel contact, which makes the cost a
  landing-quality signal rather than an in-flight braking prescription.
  """
  if start_angle <= 0.0 or max_angle <= start_angle:
    raise ValueError("late-axis-rate angles must be ordered positive values.")
  if require_wheel_contact and sensor_name is None:
    raise ValueError("sensor_name is required when wheel contact is required.")
  asset: Entity = env.scene[asset_cfg.name]
  command_term = env.command_manager.get_term(command_name)
  active = aerial_active(env, command_name) > 0.5
  qualified_flight = getattr(
    command_term, "was_airborne", torch.zeros_like(active)
  )
  progress = getattr(
    command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
  )
  launch_axis_w = getattr(
    command_term, "_launch_axis_w", torch.zeros(env.num_envs, 3, device=env.device)
  )
  axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
  late_turn = (progress >= start_angle) & (progress <= max_angle)
  contact_gate = torch.ones_like(axis_rate)
  if require_wheel_contact:
    assert sensor_name is not None
    contact_gate = torch.any(_wheel_contacts(env, sensor_name), dim=1).to(axis_rate.dtype)
  return (
    active.float()
    * qualified_flight.float()
    * late_turn.float()
    * contact_gate
    * torch.abs(axis_rate)
  )


class AerialPostTurnLandingProgress:
  """Reward new physical descent after an upright, measured full turn.

  The old four-wheel touchdown reward is necessarily sparse: it has no value
  until the robot has already found the entire landing.  This potential-like
  progress term records only a *new lower* base clearance after a full turn,
  while the robot is still upright and slowing down.  It therefore bridges
  from airborne recovery to the contact-only landing reward without dictating
  a leg pose, a time-indexed trajectory, or an actor-side phase input.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.lowest_clearance = torch.zeros(env.num_envs, device=env.device)
    self.started = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.lowest_clearance[env_ids] = 0.0
    self.started[env_ids] = False

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    target_angle: float,
    max_overrotation: float,
    gravity_std: float,
    axis_rate_clip: float,
    descent_distance: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if (
      target_angle <= 0.0
      or max_overrotation <= 0.0
      or gravity_std <= 0.0
      or axis_rate_clip <= 0.0
      or descent_distance <= 0.0
    ):
      raise ValueError("post-turn landing-progress parameters must be positive.")
    asset: Entity = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    active = aerial_active(env, command_name) > 0.5
    reset = (env.episode_length_buf == 0) | (~active)
    self.started[reset] = False
    self.lowest_clearance[reset] = 0.0

    contacts = _wheel_contacts(env, sensor_name)
    airborne = ~torch.any(contacts, dim=1)
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    in_landing_window = (
      active
      & airborne
      & (progress >= target_angle)
      & (progress <= target_angle + max_overrotation)
    )
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    clearance = asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]

    first_window_step = in_landing_window & (~self.started)
    continuing = in_landing_window & self.started
    previous_lowest = self.lowest_clearance.clone()
    improvement = torch.where(
      continuing,
      torch.clamp(previous_lowest - clearance, min=0.0),
      torch.zeros_like(clearance),
    )
    self.lowest_clearance[first_window_step] = clearance[first_window_step]
    self.lowest_clearance[continuing] = torch.minimum(
      self.lowest_clearance[continuing], clearance[continuing]
    )
    self.started |= in_landing_window

    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    launch_axis_w = getattr(
      command_term, "_launch_axis_w", torch.zeros(env.num_envs, 3, device=env.device)
    )
    axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
    upright = torch.exp(-gravity_error / gravity_std**2)
    braking = torch.clamp(1.0 - torch.abs(axis_rate) / axis_rate_clip, min=0.0, max=1.0)
    # RewardManager integrates terms over ``step_dt``.  Dividing the physical
    # clearance improvement by dt makes the total return proportional to
    # actual newly achieved descent rather than its simulation discretization.
    return (
      in_landing_window.float()
      * upright
      * braking
      * improvement
      / (descent_distance * env.step_dt)
    )


class AerialLandingRecoveryProgress:
  """Pay only new progress toward a physical, normal-wheel recovery.

  A completed aerial turn has a deliberately broad recovery basin: measured
  angular progress approaches one full turn, the body returns toward normal
  gravity, its launch-axis momentum and whole-body speed reduce, and its root
  descends toward wheel contact.  All quantities are measured state, and the
  return is a bounded potential improvement, so remaining in a favourable
  state cannot be rewarded repeatedly.  This supplies a usable ranking while
  early exploration still has large residual spin; narrow touchdown Gaussians
  are numerically zero in that regime.

  The term contains no pose target, phase clock, contact order, or prescribed
  trajectory.  It simply ranks physically better final states after the
  requested signed rotation has nearly been earned.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.best_score = torch.zeros(env.num_envs, device=env.device)
    # The command itself requires consecutive strict normal-wheel contact to
    # complete.  Track the same measured dwell here so the one recovery
    # potential can rank "kept the landing" above a single lucky touchdown.
    self.landing_dwell_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.best_score[env_ids] = 0.0
    self.landing_dwell_time[env_ids] = 0.0

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    recovery_start_angle: float,
    target_angle: float,
    max_overrotation: float,
    descent_distance: float,
    wheel_contact_weight: float,
    max_axis_rate: float,
    max_linear_speed: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if (
      recovery_start_angle < 0.0
      or target_angle <= recovery_start_angle
      or max_overrotation <= 0.0
      or descent_distance <= 0.0
      or not 0.0 <= wheel_contact_weight <= 1.0
      or max_axis_rate <= 0.0
      or max_linear_speed <= 0.0
    ):
      raise ValueError("Aerial recovery progress parameters are invalid.")

    asset: Entity = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    active = aerial_active(env, command_name) > 0.5
    reset = (env.episode_length_buf == 0) | (~active)
    self.best_score[reset] = 0.0
    self.landing_dwell_time[reset] = 0.0

    was_airborne = getattr(command_term, "was_airborne", torch.zeros_like(active))
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    launch_axis_w = getattr(
      command_term, "_launch_axis_w", torch.zeros(env.num_envs, 3, device=env.device)
    )
    in_window = (
      active
      & was_airborne
      & (progress >= recovery_start_angle)
      & (progress <= target_angle + max_overrotation)
    )
    turn_alignment = torch.clamp(
      (progress - recovery_start_angle) / (target_angle - recovery_start_angle),
      min=0.0,
      max=1.0,
    )

    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    # Projected gravity is unit length, hence this maps normal/sideways/
    # inverted attitudes to 1.0/~0.29/0.0 without a narrow exponential basin.
    upright = torch.clamp(1.0 - torch.sqrt(gravity_error) / 2.0, min=0.0, max=1.0)
    axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
    braking = torch.clamp(
      1.0 - torch.abs(axis_rate) / max_axis_rate, min=0.0, max=1.0
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    settling = torch.clamp(
      1.0 - linear_speed / max_linear_speed, min=0.0, max=1.0
    )

    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    clearance = asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]
    descent = torch.clamp(1.0 - clearance / descent_distance, min=0.0, max=1.0)
    wheel_fraction = _wheel_contacts(env, sensor_name).float().mean(dim=1)
    # Descent alone is only a bridge: the real target is wheel-first support.
    # V54 weighted these equally and therefore learned a clean-looking but
    # contact-free fall.  Keep a small pre-contact gradient while ensuring a
    # measured wheel return dominates every recovery improvement.
    approach_ground = (
      (1.0 - wheel_contact_weight) * descent
      + wheel_contact_weight * wheel_fraction
    )

    base_score = (
      in_window.float()
      * turn_alignment
      * upright
      * braking
      * settling
      * approach_ground
    )
    # ``base_score`` correctly ranks a first wheel touchdown, but its
    # best-so-far potential would otherwise saturate there.  The terminal
    # command asks for a *continuous* normal four-wheel landing.  Measure
    # exactly that same result-space condition and make every additional
    # stable control step improve the existing potential up to one.  This is
    # not a reference phase or posture: it only distinguishes a real held
    # landing from an isolated contact-sensor hit.
    strict_landing = (
      in_window
      & torch.all(_wheel_contacts(env, sensor_name), dim=1)
      & (gravity_error < command_term.cfg.landing_gravity_error_limit)
      & (linear_speed < command_term.cfg.landing_linear_velocity_limit)
      & (
        torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
        < command_term.cfg.landing_angular_velocity_limit
      )
    )
    self.landing_dwell_time = torch.where(
      strict_landing,
      self.landing_dwell_time + env.step_dt,
      torch.zeros_like(self.landing_dwell_time),
    )
    dwell_fraction = torch.clamp(
      self.landing_dwell_time / command_term.cfg.landing_settle_time,
      min=0.0,
      max=1.0,
    )
    score = base_score + (1.0 - base_score) * strict_landing.float() * dwell_fraction
    previous_best = self.best_score.clone()
    self.best_score = torch.maximum(self.best_score, score)
    # RewardManager integrates over step_dt; this makes the total return the
    # bounded physical potential increase, independent of control frequency.
    return (self.best_score - previous_best) / env.step_dt


class AerialRotationProgress:
  """Reward high-clearance new directed progress, capped at one complete turn."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.progress = torch.zeros(env.num_envs, device=env.device)
    self.best_progress = torch.zeros(env.num_envs, device=env.device)
    self.launch_axis_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.has_grounded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.airborne_time = torch.zeros(env.num_envs, device=env.device)
    self.current_flight_qualified = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.flight_rotation = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.progress[env_ids] = 0.0
    self.best_progress[env_ids] = 0.0
    self.launch_axis_w[env_ids] = 0.0
    self.was_airborne[env_ids] = False
    self.has_grounded[env_ids] = False
    self.airborne_time[env_ids] = 0.0
    self.current_flight_qualified[env_ids] = False
    self.flight_rotation[env_ids] = 0.0
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    axes: tuple[tuple[float, float, float], ...],
    target_angle: float = math.tau,
    clearance_start: float | None = None,
    clearance_full: float | None = None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if (clearance_start is None) != (clearance_full is None):
      raise ValueError("clearance_start and clearance_full must be specified together.")
    if clearance_start is None:
      command_term = env.command_manager.get_term(command_name)
      clearance_start = command_term.cfg.rotation_progress_clearance_start
      clearance_full = command_term.cfg.rotation_progress_clearance_full
    assert clearance_full is not None
    if clearance_start < 0.0 or clearance_full <= clearance_start:
      raise ValueError("Rotation-progress clearance limits must be an ordered range.")
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & (
      (~self.previous_active) | (mode != self.previous_mode) | reset
    )
    clear = new_skill | reset | (~active)
    self.progress[clear] = 0.0
    self.best_progress[clear] = 0.0
    self.was_airborne[clear] = False
    self.has_grounded[clear] = False
    self.airborne_time[clear] = 0.0
    self.current_flight_qualified[clear] = False
    self.flight_rotation[clear] = 0.0

    axes_b = torch.tensor(
      axes, dtype=asset.data.root_link_quat_w.dtype, device=env.device
    )[mode]
    self.launch_axis_w[new_skill] = quat_apply(
      asset.data.root_link_quat_w[new_skill], axes_b[new_skill]
    )
    axis_rate = torch.sum(
      asset.data.root_link_ang_vel_w * self.launch_axis_w, dim=1
    )
    command_term = env.command_manager.get_term(command_name)
    contacts = _wheel_contacts(env, sensor_name)
    (
      self.has_grounded,
      self.airborne_time,
      self.current_flight_qualified,
      self.flight_rotation,
      signed_delta,
    ) = _advance_qualified_aerial_rotation(
      env=env,
      active=active,
      contacts=contacts,
      axis_rate=axis_rate,
      has_grounded=self.has_grounded,
      airborne_time=self.airborne_time,
      current_flight_qualified=self.current_flight_qualified,
      flight_rotation=self.flight_rotation,
      min_ballistic_time=command_term.cfg.min_ballistic_time,
    )
    self.was_airborne |= self.current_flight_qualified
    self.progress = torch.clamp_min(self.progress + signed_delta, 0.0)
    old_best = self.best_progress.clone()
    self.best_progress = torch.maximum(self.best_progress, self.progress)
    progress_delta = torch.clamp(self.best_progress, max=target_angle) - torch.clamp(
      old_best, max=target_angle
    )
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    clearance = asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]
    # A low wheel-hop can no longer collect the same angular reward as a real
    # aerial maneuver.  The angle is still integrated independently above, so
    # one eventual completion is judged from raw physics rather than this gate.
    clearance_gate = torch.clamp(
      (clearance - clearance_start) / (clearance_full - clearance_start), min=0.0, max=1.0
    )
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    # Convert angular displacement back to a rate because RewardManager applies
    # dt.  The integrated return is therefore proportional to unique angle.
    return clearance_gate * progress_delta / env.step_dt


class AerialOverRotation:
  """Penalize continued positive rotation after one complete turn."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.progress = torch.zeros(env.num_envs, device=env.device)
    self.launch_axis_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.progress[env_ids] = 0.0
    self.launch_axis_w[env_ids] = 0.0
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    axes: tuple[tuple[float, float, float], ...],
    target_angle: float = math.tau,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & (
      (~self.previous_active) | (mode != self.previous_mode) | reset
    )
    clear = new_skill | reset | (~active)
    self.progress[clear] = 0.0

    axes_b = torch.tensor(
      axes, dtype=asset.data.root_link_quat_w.dtype, device=env.device
    )[mode]
    self.launch_axis_w[new_skill] = quat_apply(
      asset.data.root_link_quat_w[new_skill], axes_b[new_skill]
    )
    axis_rate = torch.sum(
      asset.data.root_link_ang_vel_w * self.launch_axis_w, dim=1
    )
    airborne = ~torch.any(_wheel_contacts(env, sensor_name), dim=1)
    signed_delta = (
      active.to(axis_rate.dtype)
      * airborne.to(axis_rate.dtype)
      * axis_rate
      * env.step_dt
    )
    self.progress = torch.clamp_min(self.progress + signed_delta, 0.0)
    excess = torch.clamp(self.progress - target_angle, min=0.0, max=math.pi)
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    return excess


class AerialRotationCompletion:
  """Reward a complete airborne turn followed by a normal four-wheel landing."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    self.progress = torch.zeros(env.num_envs, device=env.device)
    self.was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.has_grounded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.landing_settle_time = torch.zeros(env.num_envs, device=env.device)
    self.launch_axis_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.airborne_time = torch.zeros(env.num_envs, device=env.device)
    self.current_flight_qualified = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.flight_rotation = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.progress[env_ids] = 0.0
    self.was_airborne[env_ids] = False
    self.has_grounded[env_ids] = False
    self.awarded[env_ids] = False
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1
    self.landing_settle_time[env_ids] = 0.0
    self.launch_axis_w[env_ids] = 0.0
    self.airborne_time[env_ids] = 0.0
    self.current_flight_qualified[env_ids] = False
    self.flight_rotation[env_ids] = 0.0

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    axes: tuple[tuple[float, float, float], ...],
    target_angle: float = math.tau,
    landing_gravity_std: float = 0.3,
    landing_settle_time: float = 0.10,
    landing_linear_velocity_limit: float = 0.75,
    landing_angular_velocity_limit: float = 1.5,
    max_overrotation: float = 0.75,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & (
      (~self.previous_active) | (mode != self.previous_mode) | reset
    )
    clear = new_skill | reset | (~active)
    self.progress[clear] = 0.0
    self.was_airborne[clear] = False
    self.has_grounded[clear] = False
    self.awarded[clear] = False
    self.landing_settle_time[clear] = 0.0
    self.airborne_time[clear] = 0.0
    self.current_flight_qualified[clear] = False
    self.flight_rotation[clear] = 0.0

    axes_b = torch.tensor(
      axes, dtype=asset.data.root_link_quat_w.dtype, device=env.device
    )[mode]
    self.launch_axis_w[new_skill] = quat_apply(
      asset.data.root_link_quat_w[new_skill], axes_b[new_skill]
    )
    axis_rate = torch.sum(
      asset.data.root_link_ang_vel_w * self.launch_axis_w, dim=1
    )
    contacts = _wheel_contacts(env, sensor_name)
    command_term = env.command_manager.get_term(command_name)
    (
      self.has_grounded,
      self.airborne_time,
      self.current_flight_qualified,
      self.flight_rotation,
      signed_delta,
    ) = _advance_qualified_aerial_rotation(
      env=env,
      active=active,
      contacts=contacts,
      axis_rate=axis_rate,
      has_grounded=self.has_grounded,
      airborne_time=self.airborne_time,
      current_flight_qualified=self.current_flight_qualified,
      flight_rotation=self.flight_rotation,
      min_ballistic_time=command_term.cfg.min_ballistic_time,
    )
    self.was_airborne |= self.current_flight_qualified
    self.progress = torch.clamp_min(self.progress + signed_delta, 0.0)

    normal_gravity = torch.tensor((0.0, 0.0, -1.0), dtype=asset.data.projected_gravity_b.dtype, device=env.device)
    gravity_error = torch.sum(torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1)
    landed = torch.all(contacts, dim=1)
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    stable_landing = (
      active
      & self.was_airborne
      & landed
      & (gravity_error < landing_gravity_std)
      & (linear_speed < landing_linear_velocity_limit)
      & (angular_speed < landing_angular_velocity_limit)
    )
    self.landing_settle_time = torch.where(
      stable_landing,
      self.landing_settle_time + env.step_dt,
      torch.zeros_like(self.landing_settle_time),
    )
    completed = (
      stable_landing
      & (self.progress >= target_angle)
      & (self.progress <= target_angle + max_overrotation)
      & (self.landing_settle_time >= landing_settle_time)
    )
    reward = completed & (~self.awarded)
    self.awarded |= completed
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    # Preserve the configured one-shot completion value under dt scaling.
    return reward.float() / env.step_dt
