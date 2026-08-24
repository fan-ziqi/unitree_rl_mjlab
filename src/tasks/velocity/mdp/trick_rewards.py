"""Outcome rewards used by the three current Go2W trick environments.

This module deliberately contains no archived reward experiments.  Commands
remain compact proprioceptive inputs; contact geometry and target directions
below are task-side measurements, not additional actor observations.
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


def _command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def _mode_mask(
  env: ManagerBasedRlEnv,
  command_name: str,
  modes: tuple[int, ...],
  *,
  num_modes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :num_modes], dim=1)
  selected = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  for index in modes:
    selected |= mode == index
  active = torch.sum(command[:, :num_modes], dim=1) > 0.5
  return selected & active, mode


def _wheel_contacts(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1)


def _has_any_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  return torch.any(_wheel_contacts(env, sensor_name), dim=1)


def normal_four_wheel_axle_layout(
  wheel_axles: torch.Tensor,
  wheel_positions: torch.Tensor,
  *,
  line_scale: float = 0.14,
  front_inside_scale: float = 0.06,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Measure the reference video's four-wheel common-axis layout.

  All four wheel axes must lie on one horizontal line beneath the body.  The
  front pair is the inner pair on that line and the rear pair is outside it.
  This is a physical wheel-layout measurement, not a joint-position target.
  """
  if wheel_axles.ndim != 3 or wheel_axles.shape[1:] != (4, 3):
    raise ValueError("four-wheel axle layout expects [batch, 4, 3] axes.")
  if wheel_positions.shape != wheel_axles.shape:
    raise ValueError("wheel positions must match wheel axle tensor shape.")
  if line_scale <= 0.0 or front_inside_scale <= 0.0:
    raise ValueError("layout scales must be positive.")

  axes = torch.nn.functional.normalize(wheel_axles, dim=2)
  reference_axis = axes[:, :1]
  aligned_axes = axes * torch.where(
    torch.sum(axes * reference_axis, dim=2, keepdim=True) >= 0.0,
    torch.ones_like(axes[:, :, :1]),
    -torch.ones_like(axes[:, :, :1]),
  )
  common_axis = torch.nn.functional.normalize(aligned_axes.sum(dim=1), dim=1)
  parallel_score = torch.mean(
    torch.abs(torch.sum(axes * common_axis.unsqueeze(1), dim=2)), dim=1
  )
  horizontal_score = torch.linalg.vector_norm(common_axis[:, :2], dim=1)

  relative_positions = wheel_positions - wheel_positions.mean(dim=1, keepdim=True)
  axial_coordinate = torch.sum(
    relative_positions * common_axis.unsqueeze(1), dim=2
  )
  transverse_offset = relative_positions - axial_coordinate.unsqueeze(2) * common_axis.unsqueeze(1)
  transverse_rms = torch.sqrt(torch.mean(torch.sum(torch.square(transverse_offset), dim=2), dim=1))
  line_score = 1.0 / (1.0 + torch.square(transverse_rms / line_scale))

  front_radius = torch.mean(torch.abs(axial_coordinate[:, :2]), dim=1)
  rear_radius = torch.mean(torch.abs(axial_coordinate[:, 2:]), dim=1)
  front_inside_delta = rear_radius - front_radius
  front_inside_score = torch.sigmoid(front_inside_delta / front_inside_scale)
  return parallel_score * horizontal_score * line_score, front_inside_score, front_inside_delta


# ---------------------------------------------------------------------------
# Shared two-wheel support measurement.


def mode_support_score(
  env: ManagerBasedRlEnv,
  command_name: str,
  modes: tuple[int, ...],
  gravity_targets: tuple[tuple[float, float, float], ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  sensor_name: str,
  num_modes: int = 5,
  extra_contact_discount: float = 0.75,
  minimum_root_clearance: float | tuple[float, ...] | None = None,
  orientation_power: float = 1.0,
  clearance_power: float = 1.0,
  stationary_command_index: int | None = None,
  command_deadband: float = 0.0,
  static_angular_velocity_scale: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Measure the commanded contact pair, attitude, and optional height.

  This is an outcome score rather than a leg-pose target.  Multiplication
  makes a low crouch or a fallen contact pair inferior to a genuine support;
  the partial contact factors retain a discovery gradient from four wheels.
  """
  if not 0.0 <= extra_contact_discount <= 1.0:
    raise ValueError("extra_contact_discount must be in [0, 1].")
  if orientation_power <= 0.0 or clearance_power <= 0.0:
    raise ValueError("orientation_power and clearance_power must be positive.")
  if minimum_root_clearance is not None:
    clearance_values = (
      (minimum_root_clearance,)
      if isinstance(minimum_root_clearance, float | int)
      else minimum_root_clearance
    )
    if any(value <= 0.0 for value in clearance_values):
      raise ValueError("minimum_root_clearance must be positive.")
  else:
    clearance_values = ()
  if stationary_command_index is not None and command_deadband < 0.0:
    raise ValueError("command_deadband must be non-negative.")
  if static_angular_velocity_scale is not None and static_angular_velocity_scale <= 0.0:
    raise ValueError("static_angular_velocity_scale must be positive.")

  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  if stationary_command_index is not None:
    if not 0 <= stationary_command_index < command.shape[1]:
      raise ValueError("stationary_command_index is outside the command tensor.")
    active &= torch.abs(command[:, stationary_command_index]) <= command_deadband

  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  orientation = torch.pow(
    torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    ),
    orientation_power,
  )
  contacts = _wheel_contacts(env, sensor_name).float()
  masks = torch.tensor(contact_masks, dtype=contacts.dtype, device=env.device)
  target = masks[mode]
  desired = (contacts * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
  non_target = 1.0 - target
  extra = (contacts * non_target).sum(dim=1) / non_target.sum(dim=1).clamp_min(1.0)
  support = desired * (1.0 - extra_contact_discount * extra)

  clearance = torch.ones_like(orientation)
  if minimum_root_clearance is not None:
    if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
      raise ValueError("root-clearance support score needs four wheel sites.")
    wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    support_height = (wheel_height * target).sum(dim=1) / target.sum(dim=1).clamp_min(
      1.0
    )
    clearance_target = torch.tensor(
      clearance_values,
      dtype=orientation.dtype,
      device=env.device,
    )
    if clearance_target.numel() == 1:
      clearance_target = clearance_target.expand(len(gravity_targets))
    if clearance_target.numel() != len(gravity_targets):
      raise ValueError("minimum_root_clearance must be scalar or cover every mode.")
    clearance = torch.pow(
      torch.clamp(
        (asset.data.root_link_pos_w[:, 2] - support_height)
        / clearance_target[mode],
        min=0.0,
        max=1.0,
      ),
      clearance_power,
    )
  # Static one-hots (including left/right dual-wheel support) mean a held
  # support, not an unspecified spin.  Keep this inside the existing support
  # outcome rather than adding a separate regularizer.  Moving spin commands
  # are already excluded by ``stationary_command_index`` before this factor.
  stillness = torch.ones_like(orientation)
  if static_angular_velocity_scale is not None:
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    stillness = torch.clamp(
      1.0 - angular_speed / static_angular_velocity_scale,
      min=0.0,
      max=1.0,
    )
  return active.to(orientation.dtype) * orientation * support * clearance * stillness


def _stance_spin_components(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_deadband: float,
  rate_std: float,
  gravity_targets: tuple[tuple[float, float, float], ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[
  Entity,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
]:
  """Measure a commanded five-mode world-down rotation."""
  if rate_std <= 0.0:
    raise ValueError("rate_std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
    raise ValueError("stance-spin measurement needs four wheel sites.")

  command = _command(env, command_name)
  mode = torch.argmax(command[:, :5], dim=1)
  active = torch.sum(command[:, :5], dim=1) > 0.5
  # Normal/front/rear track a world-down rate.  Normal is the video's folded
  # four-wheel common-axis pivot; front/rear are their named two-wheel pivots.
  moving = active & (torch.abs(command[:, 5]) > speed_deadband)
  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * gravity, dim=1)
  rate_score = torch.clamp(
    1.0 - torch.abs(command[:, 5] - actual_rate) / rate_std,
    min=0.0,
    max=1.0,
  )

  contacts = _wheel_contacts(env, sensor_name).float()
  masks = torch.tensor(
    contact_masks,
    dtype=contacts.dtype,
    device=env.device,
  )
  target = masks[mode]
  target_count = target.sum(dim=1).clamp_min(1.0)
  desired = (contacts * target).sum(dim=1) / target_count
  non_target_count = (1.0 - target).sum(dim=1)
  extra = (contacts * (1.0 - target)).sum(dim=1) / non_target_count.clamp_min(1.0)
  extra = torch.where(non_target_count > 0.0, extra, torch.zeros_like(extra))
  contact_score = desired * (1.0 - 0.75 * extra)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
  )
  batch = torch.arange(env.num_envs, device=env.device)

  # Wheel-joint local Y is the cylinder axle.
  wheel_quat = asset.data.site_quat_w[:, asset_cfg.site_ids].reshape(-1, 4)
  local_axle = torch.tensor(
    (0.0, 1.0, 0.0), dtype=wheel_quat.dtype, device=env.device
  ).expand(wheel_quat.shape[0], -1)
  wheel_axles = quat_apply(wheel_quat, local_axle).reshape(env.num_envs, 4, 3)
  wheel_positions = asset.data.site_pos_w[:, asset_cfg.site_ids]
  def pair_coaxiality(first: int, second: int) -> torch.Tensor:
    axle_a = wheel_axles[:, first]
    axle_b = wheel_axles[:, second]
    centre_line = torch.nn.functional.normalize(
      wheel_positions[:, second] - wheel_positions[:, first], dim=1
    )
    return (
      torch.abs(torch.sum(axle_a * axle_b, dim=1))
      * torch.abs(torch.sum(centre_line * axle_a, dim=1))
      * torch.linalg.vector_norm(axle_a[:, :2], dim=1)
    )

  pair_coaxiality_for_mode = torch.stack(
    (
      pair_coaxiality(0, 1),
      pair_coaxiality(0, 1),
      pair_coaxiality(2, 3),
      pair_coaxiality(0, 2),
      pair_coaxiality(1, 3),
    ),
    dim=1,
  )[batch, mode]
  normal_coaxiality, front_inside_score, _ = normal_four_wheel_axle_layout(
    wheel_axles, wheel_positions
  )
  support_masks = masks[mode]
  normal_support_mask = torch.ones_like(support_masks)
  support_mask = torch.where((mode == 0).unsqueeze(1), normal_support_mask, support_masks)
  normal_contact_score = torch.prod(contacts, dim=1)
  contact_score = torch.where(mode == 0, normal_contact_score, contact_score)
  coaxiality = torch.where(
    mode == 0,
    normal_coaxiality * front_inside_score,
    pair_coaxiality_for_mode,
  )
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  fixed_height = (wheel_height * support_mask).sum(dim=1) / support_mask.sum(dim=1).clamp_min(1.0)
  height_score = torch.clamp(
    (asset.data.root_link_pos_w[:, 2] - fixed_height) / 0.45,
    min=0.0,
    max=1.0,
  )
  coaxial_factor = torch.where(
    mode == 0,
    coaxiality,
    0.15 + 0.85 * coaxiality,
  )
  side_alignment_score = torch.square(alignment) * (
    0.20 + 0.80 * torch.square(alignment)
  )
  alignment_score = torch.where(
    mode == 0,
    torch.square(alignment),
    torch.where(mode <= 2, torch.pow(alignment, 4.0), side_alignment_score),
  )
  support_quality = contact_score * alignment_score * height_score
  return (
    asset,
    active,
    moving,
    rate_score,
    support_quality,
    coaxial_factor,
    support_mask,
    mode,
  )


class StanceSpinPivotResult:
  """Reward one fused policy's dynamic pivots and static side supports.

  Normal uses the folded four-wheel common-axis layout.  Front/rear use their
  named horizontal support axle; left/right are static side supports.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg, env

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    speed_deadband: float,
    std: float,
    gravity_targets: tuple[tuple[float, float, float], ...],
    contact_masks: tuple[tuple[float, float, float, float], ...],
    sensor_name: str,
    pivot_speed_limit: float,
    asset_cfg: SceneEntityCfg,
    upright_support_weight: float = 0.20,
    side_support_weight: float = 0.25,
    side_pivot_speed_limit: float = 0.35,
    normal_coaxial_weight: float = 0.15,
  ) -> torch.Tensor:
    if pivot_speed_limit <= 0.0:
      raise ValueError("pivot_speed_limit must be positive.")
    if not 0.0 <= upright_support_weight < 1.0:
      raise ValueError("upright_support_weight must be in [0, 1).")
    if not 0.0 <= side_support_weight < 1.0:
      raise ValueError("side_support_weight must be in [0, 1).")
    if side_pivot_speed_limit <= 0.0:
      raise ValueError("side_pivot_speed_limit must be positive.")
    if not 0.0 <= normal_coaxial_weight < 1.0:
      raise ValueError("normal_coaxial_weight must be in [0, 1).")
    (
      asset,
      active,
      moving,
      rate_score,
      support_quality,
      coaxial_factor,
      support_mask,
      mode,
    ) = _stance_spin_components(
      env,
      command_name,
      speed_deadband,
      std,
      gravity_targets,
      contact_masks,
      sensor_name,
      asset_cfg,
    )
    wheel_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    support_centre_velocity = (
      wheel_velocity * support_mask.unsqueeze(2)
    ).sum(dim=1) / support_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    centre_speed = torch.linalg.vector_norm(support_centre_velocity, dim=1)
    # For normal this is the centroid of all four folded wheels: individual
    # wheels roll around it during a real in-place yaw spin.  Front/rear and
    # side modes retain their physical two-wheel support centroid.
    # The old late, subtractive penalty permitted a high-rate front/rear form
    # to score while its support axle travelled around a broad floor circle.
    # A local pivot is an inseparable outcome: score rate *and* a stationary
    # support centre together.  The rational form is smooth from the reset,
    # but a 0.40-m/s travelling pair receives only 8% of the result available
    # at the required 0.12-m/s local centre speed.
    pivot_stillness = 1.0 / (1.0 + torch.square(centre_speed / pivot_speed_limit))
    dynamic_modes = mode <= 2  # normal/front/rear: real high-rate pivots.
    dynamic_quality = coaxial_factor * rate_score * pivot_stillness
    # Front/rear starts at the ordinary four-wheel reset with near-zero
    # commanded-rate score.  Multiplying that zero by every support factor
    # gives PPO no path to discover the physically reachable two-wheel form.
    # Reserve a small fraction for the measured upright support itself; the
    # remaining value still requires the co-axial, commanded-rate,
    # stationary-centre pivot.
    upright_mode = (mode == 1) | (mode == 2)
    discovery_weight = torch.where(
      upright_mode,
      torch.full_like(support_quality, upright_support_weight),
      torch.zeros_like(support_quality),
    )
    upright_result = support_quality * (
      discovery_weight + (1.0 - discovery_weight) * dynamic_quality
    )
    # Normal has an analogous discovery problem, except the intermediate
    # physical result is *not* ordinary four-wheel standing: it is a common,
    # stationary wheel-axis geometry.  Reward only that measured geometry
    # before rate emerges, never the default support by itself.
    normal_geometry = support_quality * coaxial_factor * pivot_stillness
    normal_result = normal_geometry * (
      normal_coaxial_weight + (1.0 - normal_coaxial_weight) * rate_score
    )
    dynamic_result = torch.where(mode == 0, normal_result, upright_result)

    # Side support is the remaining two requested one-hots, not a failed
    # version of a pivot.  Once the base has rolled onto a left/right pair,
    # those wheel axles are vertical, so spin_rate is intentionally ignored.
    # Reward the same measured support geometry while asking it to be still;
    # this is the physically faithful result shown in the reference rather
    # than a hidden target posture or a prescribed transition.
    body_angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    static_stillness = 1.0 / (1.0 + torch.square(body_angular_speed / 1.0))
    # Static means no support-centre travel as well as no body rotation.  The
    # local-centre measurement remains the final-quality factor below, so a
    # travelling side pose cannot be the maximum-reward outcome.
    # Settling into a side stand is briefly mobile.  The old product made
    # every useful partial side support worth almost zero until it had already
    # reached the static endpoint, which is an exploration dead-end.  Reuse
    # the same support geometry as a small bridge, analogous to front/rear's
    # existing upright-support bridge.  The strict static factors still make
    # an actually still support the unique maximum; no pose, clock, or action
    # target is supplied.
    side_pivot_stillness = 1.0 / (
      1.0 + torch.square(centre_speed / side_pivot_speed_limit)
    )
    static_quality = static_stillness * side_pivot_stillness
    static_result = support_quality * (
      side_support_weight + (1.0 - side_support_weight) * static_quality
    )
    return active.to(rate_score.dtype) * torch.where(
      dynamic_modes, moving.to(rate_score.dtype) * dynamic_result, static_result
    )


# ---------------------------------------------------------------------------
# Normal/front/rear locomotion task.


def _stance_locomotion_axes(
  asset: Entity, mode: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Select world-plane forward for normal, front, and rear poses."""
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
  fallback = torch.tensor(
    (1.0, 0.0), dtype=forward.dtype, device=forward.device
  ).expand_as(forward)
  forward = torch.where(norm > 1.0e-4, forward / norm.clamp_min(1.0e-4), fallback)
  return forward, torch.stack((-forward[:, 1], forward[:, 0]), dim=1)


def _locomotion_alignment(
  env: ManagerBasedRlEnv,
  asset: Entity,
  mode: torch.Tensor,
  gravity_targets: tuple[tuple[float, float, float], ...] | None,
  gravity_power: float,
) -> torch.Tensor:
  if gravity_power < 0.0:
    raise ValueError("gravity_power must be non-negative.")
  score = torch.ones(env.num_envs, device=env.device)
  if gravity_targets is not None and gravity_power > 0.0:
    targets = torch.tensor(
      gravity_targets,
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
    score = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    ).pow(gravity_power)
  return score


def _locomotion_support_match(
  env: ManagerBasedRlEnv,
  mode: torch.Tensor,
  contact_masks: tuple[tuple[float, float, float, float], ...],
  sensor_name: str,
) -> torch.Tensor:
  """Return the measured fraction of the commanded wheel support."""
  contacts = _wheel_contacts(env, sensor_name).float()
  masks = torch.tensor(contact_masks, dtype=contacts.dtype, device=env.device)
  if masks.ndim != 2 or masks.shape[1] != contacts.shape[1]:
    raise ValueError("contact_masks must match the wheel-contact sensor layout.")
  target = masks[mode]
  desired = (contacts * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
  non_target = 1.0 - target
  extra = (contacts * non_target).sum(dim=1) / non_target.sum(dim=1).clamp_min(1.0)
  extra = torch.where(non_target.sum(dim=1) > 0.0, extra, torch.zeros_like(extra))
  return desired * (1.0 - extra)


def stance_locomotion_linear_velocity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  lateral_weight: float = 2.0,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  contact_masks: tuple[tuple[float, float, float, float], ...] | None = None,
  sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track requested forward velocity while holding lateral velocity at zero.

  This deliberately uses a non-saturating, signed distance score instead of
  an RBF.  A two-wheel policy that happens to roll in the wrong direction can
  be more than one RBF width away from a small x/yaw command; an RBF then
  returns exactly (or numerically almost) zero for *all* such actions and
  gives PPO no preference for reducing the error.  ``1 - distance / std``
  keeps the same compact outcome measurement but makes every reduction in
  command error valuable.  Validity is still supplied by the existing
  gravity gate below, so a fallen robot cannot receive a tracking result.
  """
  if std <= 0.0 or lateral_weight < 0.0:
    raise ValueError("std must be positive and lateral_weight non-negative.")
  if (contact_masks is None) != (sensor_name is None):
    raise ValueError("contact_masks and sensor_name must be provided together.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :3], dim=1)
  forward, right = _stance_locomotion_axes(asset, mode)
  velocity_xy = asset.data.root_link_lin_vel_w[:, :2]
  forward_speed = torch.sum(velocity_xy * forward, dim=1)
  lateral_speed = torch.sum(velocity_xy * right, dim=1)
  squared_error = torch.square(command[:, 3] - forward_speed) + lateral_weight * torch.square(
    lateral_speed
  )
  score = 1.0 - torch.sqrt(squared_error) / std
  result = score * _locomotion_alignment(
    env, asset, mode, gravity_targets, gravity_power
  )
  if contact_masks is not None:
    assert sensor_name is not None
    result = result * _locomotion_support_match(env, mode, contact_masks, sensor_name)
  return result


def stance_locomotion_yaw_rate_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  mode_weights: tuple[float, ...] | None = None,
  contact_masks: tuple[tuple[float, float, float, float], ...] | None = None,
  sensor_name: str | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track world-up yaw rate without an error-saturation dead zone."""
  if std <= 0.0:
    raise ValueError("std must be positive.")
  if (contact_masks is None) != (sensor_name is None):
    raise ValueError("contact_masks and sensor_name must be provided together.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :3], dim=1)
  error = torch.abs(command[:, 4] - asset.data.root_link_ang_vel_w[:, 2])
  score = 1.0 - error / std
  result = score * _locomotion_alignment(
    env, asset, mode, gravity_targets, gravity_power
  )
  if contact_masks is not None:
    assert sensor_name is not None
    result = result * _locomotion_support_match(env, mode, contact_masks, sensor_name)
  if mode_weights is not None:
    if len(mode_weights) != command.shape[1] - 2 or any(
      weight < 0.0 for weight in mode_weights
    ):
      raise ValueError("mode_weights must be non-negative and match locomotion modes.")
    weights = torch.tensor(mode_weights, dtype=result.dtype, device=env.device)
    result = result * weights[mode]
  return result


def normal_leg_default_pose_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Keep only the ordinary four-wheel command near its model default pose.

  This is deliberately a static outcome constraint, not a reference motion:
  it has no phase, time index, or desired action.  In particular, it is
  completely disabled for the front/rear modes, whose supporting legs must be
  free to find their own upright geometry.  Wheel joints are excluded by the
  supplied leg-joint selector, so rolling does not incur a posture cost.
  """
  if std <= 0.0:
    raise ValueError("std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  normal = torch.argmax(command[:, :3], dim=1) == 0
  joint_ids = asset_cfg.joint_ids
  if isinstance(joint_ids, slice):
    raise TypeError("normal default-pose reward needs explicit leg joints.")
  deviation = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[
    :, joint_ids
  ]
  score = torch.exp(-torch.mean(torch.square(deviation), dim=1) / std**2)
  return normal.to(score.dtype) * score


def aerial_airborne_clearance(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  nonwheel_sensor_name: str,
  target_clearance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward measured wheel-free height for one active aerial command.

  This is deliberately just the first physical part of a flip: leave the
  floor.  It contains no desired joint pose, takeoff time, reference state, or
  landing phase.  Multiplying by ``step_dt`` gives comparable return for the
  same physical airborne duration at different control rates.
  """
  if target_clearance <= 0.0:
    raise ValueError("target_clearance must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  active = torch.sum(command[:, :5], dim=1) > 0.5
  airborne = ~torch.any(_wheel_contacts(env, sensor_name), dim=1)
  legal = ~_has_any_contact(env, nonwheel_sensor_name)
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  clearance = torch.clamp(
    (asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2])
    / target_clearance,
    min=0.0,
    max=1.0,
  )
  return (
    active.to(clearance.dtype)
    * airborne.to(clearance.dtype)
    * legal.to(clearance.dtype)
    * clearance
    * env.step_dt
  )


class AerialNetRotationProgress:
  """Reward only new net desired-axis radians in one ballistic event.

  ``AerialRotationCommand`` already integrates signed angular displacement only
  during its first continuous wheel-free interval.  This term pays a radian
  once, when that integration reaches a new high-water mark.  A policy cannot
  collect extra reward by briefly turning forward, undoing the turn, then
  repeating it.  No pose, phase, or desired joint state is introduced.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.peak_progress = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_progress[env_ids] = 0.0
    self.previous_active[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    nonwheel_sensor_name: str,
    target_angle: float,
    target_clearance: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if target_angle <= 0.0 or target_clearance <= 0.0:
      raise ValueError("target_angle and target_clearance must be positive.")
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    # Reset at the literal idle boundary as well as the environment reset.  A
    # new one-hot event must not inherit credit from its predecessor.
    reset = (
      (~active)
      | (active & ~self.previous_active)
      | (env.episode_length_buf == 0)
    )
    self.peak_progress[reset] = 0.0
    command_term = env.command_manager.get_term(command_name)
    progress = torch.clamp(
      getattr(
        command_term,
        "_rotation_progress",
        torch.zeros(env.num_envs, device=env.device),
      ),
      min=0.0,
      max=target_angle,
    )
    increment = torch.clamp(progress - self.peak_progress, min=0.0)
    self.peak_progress = torch.where(
      active,
      torch.maximum(self.peak_progress, progress),
      torch.zeros_like(self.peak_progress),
    )
    self.previous_active = active
    legal = ~_has_any_contact(env, nonwheel_sensor_name)
    asset: Entity = env.scene[asset_cfg.name]
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    clearance = torch.clamp(
      (asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2])
      / target_clearance,
      min=0.0,
      max=1.0,
    )
    # A turn is useful only to the degree it was produced by a real jump.  The
    # separate clearance term makes takeoff discoverable; this single factor
    # prevents a low contact-edge pivot from competing with a ballistic flip.
    return (
      active.to(increment.dtype)
      * legal.to(increment.dtype)
      * torch.sqrt(clearance)
      * increment
    )


def _advance_qualified_aerial_rotation(
  *,
  env: ManagerBasedRlEnv,
  active: torch.Tensor,
  contacts: torch.Tensor,
  axis_rate: torch.Tensor,
  has_grounded: torch.Tensor,
  airborne_time: torch.Tensor,
  current_flight_qualified: torch.Tensor,
  flight_rotation: torch.Tensor,
  min_ballistic_time: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Accumulate a continuous, genuinely wheel-free flight interval."""
  if min_ballistic_time <= 0.0:
    raise ValueError("min_ballistic_time must be positive.")
  airborne = ~torch.any(contacts, dim=1)
  has_grounded = has_grounded | (active & torch.all(contacts, dim=1))
  flight_step = active & has_grounded & airborne
  airborne_time = torch.where(
    flight_step, airborne_time + env.step_dt, torch.zeros_like(airborne_time)
  )
  raw_delta = flight_step.to(axis_rate.dtype) * axis_rate * env.step_dt
  flight_rotation = torch.where(
    flight_step, flight_rotation + raw_delta, torch.zeros_like(flight_rotation)
  )
  newly_qualified = (
    flight_step & (~current_flight_qualified) & (airborne_time >= min_ballistic_time)
  )
  current_flight_qualified = torch.where(
    flight_step,
    current_flight_qualified | (airborne_time >= min_ballistic_time),
    torch.zeros_like(current_flight_qualified),
  )
  increment = torch.where(
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
    increment,
  )


class AerialRotationCompletion:
  """Reward a full-turn wheel touchdown, then its strict stable completion."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.progress = torch.zeros(env.num_envs, device=env.device)
    self.was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.has_grounded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.touchdown_awarded = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.previous_mode = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    self.post_idle_settle_time = torch.zeros(env.num_envs, device=env.device)
    self.launch_axis_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.launch_root_quat_w = torch.zeros(env.num_envs, 4, device=env.device)
    self.airborne_time = torch.zeros(env.num_envs, device=env.device)
    self.flight_qualified = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.flight_rotation = torch.zeros(env.num_envs, device=env.device)
    self.recovery_peak = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.progress[env_ids] = 0.0
    self.was_airborne[env_ids] = False
    self.has_grounded[env_ids] = False
    self.touchdown_awarded[env_ids] = False
    self.awarded[env_ids] = False
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1
    self.post_idle_settle_time[env_ids] = 0.0
    self.launch_axis_w[env_ids] = 0.0
    self.launch_root_quat_w[env_ids] = 0.0
    self.airborne_time[env_ids] = 0.0
    self.flight_qualified[env_ids] = False
    self.flight_rotation[env_ids] = 0.0
    self.recovery_peak[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    axes: tuple[tuple[float, float, float], ...],
    target_angle: float = math.tau,
    soft_touchdown_reward: float = 1.0,
    soft_touchdown_speed_scale: float = 4.0,
    landing_gravity_std: float = 0.3,
    landing_linear_velocity_limit: float = 0.75,
    landing_angular_velocity_limit: float = 1.5,
    landing_orientation_dot_min: float = 0.995,
    soft_touchdown_orientation_floor: float = 0.50,
    soft_touchdown_orientation_exponent: float = 1.0,
    soft_touchdown_turn_exponent: float = 2.0,
    max_overrotation: float = 1.25,
    settle_reward: float = 0.0,
    post_idle_settle_time: float = 0.40,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if soft_touchdown_reward < 0.0:
      raise ValueError("soft_touchdown_reward must be non-negative.")
    if soft_touchdown_speed_scale <= 0.0:
      raise ValueError("soft_touchdown_speed_scale must be positive.")
    if not 0.0 < landing_orientation_dot_min <= 1.0:
      raise ValueError("landing_orientation_dot_min must be in (0, 1].")
    if not 0.0 <= soft_touchdown_orientation_floor < landing_orientation_dot_min:
      raise ValueError(
        "soft_touchdown_orientation_floor must be non-negative and below the strict threshold."
      )
    if soft_touchdown_turn_exponent <= 0.0:
      raise ValueError("soft_touchdown_turn_exponent must be positive.")
    if soft_touchdown_orientation_exponent <= 0.0:
      raise ValueError("soft_touchdown_orientation_exponent must be positive.")
    if max_overrotation <= 0.0:
      raise ValueError("max_overrotation must be positive.")
    if settle_reward < 0.0:
      raise ValueError("settle_reward must be non-negative.")
    if post_idle_settle_time <= 0.0:
      raise ValueError("post_idle_settle_time must be positive.")
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & (
      (~self.previous_active) | (mode != self.previous_mode) | reset
    )
    # After a first landing, the command term clears the public one-hot and
    # hands control to literal default idle.  Keep the landing state through
    # that final window so the reward can reject a pose that looked correct at
    # touchdown but drifts after the maneuver has supposedly ended.
    command_term = env.command_manager.get_term(command_name)
    landing_started = getattr(
      command_term, "_landing_started", torch.zeros_like(active)
    )
    post_landing_idle = (~active) & landing_started
    clear = new_skill | reset | ((~active) & (~post_landing_idle))
    self.progress[clear] = 0.0
    self.was_airborne[clear] = False
    self.has_grounded[clear] = False
    self.touchdown_awarded[clear] = False
    self.awarded[clear] = False
    self.post_idle_settle_time[clear] = 0.0
    self.airborne_time[clear] = 0.0
    self.flight_qualified[clear] = False
    self.flight_rotation[clear] = 0.0
    self.recovery_peak[clear] = 0.0

    axes_b = torch.tensor(
      axes, dtype=asset.data.root_link_quat_w.dtype, device=env.device
    )[mode]
    self.launch_axis_w[new_skill] = quat_apply(
      asset.data.root_link_quat_w[new_skill], axes_b[new_skill]
    )
    self.launch_root_quat_w[new_skill] = asset.data.root_link_quat_w[new_skill]
    axis_rate = torch.sum(asset.data.root_link_ang_vel_w * self.launch_axis_w, dim=1)
    contacts = _wheel_contacts(env, sensor_name)
    (
      self.has_grounded,
      self.airborne_time,
      self.flight_qualified,
      self.flight_rotation,
      increment,
    ) = _advance_qualified_aerial_rotation(
      env=env,
      active=active & (~getattr(command_term, "_landing_started", torch.zeros_like(active))),
      contacts=contacts,
      axis_rate=axis_rate,
      has_grounded=self.has_grounded,
      airborne_time=self.airborne_time,
      current_flight_qualified=self.flight_qualified,
      flight_rotation=self.flight_rotation,
      min_ballistic_time=command_term.cfg.min_ballistic_time,
    )
    self.was_airborne |= self.flight_qualified
    self.progress = torch.clamp_min(self.progress + increment, 0.0)
    legal = ~_has_any_contact(env, nonwheel_sensor_name)
    # This is the learnable bridge to the exact five-frame settle event below.
    # The command intentionally becomes public idle at its *first* wheel
    # contact, so the default controller can recover instead of being asked to
    # continue a flip.  A real four-wheel touchdown often arrives one or two
    # frames later; limiting this result to ``active`` would silently throw
    # away the only endpoint signal for that recovery.  Keep the same one-shot
    # physical landing event alive through its immediate public-idle window.
    # Its value remains scaled by measured turn fraction, contact, velocity,
    # and whole-base orientation below—there is no pose, phase, or motion
    # reference introduced here.
    # A low hop at m500 was collecting the active-command touchdown signal at
    # only 0.22 turns, then displacing the useful 0.7--0.9 turn attempts.
    # The bridge is an endpoint result, not a generic landing bonus: require
    # substantial signed turn progress whether the first wheel contact is
    # still active or has already exposed public idle.  Rotation progress and
    # airborne clearance remain dense discovery signals below this threshold.
    landing_turn_eligible = self.progress >= 0.70 * target_angle
    wheel_touchdown = (
      (active | post_landing_idle)
      & landing_turn_eligible
      & self.was_airborne
      & torch.all(contacts, dim=1)
      & legal
    )
    touchdown_reward = wheel_touchdown & (~self.touchdown_awarded)
    self.touchdown_awarded |= wheel_touchdown

    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    orientation_similarity = torch.abs(
      torch.sum(asset.data.root_link_quat_w * self.launch_root_quat_w, dim=1)
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    # The original geometric touchdown bridge gave exactly the same credit to
    # a quiet landing and a high-speed wheel graze.  Preserve the *same*
    # one-shot outcome, but grade it continuously by the already public
    # landing variables.  The multiplier keeps early full-turn discoveries
    # informative; the unchanged strict five-frame event remains the only
    # high-value completion reward.
    # Before one turn, the existing power gives a smooth discovery signal.
    # After one turn, however, clamping it to one would make a 450-degree
    # bounce just as profitable as a 360-degree flip.  Multiply the same
    # first-touchdown outcome by a continuous final-angle quality instead.
    # This is an endpoint measurement, not a phase/reference trajectory.
    turn_fraction = torch.clamp(self.progress / target_angle, min=0.0, max=1.0)
    overrotation = torch.clamp(
      (self.progress - target_angle) / max_overrotation,
      min=0.0,
      max=1.0,
    )
    turn_quality = torch.pow(turn_fraction, soft_touchdown_turn_exponent) * (
      1.0 - overrotation
    )
    touchdown_quality = (
      torch.clamp(
        1.0 - gravity_error / (soft_touchdown_speed_scale * landing_gravity_std),
        min=0.0,
        max=1.0,
      )
      * torch.clamp(
        1.0
        - linear_speed
        / (soft_touchdown_speed_scale * landing_linear_velocity_limit),
        min=0.0,
        max=1.0,
      )
      * torch.clamp(
        1.0
        - angular_speed
        / (soft_touchdown_speed_scale * landing_angular_velocity_limit),
        min=0.0,
        max=1.0,
      )
      * torch.pow(
        torch.clamp(
          (orientation_similarity - soft_touchdown_orientation_floor)
          / (1.0 - soft_touchdown_orientation_floor),
          min=0.0,
          max=1.0,
        ),
        soft_touchdown_orientation_exponent,
      )
      * turn_quality
    )
    # The one-hot clears on its first wheel contact.  Requiring a simultaneous
    # four-wheel contact before *any* endpoint signal made that recovery
    # transition sparse: a policy could land a wheel pair close to the launch
    # frame but receive no preference for settling the remaining wheels.  Pay
    # only new progress of the same terminal-quality measurement while public
    # idle is recovering.  This still contains no pose, timing, or trajectory
    # target, and strict completion below remains the sole success criterion.
    recovery_eligible = (
      post_landing_idle
      & self.was_airborne
      & legal
      & (self.progress >= 0.70 * target_angle)
    )
    recovery_quality = touchdown_quality * contacts.float().mean(dim=1)
    recovery_increment = recovery_eligible.to(recovery_quality.dtype) * torch.clamp(
      recovery_quality - self.recovery_peak, min=0.0
    )
    self.recovery_peak = torch.where(
      recovery_eligible,
      torch.maximum(self.recovery_peak, recovery_quality),
      torch.zeros_like(self.recovery_peak),
    )
    # An already all-wheel touchdown has received the same endpoint quality
    # through the existing bridge; seed the monotonic recovery potential so it
    # cannot collect that result twice on the following public-idle frame.
    self.recovery_peak = torch.where(
      touchdown_reward,
      torch.maximum(self.recovery_peak, touchdown_quality),
      self.recovery_peak,
    )
    # The command clears on the first wheel contact.  From then on this same
    # reward owns the entire measured outcome: it must have completed one
    # bounded rotation and remain quietly recovered under the public idle
    # command.  No landing pose, reference action, or timing signal is added.
    turn_complete = (self.progress >= target_angle) & (
      self.progress <= target_angle + max_overrotation
    )
    idle_stable = (
      post_landing_idle
      & turn_complete
      & torch.all(contacts, dim=1)
      & legal
      & (gravity_error < landing_gravity_std)
      & (orientation_similarity >= landing_orientation_dot_min)
      & (linear_speed < landing_linear_velocity_limit)
      & (angular_speed < landing_angular_velocity_limit)
    )
    self.post_idle_settle_time = torch.where(
      idle_stable,
      self.post_idle_settle_time + env.step_dt,
      torch.zeros_like(self.post_idle_settle_time),
    )
    completed = idle_stable & (
      self.post_idle_settle_time + 0.5 * env.step_dt >= post_idle_settle_time
    )
    strict_reward = completed & (~self.awarded)
    self.awarded |= completed
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    return (
      soft_touchdown_reward
      * (touchdown_reward.float() * touchdown_quality + recovery_increment)
      # A real full-turn, all-wheel, launch-frame landing is now observable
      # before it survives the complete settle window.  Reward each moment
      # that this *same strict endpoint* remains true, so PPO can reinforce
      # braking/idle recovery instead of receiving a signal only after an
      # unlikely uninterrupted 0.6-s dwell.  It is gated by the exact
      # endpoint below; no pose, time-indexed trajectory, or new action goal
      # is introduced, and ``completed`` remains the sole success event.
      + settle_reward * idle_stable.float() * env.step_dt
      + strict_reward.float() / env.step_dt
    )
