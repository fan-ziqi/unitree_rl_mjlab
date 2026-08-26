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


def normal_two_wheel_pivot_geometry(
  wheel_axles: torch.Tensor,
  wheel_positions: torch.Tensor,
  support_is_front: torch.Tensor,
  *,
  compact_xy_radius: float = 0.48,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Measure compact all-wheel packing around a chosen tall support pair.

  A requested normal spin begins at an ordinary four-wheel idle, but the
  reference's fast pivot does *not* retain four wheel contacts.  It rises onto
  either the front or rear lateral pair, keeps the other two wheels clear, and
  folds all four wheel axes into the same compact turn envelope.  The policy is
  free to choose the front or rear pair: ``support_is_front`` is selected from
  the measured support quality, never added to the public command.

  The score intentionally measures only wheel-axis agreement and horizontal
  packing.  In particular, it permits the free pair to be vertically above the
  support pair, which is essential for the tall pivot and cannot be represented
  by the old "all centres on the floor line" condition.
  """
  if wheel_axles.ndim != 3 or wheel_axles.shape[1:] != (4, 3):
    raise ValueError("normal pivot geometry expects [batch, 4, 3] axes.")
  if wheel_positions.shape != wheel_axles.shape:
    raise ValueError("wheel positions must match wheel axle tensor shape.")
  if support_is_front.shape != (wheel_axles.shape[0],):
    raise ValueError("support_is_front must have one entry per environment.")
  if compact_xy_radius <= 0.0:
    raise ValueError("compact_xy_radius must be positive.")

  axes = torch.nn.functional.normalize(wheel_axles, dim=2)
  # Wheel-cylinder axes have an arbitrary sign in the imported model; use one
  # support wheel as the representative direction and compare every other
  # axle with an absolute dot product below.  Averaging two anti-parallel but
  # perfectly co-axial wheel axes would otherwise create a zero vector.
  front_axis = axes[:, 0]
  rear_axis = axes[:, 2]
  support_axis = torch.where(support_is_front.unsqueeze(1), front_axis, rear_axis)
  all_axis_parallel = torch.mean(
    torch.abs(torch.sum(axes * support_axis.unsqueeze(1), dim=2)), dim=1
  )
  front_centre = wheel_positions[:, :2, :2].mean(dim=1)
  rear_centre = wheel_positions[:, 2:, :2].mean(dim=1)
  support_centre = torch.where(
    support_is_front.unsqueeze(1), front_centre, rear_centre
  )
  max_radius = torch.amax(
    torch.linalg.vector_norm(wheel_positions[:, :, :2] - support_centre.unsqueeze(1), dim=2),
    dim=1,
  )
  compact_xy = 1.0 / (1.0 + torch.square(max_radius / compact_xy_radius))
  return all_axis_parallel, compact_xy


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
  orientation_progress_floor: float = 0.0,
  mode_weights: tuple[float, ...] | None = None,
  clearance_power: float = 1.0,
  stationary_command_index: int | None = None,
  static_command_start_index: int | None = None,
  command_deadband: float = 0.0,
  static_angular_velocity_scale: float | None = None,
  static_linear_velocity_scale: float | None = None,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Measure the commanded contact pair, attitude, and optional height.

  This is an outcome score rather than a leg-pose target.  Correct attitude
  remains a validity gate, while contact and trunk clearance are combined
  *inside* that gate.  A full product makes the ordinary four-wheel reset
  almost reward-free for a two-wheel request: it has halfway attitude but
  also the two extra contacts and low support clearance.  That gives PPO no
  useful return for beginning the physical rise.  The weighted combination
  preserves the same terminal optimum (correct pair, attitude, and clearance)
  without encoding a joint pose or an intermediate trajectory.
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
  if (
    stationary_command_index is not None
    or static_command_start_index is not None
  ) and command_deadband < 0.0:
    raise ValueError("command_deadband must be non-negative.")
  if static_command_start_index is not None and not 0 <= static_command_start_index < _command(
    env, command_name
  ).shape[1]:
    raise ValueError("static_command_start_index is outside the command tensor.")
  if not 0.0 <= orientation_progress_floor < 1.0:
    raise ValueError("orientation_progress_floor must be in [0, 1).")
  if mode_weights is not None and (
    len(mode_weights) != num_modes or any(weight < 0.0 for weight in mode_weights)
  ):
    raise ValueError("mode_weights must be non-negative and cover every mode.")
  if static_angular_velocity_scale is not None and static_angular_velocity_scale <= 0.0:
    raise ValueError("static_angular_velocity_scale must be positive.")
  if static_linear_velocity_scale is not None and static_linear_velocity_scale <= 0.0:
    raise ValueError("static_linear_velocity_scale must be positive.")

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
  # A zero x/yaw command means a held support, not an unspecified travelling
  # balance.  It must nevertheless be free to *reach* the support: applying
  # a zero-speed factor from ordinary four-wheel reset suppressed the pitch
  # and wheel-contact changes needed to stand up.  Activate the existing
  # stillness outcome only when the same measured attitude, target support,
  # and clearance already show that the requested two-wheel form is mostly
  # present.  This introduces neither a joint pose nor a time/phase target.
  static_command = torch.ones_like(active)
  if static_command_start_index is not None:
    static_command = torch.amax(
      torch.abs(command[:, static_command_start_index:]), dim=1
    ) <= command_deadband
  static_settling = static_command & (orientation >= 0.85) & (support >= 0.75) & (
    clearance >= 0.75
  )
  stillness = torch.ones_like(orientation)
  if static_angular_velocity_scale is not None:
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    # Do not turn a valid support's stillness measurement into a binary
    # exploration gate.  The initial Gaussian policy is often far beyond a
    # small static-speed tolerance; a clipped zero then makes every such
    # action equally uninformative to PPO.  The rational score still gives a
    # genuinely motionless stand the unique value one, limits a fast spinning
    # stand to a 10% bridge, and supplies an observable improvement path.
    angular_stillness = 0.10 + 0.90 / (
      1.0 + torch.square(angular_speed / static_angular_velocity_scale)
    )
    stillness = torch.where(static_settling, angular_stillness, stillness)
  if static_linear_velocity_scale is not None:
    planar_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
    linear_stillness = 0.10 + 0.90 / (
      1.0 + torch.square(planar_speed / static_linear_velocity_scale)
    )
    stillness = torch.where(
      static_settling, stillness * linear_stillness, stillness
    )
  # Contact is slightly more important than height: a tall robot supported by
  # the wrong wheels is not the requested stance.  Keeping both beneath the
  # measured attitude gate prevents a fallen, contact-free orientation from
  # being rewarded as a valid support.  A caller can reserve a bounded
  # attitude-progress component inside this same outcome.  That is useful
  # when the reset is ordinary four-wheel idle: it gives the policy a reason
  # to begin pitching toward the requested support before the contact pair
  # and clearance can physically improve together.  The unique maximum
  # remains correct attitude *and* contacts *and* clearance.
  support_progress = 0.65 * support + 0.35 * clearance
  result_progress = orientation_progress_floor + (
    1.0 - orientation_progress_floor
  ) * support_progress
  result = active.to(orientation.dtype) * orientation * result_progress * stillness
  if mode_weights is not None:
    weights = torch.tensor(mode_weights, dtype=result.dtype, device=env.device)
    result = result * weights[mode]
  return result


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
  # Every active mode tracks its signed world-down rate.  A nonzero normal
  # command is an unlabeled tall front-or-rear two-wheel pivot; the named
  # one-hots retain their particular support pair.
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

  front_rear_pair_coaxiality = torch.stack(
    (pair_coaxiality(0, 1), pair_coaxiality(2, 3)), dim=1
  )
  pair_coaxiality_for_mode = torch.stack(
    (
      front_rear_pair_coaxiality[:, 0],
      front_rear_pair_coaxiality[:, 0],
      front_rear_pair_coaxiality[:, 1],
      pair_coaxiality(0, 2),
      pair_coaxiality(1, 3),
    ),
    dim=1,
  )[batch, mode]

  # ``normal`` is deliberately a generic high-speed spin, rather than a
  # false four-contact mode.  Let the physical outcome choose the better of
  # the front/rear lateral support pairs.  This does not create a hidden
  # command: the actor sees only its normal one-hot and signed rate.
  pair_masks = masks[1:3]
  pair_desired = torch.sum(
    contacts.unsqueeze(1) * pair_masks.unsqueeze(0), dim=2
  ) / pair_masks.sum(dim=1).unsqueeze(0).clamp_min(1.0)
  pair_extra_count = (1.0 - pair_masks).sum(dim=1).unsqueeze(0)
  pair_extra = torch.sum(
    contacts.unsqueeze(1) * (1.0 - pair_masks).unsqueeze(0), dim=2
  ) / pair_extra_count.clamp_min(1.0)
  pair_contact_score = pair_desired * (1.0 - 0.75 * pair_extra)
  pair_alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity.unsqueeze(1) * targets[1:3].unsqueeze(0), dim=2)),
    min=0.0,
    max=1.0,
  )
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  pair_support_height = torch.sum(
    wheel_height.unsqueeze(1) * pair_masks.unsqueeze(0), dim=2
  ) / pair_masks.sum(dim=1).unsqueeze(0).clamp_min(1.0)
  pair_height_score = torch.clamp(
    (asset.data.root_link_pos_w[:, 2].unsqueeze(1) - pair_support_height) / 0.45,
    min=0.0,
    max=1.0,
  )
  pair_support_quality = pair_alignment * (
    0.65 * pair_contact_score + 0.35 * pair_height_score
  )
  normal_support_is_front = pair_support_quality[:, 0] >= pair_support_quality[:, 1]
  normal_support_quality = torch.amax(pair_support_quality, dim=1)
  normal_support_mask = torch.where(
    normal_support_is_front.unsqueeze(1),
    pair_masks[0].unsqueeze(0),
    pair_masks[1].unsqueeze(0),
  )
  normal_pair_coaxiality = torch.where(
    normal_support_is_front,
    front_rear_pair_coaxiality[:, 0],
    front_rear_pair_coaxiality[:, 1],
  )
  normal_all_axis_parallel, normal_compact_xy = normal_two_wheel_pivot_geometry(
    wheel_axles, wheel_positions, normal_support_is_front
  )
  support_masks = masks[mode]
  support_mask = torch.where((mode == 0).unsqueeze(1), normal_support_mask, support_masks)
  coaxiality = torch.where(
    mode == 0,
    normal_pair_coaxiality,
    pair_coaxiality_for_mode,
  )
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
  # A front/rear spin command begins from ordinary four-wheel idle.  Its
  # desired attitude is therefore only 0.5 aligned, its target pair has two
  # extra contacts, and its support height is modest.  Multiplying all three
  # quantities makes that physically meaningful start worth only a few
  # percent of a completed upright pivot.  PPO then finds the easier local
  # optimum: keep the normal pivot while tracking z rate under every one-hot.
  #
  # Use the same physical measurements, but combine contact and clearance
  # beneath the required upright-attitude gate for front/rear.  This preserves
  # the unique full-quality endpoint (correct pair, attitude, and height),
  # gives the policy an outcome gradient for initiating the support change,
  # and specifies neither a leg pose nor a transition trajectory.
  upright_support_progress = alignment * (0.65 * contact_score + 0.35 * height_score)
  support_quality = torch.where(
    mode != 0,
    upright_support_progress,
    normal_support_quality,
  )
  return (
    asset,
    active,
    moving,
    rate_score,
    support_quality,
    coaxial_factor,
    normal_all_axis_parallel,
    normal_compact_xy,
    support_mask,
    mode,
  )


class StanceSpinPivotResult:
  """Reward one fused policy's five commanded local pivots."""

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
    normal_final_support_weight: float = 0.02,
    normal_support_decay_start_steps: int = 38_400,
    normal_support_decay_steps: int = 25_600,
    rate_progress_weight: float = 0.75,
  ) -> torch.Tensor:
    if pivot_speed_limit <= 0.0:
      raise ValueError("pivot_speed_limit must be positive.")
    if not 0.0 <= upright_support_weight < 1.0:
      raise ValueError("upright_support_weight must be in [0, 1).")
    if not 0.0 <= normal_final_support_weight < 1.0:
      raise ValueError("normal_final_support_weight must be in [0, 1).")
    if normal_support_decay_start_steps < 0 or normal_support_decay_steps <= 0:
      raise ValueError("normal support decay steps must be non-negative/positive.")
    if not 0.0 <= rate_progress_weight <= 1.0:
      raise ValueError("rate_progress_weight must be in [0, 1].")
    (
      asset,
      active,
      moving,
      rate_score,
      support_quality,
      coaxial_factor,
      normal_all_axis_parallel,
      normal_compact_xy,
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
    # Normal selects its physically viable front or rear two-wheel centroid;
    # named modes retain their requested support-pair centroid.
    # The old late, subtractive penalty permitted a high-rate front/rear form
    # to score while its support axle travelled around a broad floor circle.
    # A local pivot is an inseparable outcome: score rate *and* a stationary
    # support centre together.  The rational form is smooth from the reset,
    # but a 0.40-m/s travelling pair receives only 8% of the result available
    # at the required 0.12-m/s local centre speed.
    pivot_stillness = 1.0 / (1.0 + torch.square(centre_speed / pivot_speed_limit))
    # ``rate_score`` is deliberately strict near the final requested speed,
    # but has no gradient once the error exceeds ``std``.  Pair it with a
    # signed triangular progress measurement: it rises from zero to the
    # requested rate and falls again after overshoot.  This preserves a dense
    # acceleration signal without rewarding a faster-than-commanded pivot.
    command = _command(env, command_name)
    gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
    actual_down_rate = torch.sum(asset.data.root_link_ang_vel_b * gravity, dim=1)
    requested_rate = command[:, 5]
    signed_rate_ratio = (
      actual_down_rate
      * torch.sign(requested_rate)
      / torch.abs(requested_rate).clamp_min(speed_deadband)
    )
    signed_rate_progress = (
      torch.clamp(signed_rate_ratio, min=0.0, max=1.0)
      * torch.clamp(2.0 - signed_rate_ratio, min=0.0, max=1.0)
    )
    speed_quality = (1.0 - rate_progress_weight) * rate_score + (
      rate_progress_weight * signed_rate_progress
    )
    dynamic_quality = coaxial_factor * pivot_stillness * (
      speed_quality
    )
    # Every named two-wheel mode starts at the ordinary four-wheel reset with near-zero
    # commanded-rate score.  Multiplying that zero by every support factor
    # gives PPO no path to discover the physically reachable two-wheel form.
    # Reserve a small fraction for the measured upright support itself; the
    # remaining value still requires the co-axial, commanded-rate,
    # stationary-centre pivot.
    upright_mode = mode != 0
    discovery_weight = torch.where(
      upright_mode,
      torch.full_like(support_quality, upright_support_weight),
      torch.zeros_like(support_quality),
    )
    upright_result = support_quality * (
      discovery_weight + (1.0 - discovery_weight) * dynamic_quality
    )
    # Fast normal pivots must become tall and two-wheel supported.  It is still
    # useful to keep the four wheel *axes* parallel and their horizontal
    # envelope compact, but requiring all four wheels on the floor selected a
    # crouched crawling turn.  This is an outcome measurement only: it does
    # not describe a joint pose, a phase, or a trajectory.
    normal_geometry = normal_all_axis_parallel * normal_compact_xy
    # A visible normal two-wheel stand is the necessary discovery bridge from
    # reset, but it cannot remain a permanent 20%-return shortcut: then PPO
    # can stand, drift across the floor, and spin in either direction while
    # never paying the signed-rate or local-pivot part of this *same* outcome.
    # Keep that bridge through the normal-only bootstrap, then decay it while
    # the already discovered support is asked to satisfy the public rate.
    normal_decay = torch.clamp(
      torch.tensor(
        (env.common_step_counter - normal_support_decay_start_steps)
        / normal_support_decay_steps,
        dtype=support_quality.dtype,
        device=env.device,
      ),
      min=0.0,
      max=1.0,
    )
    normal_support_weight = upright_support_weight + normal_decay * (
      normal_final_support_weight - upright_support_weight
    )
    normal_result = (
      support_quality
      * normal_geometry
      * (normal_support_weight + (1.0 - normal_support_weight)
         * pivot_stillness * speed_quality)
    )
    dynamic_result = torch.where(mode == 0, normal_result, upright_result)

    # A zero spin rate on an active front/rear one-hot has a useful and
    # observable meaning: make the named two-wheel support, then hold it.
    # This is the same measured contact/attitude/local-centre outcome used by
    # the rotating result—only without inventing a pose target or a separate
    # reward term.  The curriculum uses it briefly so PPO can discover the
    # support before it is asked to preserve high z-rate through a switch.
    upright_static_result = support_quality * pivot_stillness
    dynamic_or_upright_static = torch.where(
      moving,
      dynamic_result,
      torch.where(upright_mode, upright_static_result, torch.zeros_like(dynamic_result)),
    )
    return active.to(rate_score.dtype) * dynamic_or_upright_static


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


class AerialBallisticLaunch:
  """Pay launch impulse and the resulting continuous wheel-free duration.

  These are the two direct physical halves of a jump: accelerate the base
  upward while all four wheels support it, then keep every wheel clear long
  enough to rotate.  Paying only duration is correct but too sparse before the
  first useful launch; paying only velocity admits a rocking ground shortcut.
  Their equal high-water gains form one compact launch outcome, with no pose,
  desired action, phase, or reference trajectory.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.peak_upward_speed = torch.zeros(env.num_envs, device=env.device)
    self.current_duration = torch.zeros(env.num_envs, device=env.device)
    self.peak_duration = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_upward_speed[env_ids] = 0.0
    self.current_duration[env_ids] = 0.0
    self.peak_duration[env_ids] = 0.0
    self.previous_active[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    target_upward_speed: float,
    target_duration: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if target_upward_speed <= 0.0 or target_duration <= 0.0:
      raise ValueError("launch scales must be positive.")
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    reset = (
      (~active)
      | (active & ~self.previous_active)
      | (env.episode_length_buf == 0)
    )
    self.peak_upward_speed[reset] = 0.0
    self.current_duration[reset] = 0.0
    self.peak_duration[reset] = 0.0
    wheel_free = ~_has_any_contact(env, sensor_name)
    legal = ~_has_any_contact(env, nonwheel_sensor_name)
    grounded = torch.all(_wheel_contacts(env, sensor_name), dim=1)
    upward_speed = torch.clamp(
      asset.data.root_link_lin_vel_w[:, 2] / target_upward_speed,
      min=0.0,
      max=1.0,
    )
    powered = active & grounded & legal
    impulse_gain = torch.clamp(upward_speed - self.peak_upward_speed, min=0.0)
    self.peak_upward_speed = torch.where(
      active,
      torch.maximum(
        self.peak_upward_speed,
        torch.where(powered, upward_speed, torch.zeros_like(upward_speed)),
      ),
      torch.zeros_like(self.peak_upward_speed),
    )
    valid = active & wheel_free & legal
    self.current_duration = torch.where(
      valid,
      self.current_duration + env.step_dt,
      torch.zeros_like(self.current_duration),
    )
    duration = torch.clamp(
      self.current_duration / target_duration,
      min=0.0,
      max=1.0,
    )
    gain = torch.clamp(duration - self.peak_duration, min=0.0)
    self.peak_duration = torch.where(
      active,
      torch.maximum(
        self.peak_duration,
        torch.where(valid, duration, torch.zeros_like(duration)),
      ),
      torch.zeros_like(self.peak_duration),
    )
    self.previous_active = active
    # RewardManager supplies the sole dt integral.  Each high-water gain is a
    # one-off physical event rather than a per-step shaping loop.
    return (
      0.5 * powered.to(impulse_gain.dtype) * impulse_gain
      + 0.5 * valid.to(gain.dtype) * gain
    ) / env.step_dt


class AerialNetRotationProgress:
  """Pay only new *fraction* of one desired-axis turn in a ballistic event.

  ``AerialRotationCommand`` already integrates signed angular displacement only
  during its first continuous wheel-free interval.  This term pays a radian
  once, when that integration reaches a new high-water mark.  A policy cannot
  collect extra reward by briefly turning forward, undoing the turn, then
  repeating it.  No pose, phase, desired rate, or joint state is introduced.

  The command-side increment is already in radians (and therefore already
  contains one ``dt``).  Normalize it by the requested one-turn angle before
  returning its per-second form, so this result has a bounded value of one per
  completed event.  Paying raw radians made a 0.6-turn crash worth several
  times more than the entire recovery outcome, regardless of its bad landing.
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
  ) -> torch.Tensor:
    if target_angle <= 0.0:
      raise ValueError("target_angle must be positive.")
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
    return (
      active.to(increment.dtype)
      * legal.to(increment.dtype)
      * increment / target_angle
      / env.step_dt
    )


def aerial_event_failure(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_angle: float,
  non_timeout_base_cost: float = 0.0,
) -> torch.Tensor:
  """Apply one terminal cost proportional to the missing requested turn.

  A fixed failure cost taught the first corrected aerial run to suppress its
  launch: every intermediate 0.1--0.9 turn was equally bad as a zero-turn
  fall.  The only physical quantity that should reduce this cost is the
  measured desired-axis angle already used by the endpoint.  Thus a partial
  landing is still a failure, but each additional correct radian improves its
  return and PPO has a continuous route to the one-turn completion bonus.
  """
  if target_angle <= 0.0 or not 0.0 <= non_timeout_base_cost <= 1.0:
    raise ValueError("target_angle must be positive and failure cost must be in [0, 1].")
  command_term = env.command_manager.get_term(command_name)
  progress = getattr(
    command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
  )
  missing_fraction = 1.0 - torch.clamp(progress / target_angle, min=0.0, max=1.0)
  # RewardManager supplies the sole dt integral; this is one outcome cost.
  # Angle loss alone is insufficient near 2π: it made a fast trunk/leg crash
  # nearly free once a policy had learned to rotate.  A non-timeout terminal
  # event is physically invalid even at the requested angle.  A genuine
  # completed event uses the dedicated timeout boundary and has no base cost.
  invalid_terminal = env.termination_manager.terminated.to(missing_fraction.dtype)
  failure = missing_fraction + non_timeout_base_cost * invalid_terminal
  return (
    env.termination_manager.dones.to(missing_fraction.dtype)
    * failure
    / env.step_dt
  )


class AerialRotationCompletion:
  """Reward the one legal, quiet four-wheel outcome of an aerial event.

  Launch and desired-axis rotation have direct dense terms.  The only dense
  part here is the whole-body orientation return in the final part of that
  same legal flight.  It resolves the real ambiguity of a 2π *axis integral*:
  the body can still have accumulated unwanted off-axis rotation.  There is
  no braking clock, angular-rate target, joint pose, or reference trajectory.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.settle_time = torch.zeros(env.num_envs, device=env.device)
    self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.peak_orientation_return = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.settle_time[env_ids] = 0.0
    self.awarded[env_ids] = False
    self.peak_orientation_return[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    target_angle: float = math.tau,
    max_overrotation: float = 0.50,
    landing_gravity_std: float = 0.30,
    landing_orientation_dot_min: float = 0.985,
    landing_linear_velocity_limit: float = 0.75,
    landing_angular_velocity_limit: float = 1.5,
    post_idle_settle_time: float = 0.30,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if target_angle <= 0.0 or max_overrotation <= 0.0:
      raise ValueError("target_angle and max_overrotation must be positive.")
    if landing_gravity_std <= 0.0 or post_idle_settle_time <= 0.0:
      raise ValueError("landing_gravity_std and post_idle_settle_time must be positive.")
    if not 0.0 < landing_orientation_dot_min <= 1.0:
      raise ValueError("landing_orientation_dot_min must be in (0, 1].")
    if landing_linear_velocity_limit <= 0.0 or landing_angular_velocity_limit <= 0.0:
      raise ValueError("landing velocity limits must be positive.")
    asset: Entity = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    landing_started = getattr(
      command_term, "_landing_started", torch.zeros_like(active)
    )
    post_landing_idle = (~active) & landing_started
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    was_airborne = getattr(
      command_term, "was_airborne", torch.zeros_like(active)
    )
    launch_quat = getattr(
      command_term,
      "_launch_root_quat_w",
      torch.zeros_like(asset.data.root_link_quat_w),
    )
    contacts = _wheel_contacts(env, sensor_name)
    legal = ~_has_any_contact(env, nonwheel_sensor_name)
    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0), dtype=asset.data.projected_gravity_b.dtype, device=env.device
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    orientation_similarity = torch.abs(
      torch.sum(asset.data.root_link_quat_w * launch_quat, dim=1)
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)

    # Reward a legal late-flight return of the *whole* base frame, rather than
    # prescribing how any leg must brake.  The fourth power keeps the trivial
    # launch orientation from being rewarded and leaves the direct desired-axis
    # progress term responsible for the first part of the maneuver.  A
    # high-water increment makes this a bounded event signal, not a reward for
    # holding a particular airborne pose.
    flight_candidate = (
      active
      & was_airborne
      & (~landing_started)
      & (~torch.any(contacts, dim=1))
      & legal
      & (progress <= target_angle + max_overrotation)
    )
    turn_fraction = torch.clamp(progress / target_angle, min=0.0, max=1.0)
    orientation_return = torch.pow(turn_fraction, 4) * orientation_similarity
    orientation_return = torch.where(
      flight_candidate, orientation_return, torch.zeros_like(orientation_return)
    )
    orientation_gain = torch.clamp(
      orientation_return - self.peak_orientation_return, min=0.0
    )
    self.peak_orientation_return = torch.where(
      active,
      torch.maximum(self.peak_orientation_return, orientation_return),
      torch.zeros_like(self.peak_orientation_return),
    )

    stable = (
      post_landing_idle
      & was_airborne
      & (progress >= target_angle)
      & (progress <= target_angle + max_overrotation)
      & torch.all(contacts, dim=1)
      & legal
      & (gravity_error < landing_gravity_std)
      & (orientation_similarity >= landing_orientation_dot_min)
      & (linear_speed < landing_linear_velocity_limit)
      & (angular_speed < landing_angular_velocity_limit)
    )
    self.settle_time = torch.where(
      stable, self.settle_time + env.step_dt, torch.zeros_like(self.settle_time)
    )
    completed = stable & (
      self.settle_time + 0.5 * env.step_dt >= post_idle_settle_time
    )
    new_completion = completed & (~self.awarded)
    self.awarded |= completed
    return (orientation_gain + new_completion.to(orientation_gain.dtype)) / env.step_dt
