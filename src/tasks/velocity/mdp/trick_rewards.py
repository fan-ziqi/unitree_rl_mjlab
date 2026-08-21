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


def _command(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  return command


def _mode_mask(
  env: "ManagerBasedRlEnv",
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


def _wheel_contacts(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1)


# ---------------------------------------------------------------------------
# Shared two-wheel support measurement.


def mode_support_score(
  env: "ManagerBasedRlEnv",
  command_name: str,
  modes: tuple[int, ...],
  gravity_targets: tuple[tuple[float, float, float], ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  sensor_name: str,
  num_modes: int = 5,
  extra_contact_discount: float = 0.75,
  minimum_root_clearance: float | None = None,
  stationary_command_index: int | None = None,
  command_deadband: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Measure the commanded contact pair, attitude, and optional height.

  This is an outcome score rather than a leg-pose target.  Multiplication
  makes a low crouch or a fallen contact pair inferior to a genuine support;
  the partial contact factors retain a discovery gradient from four wheels.
  """
  if not 0.0 <= extra_contact_discount <= 1.0:
    raise ValueError("extra_contact_discount must be in [0, 1].")
  if minimum_root_clearance is not None and minimum_root_clearance <= 0.0:
    raise ValueError("minimum_root_clearance must be positive.")
  if stationary_command_index is not None and command_deadband < 0.0:
    raise ValueError("command_deadband must be non-negative.")

  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  if stationary_command_index is not None:
    if not 0 <= stationary_command_index < command.shape[1]:
      raise ValueError("stationary_command_index is outside the command tensor.")
    active &= torch.abs(command[:, stationary_command_index]) <= command_deadband

  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  orientation = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
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
    support_height = (wheel_height * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
    clearance = torch.clamp(
      (asset.data.root_link_pos_w[:, 2] - support_height) / minimum_root_clearance,
      min=0.0,
      max=1.0,
    )
  return active.to(orientation.dtype) * orientation * support * clearance


# ---------------------------------------------------------------------------
# Five-one-hot spin / two-wheel-pivot task.


def _dynamic_tall_pair_scores(
  env: "ManagerBasedRlEnv",
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Return support quality and its selected front/rear wheel pair.

  A moving normal command can use either transverse pair.  The outcome says
  only that a pair supports a tall body; it does not choose a joint posture or
  a transition path.
  """
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
    raise ValueError("dynamic support needs FL/FR/RL/RR wheel sites.")
  contacts = _wheel_contacts(env, sensor_name).float()
  pair_masks = torch.tensor(
    ((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)),
    dtype=contacts.dtype,
    device=env.device,
  )
  desired = (contacts.unsqueeze(1) * pair_masks).sum(dim=2) / 2.0
  extra = (contacts.unsqueeze(1) * (1.0 - pair_masks)).sum(dim=2) / 2.0
  contact_score = desired * (1.0 - 0.75 * extra)
  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  targets = torch.tensor(
    ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    dtype=gravity.dtype,
    device=env.device,
  )
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity.unsqueeze(1) * targets, dim=2)), 0.0, 1.0
  )
  pair_ids = torch.tensor(((0, 1), (2, 3)), device=env.device)
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  pair_height = 0.5 * (
    wheel_height[:, pair_ids[:, 0]] + wheel_height[:, pair_ids[:, 1]]
  )
  root_clearance = asset.data.root_link_pos_w[:, 2].unsqueeze(1) - pair_height
  height_score = torch.clamp(root_clearance / 0.35, min=0.0, max=1.0)
  scores = contact_score * alignment * height_score
  return torch.max(scores, dim=1)


def _stance_spin_components(
  env: "ManagerBasedRlEnv",
  command_name: str,
  speed_deadband: float,
  rate_std: float,
  gravity_targets: tuple[tuple[float, float, float], ...],
  sensor_name: str,
  asset_cfg: SceneEntityCfg,
) -> tuple[Entity, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
  """Measure requested yaw-about-world-down and selected support geometry."""
  if rate_std <= 0.0:
    raise ValueError("rate_std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
    raise ValueError("stance-spin measurement needs four wheel sites.")

  command = _command(env, command_name)
  mode = torch.argmax(command[:, :5], dim=1)
  moving = (mode <= 2) & (torch.abs(command[:, 5]) > speed_deadband)
  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * gravity, dim=1)
  rate_score = torch.clamp(
    1.0 - torch.abs(command[:, 5] - actual_rate) / rate_std,
    min=0.0,
    max=1.0,
  )

  dynamic_quality, dynamic_pair = _dynamic_tall_pair_scores(env, sensor_name, asset_cfg)
  pair_masks = torch.tensor(
    ((1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0)),
    dtype=gravity.dtype,
    device=env.device,
  )
  fixed_pair = torch.clamp(mode - 1, 0, 1)
  contacts = _wheel_contacts(env, sensor_name).float()
  target = pair_masks[fixed_pair]
  desired = (contacts * target).sum(dim=1) / 2.0
  extra = (contacts * (1.0 - target)).sum(dim=1) / 2.0
  contact_score = desired * (1.0 - 0.75 * extra)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
  )
  pair_ids = torch.tensor(((0, 1), (2, 3)), device=env.device)
  batch = torch.arange(env.num_envs, device=env.device)
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  fixed_height = 0.5 * (
    wheel_height[batch, pair_ids[fixed_pair, 0]]
    + wheel_height[batch, pair_ids[fixed_pair, 1]]
  )
  height_score = torch.clamp(
    (asset.data.root_link_pos_w[:, 2] - fixed_height) / 0.35,
    min=0.0,
    max=1.0,
  )
  fixed_quality = contact_score * alignment * height_score
  support_quality = torch.where(mode == 0, dynamic_quality, fixed_quality)
  support_pair = torch.where(mode == 0, dynamic_pair, fixed_pair)
  return asset, moving, rate_score, support_quality, support_pair


class StanceSpinPivotResult:
  """Reward high-rate rotation of a high support whose centre stays local.

  The supporting wheel centres, rather than root velocity, identify the
  physical pivot.  This is instantaneous measured geometry: no anchor,
  transition clock, reference path, or limb trajectory is retained in state.
  A bicycle-like support translation is explicitly worse than no spin.  A
  small support baseline is part of this same result so PPO can discover the
  handstand from four wheels; it is intentionally much lower than a correct
  stationary fast pivot.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg, env

  def reset(self, env_ids: torch.Tensor) -> None:
    del env_ids

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    speed_deadband: float,
    std: float,
    gravity_targets: tuple[tuple[float, float, float], ...],
    sensor_name: str,
    pivot_speed_limit: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    if pivot_speed_limit <= 0.0:
      raise ValueError("pivot_speed_limit must be positive.")
    asset, moving, rate_score, support_quality, pair = _stance_spin_components(
      env,
      command_name,
      speed_deadband,
      std,
      gravity_targets,
      sensor_name,
      asset_cfg,
    )
    pair_ids = torch.tensor(((0, 1), (2, 3)), device=env.device)
    batch = torch.arange(env.num_envs, device=env.device)
    wheel_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    centre_velocity = 0.5 * (
      wheel_velocity[batch, pair_ids[pair, 0]]
      + wheel_velocity[batch, pair_ids[pair, 1]]
    )
    centre_speed = torch.linalg.vector_norm(centre_velocity, dim=1)
    translation_cost = torch.clamp(
      centre_speed / pivot_speed_limit, min=0.0, max=2.0
    )
    # Before a tall support exists, random wheel velocity is part of finding
    # the transition and must not overwhelm the small support-improvement
    # signal.  Once support is high, the squared gate reaches one, so the
    # identical physical centre-speed penalty rejects bicycle translation.
    support_and_rate = support_quality * (0.35 + 0.65 * rate_score)
    translation_penalty = torch.square(support_quality) * translation_cost
    return moving.to(rate_score.dtype) * (support_and_rate - translation_penalty)


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
  env: "ManagerBasedRlEnv",
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


def stance_locomotion_linear_velocity_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  lateral_weight: float = 2.0,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track requested forward velocity while holding lateral velocity at zero."""
  if std <= 0.0 or lateral_weight < 0.0:
    raise ValueError("std must be positive and lateral_weight non-negative.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :3], dim=1)
  forward, right = _stance_locomotion_axes(asset, mode)
  velocity_xy = asset.data.root_link_lin_vel_w[:, :2]
  forward_speed = torch.sum(velocity_xy * forward, dim=1)
  lateral_speed = torch.sum(velocity_xy * right, dim=1)
  error = torch.square(command[:, 3] - forward_speed) + lateral_weight * torch.square(
    lateral_speed
  )
  return torch.exp(-error / std**2) * _locomotion_alignment(
    env, asset, mode, gravity_targets, gravity_power
  )


def stance_locomotion_yaw_rate_exp(
  env: "ManagerBasedRlEnv",
  command_name: str,
  std: float,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  gravity_power: float = 0.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track world-up yaw rate in all three commanded support modes."""
  if std <= 0.0:
    raise ValueError("std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  command = _command(env, command_name)
  mode = torch.argmax(command[:, :3], dim=1)
  error = command[:, 4] - asset.data.root_link_ang_vel_w[:, 2]
  return torch.exp(-torch.square(error) / std**2) * _locomotion_alignment(
    env, asset, mode, gravity_targets, gravity_power
  )


# ---------------------------------------------------------------------------
# Five-one-hot aerial-rotation task.


def aerial_active(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  return (torch.sum(_command(env, command_name)[:, :5], dim=1) > 0.5).float()


def aerial_rotation_overrun(
  env: "ManagerBasedRlEnv",
  command_name: str,
  target_angle: float,
  max_overrotation: float,
  activation_step: int = 0,
) -> torch.Tensor:
  """End an attempt only after its measured ballistic turn has overrun."""
  if target_angle <= 0.0 or max_overrotation <= 0.0 or activation_step < 0:
    raise ValueError("invalid aerial overrun parameters.")
  if env.common_step_counter < activation_step:
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
  command_term = env.command_manager.get_term(command_name)
  progress = getattr(
    command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
  )
  return aerial_active(env, command_name).bool() & (
    progress > target_angle + max_overrotation
  )


class AerialManeuverResultProgress:
  """Bounded result potential for a ballistic turn followed by recovery.

  Flight receives value only when wheel-free clearance and signed angular
  progress coexist.  The remaining value is available only to a near-full
  turn that returns upright, low-momentum wheel contact.  This gives PPO a
  continuous outcome signal without prescribing a limb trajectory or a phase
  schedule, and without allowing height or a partial flip to be farmed alone.
  """

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self.best_score = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.best_score[env_ids] = 0.0
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1

  def __call__(
    self,
    env: "ManagerBasedRlEnv",
    command_name: str,
    sensor_name: str,
    target_angle: float,
    target_clearance: float,
    landing_turn_start: float,
    landing_linear_velocity_limit: float,
    landing_angular_velocity_limit: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if (
      target_angle <= 0.0
      or target_clearance <= 0.0
      or not 0.0 <= landing_turn_start < target_angle
      or landing_linear_velocity_limit <= 0.0
      or landing_angular_velocity_limit <= 0.0
    ):
      raise ValueError("invalid aerial result parameters.")

    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & ((~self.previous_active) | (mode != self.previous_mode) | reset)
    self.best_score[reset | (~active) | new_skill] = 0.0

    command_term = env.command_manager.get_term(command_name)
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    was_airborne = getattr(command_term, "was_airborne", torch.zeros_like(active))
    contacts = _wheel_contacts(env, sensor_name)
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    height = torch.clamp(
      (asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]) / target_clearance,
      min=0.0,
      max=1.0,
    )
    turn = torch.clamp(progress / target_angle, min=0.0, max=1.0)
    # Preserve the height discovery gradient, but keep angular progress
    # linear: with sqrt(turn), a 0.7-turn bounce received 84% of the flight
    # value and V76 stopped improving before one full revolution.
    flight = (~torch.any(contacts, dim=1)).float() * torch.sqrt(height) * turn

    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    upright = torch.clamp(
      1.0
      - torch.linalg.vector_norm(
        asset.data.projected_gravity_b - normal_gravity, dim=1
      )
      / 2.0,
      min=0.0,
      max=1.0,
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    settled = torch.clamp(
      1.0 - linear_speed / landing_linear_velocity_limit, min=0.0, max=1.0
    ) * torch.clamp(
      1.0 - angular_speed / landing_angular_velocity_limit, min=0.0, max=1.0
    )
    landing_turn = torch.clamp(
      (progress - landing_turn_start) / (target_angle - landing_turn_start),
      min=0.0,
      max=1.0,
    )
    landing = (
      was_airborne.float()
      * landing_turn
      * contacts.float().mean(dim=1)
      * upright
      * settled
    )
    score = active.float() * (0.65 * flight + 0.35 * landing)
    old_best = self.best_score.clone()
    self.best_score = torch.maximum(self.best_score, score)
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    return (self.best_score - old_best) / env.step_dt


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
    flight_step
    & (~current_flight_qualified)
    & (airborne_time >= min_ballistic_time)
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
  return has_grounded, airborne_time, current_flight_qualified, flight_rotation, increment


class AerialRotationCompletion:
  """One-shot reward for the strict full-turn, four-wheel landing event."""

  def __init__(self, cfg: RewardTermCfg, env: "ManagerBasedRlEnv"):
    del cfg
    self.progress = torch.zeros(env.num_envs, device=env.device)
    self.was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.has_grounded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_mode = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.landing_settle_time = torch.zeros(env.num_envs, device=env.device)
    self.launch_axis_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.airborne_time = torch.zeros(env.num_envs, device=env.device)
    self.flight_qualified = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
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
    self.flight_qualified[env_ids] = False
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
    new_skill = active & ((~self.previous_active) | (mode != self.previous_mode) | reset)
    clear = new_skill | reset | (~active)
    self.progress[clear] = 0.0
    self.was_airborne[clear] = False
    self.has_grounded[clear] = False
    self.awarded[clear] = False
    self.landing_settle_time[clear] = 0.0
    self.airborne_time[clear] = 0.0
    self.flight_qualified[clear] = False
    self.flight_rotation[clear] = 0.0

    axes_b = torch.tensor(
      axes, dtype=asset.data.root_link_quat_w.dtype, device=env.device
    )[mode]
    self.launch_axis_w[new_skill] = quat_apply(
      asset.data.root_link_quat_w[new_skill], axes_b[new_skill]
    )
    axis_rate = torch.sum(asset.data.root_link_ang_vel_w * self.launch_axis_w, dim=1)
    contacts = _wheel_contacts(env, sensor_name)
    command_term = env.command_manager.get_term(command_name)
    (
      self.has_grounded,
      self.airborne_time,
      self.flight_qualified,
      self.flight_rotation,
      increment,
    ) = _advance_qualified_aerial_rotation(
      env=env,
      active=active,
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

    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    stable_landing = (
      active
      & self.was_airborne
      & torch.all(contacts, dim=1)
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
    return reward.float() / env.step_dt
