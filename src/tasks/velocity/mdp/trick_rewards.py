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
    clearance = torch.clamp(
      (asset.data.root_link_pos_w[:, 2] - support_height) / clearance_target[mode],
      min=0.0,
      max=1.0,
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
  Entity, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
  """Measure a commanded front/rear coaxial world-down pivot."""
  if rate_std <= 0.0:
    raise ValueError("rate_std must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
    raise ValueError("stance-spin measurement needs four wheel sites.")

  command = _command(env, command_name)
  mode = torch.argmax(command[:, :5], dim=1)
  active = (torch.sum(command[:, :5], dim=1) > 0.5) & (mode > 0)
  # Only front/rear supports have a horizontal transverse wheel axle when
  # their commanded body axis is vertical.  A zero rate is four-wheel idle,
  # not a request to hold a two-wheel support.
  moving = (
    active
    & (mode <= 2)
    & (torch.abs(command[:, 5]) > speed_deadband)
  )
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
  desired = (contacts * target).sum(dim=1) / 2.0
  extra = (contacts * (1.0 - target)).sum(dim=1) / 2.0
  contact_score = desired * (1.0 - 0.75 * extra)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
  )
  pair_ids = torch.tensor(((0, 1), (0, 1), (2, 3), (0, 2), (1, 3)), device=env.device)
  support_pair = pair_ids[mode]
  batch = torch.arange(env.num_envs, device=env.device)
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  fixed_height = 0.5 * (
    wheel_height[batch, support_pair[:, 0]] + wheel_height[batch, support_pair[:, 1]]
  )
  height_score = torch.clamp(
    (asset.data.root_link_pos_w[:, 2] - fixed_height) / 0.35,
    min=0.0,
    max=1.0,
  )

  # Wheel-joint local Y is the cylinder axle.  Wheel spin itself is a
  # rotation about that axis, so transforming this basis through each site
  # pose gives a stable physical axle direction.  A true balancing axle has
  # parallel wheel axes whose centre-to-centre line is along the same axis;
  # this is the missing geometry that distinguishes an in-place pivot from a
  # two-wheel circle on the floor.
  wheel_quat = asset.data.site_quat_w[:, asset_cfg.site_ids].reshape(-1, 4)
  local_axle = torch.tensor(
    (0.0, 1.0, 0.0), dtype=wheel_quat.dtype, device=env.device
  ).expand(wheel_quat.shape[0], -1)
  wheel_axles = quat_apply(wheel_quat, local_axle).reshape(env.num_envs, 4, 3)
  wheel_positions = asset.data.site_pos_w[:, asset_cfg.site_ids]
  axle_a = wheel_axles[batch, support_pair[:, 0]]
  axle_b = wheel_axles[batch, support_pair[:, 1]]
  axis_parallel = torch.abs(torch.sum(axle_a * axle_b, dim=1))
  centre_line = (
    wheel_positions[batch, support_pair[:, 1]]
    - wheel_positions[batch, support_pair[:, 0]]
  )
  centre_line = torch.nn.functional.normalize(centre_line, dim=1)
  line_on_axle = torch.abs(torch.sum(centre_line * axle_a, dim=1))
  horizontal_axle = torch.linalg.vector_norm(axle_a[:, :2], dim=1)
  coaxiality = axis_parallel * line_on_axle * horizontal_axle
  # The small nonzero baseline keeps the physical contact/attitude discovery
  # gradient alive before the front/rear pair has achieved exact co-linearity.
  # Full rate return nevertheless requires the measured coaxial geometry.
  # V45 demonstrated that a linear attitude factor leaves a lucrative local
  # optimum: the named wheels become coaxial and spin in place while the
  # trunk remains roughly half-way to its requested vertical axis.  Squaring
  # the same measured attitude keeps a discovery signal from four-wheel
  # default (0.5 -> 0.25) but makes a 0.60 low crouch materially worse than
  # the actual tall two-wheel support (1.0).  This is still an outcome
  # measurement, not a pose target.
  support_quality = (
    contact_score * torch.square(alignment) * height_score * (0.15 + 0.85 * coaxiality)
  )
  return asset, moving, rate_score, support_quality, support_pair


class StanceSpinPivotResult:
  """Reward a high-rate front/rear coaxial pivot.

  The supporting wheel centres, rather than root velocity, identify the
  physical pivot.  This is instantaneous measured geometry: no anchor,
  transition clock, reference path, or limb trajectory is retained in state.
  A bicycle-like support translation is explicitly worse than no spin.  The
  zero-speed branch is handled by default-idle action gating, so this term is
  active only for the nonzero front/rear rate request.
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
  ) -> torch.Tensor:
    if pivot_speed_limit <= 0.0:
      raise ValueError("pivot_speed_limit must be positive.")
    asset, moving, rate_score, support_quality, pair = _stance_spin_components(
      env,
      command_name,
      speed_deadband,
      std,
      gravity_targets,
      contact_masks,
      sensor_name,
      asset_cfg,
    )
    batch = torch.arange(env.num_envs, device=env.device)
    wheel_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    centre_velocity = 0.5 * (
      wheel_velocity[batch, pair[:, 0]] + wheel_velocity[batch, pair[:, 1]]
    )
    centre_speed = torch.linalg.vector_norm(centre_velocity, dim=1)
    translation_cost = torch.clamp(centre_speed / pivot_speed_limit, min=0.0, max=2.0)
    # Before a tall support exists, random wheel velocity is part of finding
    # the transition and must not overwhelm the small support-improvement
    # signal.  Once support is high, the squared gate reaches one, so the
    # identical physical centre-speed penalty rejects bicycle translation.
    support_and_rate = support_quality * (0.35 + 0.65 * rate_score)
    translation_penalty = torch.square(support_quality) * translation_cost
    dynamic_result = support_and_rate - translation_penalty
    return moving.to(rate_score.dtype) * dynamic_result


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


def stance_locomotion_linear_velocity_exp(
  env: ManagerBasedRlEnv,
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
  env: ManagerBasedRlEnv,
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


class AerialManeuverResultProgress:
  """Bounded result potential for a ballistic turn followed by recovery.

  Flight receives value only when wheel-free clearance and signed angular
  progress coexist.  The remaining value is available only to a near-full
  turn that returns upright, low-momentum wheel contact.  This gives PPO a
  continuous outcome signal without prescribing a limb trajectory or a phase
  schedule, and without allowing height or a partial flip to be farmed alone.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.previous_score = torch.zeros(env.num_envs, device=env.device)
    self.peak_clearance = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.previous_mode = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.previous_score[env_ids] = 0.0
    self.peak_clearance[env_ids] = 0.0
    self.previous_active[env_ids] = False
    self.previous_mode[env_ids] = -1

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    target_angle: float,
    target_clearance: float,
    landing_turn_start: float,
    recovery_linear_speed_scale: float,
    recovery_angular_speed_scale: float,
    potential_discount: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if (
      target_angle <= 0.0
      or target_clearance <= 0.0
      or not 0.0 <= landing_turn_start < target_angle
      or recovery_linear_speed_scale <= 0.0
      or recovery_angular_speed_scale <= 0.0
      or not 0.0 < potential_discount <= 1.0
    ):
      raise ValueError("invalid aerial result parameters.")

    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    mode = torch.argmax(command[:, :5], dim=1)
    reset = env.episode_length_buf == 0
    new_skill = active & (
      (~self.previous_active) | (mode != self.previous_mode) | reset
    )
    # Do not erase ``previous_score`` merely because the one-shot command
    # returned to idle: the first idle reward must settle the final potential
    # difference of a failed attempt.  Otherwise a bad jump could keep its
    # pre-landing score for free.  Subsequent idle frames naturally return
    # zero after this one settlement.
    clear = reset | new_skill
    self.previous_score[clear] = 0.0
    self.peak_clearance[clear] = 0.0

    command_term = env.command_manager.get_term(command_name)
    progress = getattr(
      command_term,
      "_rotation_progress",
      torch.zeros(env.num_envs, device=env.device),
    )
    was_airborne = getattr(command_term, "was_airborne", torch.zeros_like(active))
    landing_started = getattr(
      command_term, "_landing_started", torch.zeros_like(active)
    )
    contacts = _wheel_contacts(env, sensor_name)
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    height = torch.clamp(
      (asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]) / target_clearance,
      min=0.0,
      max=1.0,
    )
    airborne = ~torch.any(contacts, dim=1)
    # A full turn normally finishes while descending.  V77 used the current
    # height, so correct late angular progress could lower the potential after
    # apex and earn no return.  Keep the best wheel-free clearance achieved in
    # this attempt: high launch remains necessary, while later radians keep
    # improving the same physical result.
    self.peak_clearance = torch.maximum(
      self.peak_clearance, airborne.to(height.dtype) * height
    )
    # A partial turn is worth proportionally less than one turn.  Past the
    # target, however, the previous symmetric ramp decayed over another full
    # revolution, so 1.3--1.6 turns still retained most flight return once the
    # terminal guard was removed.  Collapse the same result potential within
    # the final 45-degree braking window instead: it rewards arriving at one
    # turn with low residual rate, while leaving the remainder of the episode
    # available for a failed attempt to recover normally.
    braking_window = target_angle / 8.0
    turn_before_target = torch.clamp(progress / target_angle, min=0.0, max=1.0)
    turn_after_target = torch.clamp(
      1.0 - (progress - target_angle) / braking_window, min=0.0, max=1.0
    )
    turn = torch.where(progress <= target_angle, turn_before_target, turn_after_target)
    # After the first touchdown this event is closed.  Do not let a later
    # bounce receive the original flight score as though it were another
    # commanded maneuver.
    flight = (
      was_airborne.float()
      * (~landing_started).to(height.dtype)
      * torch.sqrt(self.peak_clearance)
      * turn
    )

    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    upright = torch.clamp(
      1.0
      - torch.linalg.vector_norm(asset.data.projected_gravity_b - normal_gravity, dim=1)
      / 2.0,
      min=0.0,
      max=1.0,
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    # These are deliberately broader than the completion event below.  The
    # strict 0.75 m/s and 1.5-rad/s event threshold is an all-or-nothing
    # verifier; using it here made every first touchdown after a real flip
    # score exactly zero, leaving PPO no braking gradient.  This *same*
    # result potential instead ranks progressively quieter recoveries, while
    # the separate one-shot event still certifies only a quiet four-wheel
    # landing.
    linear_settled = torch.clamp(
      1.0 - linear_speed / recovery_linear_speed_scale, min=0.0, max=1.0
    )
    total_angular_settled = torch.clamp(
      1.0 - angular_speed / recovery_angular_speed_scale, min=0.0, max=1.0
    )
    # A full revolution that is still spinning rapidly cannot make the quiet
    # four-wheel landing.  The former recovery score first required contact,
    # so PPO received no directional signal to brake during the last airborne
    # quadrant: it repeatedly learned a one-turn crash.  This is one outcome
    # measurement, not a phase target—after a nearly upright full turn the
    # commanded-axis speed itself must already be reducing, whether touchdown
    # occurs on this frame or the next.
    launch_axis_w = getattr(
      command_term,
      "_launch_axis_w",
      torch.zeros(env.num_envs, 3, device=env.device),
    )
    axis_speed = torch.abs(
      torch.sum(asset.data.root_link_ang_vel_w * launch_axis_w, dim=1)
    )
    axis_settled = torch.clamp(
      1.0 - axis_speed / recovery_angular_speed_scale, min=0.0, max=1.0
    )
    landing_turn = torch.clamp(
      (progress - landing_turn_start) / (target_angle - landing_turn_start),
      min=0.0,
      max=1.0,
    )
    landing = (
      was_airborne.float()
      * landing_turn
      * turn
      * upright
      * linear_settled
      * axis_settled
      * (0.5 + 0.5 * total_angular_settled)
      # Airborne braking earns a small part of the same recovery potential;
      # the larger share arrives only with actual wheel contact and continues
      # to rank a quiet four-wheel touchdown above a soft one/two-wheel graze.
      * (0.25 + 0.75 * contacts.float().mean(dim=1))
    )
    # A complete maneuver must rank well above the already-discovered
    # one-turn crash.  Both components remain inside one bounded potential:
    # wheel-free height/turn gets the policy to the landing, and four-wheel
    # upright recovery supplies the larger final return.
    # An illegal non-wheel contact is a true terminal state, not a temporary
    # interruption of the maneuver.  The absorbing terminal state must have
    # zero potential: otherwise a robot can retain a high airborne-turn
    # potential by crashing before the next frame gets to settle it.  That
    # would violate the same discounted-potential argument used below and
    # makes a fast body-first collision a profitable local optimum.  Time
    # limits deliberately remain bootstrap-able truncations; only failures
    # erase the result potential.
    terminated = env.termination_manager.terminated
    alive = (~terminated).to(asset.data.root_link_pos_w.dtype)
    score = active.float() * (0.40 * flight + 0.60 * landing) * alive
    # Use the *discount-correct* potential difference.  PPO discounts returns
    # by ``gamma``: with a plain ``score - previous_score``, a policy can
    # collect the rise to a one-turn airborne score early, then lose it later
    # while over-rotating, yet retain a positive discounted return.  That was
    # exactly the high-rate crash local optimum in V85--V87.  The shaping
    # theorem instead requires ``gamma * Phi(next) - Phi(current)``.  This
    # keeps every dense physical discovery signal but makes an unlanded
    # intermediate result cancel out under the same PPO discount, leaving the
    # strict landing event as the only durable outcome advantage.
    #
    # A pure potential difference has one blind spot: after first touching
    # down in an almost-good result its value is exactly zero on every
    # subsequent frame.  The event verifier requires five consecutive
    # 50-Hz frames, yet PPO was paid identically for immediately launching a
    # second flip and for holding the recovery.  The command term now closes
    # the event at the first landing, so this same-result holding value is
    # available only during that one short landing decision window.
    stable_touchdown = (
      torch.all(contacts, dim=1).to(score.dtype)
      * landing_turn
      * turn
      * upright
      * linear_settled
      * axis_settled
      * (0.5 + 0.5 * total_angular_settled)
    )
    previous_score = self.previous_score
    self.previous_score = score
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    return (
      potential_discount * score - previous_score
    ) / env.step_dt + 0.25 * active.float() * stable_touchdown


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
  """One-shot reward for the strict full-turn, four-wheel landing event."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.progress = torch.zeros(env.num_envs, device=env.device)
    self.was_airborne = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.has_grounded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.previous_mode = torch.full(
      (env.num_envs,), -1, dtype=torch.long, device=env.device
    )
    self.landing_settle_time = torch.zeros(env.num_envs, device=env.device)
    self.launch_axis_w = torch.zeros(env.num_envs, 3, device=env.device)
    self.airborne_time = torch.zeros(env.num_envs, device=env.device)
    self.flight_qualified = torch.zeros(
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
    self.flight_qualified[env_ids] = False
    self.flight_rotation[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
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
    # Match ``AerialRotationCommand`` exactly: five 20-ms stable control
    # frames are the requested 0.10 s even when float32 summation represents
    # their total just below the decimal threshold.
    settled_long_enough = (
      self.landing_settle_time + 0.5 * env.step_dt >= landing_settle_time
    )
    completed = (
      stable_landing
      & (self.progress >= target_angle)
      & (self.progress <= target_angle + max_overrotation)
      & settled_long_enough
    )
    reward = completed & (~self.awarded)
    self.awarded |= completed
    self.previous_active = active
    self.previous_mode = torch.where(active, mode, self.previous_mode)
    return reward.float() / env.step_dt


class AerialFirstLandingResult:
  """Score exactly one physical result after an aerial command.

  This deliberately evaluates only the first post-flight landing window.  It
  gives PPO a graded outcome for a safe partial turn, but no reward for a
  second hop: the command term closes the event immediately afterwards.  No
  joint posture, phase, or demonstration reference is involved.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.peak_clearance = torch.zeros(env.num_envs, device=env.device)
    self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_clearance[env_ids] = 0.0
    self.awarded[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    target_angle: float,
    max_overrotation: float,
    target_clearance: float,
    landing_window_s: float,
    recovery_linear_speed_scale: float,
    recovery_angular_speed_scale: float,
    landing_gravity_error_limit: float,
    landing_settle_time: float,
    landing_linear_velocity_limit: float,
    landing_angular_velocity_limit: float,
    strict_completion_bonus: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    """Return an event reward at the end of the first landing window."""
    if (
      target_angle <= 0.0
      or max_overrotation <= 0.0
      or target_clearance <= 0.0
      or landing_window_s <= 0.0
      or recovery_linear_speed_scale <= 0.0
      or recovery_angular_speed_scale <= 0.0
      or landing_gravity_error_limit <= 0.0
      or landing_settle_time <= 0.0
      or landing_linear_velocity_limit <= 0.0
      or landing_angular_velocity_limit <= 0.0
      or strict_completion_bonus < 0.0
    ):
      raise ValueError("invalid aerial first-landing result parameters.")

    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    command_term = env.command_manager.get_term(command_name)
    contacts = _wheel_contacts(env, sensor_name)
    airborne = ~torch.any(contacts, dim=1)
    default_root_state = asset.data.default_root_state
    assert default_root_state is not None
    clearance = torch.clamp(
      (asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2])
      / target_clearance,
      min=0.0,
      max=1.0,
    )
    landing_started = getattr(
      command_term, "_landing_started", torch.zeros_like(active)
    )
    self.peak_clearance = torch.maximum(
      self.peak_clearance,
      active.to(clearance.dtype) * (~landing_started).to(clearance.dtype) * airborne.to(clearance.dtype) * clearance,
    )

    # RewardManager evaluates before CommandManager advances this control
    # step.  Include that pending step so this fires on the same frame on
    # which the command state machine clears the one-hot.
    landing_hold_time = getattr(
      command_term, "_landing_hold_time", torch.zeros_like(self.peak_clearance)
    )
    closing = active & landing_started & (
      landing_hold_time + 1.5 * env.step_dt >= landing_window_s
    )
    award = closing & ~self.awarded
    self.awarded |= closing

    progress = getattr(
      command_term, "_rotation_progress", torch.zeros_like(self.peak_clearance)
    )
    turn_before_target = torch.clamp(progress / target_angle, min=0.0, max=1.0)
    turn_after_target = torch.clamp(
      1.0 - (progress - target_angle) / max_overrotation, min=0.0, max=1.0
    )
    turn_quality = torch.where(progress <= target_angle, turn_before_target, turn_after_target)
    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=env.device,
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    upright = torch.clamp(1.0 - torch.sqrt(gravity_error) / 2.0, min=0.0, max=1.0)
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    linear_settled = torch.clamp(
      1.0 - linear_speed / recovery_linear_speed_scale, min=0.0, max=1.0
    )
    angular_settled = torch.clamp(
      1.0 - angular_speed / recovery_angular_speed_scale, min=0.0, max=1.0
    )
    # A wheel graze followed by a body impact is not a partial success.  The
    # graded signal is intentionally available only on simultaneous normal
    # four-wheel support and a non-terminal physics state; turn/height then
    # rank *safe* partial landings without rewarding a crash on its first
    # wheel contact.
    normal_wheel_support = torch.all(contacts, dim=1)
    alive = (~env.termination_manager.terminated).to(clearance.dtype)
    landing_quality = (upright + linear_settled + angular_settled) / 3.0
    graded_result = (
      getattr(command_term, "was_airborne", torch.zeros_like(active)).float()
      * torch.sqrt(self.peak_clearance)
      * turn_quality
      * (0.20 + 0.80 * landing_quality)
      * normal_wheel_support.to(clearance.dtype)
      * alive
    )
    stable = (
      normal_wheel_support
      & (gravity_error < landing_gravity_error_limit)
      & (linear_speed < landing_linear_velocity_limit)
      & (angular_speed < landing_angular_velocity_limit)
    )
    settle_time = getattr(
      command_term, "_landing_settle_time", torch.zeros_like(self.peak_clearance)
    )
    strict_completion = (
      stable
      & (settle_time + 1.5 * env.step_dt >= landing_settle_time)
      & (progress >= target_angle)
      & (progress <= target_angle + max_overrotation)
    )
    result = graded_result + strict_completion.float() * strict_completion_bonus
    return award.to(result.dtype) * result / env.step_dt
