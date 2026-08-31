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


def normal_four_wheel_pivot_geometry(
  wheel_axles: torch.Tensor,
  wheel_positions: torch.Tensor,
  *,
  axle_line_radius: float = 0.10,
  compact_axle_span: float = 0.42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Measure the compact four-wheel common-axle form for normal spin.

  A non-zero ``normal`` rate is the AS2-W-style four-wheel pivot: the trunk
  remains level, all four wheels remain on the floor, and leg motion brings
  their cylinder axes onto one transverse line below the trunk.  The previous
  normal-mode measurement instead chose a tall front/rear two-wheel support.
  That made a visually plausible balancing pose, but it is a different trick:
  it leaves two wheels high above the support pair and cannot deliver the
  compact four-wheel spin requested here.

  This is deliberately a geometry *outcome*, not a joint target or a pose
  trajectory.  It rewards three measurable properties: parallel wheel axes,
  centres close to one common axle line, and a finite transverse envelope.
  Contact and level-trunk requirements remain in the caller.
  """
  if wheel_axles.ndim != 3 or wheel_axles.shape[1:] != (4, 3):
    raise ValueError("normal pivot geometry expects [batch, 4, 3] axes.")
  if wheel_positions.shape != wheel_axles.shape:
    raise ValueError("wheel positions must match wheel axle tensor shape.")
  if axle_line_radius <= 0.0 or compact_axle_span <= 0.0:
    raise ValueError("normal-pivot geometry scales must be positive.")

  axes = torch.nn.functional.normalize(wheel_axles, dim=2)
  # Wheel-cylinder axes have an arbitrary sign in the imported model, so use
  # one wheel as a line direction and compare with absolute dot products.
  # Averaging anti-parallel but co-axial cylinders would otherwise collapse.
  reference_axis = axes[:, 0]
  all_axis_parallel = torch.mean(
    torch.abs(torch.sum(axes * reference_axis.unsqueeze(1), dim=2)), dim=1
  )
  centre = wheel_positions.mean(dim=1, keepdim=True)
  relative = wheel_positions - centre
  along_axis = torch.sum(relative * reference_axis.unsqueeze(1), dim=2)
  off_axis = relative - along_axis.unsqueeze(2) * reference_axis.unsqueeze(1)
  max_off_axis = torch.amax(torch.linalg.vector_norm(off_axis, dim=2), dim=1)
  common_axle_line = 1.0 / (1.0 + torch.square(max_off_axis / axle_line_radius))
  axle_span = torch.amax(along_axis, dim=1) - torch.amin(along_axis, dim=1)
  # The video calls for a compact envelope, not collapse of all four wheel
  # centres into one colliding point.  Keep every physically usable span at
  # full value and penalize only a clearly splayed axle.
  span_excess = torch.clamp(axle_span - compact_axle_span, min=0.0)
  compact_span = 1.0 / (1.0 + torch.square(span_excess / 0.12))
  return all_axis_parallel, common_axle_line, compact_span


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
  mode_weights: tuple[float, ...] | None = None,
  clearance_power: float = 1.0,
  soft_support_height: float | None = None,
  soft_support_height_std: float = 0.06,
  soft_support_pair_height_std: float = 0.06,
  support_leg_length_target: float | None = None,
  support_body_names: tuple[str, ...] = ("FL_hip", "FR_hip", "RL_hip", "RR_hip"),
  stationary_command_index: int | None = None,
  static_command_start_index: int | None = None,
  command_deadband: float = 0.0,
  static_angular_velocity_scale: float | None = None,
  static_linear_velocity_scale: float | None = None,
  static_support_center_speed_scale: float | None = None,
  static_stillness_floor: float = 0.10,
  static_settling_alignment_threshold: float = 0.70,
  static_settling_support_threshold: float = 0.35,
  static_settling_clearance_threshold: float = 0.40,
  attitude_progress_weight: float = 0.0,
  attitude_progress_rate_scale: float = 1.0,
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
  if soft_support_height is not None and (
    soft_support_height < 0.0
    or soft_support_height_std <= 0.0
    or soft_support_pair_height_std <= 0.0
  ):
    raise ValueError("soft support height and scales must be non-negative/positive.")
  if support_leg_length_target is not None and support_leg_length_target <= 0.0:
    raise ValueError("support_leg_length_target must be positive when provided.")
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
  if mode_weights is not None and (
    len(mode_weights) != num_modes or any(weight < 0.0 for weight in mode_weights)
  ):
    raise ValueError("mode_weights must be non-negative and cover every mode.")
  if static_angular_velocity_scale is not None and static_angular_velocity_scale <= 0.0:
    raise ValueError("static_angular_velocity_scale must be positive.")
  if static_linear_velocity_scale is not None and static_linear_velocity_scale <= 0.0:
    raise ValueError("static_linear_velocity_scale must be positive.")
  if (
    static_support_center_speed_scale is not None
    and static_support_center_speed_scale <= 0.0
  ):
    raise ValueError("static_support_center_speed_scale must be positive.")
  if not 0.0 <= static_stillness_floor < 1.0:
    raise ValueError("static_stillness_floor must be in [0, 1).")
  if not all(
    0.0 <= threshold <= 1.0
    for threshold in (
      static_settling_alignment_threshold,
      static_settling_support_threshold,
      static_settling_clearance_threshold,
    )
  ):
    raise ValueError("static settling thresholds must be in [0, 1].")
  if not 0.0 <= attitude_progress_weight <= 1.0:
    raise ValueError("attitude_progress_weight must be in [0, 1].")
  if attitude_progress_rate_scale <= 0.0:
    raise ValueError("attitude_progress_rate_scale must be positive.")

  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  command = _command(env, command_name)
  if stationary_command_index is not None:
    if not 0 <= stationary_command_index < command.shape[1]:
      raise ValueError("stationary_command_index is outside the command tensor.")
    active &= torch.abs(command[:, stationary_command_index]) <= command_deadband

  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
  )
  orientation = torch.pow(alignment, orientation_power)
  contacts = _wheel_contacts(env, sensor_name).float()
  masks = torch.tensor(contact_masks, dtype=contacts.dtype, device=env.device)
  target = masks[mode]
  desired = (contacts * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
  non_target = 1.0 - target
  extra = (contacts * non_target).sum(dim=1) / non_target.sum(dim=1).clamp_min(1.0)
  no_extra_support = 1.0 - extra_contact_discount * extra
  support = desired * no_extra_support
  if soft_support_height is not None:
    if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
      raise ValueError("soft support grounding needs four wheel sites.")
    # Contact bits give no gradient for bringing a second selected wheel down:
    # a nearly-grounded tyre and a floating tyre are both simply ``False``.
    # Once the opposite pair has lifted, score only the physical endpoint of
    # the commanded pair--both wheel centres at the terrain height and level
    # with one another.  The non-target-contact factor keeps the ordinary
    # four-wheel reset at zero, so this is not a hidden stance target.
    target_wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    target_height_error = (target_wheel_height - soft_support_height) * target
    height_score = torch.exp(
      -torch.sum(torch.square(target_height_error), dim=1)
      / target.sum(dim=1).clamp_min(1.0)
      / soft_support_height_std**2
    )
    # A two-wheel support is laterally stable only when its two selected wheel
    # centres share a ground plane.  This measures that endpoint directly and
    # still leaves all leg geometry to the policy.
    target_height_spread = torch.amax(
      torch.where(target > 0.5, target_wheel_height, -torch.inf), dim=1
    ) - torch.amin(torch.where(target > 0.5, target_wheel_height, torch.inf), dim=1)
    pair_level_score = torch.exp(
      -torch.square(target_height_spread / soft_support_pair_height_std)
    )
    soft_support = height_score * pair_level_score * no_extra_support
    # Preserve contact as the final criterion but make a physically grounded
    # wheel pair an informative bridge as it approaches that criterion.
    support = 0.5 * support + 0.5 * soft_support

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
  # Apply stillness only after the caller's measured support is sufficiently
  # close to its final result.  A task can choose strict thresholds when its
  # intermediate two-wheel contact is a low slant that still needs angular
  # motion to rise; the defaults retain the earlier generic behaviour.
  static_settling = (
    static_command
    & (alignment >= static_settling_alignment_threshold)
    & (support >= static_settling_support_threshold)
    & (clearance >= static_settling_clearance_threshold)
  )
  stillness = torch.ones_like(orientation)
  if static_angular_velocity_scale is not None:
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    # The rational score preserves a continuous route from a moving upright
    # support to a quiet one.  A caller may retain a small discovery floor,
    # but a demonstrated upright that repeatedly flings itself out of the
    # commanded wheel pair must also be able to receive no static-support
    # return at all.
    angular_stillness = static_stillness_floor + (1.0 - static_stillness_floor) / (
      1.0 + torch.square(angular_speed / static_angular_velocity_scale)
    )
    stillness = torch.where(static_settling, angular_stillness, stillness)
  if static_linear_velocity_scale is not None:
    planar_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w[:, :2], dim=1)
    linear_stillness = static_stillness_floor + (1.0 - static_stillness_floor) / (
      1.0 + torch.square(planar_speed / static_linear_velocity_scale)
    )
    stillness = torch.where(
      static_settling, stillness * linear_stillness, stillness
    )
  if static_support_center_speed_scale is not None:
    if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
      raise ValueError("static support-centre stillness needs four wheel sites.")
    wheel_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    support_velocity = (wheel_velocity * target.unsqueeze(2)).sum(dim=1) / target.sum(
      dim=1, keepdim=True
    ).clamp_min(1.0)
    support_center_speed = torch.linalg.vector_norm(support_velocity, dim=1)
    support_center_stillness = static_stillness_floor + (
      1.0 - static_stillness_floor
    ) / (
      1.0
      + torch.square(support_center_speed / static_support_center_speed_scale)
    )
    stillness = torch.where(
      static_settling, stillness * support_center_stillness, stillness
    )
  # The support itself must be a valid contact outcome.  If a caller requests
  # a minimum trunk-to-wheel clearance, make that measured endpoint part of
  # the support result once the named pair is carrying the body.  This keeps
  # a low, belly-down two-wheel contact from being scored as a finished stand,
  # while contact still provides the bridge from the ordinary four-wheel
  # reset.  Callers without a clearance requirement retain the direct contact
  # score and its original discovery gradient.
  support_progress = support if minimum_root_clearance is None else support * clearance
  if support_leg_length_target is not None:
    if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
      raise ValueError("support-leg geometry needs four explicit wheel sites.")
    hip_ids, _ = asset.find_bodies(support_body_names, preserve_order=True)
    if len(hip_ids) != 4:
      raise ValueError("support_body_names must resolve one hip body per wheel.")
    # A two-wheel contact can be held with a folded knee even while the base
    # looks partially raised.  This is the actual hip-to-wheel span of the
    # selected support legs, not a joint-angle or reference-pose target.
    leg_length = torch.linalg.vector_norm(
      asset.data.body_link_pos_w[:, hip_ids] - asset.data.site_pos_w[:, asset_cfg.site_ids],
      dim=-1,
    )
    target_leg_length = (leg_length * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
    leg_extension = torch.clamp(
      target_leg_length / support_leg_length_target, min=0.0, max=1.0
    )
    # Preserve a continuous contact/attitude route from reset, then rank a
    # genuinely load-bearing extended support above the folded local optimum.
    support_progress = support_progress * (0.5 + 0.5 * leg_extension)
  support_result = orientation * support_progress * stillness

  # A static contact outcome alone is zero at the ordinary four-wheel reset,
  # even though the robot must first rotate toward the requested gravity
  # direction before a wheel can leave the floor.  Reward instant physical
  # angular progress only while meaningfully *away* from that target.  The
  # approach factor removes the previous incentive to keep accelerating at a
  # nearly correct support, while keeping the same trajectory-free route out
  # of the reset.
  gravity_rate = -torch.linalg.cross(asset.data.root_link_ang_vel_b, gravity)
  alignment_rate = torch.sum(gravity_rate * targets[mode], dim=1)
  attitude_progress = torch.clamp(
    alignment_rate / attitude_progress_rate_scale, min=0.0, max=1.0
  )
  approach_fraction = torch.clamp(2.0 * (1.0 - alignment), min=0.0, max=1.0)
  attitude_progress = attitude_progress * approach_fraction
  result = active.to(orientation.dtype) * (
    support_result + attitude_progress_weight * attitude_progress
  )
  if mode_weights is not None:
    weights = torch.tensor(mode_weights, dtype=result.dtype, device=env.device)
    result = result * weights[mode]
  return result


def mode_root_height_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  modes: tuple[int, ...],
  target_height: float,
  scale: float,
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  contact_masks: tuple[tuple[float, float, float, float], ...] | None = None,
  sensor_name: str | None = None,
  orientation_power: float = 1.0,
  extra_contact_discount: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward the final trunk height for selected support-command outcomes.

  This is deliberately a whole-body result, not a leg configuration or a
  get-up phase.  A two-wheel command needs the trunk to rise well above the
  ordinary four-wheel reset; contact/attitude terms alone otherwise retain a
  profitable low slant.  Modes outside ``modes`` receive zero so normal idle
  remains the model-default four-wheel height.
  """
  if target_height <= 0.0 or scale <= 0.0:
    raise ValueError("target_height and scale must be positive.")
  if orientation_power <= 0.0:
    raise ValueError("orientation_power must be positive.")
  if not 0.0 <= extra_contact_discount <= 1.0:
    raise ValueError("extra_contact_discount must be in [0, 1].")
  gate_args = (gravity_targets, contact_masks, sensor_name)
  if any(value is None for value in gate_args) and not all(
    value is None for value in gate_args
  ):
    raise ValueError(
      "gravity_targets, contact_masks, and sensor_name must be supplied together."
    )
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  height_error = torch.abs(asset.data.root_link_pos_w[:, 2] - target_height)
  height_score = torch.exp(-scale * height_error)
  if gravity_targets is None:
    return active.to(height_error.dtype) * height_score

  if len(gravity_targets) != num_modes or len(contact_masks) != num_modes:
    raise ValueError("height gate targets/masks must cover every command mode.")
  assert sensor_name is not None
  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), min=0.0, max=1.0
  )
  contacts = _wheel_contacts(env, sensor_name).float()
  masks = torch.tensor(contact_masks, dtype=contacts.dtype, device=env.device)
  target = masks[mode]
  desired = (contacts * target).sum(dim=1) / target.sum(dim=1).clamp_min(1.0)
  non_target = 1.0 - target
  extra = (contacts * non_target).sum(dim=1) / non_target.sum(dim=1).clamp_min(1.0)
  support = desired * (1.0 - extra_contact_discount * extra)
  # A high base is useful only in the same physical support outcome requested
  # by the one-hot.  Without this gate, a two-wheel command can collect a
  # height return by extending into an ordinary four-wheel slant, which is
  # neither the desired gravity direction nor the named wheel pair.
  return (
    active.to(height_error.dtype)
    * height_score
    * torch.pow(alignment, orientation_power)
    * support
  )


def mode_non_support_wheel_clearance(
  env: ManagerBasedRlEnv,
  command_name: str,
  modes: tuple[int, ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  target_height: float,
  minimum_height: float = 0.10,
  support_ground_height: float = 0.086,
  support_ground_height_std: float = 0.05,
  num_modes: int = 5,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward lifting the *uncommanded* wheel pair, without a limb target.

  A two-wheel command has a simple physical endpoint: its opposite pair of
  wheel centres must be clear of the plane.  Contact bits alone provide no
  useful gradient until a tyre has already separated, which permits a low
  folded four-wheel crouch to persist even when a gravity-direction reward is
  improving.  This measures only the average height of the non-support wheels
  and is zero at their ordinary ground-level reset.  It neither selects joint
  angles nor prescribes the route by which the robot raises them.
  """
  if target_height <= minimum_height or minimum_height < 0.0:
    raise ValueError("target_height must exceed non-negative minimum_height.")
  if support_ground_height < 0.0 or support_ground_height_std <= 0.0:
    raise ValueError("support ground height and std must be non-negative/positive.")
  if len(contact_masks) != num_modes:
    raise ValueError("contact_masks must cover every command mode.")
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
    raise ValueError("non-support wheel clearance needs four wheel sites.")

  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  masks = torch.tensor(contact_masks, dtype=asset.data.site_pos_w.dtype, device=env.device)
  non_support = 1.0 - masks[mode]
  non_support_count = non_support.sum(dim=1)
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  target_count = masks[mode].sum(dim=1).clamp_min(1.0)
  target_pair_height = (wheel_height * masks[mode]).sum(dim=1) / target_count
  mean_non_support_height = (wheel_height * non_support).sum(dim=1) / non_support_count.clamp_min(1.0)
  clearance = torch.clamp(
    (mean_non_support_height - minimum_height) / (target_height - minimum_height),
    min=0.0,
    max=1.0,
  )
  # Clearance by itself is ambiguous: rolling to the *opposite* side also
  # lifts the uncommanded pair and previously earned almost the same return.
  # Gate it by the measured height of the commanded pair staying at the wheel
  # ground plane.  This is a physical support outcome, not a joint/pose target;
  # it preserves a smooth bridge at reset (where clearance is still zero) and
  # removes the wrong-side local optimum that caused right-mode collapse.
  support_ground = torch.exp(
    -torch.square(
      (target_pair_height - support_ground_height) / support_ground_height_std
    )
  )
  # Callers exclude normal mode because it has no non-support wheel pair.
  return active.to(clearance.dtype) * clearance * support_ground


def mode_static_angular_velocity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  modes: tuple[int, ...],
  angular_velocity_scale: float,
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  alignment_power: float = 4.0,
) -> torch.Tensor:
  """Reward a formed named support for actually stopping its body rotation.

  Lateral two-wheel commands are held supports, not pivots.  Contact and
  gravity terms can be satisfied while the root keeps rolling around the
  selected wheels, so this single measured outcome closes that loophole.  The
  alignment power keeps the signal small at the ordinary four-wheel reset and
  makes it strong only after the requested side attitude has formed; it is not
  a joint pose or a transition trajectory.
  """
  if angular_velocity_scale <= 0.0 or alignment_power <= 0.0:
    raise ValueError("angular velocity scale and alignment power must be positive.")
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  asset: Entity = env.scene["robot"]
  angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
  # Keep a usable braking gradient even when an exploratory side roll is
  # several scale lengths fast; a Gaussian would numerically flatten there.
  stillness = 1.0 / (
    1.0 + torch.square(angular_speed / angular_velocity_scale)
  )
  if gravity_targets is not None:
    targets = torch.tensor(
      gravity_targets, dtype=asset.data.projected_gravity_b.dtype, device=env.device
    )
    gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    # At the ordinary four-wheel reset alignment is exactly 0.5 for a side
    # command.  Do not pay a large zero-velocity bonus there: that would make
    # the actor prefer never attempting the roll.  The support-centre result
    # becomes active only as the requested side attitude is physically formed.
    alignment_rise = torch.clamp(2.0 * alignment - 1.0, 0.0, 1.0)
    stillness = stillness * alignment_rise.pow(alignment_power)
  return active.to(stillness.dtype) * stillness


def mode_static_support_center_velocity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  modes: tuple[int, ...],
  velocity_scale: float,
  contact_masks: tuple[tuple[float, float, float, float], ...],
  num_modes: int = 5,
  gravity_targets: tuple[tuple[float, float, float], ...] | None = None,
  alignment_power: float = 4.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward a static named support whose selected wheel midpoint stays local.

  Lateral two-wheel forms in the reference are balance supports, not a
  bicycle driving a circle.  Body angular stillness alone cannot detect that
  failure: a policy can keep the trunk nearly quiet while translating the
  selected wheel pair.  This outcome therefore measures the selected pair's
  planar midpoint velocity directly and gates it by the requested attitude.
  It contains no pose or transition target.
  """
  if velocity_scale <= 0.0 or alignment_power <= 0.0:
    raise ValueError("velocity_scale and alignment_power must be positive.")
  if len(contact_masks) != num_modes:
    raise ValueError("contact_masks must cover every command mode.")
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
    raise ValueError("support-centre velocity needs four wheel sites.")
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  asset: Entity = env.scene[asset_cfg.name]
  masks = torch.tensor(
    contact_masks, dtype=asset.data.site_pos_w.dtype, device=env.device
  )
  selected = masks[mode]
  wheel_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  midpoint_velocity = (wheel_velocity * selected.unsqueeze(2)).sum(dim=1) / (
    selected.sum(dim=1, keepdim=True).clamp_min(1.0)
  )
  midpoint_speed = torch.linalg.vector_norm(midpoint_velocity, dim=1)
  stillness = 1.0 / (1.0 + torch.square(midpoint_speed / velocity_scale))
  if gravity_targets is not None:
    targets = torch.tensor(
      gravity_targets, dtype=asset.data.projected_gravity_b.dtype, device=env.device
    )
    gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
    alignment = torch.clamp(
      0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), 0.0, 1.0
    )
    alignment_rise = torch.clamp(2.0 * alignment - 1.0, 0.0, 1.0)
    stillness = stillness * alignment_rise.pow(alignment_power)
  return active.to(stillness.dtype) * stillness


def mode_gravity_alignment_rise(
  env: ManagerBasedRlEnv,
  command_name: str,
  modes: tuple[int, ...],
  gravity_targets: tuple[tuple[float, float, float], ...],
  num_modes: int = 5,
  power: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward only progress beyond ordinary upright toward a commanded attitude.

  The normal four-wheel reset has a target-alignment score of exactly 0.5 for
  any front/rear/side two-wheel attitude.  Contact-quality rewards must be
  zero there—otherwise they accept the reset as a support—but a pure angular
  *rate* bridge can be collected by rocking without retaining useful pose
  progress.  This bounded state result is zero at the normal reset and one at
  the requested gravity direction.  It specifies neither a leg pose nor a
  transition timing, and leaves contact/height terms to decide how the robot
  physically realizes the attitude.
  """
  if power <= 0.0:
    raise ValueError("power must be positive.")
  if len(gravity_targets) != num_modes:
    raise ValueError("gravity_targets must cover every command mode.")
  asset: Entity = env.scene[asset_cfg.name]
  active, mode = _mode_mask(env, command_name, modes, num_modes=num_modes)
  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  targets = torch.tensor(gravity_targets, dtype=gravity.dtype, device=env.device)
  alignment = torch.clamp(
    0.5 * (1.0 + torch.sum(gravity * targets[mode], dim=1)), min=0.0, max=1.0
  )
  # Map normal reset (alignment=.5) to zero rather than paying it as a
  # partial stand; any progress in the wrong direction also remains zero.
  rise = torch.clamp(2.0 * alignment - 1.0, min=0.0, max=1.0)
  return active.to(rise.dtype) * torch.pow(rise, power)


def _stance_spin_components(
  env: ManagerBasedRlEnv,
  command_name: str,
  speed_deadband: float,
  rate_std: float,
  gravity_targets: tuple[tuple[float, float, float], ...],
  contact_masks: tuple[tuple[float, float, float, float], ...],
  sensor_name: str,
  normal_min_root_clearance: float,
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
  torch.Tensor,
]:
  """Measure a commanded five-mode world-down rotation."""
  if rate_std <= 0.0 or normal_min_root_clearance <= 0.0:
    raise ValueError("rate_std and normal_min_root_clearance must be positive.")
  asset: Entity = env.scene[asset_cfg.name]
  if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
    raise ValueError("stance-spin measurement needs four wheel sites.")

  command = _command(env, command_name)
  mode = torch.argmax(command[:, :5], dim=1)
  active = torch.sum(command[:, :5], dim=1) > 0.5
  # Normal and pitch-pair supports track signed world-down rate.  The
  # left/right pair is a static side support in the reference manoeuvre, so
  # its public rate slot is deliberately ignored rather than creating an
  # unphysical side-pivot objective.
  moving = active & (mode <= 2) & (torch.abs(command[:, 5]) > speed_deadband)
  gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
  actual_rate = torch.sum(asset.data.root_link_ang_vel_b * gravity, dim=1)
  # A fixed ``1 - error / std`` score is exactly zero for the requested
  # 10--15 rad/s rates while the policy is still at rest.  Its clamp then
  # removes the only acceleration gradient, so front/rear modes learn the
  # support shape but never learn to spin.  Normalize the error by the
  # requested magnitude and use a Gaussian-shaped score instead; it remains
  # bounded, signed-direction sensitive, and informative from zero speed.
  rate_scale = 0.75 * torch.abs(command[:, 5]) + rate_std
  normalized_rate_error = (command[:, 5] - actual_rate) / rate_scale.clamp_min(1.0e-6)
  rate_score = torch.exp(-0.5 * torch.square(normalized_rate_error))

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

  # ``normal`` is the compact, level, four-wheel form.  The named front/rear/
  # left/right one-hots retain their physically distinct two-wheel supports.
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  # In normal mode the original command target already represents the desired
  # level trunk and all four wheel contacts.  Its remaining discovery problem
  # is to reshape the wheel centres and axes into the common axle below.
  normal_root_clearance = asset.data.root_link_pos_w[:, 2] - wheel_height.mean(dim=1)
  normal_clearance_score = torch.clamp(
    normal_root_clearance / normal_min_root_clearance, min=0.0, max=1.0
  )
  normal_support_quality = alignment * contact_score * normal_clearance_score
  (
    normal_all_axis_parallel,
    normal_common_axle_line,
    normal_compact_span,
  ) = normal_four_wheel_pivot_geometry(
    wheel_axles, wheel_positions
  )
  support_mask = masks[mode]
  coaxiality = torch.where(
    mode == 0,
    normal_all_axis_parallel * normal_common_axle_line,
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
    normal_common_axle_line,
    normal_compact_span,
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
    normal_min_root_clearance: float,
    asset_cfg: SceneEntityCfg,
    upright_support_weight: float = 0.20,
    normal_final_geometry_weight: float = 0.02,
    normal_geometry_decay_start_steps: int = 38_400,
    normal_geometry_decay_steps: int = 25_600,
    rate_progress_weight: float = 0.75,
    static_support_weight: float = 1.0,
  ) -> torch.Tensor:
    if pivot_speed_limit <= 0.0 or normal_min_root_clearance <= 0.0:
      raise ValueError("pivot_speed_limit and normal_min_root_clearance must be positive.")
    if not 0.0 <= upright_support_weight < 1.0:
      raise ValueError("upright_support_weight must be in [0, 1).")
    if not 0.0 <= normal_final_geometry_weight < 1.0:
      raise ValueError("normal_final_geometry_weight must be in [0, 1).")
    if normal_geometry_decay_start_steps < 0 or normal_geometry_decay_steps <= 0:
      raise ValueError("normal geometry decay steps must be non-negative/positive.")
    if not 0.0 <= rate_progress_weight <= 1.0:
      raise ValueError("rate_progress_weight must be in [0, 1].")
    if not 0.0 <= static_support_weight <= 1.0:
      raise ValueError("static_support_weight must be in [0, 1].")
    (
      asset,
      active,
      moving,
      rate_score,
      support_quality,
      coaxial_factor,
      normal_all_axis_parallel,
      normal_common_axle_line,
      normal_compact_span,
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
      normal_min_root_clearance,
      asset_cfg,
    )
    wheel_velocity = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    support_centre_velocity = (
      wheel_velocity * support_mask.unsqueeze(2)
    ).sum(dim=1) / support_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    centre_speed = torch.linalg.vector_norm(support_centre_velocity, dim=1)
    # Normal uses the all-wheel centroid; named modes retain their requested
    # two-wheel support-pair centroid.
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
    # Keep a direct gradient toward the requested signed rate while the
    # support centre is still settling.  Multiplying by the raw stillness
    # score made any exploratory wheel motion nearly reward-free, so PPO chose
    # a static support forever.  The nonzero floor is only a discovery bridge;
    # the final outcome still ranks a local pivot through ``pivot_stillness``.
    dynamic_quality = coaxial_factor * speed_quality * (
      0.30 + 0.70 * pivot_stillness
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
    # Normal is a level four-wheel pivot.  Its all-wheel support, common axle
    # line, compact transverse span, local centroid, and signed rate are one
    # physical outcome; no individual joint angle or temporal pose is named.
    normal_geometry = (
      normal_all_axis_parallel
      * normal_common_axle_line
      * normal_compact_span
    )
    # At reset, normal has level attitude and four contacts but not the common
    # axle or spin rate.  Keep a short geometry bridge, then fade it so the
    # same measured geometry must support a requested local rotation.
    normal_decay = torch.clamp(
      torch.tensor(
        (env.common_step_counter - normal_geometry_decay_start_steps)
        / normal_geometry_decay_steps,
        dtype=support_quality.dtype,
        device=env.device,
      ),
      min=0.0,
      max=1.0,
    )
    normal_geometry_weight = upright_support_weight + normal_decay * (
      normal_final_geometry_weight - upright_support_weight
    )
    normal_result = support_quality * normal_geometry * (
      normal_geometry_weight
      + (1.0 - normal_geometry_weight)
      * speed_quality
      * (0.30 + 0.70 * pivot_stillness)
    )
    dynamic_result = torch.where(mode == 0, normal_result, upright_result)

    # A zero spin rate on an active named one-hot means make the requested
    # two-wheel support *and hold it still*.  The old static branch only
    # constrained support-centre translation, so m1200 learned to rotate the
    # body in place at zero requested rate.  Once the same measured support
    # quality is recognizably present, include whole-body angular stillness
    # in that existing outcome.  It remains inactive during the initial rise,
    # so this does not prescribe a get-up speed, pose, or trajectory.
    static_settling = support_quality >= 0.30
    static_angular_speed = torch.linalg.vector_norm(
      asset.data.root_link_ang_vel_w, dim=1
    )
    static_angular_stillness = 1.0 / (
      1.0 + torch.square(static_angular_speed / 0.8)
    )
    upright_static_result = support_quality * pivot_stillness * torch.where(
      static_settling,
      static_angular_stillness,
      torch.ones_like(static_angular_stillness),
    )
    dynamic_or_upright_static = torch.where(
      moving,
      dynamic_result,
      torch.where(
        upright_mode,
        static_support_weight * upright_static_result,
        torch.zeros_like(dynamic_result),
      ),
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
  mode_weights: tuple[float, ...] | None = None,
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
  if mode_weights is not None:
    if len(mode_weights) != command.shape[1] - 2 or any(weight < 0.0 for weight in mode_weights):
      raise ValueError("mode_weights must be non-negative and match locomotion modes.")
    weights = torch.tensor(mode_weights, dtype=result.dtype, device=env.device)
    result = result * weights[mode]
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
    mode_weights: tuple[float, ...] | None = None,
  ) -> torch.Tensor:
    if target_angle <= 0.0:
      raise ValueError("target_angle must be positive.")
    if mode_weights is not None and (
      len(mode_weights) != 5 or any(weight < 0.0 for weight in mode_weights)
    ):
      raise ValueError("mode_weights must contain five non-negative aerial values.")
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
    result = (
      active.to(increment.dtype)
      * legal.to(increment.dtype)
      * increment / target_angle
      / env.step_dt
    )
    if mode_weights is not None:
      mode = torch.argmax(command[:, :5], dim=1)
      weights = torch.tensor(mode_weights, dtype=result.dtype, device=env.device)
      result = result * weights[mode]
    return result


def aerial_event_failure(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_angle: float,
  early_missing_angle_cost: float = 1.0,
  final_missing_angle_cost: float = 1.0,
  early_non_timeout_base_cost: float = 0.0,
  final_non_timeout_base_cost: float = 0.0,
  base_cost_ramp_start_steps: int = 0,
  base_cost_ramp_steps: int = 1,
  terminal_angular_speed_scale: float | None = None,
  minimum_motion_failure_fraction: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Apply one terminal cost proportional to the missing requested turn.

  A fixed failure cost taught the first corrected aerial run to suppress its
  launch: every intermediate 0.1--0.9 turn was equally bad as a zero-turn
  fall.  The only physical quantity that should reduce this cost is the
  measured desired-axis angle already used by the endpoint.  Thus a partial
  landing is still a failure, but each additional correct radian improves its
  return and PPO has a continuous route to the one-turn completion bonus.
  """
  if (
    target_angle <= 0.0
    or not 0.0 <= early_missing_angle_cost <= 1.0
    or not 0.0 <= final_missing_angle_cost <= 1.0
    or not 0.0 <= early_non_timeout_base_cost <= 1.0
    or not 0.0 <= final_non_timeout_base_cost <= 1.0
    or base_cost_ramp_start_steps < 0
    or base_cost_ramp_steps <= 0
    or terminal_angular_speed_scale is not None and terminal_angular_speed_scale <= 0.0
    or not 0.0 < minimum_motion_failure_fraction <= 1.0
  ):
    raise ValueError("aerial failure parameters are outside their valid ranges.")
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
  # First make a legal launch and signed rotation discoverable.  Once those
  # outcomes have appeared, raise the fixed cost of an illegal touchdown so
  # the same one-shot event must trade its partial turn for a quiet landing.
  # This is a scalar task-difficulty curriculum, not an action/pose/phase
  # reference and it changes no public command.
  base_ramp = torch.clamp(
    torch.tensor(
      (env.common_step_counter - base_cost_ramp_start_steps) / base_cost_ramp_steps,
      dtype=missing_fraction.dtype,
      device=env.device,
    ),
    min=0.0,
    max=1.0,
  )
  non_timeout_base_cost = early_non_timeout_base_cost + base_ramp * (
    final_non_timeout_base_cost - early_non_timeout_base_cost
  )
  # A nearly complete spin that strikes the ground at high angular speed was
  # previously indistinguishable from a near-quiet invalid landing: both paid
  # the same terminal base cost once their desired-axis angle approached 2π.
  # That leaves no terminal outcome gradient for braking.  Scale that *same*
  # failure cost by the measured root angular speed.  The lowest-speed case
  # remains a nonzero failure, while a rapid trunk/leg crash is maximally bad.
  # This is an endpoint measurement only--not a commanded rate, pose, or
  # phase reference.
  if terminal_angular_speed_scale is not None:
    asset: Entity = env.scene[asset_cfg.name]
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    motion_fraction = 1.0 - torch.exp(
      -torch.square(angular_speed / terminal_angular_speed_scale)
    )
    non_timeout_base_cost = non_timeout_base_cost * (
      minimum_motion_failure_fraction
      + (1.0 - minimum_motion_failure_fraction) * motion_fraction
    )
  # At reset, a full missing-angle cost makes every exploratory launch much
  # worse than standing still: its terminal loss arrives before PPO has seen
  # enough legal flight to assign credit to takeoff.  Ramp only this scalar
  # outcome cost from a small value to its final strict value.  The command,
  # target angle, observations, and physical validity rules remain unchanged.
  missing_angle_cost = early_missing_angle_cost + base_ramp * (
    final_missing_angle_cost - early_missing_angle_cost
  )
  failure = (
    missing_angle_cost * missing_fraction
    + non_timeout_base_cost * invalid_terminal
  )
  return (
    env.termination_manager.dones.to(missing_fraction.dtype)
    * failure
    / env.step_dt
  )


class AerialTuckThenWheelLanding:
  """Shape the visible compact-flight-to-wheel-landing geometry.

  The failed policies spread their legs throughout a flip: their mean wheel
  distance from the trunk grows from the nominal 0.36 m to 0.54--0.61 m, then
  a non-wheel link strikes first.  The demonstrated maneuver does the
  opposite: compact wheels/legs while accumulating the turn, then extend a
  wheel-first envelope only near the landing.  Both are measured geometry
  outcomes.  This supplies neither a joint target nor a time-indexed
  reference trajectory; the switch is made only by the turn the policy has
  physically accumulated itself.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    # Tuck and wheel-first clearance are successive physical outcomes.  They
    # must retain independent high-water marks: a compact mid-air package is
    # not a substitute for making the wheels lowest again before touchdown.
    self.peak_tuck_score = torch.zeros(env.num_envs, device=env.device)
    self.peak_landing_score = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )
    self.body_ids: torch.Tensor | None = None

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_tuck_score[env_ids] = 0.0
    self.peak_landing_score[env_ids] = 0.0
    self.previous_active[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    body_names: tuple[str, ...],
    target_angle: float,
    tuck_start_turn_fraction: float = 0.08,
    tuck_ramp_end_turn_fraction: float = 0.20,
    tuck_end_turn_fraction: float = 0.62,
    tuck_target_wheel_root_distance: float = 0.30,
    tuck_max_wheel_root_distance: float = 0.40,
    landing_start_turn_fraction: float = 0.62,
    target_clearance: float = 0.10,
    minimum_clearance_for_progress: float = -0.30,
    dense_geometry_weight: float = 0.5,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if (
      target_angle <= 0.0
      or target_clearance <= 0.0
      or tuck_target_wheel_root_distance <= 0.0
      or tuck_max_wheel_root_distance <= tuck_target_wheel_root_distance
      or dense_geometry_weight < 0.0
    ):
      raise ValueError("Aerial geometry distances and target_angle must be positive.")
    if minimum_clearance_for_progress >= target_clearance:
      raise ValueError("minimum_clearance_for_progress must be below target_clearance.")
    if not (
      0.0 <= tuck_start_turn_fraction < tuck_ramp_end_turn_fraction
      <= tuck_end_turn_fraction <= landing_start_turn_fraction < 1.0
    ):
      raise ValueError("Aerial tuck/landing turn fractions must be ordered in [0, 1).")
    asset: Entity = env.scene[asset_cfg.name]
    if self.body_ids is None:
      self.body_ids, _ = asset.find_bodies(body_names, preserve_order=True)
    if isinstance(asset_cfg.site_ids, slice) or len(asset_cfg.site_ids) != 4:
      raise ValueError("wheel-first envelope needs four explicit wheel sites.")

    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    reset = (~active) | (active & ~self.previous_active) | (env.episode_length_buf == 0)
    self.peak_tuck_score[reset] = 0.0
    self.peak_landing_score[reset] = 0.0

    command_term = env.command_manager.get_term(command_name)
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    was_airborne = getattr(command_term, "was_airborne", torch.zeros_like(active))
    landing_started = getattr(command_term, "_landing_started", torch.zeros_like(active))
    wheel_free = ~_has_any_contact(env, sensor_name)
    legal = ~_has_any_contact(env, nonwheel_sensor_name)
    turn_fraction = torch.clamp(progress / target_angle, min=0.0, max=1.0)
    wheel_top = torch.amax(asset.data.site_pos_w[:, asset_cfg.site_ids, 2], dim=1)
    lowest_nonwheel_link = torch.amin(
      asset.data.body_link_pos_w[:, self.body_ids, 2], dim=1
    )
    wheel_clearance = lowest_nonwheel_link - wheel_top
    # Tuck is a body-centred geometric result: it does not select an individual
    # hip, knee, or calf angle.  The nominal reset averages 0.364 m; a failed
    # aerial spreads to roughly 0.55 m.  Reaching 0.30 m therefore describes
    # a visibly compact wheel/leg package without encoding its configuration.
    wheel_root_distance = torch.linalg.vector_norm(
      asset.data.site_pos_w[:, asset_cfg.site_ids] - asset.data.root_link_pos_w[:, None],
      dim=-1,
    ).mean(dim=1)
    # Keep a usable gradient even when an exploratory policy has thrown the
    # wheel package well outside the compact envelope.  The former linear
    # clamp became exactly zero above ``tuck_max_wheel_root_distance`` (the
    # m300 policy reached roughly 0.56 m), so PPO could not discover how to
    # pull the limbs back in after its first wide launch.  A smooth Gaussian
    # around the measured compact-distance outcome stays bounded in [0, 1]
    # while remaining informative throughout that recovery region.  The
    # scale is twice the configured envelope width: at the envelope edge the
    # score is still substantial, and the target distance remains the unique
    # maximum without naming a joint pose.
    # Keep the measured compact-distance outcome informative when an
    # exploratory policy has spread the wheel package far from the trunk.
    # The earlier two-width Gaussian made a 0.55--0.60 m package retain too
    # much return relative to the desired compact form; a narrower scale
    # supplies a stronger gradient without naming a joint pose.
    # Keep the compactness outcome informative from the wide exploratory
    # package seen in the failed body-axis branches (~0.5--0.6 m).  A scale
    # that is too narrow makes this score numerically zero before PPO can
    # discover the recovery action; the target distance remains its unique
    # maximum.
    tuck_distance_scale = 2.0 * (
      tuck_max_wheel_root_distance - tuck_target_wheel_root_distance
    )
    tuck_score = torch.exp(
      -torch.square(
        (wheel_root_distance - tuck_target_wheel_root_distance)
        / tuck_distance_scale
      )
    )
    # A hard zero while one limb is still below the wheel plane makes this
    # otherwise physical result undiscoverable.  Score the same wheel-first
    # clearance over a finite safety margin instead.
    wheel_lowest_score = torch.clamp(
      (wheel_clearance - minimum_clearance_for_progress)
      / (target_clearance - minimum_clearance_for_progress),
      min=0.0,
      max=1.0,
    )
    candidate = (
      active
      & was_airborne
      & (~landing_started)
      & wheel_free
      & legal
    )
    tuck_entry = torch.clamp(
      (turn_fraction - tuck_start_turn_fraction)
      / (tuck_ramp_end_turn_fraction - tuck_start_turn_fraction),
      min=0.0,
      max=1.0,
    )
    tuck_exit = torch.clamp(
      (tuck_end_turn_fraction - turn_fraction)
      / (tuck_end_turn_fraction - tuck_ramp_end_turn_fraction),
      min=0.0,
      max=1.0,
    )
    tuck_phase = tuck_entry * tuck_exit
    landing_phase = torch.clamp(
      (turn_fraction - landing_start_turn_fraction)
      / (1.0 - landing_start_turn_fraction),
      min=0.0,
      max=1.0,
    )
    # The two terms are consecutive outcomes of one airborne maneuver: tuck
    # while rotation is built, then wheel-lowest only in the final approach.
    # Keep their peak-gain accounting separate for a one-off discovery signal,
    # but also retain a small *continuous* measurement below.  A pure
    # high-water reward goes silent as soon as the policy has briefly tucked;
    # it therefore cannot tell PPO that the same body is extending too early
    # and will strike the floor before touchdown.  The continuous term is the
    # same measured wheel-root distance/clearance outcome, not a joint target,
    # clock, or reference trajectory.
    tuck_event_score = torch.where(
      candidate,
      tuck_phase * tuck_score,
      torch.zeros_like(tuck_score),
    )
    landing_event_score = torch.where(
      candidate,
      landing_phase * wheel_lowest_score,
      torch.zeros_like(wheel_lowest_score),
    )
    tuck_gain = torch.clamp(tuck_event_score - self.peak_tuck_score, min=0.0)
    landing_gain = torch.clamp(
      landing_event_score - self.peak_landing_score,
      min=0.0,
    )
    self.peak_tuck_score = torch.where(
      active,
      torch.maximum(self.peak_tuck_score, tuck_event_score),
      torch.zeros_like(self.peak_tuck_score),
    )
    self.peak_landing_score = torch.where(
      active,
      torch.maximum(self.peak_landing_score, landing_event_score),
      torch.zeros_like(self.peak_landing_score),
    )
    self.previous_active = active
    dense_geometry = dense_geometry_weight * torch.where(
      candidate,
      0.5 * tuck_phase * tuck_score + 0.5 * landing_phase * wheel_lowest_score,
      torch.zeros_like(tuck_score),
    )
    # RewardManager supplies the dt integral.  Keep the dense bridge bounded
    # at one per second so it guides the pre-contact package without
    # overwhelming the one-off turn and landing outcomes above.
    return (tuck_gain + landing_gain) / env.step_dt + dense_geometry


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
    self.peak_landing_quality = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.settle_time[env_ids] = 0.0
    self.awarded[env_ids] = False
    self.peak_orientation_return[env_ids] = 0.0
    self.peak_landing_quality[env_ids] = 0.0

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
    late_flight_brake_start_turn_fraction: float = 0.75,
    late_flight_brake_angular_speed_std: float = 18.0,
    partial_landing_bonus: float = 0.0,
    completion_bonus: float = 1.0,
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
    if not 0.0 < late_flight_brake_start_turn_fraction < 1.0:
      raise ValueError("late_flight_brake_start_turn_fraction must be in (0, 1).")
    if late_flight_brake_angular_speed_std <= 0.0:
      raise ValueError("late_flight_brake_angular_speed_std must be positive.")
    if partial_landing_bonus < 0.0 or completion_bonus <= 0.0:
      raise ValueError("partial_landing_bonus must be non-negative and completion_bonus positive.")
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
    # progress term responsible for the first part of the maneuver.  In the
    # final fraction of an already nearly-complete turn, blend that same score
    # into measured whole-body angular quietness.  Without this physical
    # braking condition the previous term paid an aerial that returned to its
    # launch orientation at high angular speed, although it could only crash
    # at the wheel touchdown.  No target action, joint pose, timing trace, or
    # rate command is introduced.
    flight_candidate = (
      active
      & was_airborne
      & (~landing_started)
      & (~torch.any(contacts, dim=1))
      & legal
      & (progress <= target_angle + max_overrotation)
    )
    turn_fraction = torch.clamp(progress / target_angle, min=0.0, max=1.0)
    brake_fraction = torch.clamp(
      (turn_fraction - late_flight_brake_start_turn_fraction)
      / (1.0 - late_flight_brake_start_turn_fraction),
      min=0.0,
      max=1.0,
    )
    brake_quality = torch.exp(
      -torch.square(angular_speed / late_flight_brake_angular_speed_std)
    )
    late_recovery_quality = (1.0 - brake_fraction) + brake_fraction * brake_quality
    orientation_return = (
      torch.pow(turn_fraction, 4) * orientation_similarity * late_recovery_quality
    )
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

    # The aerial command hands control to literal idle immediately after its
    # first four-wheel touchdown, so an otherwise legal landing with a small
    # residual full-frame error cannot receive any later in-flight gradient.
    # Credit the same physical endpoint once, continuously by its whole-body
    # orientation and measured quietness.  It is deliberately smaller than
    # the strict completion below: a stable but misaligned landing is a
    # useful discovery step, never an alternative task success.
    partial_landing = (
      post_landing_idle
      & was_airborne
      & (progress >= target_angle)
      & (progress <= target_angle + max_overrotation)
      & torch.all(contacts, dim=1)
      & legal
    )
    landing_quietness = torch.exp(
      -0.5
      * (
        torch.square(linear_speed / landing_linear_velocity_limit)
        + torch.square(angular_speed / landing_angular_velocity_limit)
      )
    )
    landing_quality = torch.where(
      partial_landing,
      orientation_similarity * landing_quietness,
      torch.zeros_like(orientation_similarity),
    )
    landing_quality_gain = torch.clamp(
      landing_quality - self.peak_landing_quality, min=0.0
    )
    event_open = active | landing_started
    self.peak_landing_quality = torch.where(
      event_open,
      torch.maximum(self.peak_landing_quality, landing_quality),
      torch.zeros_like(self.peak_landing_quality),
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
    return (
      orientation_gain
      + partial_landing_bonus * landing_quality_gain
      + completion_bonus * new_completion.to(orientation_gain.dtype)
    ) / env.step_dt
