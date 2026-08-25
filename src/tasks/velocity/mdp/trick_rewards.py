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
  front_inside_margin: float = 0.05,
  front_inside_scale: float = 0.06,
  inner_pair_min_spacing: float = 0.10,
  outer_pair_extra_spacing: float = 0.10,
  outer_pair_max_ratio: float = 3.0,
  outer_pair_max_bias: float = 0.08,
) -> tuple[
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
  torch.Tensor,
]:
  """Measure the reference video's four-wheel common-axis layout.

  All four wheel axes must lie on one horizontal line beneath the body.  The
  front pair is the inner pair on that line and the rear pair is outside it.
  This is a physical wheel-layout measurement, not a joint-position target.
  """
  if wheel_axles.ndim != 3 or wheel_axles.shape[1:] != (4, 3):
    raise ValueError("four-wheel axle layout expects [batch, 4, 3] axes.")
  if wheel_positions.shape != wheel_axles.shape:
    raise ValueError("wheel positions must match wheel axle tensor shape.")
  if (
    line_scale <= 0.0
    or front_inside_margin < 0.0
    or front_inside_scale <= 0.0
    or inner_pair_min_spacing <= 0.0
    or outer_pair_extra_spacing <= 0.0
    or outer_pair_max_ratio <= 1.0
    or outer_pair_max_bias < 0.0
  ):
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

  # The reference form is not satisfied by *one* good wheel from each pair:
  # both front wheels must be inside both rear wheels on the shared axle.  An
  # average pair radius allowed crossed/stepping arrangements to receive the
  # same score as the nested four-wheel layout.  Compare the outermost front
  # wheel to the innermost rear wheel instead; this is still only measured
  # wheel geometry, never a prescribed leg-joint posture.
  front_outer_radius = torch.amax(torch.abs(axial_coordinate[:, :2]), dim=1)
  rear_inner_radius = torch.amin(torch.abs(axial_coordinate[:, 2:]), dim=1)
  front_inside_delta = rear_inner_radius - front_outer_radius
  # A zero-gap default rectangle must not be a half-credit answer.  The
  # 5-cm margin is the visible "front inside, rear outside" separation used
  # by the fixed evaluator, while the sigmoid keeps a dense approach signal.
  front_inside_score = torch.sigmoid(
    (front_inside_delta - front_inside_margin) / front_inside_scale
  )
  # A nesting-only score admits the failure seen in the recording: the two
  # inner wheels collapse together while the outer pair spreads excessively.
  # The reference has four distinct centres ordered along one axle.  These
  # are measured wheel-centre spacings, never desired joint angles.
  front_pair_spacing = torch.abs(axial_coordinate[:, 0] - axial_coordinate[:, 1])
  rear_pair_spacing = torch.abs(axial_coordinate[:, 2] - axial_coordinate[:, 3])
  inner_separation_score = torch.sigmoid(
    (front_pair_spacing - inner_pair_min_spacing) / 0.025
  )
  outer_order_score = torch.sigmoid(
    (rear_pair_spacing - front_pair_spacing - outer_pair_extra_spacing) / 0.04
  )
  outer_bound_score = torch.sigmoid(
    (
      outer_pair_max_ratio * front_pair_spacing
      + outer_pair_max_bias
      - rear_pair_spacing
    )
    / 0.06
  )
  nested_spacing_score = (
    front_inside_score
    * inner_separation_score
    * outer_order_score
    * outer_bound_score
  )
  return (
    parallel_score * horizontal_score * line_score,
    nested_spacing_score,
    front_inside_delta,
    front_pair_spacing,
    rear_pair_spacing,
  )


def normal_four_wheel_spacing_ok(
  front_pair_spacing: torch.Tensor,
  rear_pair_spacing: torch.Tensor,
  *,
  inner_pair_min_spacing: float = 0.10,
  outer_pair_extra_spacing: float = 0.10,
  outer_pair_max_ratio: float = 3.0,
  outer_pair_max_bias: float = 0.08,
) -> torch.Tensor:
  """Check that normal-spin inner/outer pairs are distinct and compact."""
  return (
    (front_pair_spacing >= inner_pair_min_spacing)
    & (rear_pair_spacing >= front_pair_spacing + outer_pair_extra_spacing)
    & (
      rear_pair_spacing
      <= outer_pair_max_ratio * front_pair_spacing + outer_pair_max_bias
    )
  )


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
  return active.to(orientation.dtype) * orientation * result_progress * stillness


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
  normal_coaxiality, front_inside_score, _, _, _ = normal_four_wheel_axle_layout(
    wheel_axles, wheel_positions
  )
  # The strict product below is the final reference-layout test, but at the
  # four-wheel reset both its line and nesting factors are small.  Feeding
  # only that product to PPO made every useful first deformation effectively
  # reward-free.  This bounded average uses the *same two measured outcomes*
  # as a formation-progress signal; it does not choose a joint posture.
  # The reference layout needs *both* a common axle and the front-inside/rear-
  # outside order.  Arithmetic weighting let PPO trade one against the other:
  # it first learned a crossed common line, then a correctly nested but
  # non-coaxial static form.  A small-floor geometric mean is continuous at
  # reset yet treats either missing physical property as a real limitation.
  formation_floor = 0.05
  normal_formation_progress = torch.sqrt(
    (formation_floor + (1.0 - formation_floor) * normal_coaxiality)
    * (formation_floor + (1.0 - formation_floor) * front_inside_score)
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
  # Normal's common-axis pivot is a level four-wheel form.  Its previous
  # quadratic tolerance still made a visibly nose-up pivot worthwhile.
  normal_alignment_score = torch.pow(alignment, 16.0)
  alignment_score = torch.where(
    mode == 0,
    normal_alignment_score,
    # Front/rear start in the ordinary four-wheel reset at alignment 0.5.
    # The previous fourth power suppressed that already contact/height-gated
    # discovery return to 6.25%, so a zero-rate support curriculum could not
    # discover the very stance needed before asking for high z-rate.  Keep
    # the same measured attitude/contact/clearance outcome but expose its
    # linear physical progress; final validation still requires the strict
    # two-wheel pose and local pivot.
    torch.where(mode <= 2, alignment, side_alignment_score),
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
  # and specifies neither a leg pose nor a transition trajectory.  Normal and
  # side modes retain their existing stricter geometry products.
  upright_support_progress = alignment * (0.65 * contact_score + 0.35 * height_score)
  support_quality = torch.where(
    (mode == 1) | (mode == 2),
    upright_support_progress,
    contact_score * alignment_score * height_score,
  )
  return (
    asset,
    active,
    moving,
    rate_score,
    support_quality,
    coaxial_factor,
    normal_formation_progress,
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
    normal_formation_weight: float = 0.45,
    rate_progress_weight: float = 0.75,
  ) -> torch.Tensor:
    if pivot_speed_limit <= 0.0:
      raise ValueError("pivot_speed_limit must be positive.")
    if not 0.0 <= upright_support_weight < 1.0:
      raise ValueError("upright_support_weight must be in [0, 1).")
    if not 0.0 <= side_support_weight < 1.0:
      raise ValueError("side_support_weight must be in [0, 1).")
    if side_pivot_speed_limit <= 0.0:
      raise ValueError("side_pivot_speed_limit must be positive.")
    if not 0.0 <= normal_formation_weight < 1.0:
      raise ValueError("normal_formation_weight must be in [0, 1).")
    if not 0.0 <= rate_progress_weight <= 1.0:
      raise ValueError("rate_progress_weight must be in [0, 1].")
    (
      asset,
      active,
      moving,
      rate_score,
      support_quality,
      coaxial_factor,
      normal_formation_progress,
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
    # ``rate_score`` is deliberately strict near the final requested speed,
    # but has no gradient once the error exceeds ``std``.  That left the
    # policy satisfied with the common-axis geometry while rotating at only
    # about 1 rad/s for a 10--18 rad/s request.  The signed, normalized
    # measured z-rate supplies the missing dense route to that same final
    # target.  It is gated by the identical support/coaxial/local-pivot
    # outcome, so spinning in the air, travelling on a floor circle, or
    # reversing direction never earns it.  This is active continuously in
    # both one-hot segments: a switch that brakes or reverses necessarily
    # loses the reward rather than being treated as a fresh manoeuvre.
    command = _command(env, command_name)
    gravity = torch.nn.functional.normalize(asset.data.projected_gravity_b, dim=1)
    actual_down_rate = torch.sum(asset.data.root_link_ang_vel_b * gravity, dim=1)
    requested_rate = command[:, 5]
    signed_rate_progress = torch.clamp(
      actual_down_rate * torch.sign(requested_rate)
      / torch.abs(requested_rate).clamp_min(speed_deadband),
      min=0.0,
      max=1.0,
    )
    dynamic_quality = coaxial_factor * pivot_stillness * (
      rate_score + rate_progress_weight * signed_rate_progress
    )
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
    # Formation must be discoverable before the strict common-axis product is
    # non-zero.  Use the same continuous measured formation to expose the
    # *rate* route as well: gating it by the strict product made a quiet
    # four-wheel default locally optimal, because it could never receive a
    # first rotation gradient.  The strict product is still part of final
    # geometry quality through the formation term's full endpoint and the
    # evaluator remains the acceptance test.
    normal_dynamic_quality = rate_score + rate_progress_weight * signed_rate_progress
    # The old bridge paid for common-axis formation even while the complete
    # wheel footprint travelled across the plane.  That exactly recreates the
    # bicycle-like shortcut seen in f50.  A normal-mode formation is valuable
    # only when its wheel-centre centroid is local; keep that direct physical
    # condition on both the geometry-discovery and rate-tracking portions.
    normal_result = (
      support_quality
      * normal_formation_progress
      * pivot_stillness
      * (
        normal_formation_weight
        + (1.0 - normal_formation_weight) * normal_dynamic_quality
      )
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
    # A zero spin rate on an active front/rear one-hot has a useful and
    # observable meaning: make the named two-wheel support, then hold it.
    # This is the same measured contact/attitude/local-centre outcome used by
    # the rotating result—only without inventing a pose target or a separate
    # reward term.  The curriculum uses it briefly so PPO can discover the
    # support before it is asked to preserve high z-rate through a switch.
    upright_static_result = support_quality * static_quality
    static_result = support_quality * (
      side_support_weight + (1.0 - side_support_weight) * static_quality
    )
    dynamic_or_upright_static = torch.where(
      moving,
      dynamic_result,
      torch.where(upright_mode, upright_static_result, torch.zeros_like(dynamic_result)),
    )
    return active.to(rate_score.dtype) * torch.where(
      dynamic_modes, dynamic_or_upright_static, static_result
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


class AerialBallisticDuration:
  """Pay each new amount of legal, continuous four-wheel flight once.

  A brief all-wheel contact gap is not enough time to rotate the body.  The
  previous height high-water reward correctly rejected ground pivots but still
  paid a micro-flight once, leaving PPO no direct reason to extend it.  Flight
  duration is the minimal physical quantity that expresses the needed launch
  impulse: hold all wheels clear long enough to complete a turn.  It contains
  no pose, desired action, phase, or reference trajectory.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.current_duration = torch.zeros(env.num_envs, device=env.device)
    self.peak_duration = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.current_duration[env_ids] = 0.0
    self.peak_duration[env_ids] = 0.0
    self.previous_active[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    target_duration: float,
  ) -> torch.Tensor:
    if target_duration <= 0.0:
      raise ValueError("target_duration must be positive.")
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    reset = (
      (~active)
      | (active & ~self.previous_active)
      | (env.episode_length_buf == 0)
    )
    self.current_duration[reset] = 0.0
    self.peak_duration[reset] = 0.0
    wheel_free = ~_has_any_contact(env, sensor_name)
    legal = ~_has_any_contact(env, nonwheel_sensor_name)
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
    # RewardManager supplies the dt integral.  Divide once so this remains a
    # discrete launch impulse rather than being attenuated by control rate.
    return valid.to(gain.dtype) * gain / env.step_dt


class AerialBallisticSpinRate:
  """Pay each new amount of correct, wheel-free spin rate once.

  A full turn needs a strong instantaneous angular launch, not a tiny average
  score over a still-undiscovered short flight.  The former per-step rate term
  was therefore one order of magnitude weaker than the one-shot height term:
  PPO learned to lift one end of the robot but had no comparably useful signal
  to snap around the commanded axis.  This companion high-water reward pays
  the best signed base rate observed while every wheel is clear.  Ground
  steering, body contact, and opposite-direction motion receive zero.  As
  with ballistic height, it specifies neither pose nor trajectory.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.peak_rate = torch.zeros(env.num_envs, device=env.device)
    self.previous_active = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.peak_rate[env_ids] = 0.0
    self.previous_active[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    target_angular_speed: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  ) -> torch.Tensor:
    if target_angular_speed <= 0.0:
      raise ValueError("target_angular_speed must be positive.")
    asset: Entity = env.scene[asset_cfg.name]
    command = _command(env, command_name)
    active = torch.sum(command[:, :5], dim=1) > 0.5
    reset = (
      (~active)
      | (active & ~self.previous_active)
      | (env.episode_length_buf == 0)
    )
    self.peak_rate[reset] = 0.0
    command_term = env.command_manager.get_term(command_name)
    launch_axis = getattr(
      command_term, "_launch_axis_w", torch.zeros(env.num_envs, 3, device=env.device)
    )
    signed_axis_rate = torch.sum(asset.data.root_link_ang_vel_w * launch_axis, dim=1)
    rate = torch.clamp(
      signed_axis_rate / target_angular_speed,
      min=0.0,
      max=1.0,
    )
    wheel_free = ~_has_any_contact(env, sensor_name)
    legal = ~_has_any_contact(env, nonwheel_sensor_name)
    valid = active & wheel_free & legal
    gain = torch.clamp(rate - self.peak_rate, min=0.0)
    self.peak_rate = torch.where(
      active,
      torch.maximum(
        self.peak_rate,
        torch.where(valid, rate, torch.zeros_like(rate)),
      ),
      torch.zeros_like(self.peak_rate),
    )
    self.previous_active = active
    return valid.to(gain.dtype) * gain / env.step_dt


class AerialNetRotationProgress:
  """Pay only new net desired-axis radians in one ballistic event.

  ``AerialRotationCommand`` already integrates signed angular displacement only
  during its first continuous wheel-free interval.  This term pays a radian
  once, when that integration reaches a new high-water mark.  A policy cannot
  collect extra reward by briefly turning forward, undoing the turn, then
  repeating it.  No pose, phase, desired rate, or joint state is introduced.

  The command-side increment is already in radians (and therefore already
  contains one ``dt``).  Return it in per-second form because RewardManager
  performs the sole time integration.  The prior extra integration reduced a
  complete-turn signal by fifty times and made short hops deceptively cheap.
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
      * increment
      / env.step_dt
    )


def aerial_event_failure(
  env: ManagerBasedRlEnv,
  command_name: str,
  target_angle: float,
) -> torch.Tensor:
  """Apply one terminal cost proportional to the missing requested turn.

  A fixed failure cost taught the first corrected aerial run to suppress its
  launch: every intermediate 0.1--0.9 turn was equally bad as a zero-turn
  fall.  The only physical quantity that should reduce this cost is the
  measured desired-axis angle already used by the endpoint.  Thus a partial
  landing is still a failure, but each additional correct radian improves its
  return and PPO has a continuous route to the one-turn completion bonus.
  """
  if target_angle <= 0.0:
    raise ValueError("target_angle must be positive.")
  command_term = env.command_manager.get_term(command_name)
  progress = getattr(
    command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
  )
  missing_fraction = 1.0 - torch.clamp(progress / target_angle, min=0.0, max=1.0)
  # RewardManager supplies the sole dt integral; this is one outcome cost.
  # ``time_out`` is also an incomplete aerial outcome unless the turn has
  # already reached its exact target, in which case ``missing_fraction`` is
  # zero.  Omitting it made a low, unqualified hop profitable: it collected
  # ground-contact lift reward then escaped through the ordinary episode
  # timeout without ever paying for its missing rotation.
  return (
    env.termination_manager.dones.to(missing_fraction.dtype)
    * missing_fraction
    / env.step_dt
  )


class AerialRotationCompletion:
  """Pay only a quiet, one-turn return to four-wheel default idle.

  Rotation progress and the launch frame are already maintained by the command
  term.  Recomputing them here, then adding partial-touchdown bridges, made a
  0.7--0.95 turn hop profitable even though the requested event had failed.
  Keep one dense reward (new desired-axis radians) and one strict endpoint.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    del cfg
    self.settle_time = torch.zeros(env.num_envs, device=env.device)
    self.awarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.settle_time[env_ids] = 0.0
    self.awarded[env_ids] = False

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
    return new_completion.to(asset.data.root_link_pos_w.dtype) / env.step_dt
