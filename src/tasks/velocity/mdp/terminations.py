from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def illegal_contact_after_grace(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
  grace_period_s: float = 0.4,
) -> torch.Tensor:
  """Enable illegal-contact termination after a short reset settling window.

  A two-wheel upright reset is physically valid but is far less tolerant of
  the policy's initial Gaussian actions than a four-wheel reset.  Without a
  brief window, every untrained rollout ends at its first control step and PPO
  has no state-action sequence from which to learn balance.  Contacts remain
  fully simulated and rewarded during the window; only the terminal label is
  delayed.
  """
  if grace_period_s < 0.0:
    raise ValueError("grace_period_s must be non-negative.")
  grace_steps = round(grace_period_s / env.step_dt)
  return (env.episode_length_buf >= grace_steps) & illegal_contact(
    env, sensor_name=sensor_name, force_threshold=force_threshold
  )


def illegal_contact_after_mode_switch_grace(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  switch_grace_period_s: float = 0.20,
  initial_transition_grace_period_s: float = 0.45,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Allow brief load-transfer windows when entering or switching modes.

  A side-support change has to unload one wheel pair before the other pair can
  catch the body.  The same is true for the first idle -> one-hot entry: the
  policy must be allowed to roll away from the four-wheel reset before a
  selected pair can carry the trunk.  Terminating on the first transient
  thigh/trunk contact cuts either sequence off before PPO can observe the new
  support.  The entry window is restricted to the command term's initial
  active phase; the shorter post-switch window is restricted to its actual
  second phase.  Outside those windows non-wheel contact remains an immediate
  failure.
  """
  if switch_grace_period_s < 0.0:
    raise ValueError("switch_grace_period_s must be non-negative.")
  if initial_transition_grace_period_s < 0.0:
    raise ValueError("initial_transition_grace_period_s must be non-negative.")
  command_term = env.command_manager.get_term(command_name)
  phase = getattr(
    command_term,
    "_transition_phase",
    torch.full((env.num_envs,), 3, dtype=torch.int8, device=env.device),
  )
  transition_time = getattr(
    command_term,
    "_transition_time",
    torch.zeros(env.num_envs, device=env.device),
  )
  scheduled = getattr(command_term, "_scheduled_command", None)
  next_scheduled = getattr(command_term, "_next_scheduled_command", None)
  if scheduled is None or next_scheduled is None:
    return illegal_contact(env, sensor_name=sensor_name, force_threshold=force_threshold)
  changed = torch.argmax(scheduled[:, :5], dim=1) != torch.argmax(
    next_scheduled[:, :5], dim=1
  )
  active_time = float(getattr(command_term.cfg, "transition_active_time", 0.0))
  switch_start = 0.5 * active_time
  initial_active = torch.sum(scheduled[:, :5], dim=1) > 0.5
  in_initial_window = (
    initial_active
    & (phase == 1)
    & (transition_time <= initial_transition_grace_period_s)
  )
  in_window = (
    in_initial_window
    | (
      changed
      & (phase == 2)
      & (transition_time <= switch_start + switch_grace_period_s)
    )
  )
  return (~in_window) & illegal_contact(
    env, sensor_name=sensor_name, force_threshold=force_threshold
  )


def terrain_contact_after_grace(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  grace_period_s: float = 0.0,
) -> torch.Tensor:
  """Terminate on any matching terrain contact, independent of its force.

  This is deliberately stricter than :func:`illegal_contact_after_grace`.
  A base or thigh resting on the ground can be a mechanically useful prop even
  when its net normal force falls below a chosen threshold, so it must not be
  admitted as a solution to a wheel-only support task.
  """
  if grace_period_s < 0.0:
    raise ValueError("grace_period_s must be non-negative.")
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  contact = (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=(1, 2))
  grace_steps = round(grace_period_s / env.step_dt)
  return (env.episode_length_buf >= grace_steps) & contact


def upright_illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
  minimum_root_height: float = 0.0,
  force_threshold: float = 0.5,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Terminate a near-upright robot that props itself on a forbidden body.

  This permits an exploratory transition to pass briefly near the floor, but
  makes it impossible to settle or drive in a pose supported by anything other
  than the configured wheel geoms.  It is a final-support validity condition,
  not a staged get-up instruction.
  """
  if upright_gate_error <= 0.0:
    raise ValueError("upright_gate_error must be positive.")
  if minimum_root_height < 0.0:
    raise ValueError("minimum_root_height must be non-negative.")
  asset: Entity = env.scene[asset_name]
  target = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target), dim=1
  )
  # Attitude alone is not the completed two-wheel stance: while the body is
  # still low, the front wheels can briefly remain in contact as the rear legs
  # extend.  Require the final root-height region too, so this remains a
  # final-support validity check rather than cutting off that physical motion.
  final_height_reached = asset.data.root_link_pos_w[:, 2] >= minimum_root_height
  return (gravity_error < upright_gate_error) & final_height_reached & illegal_contact(
    env, sensor_name=sensor_name, force_threshold=force_threshold
  )


def command_gravity_fall(
  env: ManagerBasedRlEnv,
  command_name: str,
  gravity_targets: tuple[tuple[float, float, float], ...],
  min_alignment: float,
  grace_period_s: float = 0.4,
  asset_name: str = "robot",
) -> torch.Tensor:
  """Terminate only when the body has genuinely fallen from its commanded pose.

  For a wheel handstand, link/terrain contacts around the support wheels are
  part of the valid geometry, so a generic non-wheel contact sensor cannot be
  used as a fall proxy.  The command already selects the desired gravity
  direction; losing that direction is the task-relevant definition of a fall.
  """
  if not -1.0 < min_alignment < 1.0 or grace_period_s < 0.0:
    raise ValueError("min_alignment must be in (-1, 1) and grace_period_s non-negative.")
  command = env.command_manager.get_command(command_name)
  num_modes = len(gravity_targets)
  if command.shape[1] < num_modes:
    raise ValueError("command does not contain enough one-hot mode entries.")
  mode = torch.argmax(command[:, :num_modes], dim=1)
  target = torch.tensor(gravity_targets, device=env.device, dtype=command.dtype)[mode]
  asset: Entity = env.scene[asset_name]
  gravity = asset.data.projected_gravity_b
  gravity = gravity / torch.linalg.vector_norm(gravity, dim=1, keepdim=True).clamp_min(1.0e-6)
  alignment = torch.sum(gravity * target, dim=1)
  grace_steps = round(grace_period_s / env.step_dt)
  return (env.episode_length_buf >= grace_steps) & (alignment < min_alignment)


def command_support_lost(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  contact_masks: tuple[tuple[float, float, float, float], ...],
  grace_period_s: float = 0.12,
) -> torch.Tensor:
  """End a stance episode when either commanded support wheel leaves ground."""
  if grace_period_s < 0.0:
    raise ValueError("grace_period_s must be non-negative.")
  command = env.command_manager.get_command(command_name)
  num_modes = len(contact_masks)
  mode = torch.argmax(command[:, :num_modes], dim=1)
  target = torch.tensor(contact_masks, device=env.device, dtype=torch.bool)[mode]
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  contacts = (sensor.data.found.reshape(env.num_envs, sensor.data.found.shape[1], -1) > 0).any(dim=-1)
  desired_present = torch.all(contacts | ~target, dim=1)
  grace_steps = round(grace_period_s / env.step_dt)
  return (env.episode_length_buf >= grace_steps) & ~desired_present


class AerialPostLandingRelaunch:
  """Terminate an aerial event that bounces into a second flight.

  A first individual wheel graze is not yet a landed robot: ending the event
  as soon as that grazing contact rebounds removes the only contact-braking
  evidence from PPO.  Arm the one-shot relaunch check only after a short,
  continuous four-wheel landing.  A later genuinely wheel-free interval then
  is a second jump; an ordinary first-impact bounce remains part of the same
  event and can recover or fail through its normal landing outcome.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.airborne_time = torch.zeros(env.num_envs, device=env.device)
    self.landed_time = torch.zeros(env.num_envs, device=env.device)
    self.relaunch_armed = torch.zeros(
      env.num_envs, dtype=torch.bool, device=env.device
    )

  def reset(self, env_ids: torch.Tensor) -> None:
    self.airborne_time[env_ids] = 0.0
    self.landed_time[env_ids] = 0.0
    self.relaunch_armed[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    min_ballistic_time: float,
    arming_settle_time: float,
  ) -> torch.Tensor:
    if min_ballistic_time <= 0.0 or arming_settle_time <= 0.0:
      raise ValueError("aerial relaunch times must be positive.")
    command_term = env.command_manager.get_term(command_name)
    landed = getattr(
      command_term,
      "_landing_started",
      torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    sensor: ContactSensor = env.scene[sensor_name]
    found = sensor.data.found
    assert found is not None
    contacts = (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1)
    four_wheel_landed = torch.all(contacts, dim=1)
    self.landed_time = torch.where(
      landed & (~self.relaunch_armed) & four_wheel_landed,
      self.landed_time + env.step_dt,
      torch.where(
        self.relaunch_armed,
        self.landed_time,
        torch.zeros_like(self.landed_time),
      ),
    )
    self.relaunch_armed |= landed & (
      self.landed_time + 0.5 * env.step_dt >= arming_settle_time
    )
    airborne = ~torch.any(contacts, dim=1)
    self.airborne_time = torch.where(
      self.relaunch_armed & airborne,
      self.airborne_time + env.step_dt,
      torch.zeros_like(self.airborne_time),
    )
    return self.relaunch_armed & (self.airborne_time >= min_ballistic_time)


class AerialEventFinished:
  """Truncate after a one-shot aerial event has *stably* returned to idle.

  The command term keeps its one-hot through the short first-landing verdict,
  then exposes the literal zero-command/default controller.  One additional
  idle interval preserves the normal default controller after touchdown before
  this condition is reached.  Treating the resulting reset as a timeout,
  rather than a failure, avoids spending
  most of every three-second PPO rollout on an already-finished event.

  An episode that was sampled idle does not satisfy ``_landing_started`` and
  therefore retains the ordinary timeout path.  The physical landing checks
  below are the same measured endpoint used by the completion reward; they
  prevent a one-hot clear during a rebound from truncating the useful
  post-contact learning window.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.idle_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.idle_time[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    post_idle_settle_time_s: float,
    target_angle: float = math.tau,
    max_overrotation: float = 0.50,
  ) -> torch.Tensor:
    if post_idle_settle_time_s <= 0.0 or target_angle <= 0.0 or max_overrotation < 0.0:
      raise ValueError("aerial event-finish parameters are invalid.")
    command_term = env.command_manager.get_term(command_name)
    active = torch.sum(command_term.command, dim=1) > 0.5
    landed = getattr(
      command_term,
      "_landing_started",
      torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    valid_turn = (progress >= target_angle) & (
      progress <= target_angle + max_overrotation
    )
    idle_after_landing = landed & (~active)

    # Do not end a valid-turn episode merely because the public command has
    # cleared.  A first wheel contact can still be a rebound, and a body/leg
    # can be resting on the floor while all four wheel bits are not present.
    # Require the measured quiet wheel-supported endpoint before truncating;
    # this is deliberately the same endpoint used by AerialRotationCompletion.
    wheel_sensor: ContactSensor = env.scene[sensor_name]
    found = wheel_sensor.data.found
    assert found is not None
    wheel_contacts = (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1)
    nonwheel_sensor: ContactSensor = env.scene[nonwheel_sensor_name]
    nonwheel_found = nonwheel_sensor.data.found
    assert nonwheel_found is not None
    nonwheel_contact = (
      nonwheel_found.reshape(env.num_envs, nonwheel_found.shape[1], -1) > 0
    ).any(dim=(1, 2))
    asset = env.scene[command_term.cfg.entity_name]
    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0), dtype=asset.data.projected_gravity_b.dtype, device=env.device
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    launch_quat = getattr(
      command_term, "_launch_root_quat_w", torch.zeros_like(asset.data.root_link_quat_w)
    )
    orientation_similarity = torch.abs(
      torch.sum(asset.data.root_link_quat_w * launch_quat, dim=1)
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    stable_landing = (
      torch.all(wheel_contacts, dim=1)
      & ~nonwheel_contact
      & (gravity_error < command_term.cfg.landing_gravity_error_limit)
      & (orientation_similarity >= command_term.cfg.landing_orientation_dot_min)
      & (linear_speed < command_term.cfg.landing_linear_velocity_limit)
      & (angular_speed < command_term.cfg.landing_angular_velocity_limit)
    )
    self.idle_time = torch.where(
      idle_after_landing & stable_landing,
      self.idle_time + env.step_dt,
      torch.zeros_like(self.idle_time),
    )
    # This manager runs immediately before the reward manager.  Half a policy
    # step of tolerance agrees with the landing-result's five-frame test, so
    # the final idle frame is rewarded before the truncation reset occurs.
    # A partial hop must never receive the benign ``time_out`` label merely
    # because it returned to default idle.  That previous behaviour paid the
    # first 0.2--0.4 revolution through the dense progress term, then ended
    # the episode without a failure signal—a stable local optimum that the
    # m800 recording exposes directly.  Only a legal one-turn event gets the
    # normal timeout boundary; the complementary case is handled by the
    # non-timeout termination below.
    return idle_after_landing & valid_turn & stable_landing & (
      self.idle_time + 0.5 * env.step_dt >= post_idle_settle_time_s
    )


class AerialIncompleteLanding:
  """Fail a landed aerial event that did not complete its one requested turn.

  This is an outcome validity rule, not a landing trajectory: after the same
  short idle window used by :class:`AerialEventFinished`, distinguish only the
  measured total desired-axis angle.  It prevents a safe-looking partial hop
  from receiving rotation reward plus an unpenalized timeout.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.idle_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.idle_time[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    nonwheel_sensor_name: str,
    post_idle_settle_time_s: float,
    target_angle: float = math.tau,
    max_overrotation: float = 0.50,
  ) -> torch.Tensor:
    if post_idle_settle_time_s <= 0.0 or target_angle <= 0.0 or max_overrotation < 0.0:
      raise ValueError("aerial incomplete-landing parameters are invalid.")
    command_term = env.command_manager.get_term(command_name)
    active = torch.sum(command_term.command, dim=1) > 0.5
    landed = getattr(
      command_term,
      "_landing_started",
      torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    progress = getattr(
      command_term, "_rotation_progress", torch.zeros(env.num_envs, device=env.device)
    )
    valid_turn = (progress >= target_angle) & (
      progress <= target_angle + max_overrotation
    )
    idle_after_landing = landed & (~active)
    wheel_sensor: ContactSensor = env.scene[sensor_name]
    found = wheel_sensor.data.found
    assert found is not None
    wheel_contacts = (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1)
    nonwheel_sensor: ContactSensor = env.scene[nonwheel_sensor_name]
    nonwheel_found = nonwheel_sensor.data.found
    assert nonwheel_found is not None
    nonwheel_contact = (
      nonwheel_found.reshape(env.num_envs, nonwheel_found.shape[1], -1) > 0
    ).any(dim=(1, 2))
    asset = env.scene[command_term.cfg.entity_name]
    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0), dtype=asset.data.projected_gravity_b.dtype, device=env.device
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    launch_quat = getattr(
      command_term, "_launch_root_quat_w", torch.zeros_like(asset.data.root_link_quat_w)
    )
    orientation_similarity = torch.abs(
      torch.sum(asset.data.root_link_quat_w * launch_quat, dim=1)
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    stable_landing = (
      torch.all(wheel_contacts, dim=1)
      & ~nonwheel_contact
      & (gravity_error < command_term.cfg.landing_gravity_error_limit)
      & (orientation_similarity >= command_term.cfg.landing_orientation_dot_min)
      & (linear_speed < command_term.cfg.landing_linear_velocity_limit)
      & (angular_speed < command_term.cfg.landing_angular_velocity_limit)
    )
    # For an incomplete/unstable attempt the clock must continue through the
    # whole post-landing verdict window; unlike the successful event above it
    # is intentionally not gated by ``stable_landing``.
    self.idle_time = torch.where(
      idle_after_landing,
      self.idle_time + env.step_dt,
      torch.zeros_like(self.idle_time),
    )
    return idle_after_landing & ((~valid_turn) | (valid_turn & ~stable_landing)) & (
      self.idle_time + 0.5 * env.step_dt >= post_idle_settle_time_s
    )
