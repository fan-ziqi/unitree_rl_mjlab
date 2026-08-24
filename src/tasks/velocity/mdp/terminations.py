from __future__ import annotations

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


def normal_spin_support_lost(
  env: ManagerBasedRlEnv,
  command_name: str,
  sensor_name: str,
  speed_deadband: float = 0.20,
  grace_period_s: float = 2.0,
) -> torch.Tensor:
  """Reject a normal spin that turns by lifting one or more wheels.

  The normal one-hot is the video's folded four-wheel pivot, not a hopping
  steering turn.  PPO is free to reorganize during the short initial grace
  interval; after that all four wheels must remain grounded whenever the
  spin-rate command is active.  This is only a physical contact-validity
  condition and supplies neither a joint posture nor a motion reference.
  """
  if speed_deadband < 0.0 or grace_period_s < 0.0:
    raise ValueError("speed_deadband and grace_period_s must be non-negative.")
  command = env.command_manager.get_command(command_name)
  if command.shape[1] < 6:
    raise ValueError("normal spin requires five one-hots and a rate channel.")
  normal_spinning = (command[:, 0] > 0.5) & (
    torch.abs(command[:, 5]) > speed_deadband
  )
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  contacts = (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1)
  grace_steps = round(grace_period_s / env.step_dt)
  return (
    normal_spinning
    & (env.episode_length_buf >= grace_steps)
    & ~torch.all(contacts, dim=1)
  )


class AerialPostLandingRelaunch:
  """Terminate an aerial event that bounces into a second flight.

  The first post-flight wheel contact closes the physical event immediately,
  even though ``AerialRotationCommand`` keeps its one-hot for a brief landing
  verdict.  Watch from that contact rather than from the later command clear:
  otherwise a rebound within the verdict window would escape the one-shot
  rule.  A short contact glitch is ignored by the same threshold used for the
  initial flight.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    del cfg
    self.airborne_time = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids: torch.Tensor) -> None:
    self.airborne_time[env_ids] = 0.0

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    min_ballistic_time: float,
  ) -> torch.Tensor:
    if min_ballistic_time <= 0.0:
      raise ValueError("min_ballistic_time must be positive.")
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
    airborne = ~torch.any(contacts, dim=1)
    self.airborne_time = torch.where(
      landed & airborne,
      self.airborne_time + env.step_dt,
      torch.zeros_like(self.airborne_time),
    )
    return landed & (self.airborne_time >= min_ballistic_time)


class AerialEventFinished:
  """Truncate after the one-shot aerial event has returned to public idle.

  The command term keeps its one-hot through the short first-landing verdict,
  then exposes the literal zero-command/default controller.  One additional
  idle interval preserves the normal default controller after touchdown before
  this condition is reached.  Treating the resulting reset as a timeout,
  rather than a failure, avoids spending
  most of every three-second PPO rollout on an already-finished event.

  An episode that was sampled idle does not satisfy ``_landing_started`` and
  therefore retains the ordinary timeout path.  This is only an event
  boundary; it supplies no pose, action, or reward target.
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
    post_idle_settle_time_s: float,
  ) -> torch.Tensor:
    if post_idle_settle_time_s <= 0.0:
      raise ValueError("post_idle_settle_time_s must be positive.")
    command_term = env.command_manager.get_term(command_name)
    active = torch.sum(command_term.command, dim=1) > 0.5
    landed = getattr(
      command_term,
      "_landing_started",
      torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    idle_after_landing = landed & (~active)
    self.idle_time = torch.where(
      idle_after_landing,
      self.idle_time + env.step_dt,
      torch.zeros_like(self.idle_time),
    )
    # This manager runs immediately before the reward manager.  Half a policy
    # step of tolerance agrees with the landing-result's five-frame test, so
    # the final idle frame is rewarded before the truncation reset occurs.
    return idle_after_landing & (
      self.idle_time + 0.5 * env.step_dt >= post_idle_settle_time_s
    )
