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
  grace_steps = int(round(grace_period_s / env.step_dt))
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
  grace_steps = int(round(grace_period_s / env.step_dt))
  return (env.episode_length_buf >= grace_steps) & contact


def upright_illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  target_gravity: tuple[float, float, float],
  upright_gate_error: float,
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
  asset: Entity = env.scene[asset_name]
  target = torch.tensor(
    target_gravity, dtype=asset.data.projected_gravity_b.dtype, device=env.device
  )
  gravity_error = torch.sum(
    torch.square(asset.data.projected_gravity_b - target), dim=1
  )
  return (gravity_error < upright_gate_error) & illegal_contact(
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
  grace_steps = int(round(grace_period_s / env.step_dt))
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
  grace_steps = int(round(grace_period_s / env.step_dt))
  return (env.episode_length_buf >= grace_steps) & ~desired_present
