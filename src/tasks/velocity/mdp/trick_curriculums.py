"""Shared command curricula for support, locomotion, and aerial branches.

The final policies have different command semantics.  These helpers only alter
sampling difficulty; they never add a task phase, axis, or hidden target to the
actor command.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _active_stage(
  common_step: int, stages: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
  if not stages:
    raise ValueError("At least one curriculum stage is required.")
  selected = stages[0]
  for stage in stages:
    if common_step >= stage["step"]:
      selected = stage
  return selected


def stance_spin_command_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[dict[str, Any], ...],
) -> dict[str, torch.Tensor]:
  """Stage support discovery, then train command-to-command transitions."""
  del env_ids
  stage = _active_stage(env.common_step_counter, stages)
  command = env.command_manager.get_term(command_name)
  command.set_curriculum(
    mode_probabilities=stage.get("mode_probabilities"),
    spin_idle_probability=stage.get("spin_idle_probability"),
    upright_static_probability=stage.get("upright_static_probability"),
    spin_rate_range=stage.get("spin_rate_range"),
    resampling_time_range=stage.get("resampling_time_range"),
  )
  return {
    "spin_rate_max": torch.tensor(command.cfg.spin_rate_range[1]),
    "spin_idle_probability": torch.tensor(command.cfg.spin_idle_probability),
    "upright_static_probability": torch.tensor(
      command.cfg.upright_static_probability
    ),
    "spin_command_time": torch.tensor(command.cfg.resampling_time_range[1]),
  }


def stance_locomotion_command_stages(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[dict[str, Any], ...],
) -> dict[str, torch.Tensor]:
  """Increase x/yaw ranges for the shared normal/front/rear walking task."""
  del env_ids
  stage = _active_stage(env.common_step_counter, stages)
  command = env.command_manager.get_term(command_name)
  command.set_curriculum(
    mode_probabilities=stage.get("mode_probabilities"),
    idle_probability=stage.get("idle_probability"),
    mode_idle_probabilities=stage.get("mode_idle_probabilities"),
    lin_vel_x_range=stage.get("lin_vel_x_range"),
    yaw_rate_range=stage.get("yaw_rate_range"),
    resampling_time_range=stage.get("resampling_time_range"),
  )
  return {
    "stance_locomotion_x_max": torch.tensor(command.cfg.lin_vel_x_range[1]),
    "stance_locomotion_yaw_max": torch.tensor(command.cfg.yaw_rate_range[1]),
    "stance_locomotion_idle_probability": torch.tensor(command.cfg.idle_probability),
    "stance_locomotion_command_time": torch.tensor(
      command.cfg.resampling_time_range[1]
    ),
  }
