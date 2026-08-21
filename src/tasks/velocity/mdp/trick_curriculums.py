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
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[dict[str, Any], ...],
) -> dict[str, torch.Tensor]:
  """Progress from static two-wheel poses to a fast support-changing orbit."""
  del env_ids
  stage = _active_stage(env.common_step_counter, stages)
  command = env.command_manager.get_term(command_name)
  command.set_curriculum(
    mode_probabilities=stage.get("mode_probabilities"),
    spin_idle_probability=stage.get("spin_idle_probability"),
    spin_rate_range=stage.get("spin_rate_range"),
  )
  return {
    "spin_rate_max": torch.tensor(command.cfg.spin_rate_range[1]),
    "spin_idle_probability": torch.tensor(command.cfg.spin_idle_probability),
  }


def stance_locomotion_command_stages(
  env: "ManagerBasedRlEnv",
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
  )
  return {
    "stance_locomotion_x_max": torch.tensor(command.cfg.lin_vel_x_range[1]),
    "stance_locomotion_yaw_max": torch.tensor(command.cfg.yaw_rate_range[1]),
    "stance_locomotion_idle_probability": torch.tensor(command.cfg.idle_probability),
  }


def aerial_rotation_command_stages(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor,
  command_name: str,
  stages: tuple[dict[str, Any], ...],
) -> dict[str, torch.Tensor]:
  """Tighten airborne rotation-quality gates without changing the full turn.

  Early PPO exploration receives useful credit for the commanded rotation after
  a modest but genuine jump.  The gate then advances to the final ballistic
  height.  The actor still sees only its one-hot, and command completion stays
  one normal four-wheel full turn at all curriculum stages.
  """
  del env_ids
  stage = _active_stage(env.common_step_counter, stages)
  command = env.command_manager.get_term(command_name)
  command.set_curriculum(
    idle_probability=stage.get("idle_probability"),
    mode_probabilities=stage.get("mode_probabilities"),
    rotation_progress_clearance_start=stage.get("rotation_progress_clearance_start"),
    rotation_progress_clearance_full=stage.get("rotation_progress_clearance_full"),
    rotation_rate_clearance_start=stage.get("rotation_rate_clearance_start"),
    rotation_rate_clearance_full=stage.get("rotation_rate_clearance_full"),
  )
  return {
    "aerial_idle_probability": torch.tensor(command.cfg.idle_probability),
    "aerial_progress_clearance_full": torch.tensor(
      command.cfg.rotation_progress_clearance_full
    ),
    "aerial_rate_clearance_full": torch.tensor(
      command.cfg.rotation_rate_clearance_full
    ),
  }
