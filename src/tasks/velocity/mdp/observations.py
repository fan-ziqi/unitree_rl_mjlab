from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def joint_pos_rel_without_wheel(
  env: ManagerBasedRlEnv,
  wheel_asset_cfg: SceneEntityCfg,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Relative joint position observation with unbounded wheel angles masked.

  Wheel joints are continuous, so their absolute position has no useful meaning
  for the policy and grows without bound during rolling.  Keep the observation
  layout identical to the complete joint-state vector but set those entries to
  zero, matching the Go2W observation convention.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos
  assert default_joint_pos is not None
  obs = asset.data.joint_pos[:, asset_cfg.joint_ids] - default_joint_pos[:, asset_cfg.joint_ids]

  wheel_ids = wheel_asset_cfg.joint_ids
  assert isinstance(wheel_ids, list), "wheel_asset_cfg must select explicit wheel joints."
  if isinstance(asset_cfg.joint_ids, slice):
    obs[:, wheel_ids] = 0.0
  else:
    wheel_id_set = set(wheel_ids)
    wheel_obs_ids = [i for i, joint_id in enumerate(asset_cfg.joint_ids) if joint_id in wheel_id_set]
    obs[:, wheel_obs_ids] = 0.0
  return obs


def foot_height(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # (num_envs, num_sites)


def foot_air_time(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  current_air_time = sensor_data.current_air_time
  assert current_air_time is not None
  return current_air_time


def foot_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def foot_contact_forces(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.force is not None
  forces_flat = sensor_data.force.flatten(start_dim=1)  # [B, N*3]
  return torch.sign(forces_flat) * torch.log1p(torch.abs(forces_flat))


def base_height_above_default(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  """Root height relative to this robot model's nominal reset height.

  The quantity is a generic proprioceptive signal: it lets a policy separate
  ground contact, takeoff, apex, and landing without exposing a task phase or
  a trick-specific target.
  """
  asset: Entity = env.scene[asset_cfg.name]
  default_root_state = asset.data.default_root_state
  assert default_root_state is not None
  return (asset.data.root_link_pos_w[:, 2] - default_root_state[:, 2]).unsqueeze(1)


def contact_state(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Binary contact state, flattened to one value per selected contact body."""
  sensor: ContactSensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return (found.reshape(env.num_envs, found.shape[1], -1) > 0).any(dim=-1).float()


def phase(env: ManagerBasedRlEnv, period: float, command_name: str) -> torch.Tensor:
    global_phase = (env.episode_length_buf * env.step_dt) % period / period
    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
    stand_mask = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
    return phase
