"""Fixed-command validation for the fused Go2W five-mode spin policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from src.tasks.velocity.mdp.trick_commands import StanceSpinCommandCfg


TASK_ID = "Unitree-Go2W-Spin-Stance-Flat"
MODE_NAMES = ("normal", "front", "rear", "left", "right")
GRAVITY_TARGETS = (
  (0.0, 0.0, -1.0),
  (1.0, 0.0, 0.0),
  (-1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, -1.0, 0.0),
)
CONTACT_MASKS = (
  (1, 1, 1, 1),
  (1, 1, 0, 0),
  (0, 0, 1, 1),
  (1, 0, 1, 0),
  (0, 1, 0, 1),
)


@dataclass
class EvalConfig:
  checkpoint_file: Path
  mode: int = 0
  spin_rate: float = 2.0
  all_modes: bool = False
  num_envs: int = 250
  duration_s: float = 6.0
  settle_s: float = 2.0
  device: str = "cuda:0"
  seed: int = 42
  emit_metrics: bool = False


def _configure_command(cfg, duration_s: float) -> None:
  command = cfg.commands["trick"]
  assert isinstance(command, StanceSpinCommandCfg)
  command.resampling_time_range = (duration_s + 1.0, duration_s + 1.0)


def _mode_rates(modes: torch.Tensor, spin_rate: float) -> torch.Tensor:
  """Every one-hot can request its signed spin about the current down axis."""
  return torch.full_like(modes, spin_rate, dtype=torch.float32)


def _pin_modes(command_term, modes: torch.Tensor, spin_rate: float) -> None:
  command_term.command_buf.zero_()
  command_term.command_buf[
    torch.arange(command_term.num_envs, device=modes.device), modes
  ] = 1.0
  rates = _mode_rates(modes, spin_rate).to(command_term.command_buf.dtype)
  command_term.command_buf[:, 5] = rates
  command_term._target_spin_rate.copy_(rates)


def _fixed_reset_observation(base_env: ManagerBasedRlEnv) -> TensorDict:
  base_env.observation_manager._obs_buffer = None
  observations = None
  for _ in range(10):
    observations = base_env.observation_manager.compute(update_history=True)
  assert observations is not None
  base_env.obs_buf = observations
  return TensorDict(observations, batch_size=[base_env.num_envs])


def run(cfg: EvalConfig) -> dict[str, float] | list[dict[str, float]]:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if not 0 <= cfg.mode < len(MODE_NAMES):
    raise ValueError("mode must be in [0, 4].")
  if cfg.num_envs <= 0 or cfg.duration_s <= 0.0 or cfg.settle_s < 0.0:
    raise ValueError("num_envs/duration_s must be positive; settle_s non-negative.")
  if cfg.all_modes and (cfg.num_envs < 5 or cfg.num_envs % 5):
    raise ValueError("all_modes requires a positive multiple of five environments.")

  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = cfg.num_envs
  _configure_command(env_cfg, cfg.duration_s)
  agent_cfg = load_rl_cfg(TASK_ID)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device=cfg.device)

  env.reset()
  if cfg.all_modes:
    modes = torch.arange(cfg.num_envs, device=base_env.device) // (cfg.num_envs // 5)
  else:
    modes = torch.full(
      (cfg.num_envs,), cfg.mode, dtype=torch.long, device=base_env.device
    )
  command_term = base_env.command_manager.get_term("trick")
  _pin_modes(command_term, modes, cfg.spin_rate)
  obs = _fixed_reset_observation(base_env)

  robot = base_env.scene["robot"]
  wheel_sensor = base_env.scene["wheel_ground_contact"]
  nonwheel_sensor = base_env.scene["nonwheel_ground_contact"]
  targets = torch.tensor(GRAVITY_TARGETS, device=base_env.device)
  target_contacts = torch.tensor(CONTACT_MASKS, device=base_env.device, dtype=torch.bool)
  rates = _mode_rates(modes, cfg.spin_rate).to(base_env.device)
  dynamic = (modes == 0) & (rates.abs() > 0.20)
  fixed_spin = (modes >= 1) & (rates.abs() > 0.20)
  static = ~(dynamic | fixed_spin)
  trial_open = torch.ones(cfg.num_envs, dtype=torch.bool, device=base_env.device)
  failed = torch.zeros_like(trial_open)
  sample_count = torch.zeros(cfg.num_envs, device=base_env.device)
  alignment_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  support_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  horizontal_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  rate_error_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  nonwheel_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  side_geometry_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  side_center_speed_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  steady_count = torch.zeros(cfg.num_envs, device=base_env.device)
  steady_success = torch.zeros(cfg.num_envs, device=base_env.device)

  num_steps = round(cfg.duration_s / base_env.step_dt)
  wheel_site_ids, _ = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  previous_side_center = torch.zeros(cfg.num_envs, 2, device=base_env.device)
  side_center_initialized = torch.zeros(cfg.num_envs, dtype=torch.bool, device=base_env.device)
  with torch.inference_mode():
    for step in range(num_steps):
      obs, _, dones, _ = env.step(policy(obs))
      valid = trial_open
      found = wheel_sensor.data.found
      assert found is not None
      contacts = (found.reshape(cfg.num_envs, found.shape[1], -1) > 0).any(dim=-1)
      gravity = torch.nn.functional.normalize(robot.data.projected_gravity_b, dim=1)
      alignment = 0.5 * (1.0 + torch.sum(gravity * targets[modes], dim=1))
      static_contact = torch.all(contacts == target_contacts[modes], dim=1)
      dynamic_contact = torch.sum(contacts, dim=1) == 2
      horizontal = torch.abs(gravity[:, 2]) < 0.35
      down_rate = torch.sum(robot.data.root_link_ang_vel_b * gravity, dim=1)
      rate_error = torch.abs(down_rate - rates)
      nonwheel_found = nonwheel_sensor.data.found
      assert nonwheel_found is not None
      nonwheel = (
        nonwheel_found.reshape(cfg.num_envs, nonwheel_found.shape[1], -1) > 0
      ).any(dim=(1, 2))
      side = modes >= 3
      side_index = torch.clamp(modes - 3, 0, 1)
      wheel_pos = robot.data.site_pos_w[:, wheel_site_ids, :2]
      pair_indices = torch.tensor(((0, 2), (1, 3)), device=base_env.device)
      batch = torch.arange(cfg.num_envs, device=base_env.device)
      first = pair_indices[side_index, 0]
      second = pair_indices[side_index, 1]
      delta_xy = wheel_pos[batch, second] - wheel_pos[batch, first]
      quat = robot.data.root_link_quat_w
      body_x = quat_apply(
        quat,
        torch.tensor((1.0, 0.0, 0.0), device=base_env.device, dtype=quat.dtype).expand(cfg.num_envs, -1),
      )[:, :2]
      body_z = quat_apply(
        quat,
        torch.tensor((0.0, 0.0, 1.0), device=base_env.device, dtype=quat.dtype).expand(cfg.num_envs, -1),
      )[:, :2]
      body_x = body_x / torch.linalg.vector_norm(body_x, dim=1, keepdim=True).clamp_min(1.0e-6)
      body_z = body_z / torch.linalg.vector_norm(body_z, dim=1, keepdim=True).clamp_min(1.0e-6)
      transverse = torch.abs(torch.sum(delta_xy * body_z, dim=1))
      longitudinal = torch.abs(torch.sum(delta_xy * body_x, dim=1))
      side_geometry = (transverse >= 0.08) & (longitudinal <= 0.10)
      side_center = 0.5 * (wheel_pos[batch, first] + wheel_pos[batch, second])
      side_center_speed = torch.linalg.vector_norm(
        (side_center - previous_side_center) / base_env.step_dt, dim=1
      )
      previous_side_center.copy_(side_center)
      side_center_initialized[:] = True
      side_in_place = side_center_speed <= 0.20
      support_ok = torch.where(dynamic, dynamic_contact, static_contact)
      rate_ok = torch.where(dynamic | fixed_spin, rate_error < 0.75, torch.ones_like(dynamic))
      pose_ok = torch.where(dynamic, horizontal, alignment >= 0.97)
      success = support_ok & rate_ok & pose_ok & ~nonwheel & (~side | (side_geometry & side_in_place))

      sample_count += valid.float()
      alignment_sum += valid.float() * alignment
      support_sum += valid.float() * support_ok.float()
      horizontal_sum += valid.float() * horizontal.float()
      rate_error_sum += valid.float() * rate_error
      nonwheel_sum += valid.float() * nonwheel.float()
      side_geometry_sum += valid.float() * side.float() * side_geometry.float()
      side_center_speed_sum += valid.float() * side.float() * side_center_speed
      if (step + 1) * base_env.step_dt >= cfg.settle_s:
        steady_count += valid.float()
        steady_success += valid.float() * success.float()
      failed |= valid & dones.bool()
      trial_open &= ~dones.bool()

  def summarize(mode: int) -> dict[str, float]:
    mask = modes == mode
    denom = sample_count[mask].clamp_min(1.0)
    steady_denom = steady_count[mask].clamp_min(1.0)
    return {
      "mode": MODE_NAMES[mode],
      "mode_index": float(mode),
      "num_envs": float(mask.sum().item()),
      "spin_rate": cfg.spin_rate,
      "mean_gravity_alignment": (alignment_sum[mask] / denom).mean().item(),
      "mean_support_match_rate": (support_sum[mask] / denom).mean().item(),
      "mean_horizontal_rate": (horizontal_sum[mask] / denom).mean().item(),
      "mean_spin_rate_abs_error": (rate_error_sum[mask] / denom).mean().item(),
      "mean_nonwheel_contact_rate": (nonwheel_sum[mask] / denom).mean().item(),
      "mean_side_balancer_geometry_rate": (side_geometry_sum[mask] / denom).mean().item(),
      "mean_side_support_center_speed": (side_center_speed_sum[mask] / denom).mean().item(),
      "steady_success_rate": (steady_success[mask] / steady_denom).mean().item(),
      "failed_rate": failed[mask].float().mean().item(),
      "unfinished_rate": trial_open[mask].float().mean().item(),
    }

  metrics = [summarize(mode) for mode in range(5)] if cfg.all_modes else summarize(cfg.mode)
  env.close()
  return metrics


def main() -> None:
  cfg = tyro.cli(EvalConfig)
  metrics = run(cfg)
  if cfg.emit_metrics:
    print(json.dumps(metrics, sort_keys=True))
  elif isinstance(metrics, list):
    for metric in metrics:
      print(" ".join(f"{key}: {value}" for key, value in metric.items()))
  else:
    print(" ".join(f"{key}: {value}" for key, value in metrics.items()))


if __name__ == "__main__":
  main()
