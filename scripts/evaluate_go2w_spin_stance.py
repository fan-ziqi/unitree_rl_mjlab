"""Fixed-command validation for the fused Go2W five-mode spin policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from tensordict import TensorDict

from src.tasks.velocity.mdp.trick_commands import StanceSpinCommandCfg
from src.tasks.velocity.mdp.trick_rewards import (
  normal_four_wheel_pivot_geometry,
)

TASK_ID = "Unitree-Go2W-Spin-Stance-Flat"
MODE_NAMES = ("normal", "front", "rear", "left", "right")
GRAVITY_TARGETS = (
  # Normal preserves a level body while its four wheels form one compact axle.
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
DYNAMIC_PIVOT_MODES = (0, 1, 2, 3, 4)


def _pair_coaxiality(
  wheel_axles: torch.Tensor, wheel_positions: torch.Tensor, first: int, second: int
) -> torch.Tensor:
  """Measure whether two wheel centres lie on their common horizontal axle."""
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


@dataclass
class EvalConfig:
  checkpoint_file: Path
  mode: int = 0
  # This is the reference-speed normal pivot, replacing the visibly slow
  # 4--6-rad/s prototype range.
  spin_rate: float = 15.0
  all_modes: bool = False
  num_envs: int = 250
  duration_s: float = 6.0
  settle_s: float = 2.0
  ramp_spin_rate: bool = True
  device: str = "cuda:0"
  seed: int = 42
  emit_metrics: bool = False


def _configure_command(cfg, duration_s: float) -> None:
  command = cfg.commands["trick"]
  assert isinstance(command, StanceSpinCommandCfg)
  command.resampling_time_range = (duration_s + 1.0, duration_s + 1.0)


def _mode_rates(modes: torch.Tensor, spin_rate: float) -> torch.Tensor:
  """Return the signed public rate carried by every active one-hot."""
  rates = torch.full_like(modes, spin_rate, dtype=torch.float32)
  # Side one-hots are held two-wheel supports; their public rate is ignored.
  return torch.where(modes >= 3, torch.zeros_like(rates), rates)


def _expected_down_rates(modes: torch.Tensor, spin_rate: float) -> torch.Tensor:
  """Every physical pivot tracks the signed world-down command rate."""
  return _mode_rates(modes, spin_rate)


def _pin_modes(
  command_term,
  modes: torch.Tensor,
  spin_rate: float,
  ramp_spin_rate: bool = True,
  preserve_spin_rate: bool = False,
) -> None:
  """Pin one-hot targets, optionally reproducing the training-rate ramp."""
  current_rate = command_term.command_buf[:, 5].clone()
  command_term.command_buf.zero_()
  rates = _mode_rates(modes, spin_rate).to(command_term.command_buf.dtype)
  active = torch.ones_like(modes, dtype=torch.bool)
  command_term.command_buf[
    torch.arange(command_term.num_envs, device=modes.device)[active], modes[active]
  ] = 1.0
  if not ramp_spin_rate:
    command_term.command_buf[:, 5] = rates
  elif preserve_spin_rate:
    # A direct dynamic-mode switch must retain the public signed rate.  The
    # rate ramp remains available for the initial idle -> active entry only.
    command_term.command_buf[:, 5] = current_rate
  command_term._target_spin_rate.copy_(rates)
  # A fixed-command evaluation must not be overwritten by the training
  # sampler's idle -> A -> B transition sequence after 0.8 seconds.  Leave
  # the rate ramp live when requested, but mark the sequence itself complete.
  command_term._transition_phase.fill_(3)
  command_term._transition_time.zero_()


def _fixed_reset_observation(base_env: ManagerBasedRlEnv) -> TensorDict:
  base_env.observation_manager._obs_buffer = None
  observations = None
  history_length = base_env.cfg.observations["actor"].history_length or 1
  for _ in range(history_length):
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
  _pin_modes(command_term, modes, cfg.spin_rate, cfg.ramp_spin_rate)
  obs = _fixed_reset_observation(base_env)

  robot = base_env.scene["robot"]
  wheel_sensor = base_env.scene["wheel_ground_contact"]
  nonwheel_sensor = base_env.scene["nonwheel_ground_contact"]
  wheel_site_ids, _ = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  targets = torch.tensor(GRAVITY_TARGETS, device=base_env.device)
  target_contacts = torch.tensor(
    CONTACT_MASKS, device=base_env.device, dtype=torch.bool
  )
  expected_rates = _expected_down_rates(modes, cfg.spin_rate).to(base_env.device)
  dynamic_pivot = torch.ones_like(modes, dtype=torch.bool)
  active_spin = dynamic_pivot & (expected_rates.abs() > 0.20)
  trial_open = torch.ones(cfg.num_envs, dtype=torch.bool, device=base_env.device)
  failed = torch.zeros_like(trial_open)
  sample_count = torch.zeros(cfg.num_envs, device=base_env.device)
  alignment_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  support_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  axle_coaxiality_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  normal_axis_parallel_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  normal_common_axle_line_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  normal_compact_span_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  normal_root_clearance_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  selected_root_clearance_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  non_support_wheel_height_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  rate_error_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  down_rate_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  support_center_speed_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  nonwheel_sum = torch.zeros(cfg.num_envs, device=base_env.device)
  steady_count = torch.zeros(cfg.num_envs, device=base_env.device)
  steady_success = torch.zeros(cfg.num_envs, device=base_env.device)
  steady_normal_four_wheel_support = torch.ones(
    cfg.num_envs, dtype=torch.bool, device=base_env.device
  )
  previous_center = torch.zeros(cfg.num_envs, 2, device=base_env.device)
  center_initialized = torch.zeros(
    cfg.num_envs, dtype=torch.bool, device=base_env.device
  )

  num_steps = round(cfg.duration_s / base_env.step_dt)
  with torch.inference_mode():
    for step in range(num_steps):
      obs, _, dones, _ = env.step(policy(obs))
      valid = trial_open
      found = wheel_sensor.data.found
      assert found is not None
      contacts = (found.reshape(cfg.num_envs, found.shape[1], -1) > 0).any(dim=-1)
      gravity = torch.nn.functional.normalize(robot.data.projected_gravity_b, dim=1)
      effective_modes = modes
      target_alignment = 0.5 * (1.0 + torch.sum(gravity * targets[effective_modes], dim=1))
      down_rate = torch.sum(robot.data.root_link_ang_vel_b * gravity, dim=1)
      # The rate channel is intentionally ramped in the training command
      # term.  Measure against the actual public command seen by the actor at
      # this frame, not its eventual target, so default validation is not an
      # out-of-distribution 0 -> 8 rad/s step.
      expected_rate = torch.where(
        dynamic_pivot,
        command_term.command[:, 5],
        torch.zeros_like(command_term.command[:, 5]),
      )
      rate_error = torch.abs(down_rate - expected_rate)
      batch = torch.arange(cfg.num_envs, device=base_env.device)
      nonwheel_found = nonwheel_sensor.data.found
      assert nonwheel_found is not None
      nonwheel = (
        nonwheel_found.reshape(cfg.num_envs, nonwheel_found.shape[1], -1) > 0
      ).any(dim=(1, 2))

      wheel_quat = robot.data.site_quat_w[:, wheel_site_ids].reshape(-1, 4)
      local_axle = torch.tensor(
        (0.0, 1.0, 0.0), dtype=wheel_quat.dtype, device=base_env.device
      ).expand(wheel_quat.shape[0], -1)
      wheel_axles = quat_apply(wheel_quat, local_axle).reshape(cfg.num_envs, 4, 3)
      wheel_positions = robot.data.site_pos_w[:, wheel_site_ids]
      (
        normal_axis_parallel,
        normal_common_axle_line,
        normal_compact_span,
      ) = normal_four_wheel_pivot_geometry(wheel_axles, wheel_positions)
      support_masks = target_contacts[effective_modes]
      support_mask = support_masks
      support_ok = torch.all(contacts == target_contacts[effective_modes], dim=1)
      wheel_xy = wheel_positions[:, :, :2]
      center = (wheel_xy * support_mask.unsqueeze(2)).sum(dim=1) / support_mask.sum(
        dim=1, keepdim=True
      ).clamp_min(1.0)
      raw_center_speed = torch.linalg.vector_norm(
        (center - previous_center) / base_env.step_dt, dim=1
      )
      # ``site_pos_w`` includes the tiled world origin of each parallel
      # environment.  The first sample therefore is not a physical velocity:
      # compare only after one measured centre has been stored for this
      # environment.  The PPO reward uses MuJoCo site velocity directly and
      # was never affected; this fixes validation-only false drift.
      center_speed = torch.where(
        center_initialized, raw_center_speed, torch.zeros_like(raw_center_speed)
      )
      center_measured = center_initialized
      previous_center.copy_(center)
      center_initialized[:] = True
      front_rear_coaxiality = torch.stack(
        (
          _pair_coaxiality(wheel_axles, wheel_positions, 0, 1),
          _pair_coaxiality(wheel_axles, wheel_positions, 2, 3),
        ),
        dim=1,
      )
      pair_coaxiality_for_mode = torch.stack(
        (
          front_rear_coaxiality[:, 0],
          front_rear_coaxiality[:, 0],
          front_rear_coaxiality[:, 1],
          _pair_coaxiality(wheel_axles, wheel_positions, 0, 2),
          _pair_coaxiality(wheel_axles, wheel_positions, 1, 3),
        ),
        dim=1,
      )[batch, modes]
      coaxiality = torch.where(
        modes == 0,
        normal_axis_parallel * normal_common_axle_line,
        pair_coaxiality_for_mode,
      )
      total_angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=1)
      rate_ok = torch.where(active_spin, rate_error < 0.75, total_angular_speed < 1.0)
      normal_root_clearance = (
        robot.data.root_link_pos_w[:, 2] - wheel_positions[:, :, 2].mean(dim=1)
      )
      selected_wheel_height = (
        wheel_positions[:, :, 2] * support_mask
      ).sum(dim=1) / support_mask.sum(dim=1).clamp_min(1.0)
      selected_root_clearance = robot.data.root_link_pos_w[:, 2] - selected_wheel_height
      non_support_mask = (~support_mask).to(wheel_positions.dtype)
      non_support_wheel_height = (
        wheel_positions[:, :, 2] * non_support_mask
      ).sum(dim=1) / non_support_mask.sum(dim=1).clamp_min(1.0)
      pose_ok = torch.where(
        modes == 0,
        (target_alignment >= 0.97) & (normal_root_clearance >= 0.35),
        target_alignment >= 0.97,
      )
      pivot_ok = center_measured & (center_speed < 0.08)
      normal_layout_ok = (
        (normal_axis_parallel >= 0.90)
        & (normal_common_axle_line >= 0.85)
        & (normal_compact_span >= 0.50)
      )
      axle_ok = torch.where(
        active_spin,
        torch.where(modes == 0, normal_layout_ok, coaxiality >= 0.90),
        torch.ones_like(active_spin),
      )
      success = support_ok & rate_ok & pose_ok & pivot_ok & axle_ok & ~nonwheel

      sample_count += valid.float()
      alignment_sum += valid.float() * target_alignment
      support_sum += valid.float() * support_ok.float()
      axle_coaxiality_sum += valid.float() * active_spin.float() * coaxiality
      normal_axis_parallel_sum += (
        valid.float() * (modes == 0).float() * normal_axis_parallel
      )
      normal_common_axle_line_sum += (
        valid.float() * (modes == 0).float() * normal_common_axle_line
      )
      normal_compact_span_sum += (
        valid.float() * (modes == 0).float() * normal_compact_span
      )
      normal_root_clearance_sum += (
        valid.float() * (modes == 0).float() * normal_root_clearance
      )
      selected_root_clearance_sum += valid.float() * selected_root_clearance
      non_support_wheel_height_sum += valid.float() * non_support_wheel_height
      rate_error_sum += valid.float() * rate_error
      down_rate_sum += valid.float() * down_rate
      support_center_speed_sum += valid.float() * center_speed
      nonwheel_sum += valid.float() * nonwheel.float()
      if (step + 1) * base_env.step_dt >= cfg.settle_s:
        steady_count += valid.float()
        steady_success += valid.float() * success.float()
        steady_normal_four_wheel_support &= ~(
          valid & (modes == 0) & ~support_ok
        )
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
      "spin_rate": _expected_down_rates(
        torch.tensor([mode], device=base_env.device), cfg.spin_rate
      ).item(),
      "mean_gravity_alignment": (alignment_sum[mask] / denom).mean().item(),
      "mean_support_match_rate": (support_sum[mask] / denom).mean().item(),
      "mean_axle_coaxiality": (axle_coaxiality_sum[mask] / denom).mean().item(),
      "mean_normal_all_axis_parallel": (
        normal_axis_parallel_sum[mask] / denom
      ).mean().item(),
      "mean_normal_common_axle_line_score": (
        normal_common_axle_line_sum[mask] / denom
      ).mean().item(),
      "mean_normal_compact_span_score": (
        normal_compact_span_sum[mask] / denom
      ).mean().item(),
      "mean_normal_root_clearance_m": (
        normal_root_clearance_sum[mask] / denom
      ).mean().item(),
      "mean_selected_root_clearance_m": (
        selected_root_clearance_sum[mask] / denom
      ).mean().item(),
      "mean_non_support_wheel_height_m": (
        non_support_wheel_height_sum[mask] / denom
      ).mean().item(),
      "steady_normal_four_wheel_support_rate": steady_normal_four_wheel_support[mask]
      .float()
      .mean()
      .item(),
      "mean_spin_rate_abs_error": (rate_error_sum[mask] / denom).mean().item(),
      "mean_world_down_spin_rate": (down_rate_sum[mask] / denom).mean().item(),
      "mean_support_center_speed_m_s": (support_center_speed_sum[mask] / denom)
      .mean()
      .item(),
      "mean_nonwheel_contact_rate": (nonwheel_sum[mask] / denom).mean().item(),
      "steady_success_rate": (steady_success[mask] / steady_denom).mean().item(),
      "failed_rate": failed[mask].float().mean().item(),
      "unfinished_rate": trial_open[mask].float().mean().item(),
    }

  metrics = (
    [summarize(mode) for mode in range(5)] if cfg.all_modes else summarize(cfg.mode)
  )
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
