"""Headless, per-command validation for Go2W one-shot aerial rotations.

The evaluator starts every trial from the normal four-wheel reset.  It pins one
of the five compact aerial one-hot commands only for this evaluation rollout;
the policy never receives target joint positions or a reference trajectory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import torch
import tyro
from tensordict import TensorDict
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from src.assets.robots.unitree_go2w.go2w_constants import GO2W_LEG_JOINTS


TASK_ID = "Unitree-Go2W-Aerial-Rotation-Flat"
MODE_NAMES = ("front", "back", "left", "right", "yaw")
TARGET_ANGLE = math.tau


@dataclass
class EvalConfig:
  checkpoint_file: Path
  task_id: str = TASK_ID
  mode: int = 0
  num_envs: int = 256
  duration_s: float = 3.0
  device: str = "cuda:0"
  seed: int = 42
  emit_metrics: bool = False
  quiet: bool = False


def _wheel_contacts(sensor) -> torch.Tensor:
  """Return one boolean contact state for every Go2W wheel."""
  found = sensor.data.found
  assert found is not None
  return (found.reshape(found.shape[0], found.shape[1], -1) > 0).any(dim=-1)


def _pin_mode(command_term, mode: int) -> None:
  """Set one evaluation command and reset only that command's event state."""
  command_term.command_buf.zero_()
  command_term.command_buf[:, mode] = 1.0
  command_term.was_airborne.zero_()
  command_term._landing_settle_time.zero_()
  command_term._rotation_progress.zero_()
  command_term._launch_axis_w.zero_()
  # The command term captures the launch axis from the normal reset attitude
  # on its first control step, just as it does during training.
  command_term._new_skill.fill_(True)


def _fixed_reset_observation(base_env: ManagerBasedRlEnv, mode: int) -> TensorDict:
  """Pin one command and rebuild the ordinary observation-history window.

  The aerial actor consumes ten consecutive observations.  ``env.reset()``
  necessarily constructs that history using the command sampled by the normal
  reset path; simply overwriting the command afterwards therefore evaluates a
  policy that sees nine stale one-hots.  A stationary pre-roll at the same
  physical reset state gives the requested command its proper 200-ms history
  without changing simulation state or granting privileged information.
  """
  command_term = base_env.command_manager.get_term("trick")
  _pin_mode(command_term, mode)
  base_env.observation_manager._obs_buffer = None
  observations = None
  for _ in range(10):
    observations = base_env.observation_manager.compute(update_history=True)
  assert observations is not None
  base_env.obs_buf = observations
  return TensorDict(observations, batch_size=[base_env.num_envs])


def run(cfg: EvalConfig) -> dict[str, float | int | str]:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if not 0 <= cfg.mode < len(MODE_NAMES):
    raise ValueError(f"mode must be in [0, {len(MODE_NAMES) - 1}]")
  if cfg.num_envs <= 0 or cfg.duration_s <= 0.0:
    raise ValueError("num_envs and duration_s must be positive")

  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  # A trial's desired mode must not be replaced by the normal periodic command
  # sampler before its evaluation interval has elapsed.
  command_cfg = env_cfg.commands["trick"]
  command_cfg.idle_probability = 0.0
  command_cfg.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)
  env_cfg.episode_length_s = cfg.duration_s + 0.5

  agent_cfg = load_rl_cfg(cfg.task_id)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(cfg.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device=cfg.device)

  env.reset()
  obs = _fixed_reset_observation(base_env, cfg.mode)
  command_term = base_env.command_manager.get_term("trick")
  robot = base_env.scene["robot"]
  wheel_sensor = base_env.scene[command_cfg.sensor_name]
  normal_gravity = torch.tensor((0.0, 0.0, -1.0), device=base_env.device)
  default_height = robot.data.default_root_state[:, 2]
  leg_joint_ids, _ = robot.find_joints(GO2W_LEG_JOINTS, preserve_order=True)

  trial_open = torch.ones(cfg.num_envs, dtype=torch.bool, device=base_env.device)
  ever_airborne = torch.zeros_like(trial_open)
  completed = torch.zeros_like(trial_open)
  failed = torch.zeros_like(trial_open)
  peak_progress = torch.zeros(cfg.num_envs, device=base_env.device)
  peak_height_delta = torch.zeros(cfg.num_envs, device=base_env.device)
  peak_axis_rate = torch.zeros(cfg.num_envs, device=base_env.device)
  # This does not prescribe a motion.  It makes the compact-wheel-leg
  # requirement measurable during validation: the task rejects an excursion
  # above 0.55 rad, so values near that bound show a large leg swing.
  peak_leg_deviation = torch.zeros(cfg.num_envs, device=base_env.device)
  peak_leg_excess_l2 = torch.zeros(cfg.num_envs, device=base_env.device)
  leg_envelope_violated = torch.zeros_like(trial_open)
  completion_gravity_error = torch.zeros(cfg.num_envs, device=base_env.device)
  completion_linear_speed = torch.zeros(cfg.num_envs, device=base_env.device)
  completion_angular_speed = torch.zeros(cfg.num_envs, device=base_env.device)
  completion_all_wheels = torch.zeros_like(trial_open)

  num_steps = round(cfg.duration_s / base_env.step_dt)
  with torch.inference_mode():
    for _ in range(num_steps):
      # The term stores signed rotation accumulated while airborne.  Record it
      # before a successful completion deliberately clears the one-shot state.
      peak_progress = torch.maximum(
        peak_progress,
        torch.where(trial_open, command_term._rotation_progress, torch.zeros_like(peak_progress)),
      )
      pre_active = torch.sum(command_term.command, dim=1) > 0.5
      obs, _, dones, _ = env.step(policy(obs))
      contacts = _wheel_contacts(wheel_sensor)
      airborne = ~torch.any(contacts, dim=1)
      ever_airborne |= trial_open & airborne
      height_delta = robot.data.root_link_pos_w[:, 2] - default_height
      peak_height_delta = torch.maximum(
        peak_height_delta,
        torch.where(trial_open, height_delta, torch.zeros_like(height_delta)),
      )
      axis_rate = torch.sum(robot.data.root_link_ang_vel_w * command_term._launch_axis_w, dim=1)
      peak_axis_rate = torch.maximum(
        peak_axis_rate,
        torch.where(trial_open & airborne, axis_rate, torch.zeros_like(axis_rate)),
      )
      leg_deviation = torch.abs(
        robot.data.joint_pos[:, leg_joint_ids]
        - robot.data.default_joint_pos[:, leg_joint_ids]
      )
      peak_leg_deviation = torch.maximum(
        peak_leg_deviation,
        torch.where(
          trial_open,
          torch.max(leg_deviation, dim=1).values,
          torch.zeros_like(peak_leg_deviation),
        ),
      )
      leg_excess_l2 = torch.sum(torch.square(torch.relu(leg_deviation - 0.12)), dim=1)
      peak_leg_excess_l2 = torch.maximum(
        peak_leg_excess_l2,
        torch.where(trial_open, leg_excess_l2, torch.zeros_like(peak_leg_excess_l2)),
      )
      # Keep this separate from the generic failure rate.  The environment
      # ends these episodes immediately, but the terminal state is still the
      # clearest validation evidence that a policy tried to use a large swing.
      leg_envelope_violated |= trial_open & torch.any(leg_deviation > 0.55, dim=1)

      post_active = torch.sum(command_term.command, dim=1) > 0.5
      # AerialRotationCommand clears a nonzero one-hot only after it has
      # observed the requested full turn, four-wheel contact, normal gravity,
      # velocity limits, and the configured settle interval.
      completed_now = trial_open & pre_active & ~post_active & ~dones.bool()
      if torch.any(completed_now):
        gravity_error = torch.sum(
          torch.square(robot.data.projected_gravity_b - normal_gravity), dim=1
        )
        completion_gravity_error[completed_now] = gravity_error[completed_now]
        completion_linear_speed[completed_now] = torch.linalg.vector_norm(
          robot.data.root_link_lin_vel_w[completed_now], dim=1
        )
        completion_angular_speed[completed_now] = torch.linalg.vector_norm(
          robot.data.root_link_ang_vel_w[completed_now], dim=1
        )
        completion_all_wheels[completed_now] = torch.all(contacts[completed_now], dim=1)
      completed |= completed_now
      failed |= trial_open & dones.bool() & ~completed_now
      trial_open &= ~(completed_now | dones.bool())

  peak_progress = torch.maximum(peak_progress, command_term._rotation_progress)
  completed_count = completed.sum().item()
  completion_mask = completed
  metrics: dict[str, float | int | str] = {
    "mode": MODE_NAMES[cfg.mode],
    "mode_index": cfg.mode,
    "num_envs": cfg.num_envs,
    "duration_s": cfg.duration_s,
    "airborne_rate": ever_airborne.float().mean().item(),
    "full_turn_rate": (peak_progress >= TARGET_ANGLE).float().mean().item(),
    "completion_rate": completed.float().mean().item(),
    "illegal_reset_rate": failed.float().mean().item(),
    "unfinished_rate": trial_open.float().mean().item(),
    "mean_peak_rotation_rad": peak_progress.mean().item(),
    "mean_peak_rotation_turns": (peak_progress / TARGET_ANGLE).mean().item(),
    "mean_peak_height_delta_m": peak_height_delta.mean().item(),
    "mean_peak_axis_rate_rad_s": peak_axis_rate.mean().item(),
    "mean_peak_leg_deviation_rad": peak_leg_deviation.mean().item(),
    "p95_peak_leg_deviation_rad": torch.quantile(peak_leg_deviation, 0.95).item(),
    "max_peak_leg_deviation_rad": peak_leg_deviation.max().item(),
    "mean_peak_leg_excess_l2": peak_leg_excess_l2.mean().item(),
    "leg_envelope_violation_rate": leg_envelope_violated.float().mean().item(),
    "completion_four_wheel_contact_rate": completion_all_wheels.float().sum().item()
    / max(completed_count, 1),
    "completion_mean_gravity_error": completion_gravity_error[completion_mask].mean().item()
    if completed_count
    else float("inf"),
    "completion_mean_linear_speed": completion_linear_speed[completion_mask].mean().item()
    if completed_count
    else float("inf"),
    "completion_mean_angular_speed": completion_angular_speed[completion_mask].mean().item()
    if completed_count
    else float("inf"),
  }
  env.close()
  return metrics


def main() -> None:
  cfg = tyro.cli(EvalConfig)
  metrics = run(cfg)
  if cfg.emit_metrics:
    print(json.dumps(metrics, sort_keys=True))
  elif not cfg.quiet:
    for name, value in metrics.items():
      print(f"{name}: {value}")


if __name__ == "__main__":
  main()
