"""Record one fixed-command Go2W stance-locomotion rollout headlessly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

from evaluate_go2w_stance_locomotion import EvalConfig, TASK_ID, _configure_command


@dataclass
class RecordConfig:
  checkpoint_file: Path
  output_dir: Path
  mode: int = 1
  lin_vel_x: float = 0.0
  yaw_rate: float = 0.0
  duration_s: float = 10.0
  width: int = 960
  height: int = 720
  device: str = "cuda:0"
  name: str = "go2w-stance-locomotion"
  seed: int = 42
  # Optional continuous one-hot schedule, for example:
  # ``normal:2,front:4,normal:2,rear:4,normal:2``.  Unlike concatenating
  # independent clips, this changes the command in one physical rollout.
  switch_schedule: str = ""


_MODE_NAMES = {"normal": 0, "front": 1, "rear": 2}


def _parse_schedule(value: str, fallback_mode: int, fallback_duration: float) -> list[tuple[int, float]]:
  if not value.strip():
    return [(fallback_mode, fallback_duration)]
  schedule: list[tuple[int, float]] = []
  for item in value.split(","):
    name, separator, duration_text = item.strip().partition(":")
    if not separator or name not in _MODE_NAMES:
      raise ValueError(
        "switch_schedule must be comma-separated normal/front/rear durations, "
        "for example 'normal:2,front:4,rear:4'."
      )
    duration = float(duration_text)
    if duration <= 0.0:
      raise ValueError("Every switch_schedule duration must be positive.")
    schedule.append((_MODE_NAMES[name], duration))
  return schedule


def _write_fixed_command(command_buf: torch.Tensor, mode: int, lin_vel_x: float, yaw_rate: float) -> None:
  command_buf.zero_()
  command_buf[:, mode] = 1.0
  command_buf[:, 3] = lin_vel_x
  command_buf[:, 4] = yaw_rate


def run(cfg: RecordConfig) -> Path:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  schedule = _parse_schedule(cfg.switch_schedule, cfg.mode, cfg.duration_s)
  total_duration_s = sum(duration for _, duration in schedule)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  # Match the deterministic fixed-command evaluator so a visual audit and its
  # contact/transition metrics refer to the same reset distribution.
  env_cfg.seed = cfg.seed
  torch.manual_seed(cfg.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.distance = 2.6
  env_cfg.viewer.elevation = -10.0
  _configure_command(
    env_cfg,
    EvalConfig(
      checkpoint_file=cfg.checkpoint_file,
      mode=schedule[0][0],
      lin_vel_x=cfg.lin_vel_x,
      yaw_rate=cfg.yaw_rate,
      duration_s=total_duration_s,
      num_envs=1,
      device=cfg.device,
    ),
  )

  agent_cfg = load_rl_cfg(TASK_ID)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode="rgb_array")
  num_steps = round(total_duration_s / base_env.step_dt)
  recorder = VideoRecorder(
    base_env,
    video_folder=cfg.output_dir,
    step_trigger=lambda step: step == 0,
    video_length=num_steps,
    name_prefix=cfg.name,
  )
  env = RslRlVecEnvWrapper(recorder, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device=cfg.device)
  obs, _ = env.reset()
  # Keep visual rollouts semantically identical to the fixed-command
  # evaluator.  The command term's training sampler otherwise splits active
  # samples into x-only/yaw-only/combined cases, which can silently turn a
  # requested non-zero x or yaw video into a zero-command recording.
  command_term = base_env.command_manager.get_term("trick")
  command_buf = command_term.command_buf
  schedule_index = 0
  next_switch_s = schedule[0][1]
  _write_fixed_command(command_buf, schedule[0][0], cfg.lin_vel_x, cfg.yaw_rate)
  # ``commands`` is the final term in the required 59-D observation layout.
  # Update the reset observation too, so even the first action sees the first
  # schedule item rather than the command that happened to be sampled at reset.
  obs[:, -5:] = command_buf
  with torch.inference_mode():
    for step in range(num_steps):
      elapsed_s = step * base_env.step_dt
      while schedule_index + 1 < len(schedule) and elapsed_s >= next_switch_s:
        schedule_index += 1
        next_switch_s += schedule[schedule_index][1]
        _write_fixed_command(
          command_buf, schedule[schedule_index][0], cfg.lin_vel_x, cfg.yaw_rate
        )
        obs[:, -5:] = command_buf
      obs, _, _, _ = env.step(policy(obs))
  env.close()
  video_path = cfg.output_dir / f"{cfg.name}-step-0.mp4"
  if not video_path.is_file():
    raise RuntimeError(f"Video was not created: {video_path}")
  return video_path


def main() -> None:
  print(f"Saved video: {run(tyro.cli(RecordConfig))}")


if __name__ == "__main__":
  main()
