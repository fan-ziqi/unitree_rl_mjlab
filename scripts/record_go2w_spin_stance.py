"""Record fixed or direct-switch Go2W five-mode spin rollouts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import tyro
from evaluate_go2w_spin_stance import (
  MODE_NAMES,
  TASK_ID,
  _configure_command,
  _fixed_reset_observation,
  _pin_modes,
)
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from tensordict import TensorDict


@dataclass
class RecordConfig:
  checkpoint_file: Path
  output_dir: Path
  mode: int = 0
  # Match the reference-speed fixed-command evaluator and high-rate stage.
  spin_rate: float = 15.0
  duration_s: float = 6.0
  ramp_spin_rate: bool = True
  width: int = 960
  height: int = 720
  device: str = "cuda:0"
  name: str = "go2w-five-mode-spin"
  seed: int = 42
  # Consecutive one-hot intervals in the same physical rollout, e.g.
  # ``normal:2,front:3,rear:3``.  Dynamic normal/front/rear intervals retain
  # one signed z-spin request through every boundary.
  switch_schedule: str = ""


def _parse_schedule(value: str, mode: int, duration_s: float) -> list[tuple[int, float]]:
  if not value:
    return [(mode, duration_s)]
  schedule: list[tuple[int, float]] = []
  for item in value.split(","):
    name, separator, duration_text = item.strip().partition(":")
    if not separator or name not in MODE_NAMES:
      raise ValueError(
        "switch_schedule must use normal/front/rear/left/right durations, "
        "for example 'normal:2,front:3,rear:3'."
      )
    duration = float(duration_text)
    if duration <= 0.0:
      raise ValueError("Every switch_schedule duration must be positive.")
    schedule.append((MODE_NAMES.index(name), duration))
  return schedule


def run(cfg: RecordConfig) -> Path:
  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if not 0 <= cfg.mode < len(MODE_NAMES):
    raise ValueError("mode must be in [0, 4].")
  schedule = _parse_schedule(cfg.switch_schedule, cfg.mode, cfg.duration_s)
  total_duration_s = sum(duration for _, duration in schedule)
  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.distance = 3.0
  env_cfg.viewer.elevation = -12.0
  _configure_command(env_cfg, total_duration_s)
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
  env.reset()
  command_term = base_env.command_manager.get_term("trick")
  modes = torch.full((1,), schedule[0][0], dtype=torch.long, device=base_env.device)
  _pin_modes(command_term, modes, cfg.spin_rate, cfg.ramp_spin_rate)
  obs = _fixed_reset_observation(base_env)
  schedule_index = 0
  next_switch_s = schedule[0][1]
  command_trace: list[dict[str, float | str]] = []
  robot = base_env.scene["robot"]
  with torch.inference_mode():
    for step in range(num_steps):
      elapsed_s = step * base_env.step_dt
      while schedule_index + 1 < len(schedule) and elapsed_s >= next_switch_s:
        schedule_index += 1
        next_switch_s += schedule[schedule_index][1]
        next_mode = schedule[schedule_index][0]
        _pin_modes(
          command_term,
          torch.full((1,), next_mode, dtype=torch.long, device=base_env.device),
          cfg.spin_rate,
          cfg.ramp_spin_rate,
          preserve_spin_rate=True,
        )
        observations = base_env.observation_manager.compute(update_history=True)
        base_env.obs_buf = observations
        obs = TensorDict(observations, batch_size=[base_env.num_envs])
      gravity_down_b = torch.nn.functional.normalize(
        robot.data.projected_gravity_b, dim=1
      )
      actual_down_rate = torch.sum(robot.data.root_link_ang_vel_b * gravity_down_b, dim=1)
      command_trace.append(
        {
          "time_s": elapsed_s,
          "mode": MODE_NAMES[int(torch.argmax(command_term.command_buf[0, :5]))],
          "requested_rate": float(command_term.command_buf[0, 5]),
          "actual_down_rate": float(actual_down_rate[0]),
        }
      )
      obs, _, dones, _ = env.step(policy(obs))
      if torch.any(dones):
        break
  env.close()
  video_path = cfg.output_dir / f"{cfg.name}-step-0.mp4"
  if not video_path.is_file():
    raise RuntimeError(f"Video was not created: {video_path}")
  if cfg.switch_schedule:
    trace_path = cfg.output_dir / f"{cfg.name}-spin-trace.json"
    trace_path.write_text(json.dumps(command_trace, indent=2) + "\n")
  return video_path


def main() -> None:
  print(f"Saved video: {run(tyro.cli(RecordConfig))}")


if __name__ == "__main__":
  main()
