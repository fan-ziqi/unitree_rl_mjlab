"""Record one deterministic fixed-command Go2W five-mode spin rollout."""

from __future__ import annotations

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


def run(cfg: RecordConfig) -> Path:
  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if not 0 <= cfg.mode < len(MODE_NAMES):
    raise ValueError("mode must be in [0, 4].")
  torch.manual_seed(cfg.seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.distance = 3.0
  env_cfg.viewer.elevation = -12.0
  _configure_command(env_cfg, cfg.duration_s)
  agent_cfg = load_rl_cfg(TASK_ID)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode="rgb_array")
  num_steps = round(cfg.duration_s / base_env.step_dt)
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
  modes = torch.full((1,), cfg.mode, dtype=torch.long, device=base_env.device)
  _pin_modes(command_term, modes, cfg.spin_rate, cfg.ramp_spin_rate)
  obs = _fixed_reset_observation(base_env)
  with torch.inference_mode():
    for _ in range(num_steps):
      obs, _, dones, _ = env.step(policy(obs))
      if torch.any(dones):
        break
  env.close()
  video_path = cfg.output_dir / f"{cfg.name}-step-0.mp4"
  if not video_path.is_file():
    raise RuntimeError(f"Video was not created: {video_path}")
  return video_path


def main() -> None:
  print(f"Saved video: {run(tyro.cli(RecordConfig))}")


if __name__ == "__main__":
  main()
