"""Record a headless visual rollout of the Go2W upright policy."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer.viewer_config import ViewerConfig


TASK_ID = "Unitree-Go2W-Upright-Flat"


@dataclass
class RecordConfig:
  checkpoint_file: Path
  output_dir: Path
  command_x: float = 0.0
  duration_s: float = 8.0
  width: int = 960
  height: int = 720
  follow_robot: bool = False
  device: str = "cuda:0"
  name: str = "go2w-upright"


def run(cfg: RecordConfig) -> Path:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if cfg.duration_s <= 0:
    raise ValueError("duration_s must be positive")

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = 1
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  if cfg.follow_robot:
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.ASSET_BODY
    env_cfg.viewer.entity_name = "robot"
    env_cfg.viewer.body_name = "base_link"
    env_cfg.viewer.distance = 2.6
  else:
    # A fixed world camera makes actual forward travel visible; a body-tracking
    # camera would keep the grid stationary and make a moving robot look still.
    env_cfg.viewer.origin_type = ViewerConfig.OriginType.WORLD
    env_cfg.viewer.entity_name = None
    env_cfg.viewer.body_name = None
    env_cfg.viewer.lookat = (1.0, 0.0, 0.35)
    env_cfg.viewer.distance = 4.2
  env_cfg.viewer.elevation = -12.0

  twist = env_cfg.commands["twist"]
  twist.ranges.lin_vel_x = (cfg.command_x, cfg.command_x)
  twist.ranges.lin_vel_y = (0.0, 0.0)
  twist.ranges.ang_vel_z = (0.0, 0.0)
  twist.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)

  agent_cfg = load_rl_cfg(TASK_ID)
  base_env = ManagerBasedRlEnv(
    cfg=env_cfg, device=cfg.device, render_mode="rgb_array"
  )
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
  runner.load(
    str(cfg.checkpoint_file), load_cfg={"actor": True}, strict=True, map_location=cfg.device
  )
  policy = runner.get_inference_policy(device=cfg.device)

  obs, _ = env.reset()
  with torch.inference_mode():
    for _ in range(num_steps):
      obs, _, _, _ = env.step(policy(obs))
  env.close()

  video_path = cfg.output_dir / f"{cfg.name}-step-0.mp4"
  if not video_path.is_file():
    raise RuntimeError(f"Video was not created: {video_path}")
  return video_path


def main() -> None:
  video_path = run(tyro.cli(RecordConfig))
  print(f"Saved video: {video_path}")


if __name__ == "__main__":
  main()
