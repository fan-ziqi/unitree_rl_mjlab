"""Record one deterministic fixed-one-hot Go2W aerial-rotation rollout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

# A training server normally has no X11 display.  Choose MuJoCo's GPU-backed
# EGL renderer before importing any package that can import ``mujoco``.  An
# explicit caller setting is respected, so local interactive recording keeps
# its chosen backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import tyro
from tensordict import TensorDict

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.wrappers import VideoRecorder

from evaluate_go2w_aerial_rotation import MODE_NAMES, TASK_ID, _pin_mode


@dataclass
class RecordConfig:
  checkpoint_file: Path
  output_dir: Path
  mode: int = 0
  duration_s: float = 3.0
  width: int = 960
  height: int = 720
  device: str = "cuda:0"
  name: str = "go2w-aerial-rotation"
  seed: int = 42


def _fixed_reset_observation(base_env: ManagerBasedRlEnv, mode: int) -> TensorDict:
  """Pin the command before inference and replace reset-time random history.

  The policy's 10-frame history is an ordinary observation input, so the
  initial history must contain the requested one-hot rather than a command
  sampled by the training reset.  Repeating the unchanged reset state is
  equivalent to a stationary 200-ms pre-roll and does not change physics.
  """
  command_term = base_env.command_manager.get_term("trick")
  _pin_mode(command_term, mode)
  history_length = 10
  base_env.observation_manager._obs_buffer = None
  for _ in range(history_length):
    observations = base_env.observation_manager.compute(update_history=True)
  base_env.obs_buf = observations
  return TensorDict(observations, batch_size=[base_env.num_envs])


def run(cfg: RecordConfig) -> Path:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if not 0 <= cfg.mode < len(MODE_NAMES):
    raise ValueError(f"mode must be in [0, {len(MODE_NAMES) - 1}]")
  if cfg.duration_s <= 0.0:
    raise ValueError("duration_s must be positive")

  torch.manual_seed(cfg.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 1
  env_cfg.episode_length_s = cfg.duration_s + 0.5
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.distance = 3.4
  env_cfg.viewer.elevation = -12.0
  env_cfg.viewer.lookat = (0.0, 0.0, 0.32)
  command_cfg = env_cfg.commands["trick"]
  command_cfg.idle_probability = 0.0
  command_cfg.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)

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
  obs = _fixed_reset_observation(base_env, cfg.mode)
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
