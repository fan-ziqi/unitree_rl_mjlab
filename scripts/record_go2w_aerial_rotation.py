"""Record fixed or consecutive-one-hot Go2W aerial-rotation rollouts."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

# A training server normally has no X11 display.  Choose MuJoCo's GPU-backed
# EGL renderer before importing any package that can import ``mujoco``.  An
# explicit caller setting is respected, so local interactive recording keeps
# its chosen backend.
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import tyro
from evaluate_go2w_aerial_rotation import MODE_NAMES, TASK_ID, _pin_mode
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
  idle: bool = False
  duration_s: float = 3.0
  width: int = 960
  height: int = 720
  device: str = "cuda:0"
  name: str = "go2w-aerial-rotation"
  seed: int = 42
  actor_hidden_dims: tuple[int, ...] | None = None
  critic_hidden_dims: tuple[int, ...] | None = None
  actor_class_name: str | None = None
  critic_class_name: str | None = None
  observation_history_length: int | None = None
  # An optional real-time command schedule.  Unlike concatenating independent
  # clips, one simulator instance receives each next one-hot only after the
  # previous one-shot event has returned to upright four-wheel idle.
  sequence: tuple[str, ...] = ()
  # Match ``AerialRotationCommand.trigger_idle_time`` used in training.  The
  # command history is therefore initially all-zero and the one-hot enters it
  # one physical control step at a time, rather than being prefilled.
  initial_idle_s: float = 0.5
  sequence_idle_s: float = 0.6


def _fixed_reset_observation(
  base_env: ManagerBasedRlEnv, mode: int | None
) -> TensorDict:
  """Pin the command before inference and replace reset-time random history.

  The policy's 10-frame history is an ordinary observation input, so the
  initial history must contain the requested one-hot rather than a command
  sampled by the training reset.  Repeating the unchanged reset state is
  equivalent to a stationary 200-ms pre-roll and does not change physics.
  """
  command_term = base_env.command_manager.get_term("trick")
  if mode is None:
    # The aerial idle command is deliberately all-zero, unlike a maneuver
    # one-hot.  Retain the ordinary physical reset and only rebuild the actor
    # history so this is a true untriggered-command rollout.
    command_term.command_buf.zero_()
    command_term.was_airborne.zero_()
    command_term.has_grounded.zero_()
    command_term._airborne_time.zero_()
    command_term._flight_rotation.zero_()
    command_term._current_flight_qualified.zero_()
    command_term._landing_started.zero_()
    command_term._rotation_progress.zero_()
    command_term._launch_axis_w.zero_()
    command_term._launch_root_quat_w.zero_()
    command_term._new_skill.fill_(False)
    command_term._pending_trigger.fill_(False)
    command_term._trigger_time.zero_()
  else:
    _pin_mode(command_term, mode)
  history_length = base_env.cfg.observations["actor"].history_length or 1
  base_env.observation_manager._obs_buffer = None
  for _ in range(history_length):
    observations = base_env.observation_manager.compute(update_history=True)
  base_env.obs_buf = observations
  return TensorDict(observations, batch_size=[base_env.num_envs])


def _append_current_observation(base_env: ManagerBasedRlEnv) -> TensorDict:
  """Expose an externally triggered command to the recurrent actor now."""
  observations = base_env.observation_manager.compute(update_history=True)
  base_env.obs_buf = observations
  return TensorDict(observations, batch_size=[base_env.num_envs])


def _upright_four_wheel_idle(base_env: ManagerBasedRlEnv, sensor_name: str) -> bool:
  """Gate the next one-hot on the public four-wheel idle state."""
  sensor = base_env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  contacts = (found.reshape(1, found.shape[1], -1) > 0).any(dim=-1)
  robot = base_env.scene["robot"]
  gravity = torch.nn.functional.normalize(robot.data.projected_gravity_b, dim=1)
  return bool(torch.all(contacts) and (-gravity[0, 2] >= 0.98))


def run(cfg: RecordConfig) -> Path:
  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if not cfg.idle and not 0 <= cfg.mode < len(MODE_NAMES):
    raise ValueError(f"mode must be in [0, {len(MODE_NAMES) - 1}]")
  if cfg.idle and cfg.sequence:
    raise ValueError("idle and sequence cannot be requested together")
  if (
    cfg.duration_s <= 0.0
    or cfg.initial_idle_s < 0.0
    or cfg.sequence_idle_s < 0.0
  ):
    raise ValueError("durations must be non-negative, with duration_s positive")
  unknown_modes = set(cfg.sequence).difference(MODE_NAMES)
  if unknown_modes:
    raise ValueError(f"Unknown aerial modes in sequence: {sorted(unknown_modes)}")
  sequence_modes = tuple(MODE_NAMES.index(name) for name in cfg.sequence)
  # Use the same real-time command boundary for a single fixed command and a
  # multi-command recording.  Previously a one-mode clip prefilled all ten
  # history slots with its one-hot, unlike aerial training; a sequence did
  # not.  That made the two recorders evaluate different policies.
  scheduled_modes = (
    sequence_modes
    if sequence_modes
    else (() if cfg.idle else (cfg.mode,))
  )

  torch.manual_seed(cfg.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.seed = cfg.seed
  env_cfg.scene.num_envs = 1
  if cfg.observation_history_length is not None:
    if cfg.observation_history_length <= 0:
      raise ValueError("observation_history_length must be positive")
    for group_name in ("actor", "critic"):
      env_cfg.observations[group_name].history_length = cfg.observation_history_length
  total_duration_s = (
    cfg.duration_s
    if not scheduled_modes
    else (
      cfg.initial_idle_s
      + len(scheduled_modes) * cfg.duration_s
      + max(0, len(scheduled_modes) - 1) * cfg.sequence_idle_s
      + 0.8
    )
  )
  env_cfg.episode_length_s = total_duration_s + 0.5
  env_cfg.viewer.width = cfg.width
  env_cfg.viewer.height = cfg.height
  env_cfg.viewer.distance = 3.4
  env_cfg.viewer.elevation = -12.0
  env_cfg.viewer.lookat = (0.0, 0.0, 0.32)
  command_cfg = env_cfg.commands["trick"]
  command_cfg.idle_probability = 0.0
  command_cfg.resampling_time_range = (total_duration_s + 1.0, total_duration_s + 1.0)
  # A training rollout intentionally terminates on an illegal contact or a
  # post-landing rebound.  Those boundaries are useful for PPO, but they
  # truncate a real-time multi-command demonstration before the next one-hot
  # can be issued.  Keep the ordinary fixed-mode recorder strict; only a
  # sequence recorder is allowed to run until its requested wall-clock end so
  # that successful landings can transition to the next command.
  if sequence_modes:
    env_cfg.terminations.pop("illegal_contact", None)
    env_cfg.terminations.pop("post_landing_relaunch", None)

  agent_cfg = load_rl_cfg(TASK_ID)
  if cfg.actor_hidden_dims is not None:
    if not cfg.actor_hidden_dims or any(dim <= 0 for dim in cfg.actor_hidden_dims):
      raise ValueError("actor_hidden_dims must contain positive dimensions.")
    agent_cfg.actor.hidden_dims = cfg.actor_hidden_dims
  if cfg.critic_hidden_dims is not None:
    if not cfg.critic_hidden_dims or any(dim <= 0 for dim in cfg.critic_hidden_dims):
      raise ValueError("critic_hidden_dims must contain positive dimensions.")
    agent_cfg.critic.hidden_dims = cfg.critic_hidden_dims
  if cfg.actor_class_name is not None:
    agent_cfg.actor.class_name = cfg.actor_class_name
  if cfg.critic_class_name is not None:
    agent_cfg.critic.class_name = cfg.critic_class_name
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
  # Every clip begins from the same all-zero default state exposed to users.
  # The requested one-hot is inserted only at its real-time trigger below.
  obs = _fixed_reset_observation(base_env, None)
  command_term = base_env.command_manager.get_term("trick")
  sequence_next = 0
  idle_elapsed = 0.0
  trigger_log: list[dict[str, float | int | str]] = []
  with torch.inference_mode():
    for step in range(num_steps):
      obs, _, dones, _ = env.step(policy(obs))
      if torch.any(dones):
        break
      if not scheduled_modes or sequence_next >= len(scheduled_modes):
        continue
      active = bool(torch.linalg.vector_norm(command_term.command_buf, dim=1)[0] > 0.5)
      if active:
        idle_elapsed = 0.0
        continue
      idle_elapsed += base_env.step_dt
      required_idle = cfg.initial_idle_s if sequence_next == 0 else cfg.sequence_idle_s
      if idle_elapsed < required_idle or not _upright_four_wheel_idle(
        base_env, command_cfg.sensor_name
      ):
        continue
      mode = scheduled_modes[sequence_next]
      _pin_mode(command_term, mode)
      # Training closes each one-shot event as an environment episode, which
      # clears the recurrent actor state.  The real-time sequence preserves
      # physics but reproduces that same command-event boundary for the LSTM;
      # otherwise an idle-period hidden state is an out-of-distribution input
      # that never appeared at a training event's first control step.
      if getattr(policy, "is_recurrent", False):
        policy.reset()
      # Append the new public command right away so the next policy action
      # responds to the trigger rather than a stale all-zero observation.
      # Physical state is never reset here.
      obs = _append_current_observation(base_env)
      trigger_log.append(
        {"mode": MODE_NAMES[mode], "step": step + 1, "time_s": (step + 1) * base_env.step_dt}
      )
      sequence_next += 1
      idle_elapsed = 0.0
  env.close()
  video_path = cfg.output_dir / f"{cfg.name}-step-0.mp4"
  if not video_path.is_file():
    raise RuntimeError(f"Video was not created: {video_path}")
  if scheduled_modes:
    schedule_path = cfg.output_dir / f"{cfg.name}-triggers.json"
    schedule_path.write_text(
      json.dumps(
        {
          "initial_idle_s": cfg.initial_idle_s,
          "sequence_idle_s": cfg.sequence_idle_s,
          "triggers": trigger_log,
        },
        indent=2,
      )
      + "\n"
    )
  return video_path


def main() -> None:
  print(f"Saved video: {run(tyro.cli(RecordConfig))}")


if __name__ == "__main__":
  main()
