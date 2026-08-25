"""Measure the native Go2W aerial action envelope without training a policy.

This is a diagnostic, not an imitation source: every rollout holds a simple
bounded leg-residual pattern from the ordinary four-wheel reset and reports
the resulting takeoff/flight/axis-rate envelope.  It tells us whether PPO's
outcome thresholds are physically reachable before spending training time on
them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from evaluate_go2w_aerial_rotation import _pin_modes, _wheel_contacts


@dataclass
class Config:
  """Structured, constant-action probe settings."""

  device: str = "cuda:0"
  replicas_per_pattern: int = 64
  duration_s: float = 0.60
  action_limit: float = 1.5


def _patterns(
  limit: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str], torch.Tensor, torch.Tensor]:
  """Build constant and lead-pulse front/rear, left/right launch combinations."""
  pairs = ((-limit, limit), (-limit, -limit), (limit, limit), (limit, -limit))
  initial_actions: list[torch.Tensor] = []
  final_actions: list[torch.Tensor] = []
  names: list[str] = []
  modes: list[int] = []
  switch_steps: list[int] = []
  # The action ordering is [FL hip/thigh/calf, FR ..., RL ..., RR ..., wheels].
  for kind, groups, mode in (
    ("pitch", ((0, 1), (2, 3)), 0),
    ("roll", ((0, 2), (1, 3)), 2),
  ):
    for first_i, first in enumerate(pairs):
      for second_i, second in enumerate(pairs):
        action = torch.zeros(16)
        for leg in groups[0]:
          action[3 * leg + 1 : 3 * leg + 3] = torch.tensor(first)
        for leg in groups[1]:
          action[3 * leg + 1 : 3 * leg + 3] = torch.tensor(second)
        initial_actions.append(action)
        final_actions.append(action)
        names.append(f"{kind}:a{first_i}:b{second_i}")
        modes.append(mode)
        switch_steps.append(0)
    # A full-strength leading support pair followed by the remaining pair is
    # the minimal open-loop test of ground-contact torque.  It is deliberately
    # not used as a policy target or a training phase.
    all_launch = torch.zeros(16)
    for leg in range(4):
      all_launch[3 * leg + 1 : 3 * leg + 3] = torch.tensor(pairs[0])
    for lead_index, lead_name in enumerate(("first", "second")):
      for delay in (1, 3, 5, 7):
        lead = torch.zeros(16)
        for leg in groups[lead_index]:
          lead[3 * leg + 1 : 3 * leg + 3] = torch.tensor(pairs[0])
        initial_actions.append(lead)
        final_actions.append(all_launch)
        names.append(f"{kind}:{lead_name}:d{delay}")
        modes.append(mode)
        switch_steps.append(delay)
  return (
    torch.stack(initial_actions),
    torch.stack(final_actions),
    names,
    torch.tensor(modes, dtype=torch.long),
    torch.tensor(switch_steps, dtype=torch.long),
  )


def run(cfg: Config) -> None:
  if cfg.replicas_per_pattern <= 0 or cfg.duration_s <= 0.0 or cfg.action_limit <= 0.0:
    raise ValueError("replicas_per_pattern, duration_s, and action_limit must be positive")
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  initial_patterns, final_patterns, names, pattern_modes, pattern_switches = _patterns(
    cfg.action_limit
  )
  num_patterns = len(names)
  env_cfg = load_env_cfg("Unitree-Go2W-Aerial-Rotation-Flat", play=True)
  env_cfg.events["foot_friction"].params["ranges"] = (1.0, 1.0)
  env_cfg.scene.num_envs = num_patterns * cfg.replicas_per_pattern
  env_cfg.episode_length_s = cfg.duration_s + 0.5
  agent_cfg = load_rl_cfg("Unitree-Go2W-Aerial-Rotation-Flat")
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  try:
    env.reset()
    initial_action = initial_patterns.repeat_interleave(
      cfg.replicas_per_pattern, dim=0
    ).to(base_env.device)
    final_action = final_patterns.repeat_interleave(
      cfg.replicas_per_pattern, dim=0
    ).to(base_env.device)
    switches = pattern_switches.repeat_interleave(cfg.replicas_per_pattern).to(
      base_env.device
    )
    modes = pattern_modes.repeat_interleave(cfg.replicas_per_pattern).to(base_env.device)
    if initial_action.shape[1] != base_env.action_manager.action.shape[1]:
      raise RuntimeError(
        f"probe action has {initial_action.shape[1]} dimensions, expected "
        f"{base_env.action_manager.action.shape[1]}"
      )
    command = base_env.command_manager.get_term("trick")
    _pin_modes(command, modes)
    wheel_sensor = base_env.scene[env_cfg.commands["trick"].sensor_name]
    robot = base_env.scene["robot"]
    axes = torch.tensor(env_cfg.commands["trick"].axes, device=base_env.device)
    axes = axes[modes]
    alive = torch.ones(base_env.num_envs, dtype=torch.bool, device=base_env.device)
    all_grounded = torch.zeros_like(alive)
    air_time = torch.zeros(base_env.num_envs, device=base_env.device)
    peak_air_time = torch.zeros_like(air_time)
    peak_up = torch.zeros_like(air_time)
    peak_axis_rate = torch.zeros_like(air_time)
    peak_turn = torch.zeros_like(air_time)
    turns = torch.zeros_like(air_time)
    with torch.inference_mode():
      for step in range(round(cfg.duration_s / base_env.step_dt)):
        action = torch.where((step >= switches).unsqueeze(1), final_action, initial_action)
        _, _, dones, _ = env.step(action)
        contacts = _wheel_contacts(wheel_sensor)
        grounded = torch.all(contacts, dim=1)
        airborne = ~torch.any(contacts, dim=1)
        all_grounded |= alive & grounded
        valid_air = alive & all_grounded & airborne
        air_time = torch.where(valid_air, air_time + base_env.step_dt, torch.zeros_like(air_time))
        peak_air_time = torch.maximum(peak_air_time, air_time)
        peak_up = torch.maximum(
          peak_up, torch.where(alive & grounded, robot.data.root_link_lin_vel_w[:, 2], torch.zeros_like(peak_up))
        )
        axis_rate = torch.sum(robot.data.root_link_ang_vel_w * axes, dim=1)
        peak_axis_rate = torch.maximum(
          peak_axis_rate, torch.where(valid_air, torch.abs(axis_rate), torch.zeros_like(axis_rate))
        )
        turns = torch.where(valid_air, turns + axis_rate * base_env.step_dt, turns)
        peak_turn = torch.maximum(peak_turn, torch.abs(turns))
        alive &= ~dones.bool()
    for pattern_id, name in enumerate(names):
      mask = torch.arange(base_env.num_envs, device=base_env.device) // cfg.replicas_per_pattern == pattern_id
      def q95(values: torch.Tensor) -> float:
        return float(torch.quantile(values[mask], 0.95).cpu())
      print(
        f"{name:17s} up_p95={q95(peak_up):5.2f} "
        f"air_p95={q95(peak_air_time):4.2f} "
        f"axis_rate_p95={q95(peak_axis_rate):6.2f} "
        f"turn_p95={q95(peak_turn):5.2f}",
        flush=True,
      )
  finally:
    env.close()


if __name__ == "__main__":
  run(tyro.cli(Config))
