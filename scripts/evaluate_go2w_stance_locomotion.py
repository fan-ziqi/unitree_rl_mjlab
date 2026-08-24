"""Headless, fixed-command evaluation for Go2W normal/front/rear locomotion."""

from __future__ import annotations

import contextlib
import io
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply
from tensordict import TensorDict

from src.tasks.velocity.mdp.trick_commands import StanceLocomotionCommandCfg

TASK_ID = "Unitree-Go2W-Stance-Locomotion-Flat"
_GRAVITY_TARGETS = (
  (0.0, 0.0, -1.0),
  (1.0, 0.0, 0.0),
  (-1.0, 0.0, 0.0),
)
_CONTACT_MASKS = (
  (1.0, 1.0, 1.0, 1.0),
  (1.0, 1.0, 0.0, 0.0),
  (0.0, 0.0, 1.0, 1.0),
)


@dataclass
class EvalConfig:
  checkpoint_file: Path
  mode: int = 1
  lin_vel_x: float = 0.0
  yaw_rate: float = 0.0
  num_envs: int = 256
  duration_s: float = 10.0
  settle_s: float = 2.0
  zero_actions: bool = False
  emit_metrics: bool = False
  quiet: bool = False
  device: str = "cuda:0"
  # A command-response comparison must replay the same reset randomization
  # for every request.  Without an explicit seed, separate evaluator
  # processes can attribute reset noise to a change in x/yaw command.
  seed: int = 42


def _configure_command(cfg, evaluation: EvalConfig) -> None:
  if not 0 <= evaluation.mode < 3:
    raise ValueError("mode must be 0=normal, 1=front, or 2=rear.")
  command = cfg.commands["trick"]
  assert isinstance(command, StanceLocomotionCommandCfg)
  probabilities = [0.0, 0.0, 0.0]
  probabilities[evaluation.mode] = 1.0
  command.mode_probabilities = tuple(probabilities)
  command.idle_probability = 0.0
  command.lin_vel_x_range = (evaluation.lin_vel_x, evaluation.lin_vel_x)
  command.yaw_rate_range = (evaluation.yaw_rate, evaluation.yaw_rate)
  command.resampling_time_range = (
    evaluation.duration_s + 1.0,
    evaluation.duration_s + 1.0,
  )


def _fixed_reset_observation(base_env: ManagerBasedRlEnv) -> TensorDict:
  """Build the complete configured history after pinning a fixed command.

  Replacing only the most recent reset observation leaves stale one-hots
  in the actor input, which makes a per-mode evaluation test the reset sampler
  rather than the requested command.
  """
  base_env.observation_manager._obs_buffer = None
  observations = None
  history_length = base_env.cfg.observations["actor"].history_length or 1
  for _ in range(history_length):
    observations = base_env.observation_manager.compute(update_history=True)
  assert observations is not None
  base_env.obs_buf = observations
  return TensorDict(observations, batch_size=[base_env.num_envs])


def _forward_right_axes(robot, mode: int) -> tuple[torch.Tensor, torch.Tensor]:
  quat = robot.data.root_link_quat_w
  body_x = quat_apply(
    quat,
    torch.tensor((1.0, 0.0, 0.0), device=quat.device, dtype=quat.dtype).expand(
      quat.shape[0], -1
    ),
  )[:, :2]
  body_z = quat_apply(
    quat,
    torch.tensor((0.0, 0.0, 1.0), device=quat.device, dtype=quat.dtype).expand(
      quat.shape[0], -1
    ),
  )[:, :2]
  forward = body_x if mode == 0 else (body_z if mode == 1 else -body_z)
  forward = forward / torch.linalg.vector_norm(
    forward, dim=1, keepdim=True
  ).clamp_min(1.0e-6)
  return forward, torch.stack((-forward[:, 1], forward[:, 0]), dim=1)


def run(cfg: EvalConfig) -> dict[str, float]:
  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)
  if cfg.num_envs <= 0 or cfg.duration_s <= 0.0 or cfg.settle_s < 0.0:
    raise ValueError("num_envs and duration_s must be positive; settle_s non-negative.")

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.seed = cfg.seed
  torch.manual_seed(cfg.seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(cfg.seed)
  env_cfg.scene.num_envs = cfg.num_envs
  _configure_command(env_cfg, cfg)
  agent_cfg = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=cfg.device)
  runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device=cfg.device)

  target_gravity = torch.tensor(_GRAVITY_TARGETS[cfg.mode], device=env.device)
  target_contacts = torch.tensor(_CONTACT_MASKS[cfg.mode], device=env.device)
  robot = env.scene["robot"]
  wrapped.reset()
  # ``StanceLocomotionCommand`` deliberately samples x-only, yaw-only, and
  # combined training requests.  A fixed-range evaluator must not inherit
  # that *training distribution*: otherwise a nominal ``x=-0.08`` trial
  # silently contains yaw-only environments whose x command is zero.  Write
  # the requested complete command after reset, before the first action, so
  # every reported parallel rollout tests exactly the CLI command.
  command_term = env.command_manager.get_term("trick")
  command_buf = command_term.command_buf
  command_buf.zero_()
  command_buf[:, cfg.mode] = 1.0
  command_buf[:, 3] = cfg.lin_vel_x
  command_buf[:, 4] = cfg.yaw_rate
  # Do not let the training-only idle -> A -> B schedule overwrite this
  # explicit fixed evaluator command on its first timed transition.
  command_term._transition_phase.fill_(3)
  command_term._transition_time.zero_()
  obs = _fixed_reset_observation(env)
  # Every command must begin from the same normal four-wheel idle reset.  Keep
  # this explicit in the report so a visually convincing terminal frame can
  # never be mistaken for a learned normal-to-two-wheel transition.
  initial_gravity = robot.data.projected_gravity_b
  initial_gravity = initial_gravity / torch.linalg.vector_norm(
    initial_gravity, dim=1, keepdim=True
  ).clamp_min(1.0e-6)
  normal_gravity = torch.tensor((0.0, 0.0, -1.0), device=env.device)
  initial_normal_gravity_alignment = (
    0.5 * (1.0 + torch.sum(initial_gravity * normal_gravity, dim=1))
  )
  num_steps = round(cfg.duration_s / env.step_dt)
  gravity_alignment_sum = torch.zeros((), device=env.device)
  contact_exact_sum = torch.zeros((), device=env.device)
  steady_gravity_alignment_sum = torch.zeros((), device=env.device)
  steady_contact_exact_sum = torch.zeros((), device=env.device)
  steady_nonwheel_contact_sum = torch.zeros((), device=env.device)
  steady_steps = 0
  forward_sum = torch.zeros((), device=env.device)
  lateral_abs_sum = torch.zeros((), device=env.device)
  yaw_sum = torch.zeros((), device=env.device)
  forward_abs_error_sum = torch.zeros((), device=env.device)
  yaw_abs_error_sum = torch.zeros((), device=env.device)
  target_wheel_height_sum = torch.zeros((), device=env.device)
  non_target_wheel_height_sum = torch.zeros((), device=env.device)
  root_height_sum = torch.zeros((), device=env.device)
  support_leg_clearance_sum = torch.zeros((), device=env.device)
  support_leg_length_sum = torch.zeros((), device=env.device)
  shortest_support_leg_length_sum = torch.zeros((), device=env.device)
  normal_leg_default_deviation_sum = torch.zeros((), device=env.device)
  done_count = torch.zeros((), device=env.device)
  # First legal target stance is a direct measure of the requested transition,
  # not merely a steady-state snapshot obtained after an undisclosed reset.
  first_target_time_s = torch.full(
    (cfg.num_envs,), float("nan"), device=env.device
  )

  sensor = env.scene["wheel_ground_contact"]
  nonwheel_sensor = env.scene["nonwheel_ground_contact"]
  wheel_site_ids, _ = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)
  hip_body_ids, _ = robot.find_bodies(
    ("FL_hip", "FR_hip", "RL_hip", "RR_hip"), preserve_order=True
  )
  leg_joint_ids, _ = robot.find_joints(
    (
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
      "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ),
    preserve_order=True,
  )
  _, nonwheel_geom_names = robot.find_geoms(r".*_collision\d*$")
  nonwheel_geom_names = tuple(
    name for name in nonwheel_geom_names if name not in {"FL_wheel_collision", "FR_wheel_collision", "RL_wheel_collision", "RR_wheel_collision"}
  )
  nonwheel_contact_by_geom_sum = torch.zeros(
    len(nonwheel_geom_names), device=env.device
  )
  with torch.inference_mode():
    for step in range(num_steps):
      action = torch.zeros(cfg.num_envs, env.action_manager.total_action_dim, device=env.device) if cfg.zero_actions else policy(obs)
      obs, _, dones, _ = wrapped.step(action)
      gravity = robot.data.projected_gravity_b
      gravity = gravity / torch.linalg.vector_norm(
        gravity, dim=1, keepdim=True
      ).clamp_min(1.0e-6)
      gravity_alignment = 0.5 * (1.0 + torch.sum(gravity * target_gravity, dim=1))
      found = sensor.data.found
      assert found is not None
      contacts = (found.reshape(cfg.num_envs, found.shape[1], -1) > 0).any(dim=-1)
      contact_exact = torch.all(contacts == target_contacts.bool(), dim=1)
      nonwheel_found = nonwheel_sensor.data.found
      assert nonwheel_found is not None
      nonwheel_contact = (
        nonwheel_found.reshape(cfg.num_envs, nonwheel_found.shape[1], -1) > 0
      ).any(dim=(1, 2))
      reached_target = (
        (gravity_alignment >= 0.97) & contact_exact & ~nonwheel_contact
      )
      first_reached = torch.isnan(first_target_time_s) & reached_target
      first_target_time_s[first_reached] = (step + 1) * env.step_dt
      nonwheel_by_geom = (
        nonwheel_found.reshape(cfg.num_envs, nonwheel_found.shape[1], -1) > 0
      ).any(dim=-1)
      if nonwheel_by_geom.shape[1] != len(nonwheel_geom_names):
        raise RuntimeError("Non-wheel contact sensor/geometries are inconsistent.")
      wheel_height = robot.data.site_pos_w[:, wheel_site_ids, 2]
      target_wheel_height = (wheel_height * target_contacts).sum(dim=1) / target_contacts.sum()
      non_target_mask = 1.0 - target_contacts
      non_target_wheel_height = (
        (wheel_height * non_target_mask).sum(dim=1) / non_target_mask.sum().clamp_min(1.0)
      )
      root_height = robot.data.root_link_pos_w[:, 2]
      # This is an observable geometry outcome: base height above the chosen
      # support wheel centres.  It is retained as a global posture diagnostic;
      # the explicit hip-to-wheel length below is the real leg-extension
      # measurement, because a high vertical trunk can still have folded knees.
      support_leg_clearance = root_height - target_wheel_height
      leg_length = torch.linalg.vector_norm(
        robot.data.body_link_pos_w[:, hip_body_ids] - robot.data.site_pos_w[:, wheel_site_ids],
        dim=-1,
      )
      support_leg_length = (leg_length * target_contacts).sum(dim=1) / target_contacts.sum()
      shortest_support_leg_length = torch.where(
        target_contacts.bool(), leg_length, torch.inf
      ).amin(dim=1)
      normal_leg_default_deviation = torch.mean(
        torch.abs(
          robot.data.joint_pos[:, leg_joint_ids]
          - robot.data.default_joint_pos[:, leg_joint_ids]
        ),
        dim=1,
      )
      forward, right = _forward_right_axes(robot, cfg.mode)
      velocity_xy = robot.data.root_link_lin_vel_w[:, :2]
      actual_x = torch.sum(velocity_xy * forward, dim=1)
      actual_y = torch.sum(velocity_xy * right, dim=1)

      gravity_alignment_sum += gravity_alignment.mean()
      contact_exact_sum += contact_exact.float().mean()
      if (step + 1) * env.step_dt >= cfg.settle_s:
        steady_steps += 1
        steady_gravity_alignment_sum += gravity_alignment.mean()
        steady_contact_exact_sum += contact_exact.float().mean()
        steady_nonwheel_contact_sum += nonwheel_contact.float().mean()
      forward_sum += actual_x.mean()
      lateral_abs_sum += actual_y.abs().mean()
      yaw_sum += robot.data.root_link_ang_vel_w[:, 2].mean()
      forward_abs_error_sum += (actual_x - cfg.lin_vel_x).abs().mean()
      yaw_abs_error_sum += (robot.data.root_link_ang_vel_w[:, 2] - cfg.yaw_rate).abs().mean()
      target_wheel_height_sum += target_wheel_height.mean()
      non_target_wheel_height_sum += non_target_wheel_height.mean()
      root_height_sum += root_height.mean()
      support_leg_clearance_sum += support_leg_clearance.mean()
      support_leg_length_sum += support_leg_length.mean()
      shortest_support_leg_length_sum += shortest_support_leg_length.mean()
      normal_leg_default_deviation_sum += normal_leg_default_deviation.mean()
      nonwheel_contact_by_geom_sum += nonwheel_by_geom.float().mean(dim=0)
      done_count += dones.float().sum()

  scale = float(num_steps)
  steady_scale = float(max(steady_steps, 1))
  metrics = {
    "initial_normal_gravity_alignment": initial_normal_gravity_alignment.mean().item(),
    "target_stance_reached_rate": (~torch.isnan(first_target_time_s)).float().mean().item(),
    "mean_first_target_stance_time_s": torch.nan_to_num(
      first_target_time_s, nan=cfg.duration_s
    ).mean().item(),
    "gravity_alignment": (gravity_alignment_sum / scale).item(),
    "exact_support_contact_rate": (contact_exact_sum / scale).item(),
    "steady_gravity_alignment": (steady_gravity_alignment_sum / steady_scale).item(),
    "steady_exact_support_contact_rate": (
      steady_contact_exact_sum / steady_scale
    ).item(),
    "steady_nonwheel_contact_rate": (
      steady_nonwheel_contact_sum / steady_scale
    ).item(),
    "mean_forward_velocity": (forward_sum / scale).item(),
    "mean_lateral_abs_velocity": (lateral_abs_sum / scale).item(),
    "mean_yaw_rate": (yaw_sum / scale).item(),
    "mean_forward_abs_error": (forward_abs_error_sum / scale).item(),
    "mean_yaw_abs_error": (yaw_abs_error_sum / scale).item(),
    "mean_target_wheel_center_height": (target_wheel_height_sum / scale).item(),
    "mean_non_target_wheel_center_height": (
      non_target_wheel_height_sum / scale
    ).item(),
    "mean_root_height": (root_height_sum / scale).item(),
    "mean_support_leg_clearance": (support_leg_clearance_sum / scale).item(),
    "mean_support_leg_length": (support_leg_length_sum / scale).item(),
    "mean_shortest_support_leg_length": (
      shortest_support_leg_length_sum / scale
    ).item(),
    "mean_leg_default_joint_deviation": (
      normal_leg_default_deviation_sum / scale
    ).item(),
    "resets_per_env": (done_count / cfg.num_envs).item(),
    "duration_s": cfg.duration_s,
  }
  metrics.update(
    {
      f"nonwheel_contact_{name}": (nonwheel_contact_by_geom_sum[idx] / scale).item()
      for idx, name in enumerate(nonwheel_geom_names)
    }
  )
  if cfg.emit_metrics:
    for key, value in metrics.items():
      print(f"{key}: {value:.6f}", file=sys.stderr)
  wrapped.close()
  return metrics


def main() -> None:
  cfg = tyro.cli(EvalConfig)
  if cfg.quiet:
    # Manager construction is intentionally verbose.  A compact evaluator is
    # essential when checkpoint acceptance is based on the actual contact and
    # transition metrics rather than on the training reward alone.
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
      metrics = run(cfg)
  else:
    metrics = run(cfg)
  for key, value in metrics.items():
    print(f"{key}: {value:.6f}")


if __name__ == "__main__":
  main()
