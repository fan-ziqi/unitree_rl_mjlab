"""Headless evaluation for a Go2W upright-walking checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.lab_api.math import quat_apply_inverse

TASK_ID = "Unitree-Go2W-Upright-Flat"
FRONT_LEG_JOINTS = (
  "FL_hip_joint",
  "FL_thigh_joint",
  "FL_calf_joint",
  "FR_hip_joint",
  "FR_thigh_joint",
  "FR_calf_joint",
)
FRONT_LEG_HANGING_POSE = (0.0, 1.75, -1.30, 0.0, 1.75, -1.30)
FRONT_ARM_BEND_POSE = (1.75, -1.30, 1.75, -1.30)
FRONT_WHEEL_SITES = ("FL", "FR")
REAR_WHEEL_SITES = ("RL", "RR")
FRONT_WHEEL_MIN_HEIGHT = 0.41
FRONT_WHEEL_HANGING_POSITIONS_B = (
  (-0.11467, 0.14200, 0.00000),
  (-0.11467, -0.14200, 0.00000),
)
REAR_LEG_JOINTS = (
  "RL_hip_joint",
  "RL_thigh_joint",
  "RL_calf_joint",
  "RR_hip_joint",
  "RR_thigh_joint",
  "RR_calf_joint",
)
REAR_LEG_SUPPORT_POSE = (0.0, 2.425, -1.75, 0.0, 2.425, -1.75)


@dataclass
class EvalConfig:
  checkpoint_file: Path
  num_envs: int = 256
  duration_s: float = 12.0
  command_x: float = 0.0
  command_y: float = 0.0
  command_yaw: float = 0.0
  device: str = "cuda:0"


def run(cfg: EvalConfig) -> dict[str, float]:
  import mjlab.tasks  # noqa: F401

  import src.tasks  # noqa: F401

  if not cfg.checkpoint_file.is_file():
    raise FileNotFoundError(cfg.checkpoint_file)

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = cfg.num_envs
  twist = env_cfg.commands["twist"]
  twist.ranges.lin_vel_x = (cfg.command_x, cfg.command_x)
  twist.ranges.lin_vel_y = (cfg.command_y, cfg.command_y)
  twist.ranges.ang_vel_z = (cfg.command_yaw, cfg.command_yaw)
  # A nonzero evaluation command must be issued to every environment.  The
  # training configuration intentionally mixes in standing commands, but
  # leaving that mixture enabled here silently halves a requested forward
  # command and invalidates the tracking measurement.
  twist.rel_standing_envs = (
    1.0
    if abs(cfg.command_x) < 1.0e-6 and abs(cfg.command_y) < 1.0e-6 and abs(cfg.command_yaw) < 1.0e-6
    else 0.0
  )
  # Evaluation must issue exactly the requested command to every environment;
  # the task's training sampler otherwise intentionally replaces a portion
  # with pure-yaw or pure-linear commands.
  twist.rel_yaw_only_envs = 0.0
  twist.rel_linear_only_envs = 0.0
  twist.resampling_time_range = (cfg.duration_s + 1.0, cfg.duration_s + 1.0)

  agent_cfg = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
  wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped_env, asdict(agent_cfg), device=cfg.device)
  runner.load(str(cfg.checkpoint_file), load_cfg={"actor": True}, strict=True)
  policy = runner.get_inference_policy(device=cfg.device)

  obs, _ = wrapped_env.reset()
  num_steps = round(cfg.duration_s / env.step_dt)
  target_gravity = torch.tensor((-1.0, 0.0, 0.0), device=env.device)
  robot = env.scene["robot"]
  # The final state allows only RL/RR contact.  This sensor therefore catches
  # a front wheel, calf, hip, or trunk used as a final support while leaving
  # the valid initial four-wheel reset measurable as a transition state.
  forbidden_sensor = env.scene["forbidden_ground_contact"]
  front_leg_ids, _ = robot.find_joints(FRONT_LEG_JOINTS, preserve_order=True)
  front_hip_ids = [front_leg_ids[0], front_leg_ids[3]]
  front_arm_bend_ids = [front_leg_ids[1], front_leg_ids[2], front_leg_ids[4], front_leg_ids[5]]
  rear_leg_ids, _ = robot.find_joints(REAR_LEG_JOINTS, preserve_order=True)
  rear_hip_ids = [rear_leg_ids[0], rear_leg_ids[3]]
  all_hip_ids = front_hip_ids + rear_hip_ids
  rear_thigh_ids = [rear_leg_ids[1], rear_leg_ids[4]]
  rear_calf_ids = [rear_leg_ids[2], rear_leg_ids[5]]
  front_wheel_site_ids, _ = robot.find_sites(FRONT_WHEEL_SITES, preserve_order=True)
  rear_wheel_site_ids, _ = robot.find_sites(REAR_WHEEL_SITES, preserve_order=True)
  initial_gravity_error = torch.sum(
    torch.square(robot.data.projected_gravity_b - target_gravity), dim=1
  )
  initial_upright = initial_gravity_error < 0.12
  ever_upright = initial_upright.clone()
  first_upright_time = torch.full(
    (cfg.num_envs,), float("inf"), dtype=torch.float32, device=env.device
  )
  first_upright_time[initial_upright] = 0.0
  ever_done = torch.zeros(cfg.num_envs, dtype=torch.bool, device=env.device)
  action_delta_sum = torch.zeros((), device=env.device)
  velocity_error_sum = torch.zeros((), device=env.device)
  forward_velocity_sum = torch.zeros((), device=env.device)
  yaw_velocity_error_sum = torch.zeros((), device=env.device)
  yaw_velocity_sum = torch.zeros((), device=env.device)
  wheel_action_abs_sum = torch.zeros((), device=env.device)
  applied_wheel_action_abs_sum = torch.zeros((), device=env.device)
  pre_upright_forward_abs_sum = torch.zeros((), device=env.device)
  front_leg_velocity_sum = torch.zeros((), device=env.device)
  transition_leg_speed_sum = torch.zeros((), device=env.device)
  transition_action_delta_sum = torch.zeros((), device=env.device)
  transition_sample_count = torch.zeros((), device=env.device)
  forbidden_contact_count_sum = torch.zeros((), device=env.device)
  forbidden_contact_after_upright_sum = torch.zeros((), device=env.device)
  upright_sample_count = torch.zeros((), device=env.device)
  ever_forbidden_contact = torch.zeros(cfg.num_envs, dtype=torch.bool, device=env.device)
  command_x_sum = torch.zeros((), device=env.device)
  previous_actions: torch.Tensor | None = None
  wheel_action_term = env.action_manager.get_term("joint_vel")

  with torch.inference_mode():
    for step in range(num_steps):
      actions = policy(obs)
      if previous_actions is not None:
        action_delta = torch.linalg.vector_norm(actions - previous_actions, dim=1)
        action_delta_sum += action_delta.mean()
      else:
        action_delta = torch.zeros(cfg.num_envs, device=env.device)
      previous_actions = actions
      obs, _, dones, _ = wrapped_env.step(actions)
      ever_done |= dones.bool()

      body_z_w = torch.nn.functional.normalize(
        torch.stack(
          (
            2 * (robot.data.root_link_quat_w[:, 1] * robot.data.root_link_quat_w[:, 3]
                 + robot.data.root_link_quat_w[:, 0] * robot.data.root_link_quat_w[:, 2]),
            2 * (robot.data.root_link_quat_w[:, 2] * robot.data.root_link_quat_w[:, 3]
                 - robot.data.root_link_quat_w[:, 0] * robot.data.root_link_quat_w[:, 1]),
          ),
          dim=1,
        ),
        dim=1,
      )
      velocity_xy = robot.data.root_link_lin_vel_w[:, :2]
      actual_x = torch.sum(velocity_xy * body_z_w, dim=1)
      velocity_error_sum += torch.abs(actual_x - cfg.command_x).mean()
      forward_velocity_sum += actual_x.mean()
      actual_yaw = robot.data.root_link_ang_vel_w[:, 2]
      yaw_velocity_error_sum += torch.abs(actual_yaw - cfg.command_yaw).mean()
      yaw_velocity_sum += actual_yaw.mean()
      # The upright task exposes only RL/RR wheel velocity actions; FL/FR are
      # intentionally passive hanging-arm wheels.
      wheel_action_abs_sum += actions[:, -2:].abs().mean()
      applied_wheel_action_abs_sum += wheel_action_term._processed_actions.abs().mean()
      front_leg_velocity_sum += torch.linalg.vector_norm(
        robot.data.joint_vel[:, front_leg_ids], dim=1
      ).mean()
      command_x_sum += env.command_manager.get_command("twist")[:, 0].mean()

      # ``ever_upright`` still represents the attitude before this action, so
      # these statistics isolate the four-foot-to-upright transition itself.
      transition_mask = ~ever_upright
      transition_leg_speed_sum += torch.linalg.vector_norm(
        robot.data.joint_vel[:, :12], dim=1
      )[transition_mask].sum()
      transition_action_delta_sum += action_delta[transition_mask].sum()
      pre_upright_forward_abs_sum += actual_x[transition_mask].abs().sum()
      transition_sample_count += transition_mask.sum()

      gravity_error_now = torch.sum(
        torch.square(robot.data.projected_gravity_b - target_gravity), dim=1
      )
      # Use force history instead of the raw match count: Warp can report a
      # persistent resting contact with ``found == 0`` while retaining the
      # physical force in the sensor history.
      forbidden_history = forbidden_sensor.data.force_history
      if forbidden_history is not None:
        forbidden_force = torch.linalg.vector_norm(forbidden_history, dim=-1)
        forbidden_contact = torch.any(forbidden_force > 0.5, dim=(1, 2))
        forbidden_count = torch.any(forbidden_force > 0.5, dim=-1).sum(dim=1)
      else:
        assert forbidden_sensor.data.force is not None
        forbidden_force = torch.linalg.vector_norm(forbidden_sensor.data.force, dim=-1)
        forbidden_contact = torch.any(forbidden_force > 0.5, dim=1)
        forbidden_count = (forbidden_force > 0.5).sum(dim=1)
      upright_now = gravity_error_now < 0.12
      forbidden_contact_count_sum += forbidden_count.float().mean()
      forbidden_contact_after_upright_sum += (
        forbidden_contact & upright_now
      ).float().sum()
      upright_sample_count += upright_now.sum()
      ever_forbidden_contact |= forbidden_contact
      newly_upright = (gravity_error_now < 0.12) & ~ever_upright
      first_upright_time[newly_upright] = (step + 1) * env.step_dt
      ever_upright |= gravity_error_now < 0.12

  robot = env.scene["robot"]
  gravity_error = torch.sum(
    torch.square(robot.data.projected_gravity_b - target_gravity), dim=1
  )
  upright = gravity_error < 0.12
  survived = ~ever_done
  front_target = torch.tensor(
    FRONT_LEG_HANGING_POSE, dtype=robot.data.joint_pos.dtype, device=env.device
  )
  front_leg_pose_error = torch.linalg.vector_norm(
    robot.data.joint_pos[:, front_leg_ids] - front_target, dim=1
  )
  front_hip_parallel_error = torch.linalg.vector_norm(
    robot.data.joint_pos[:, front_hip_ids], dim=1
  )
  # Hip joints are the lateral swing DoFs.  Check every leg, not only the
  # hanging front pair, so a rear support that splays its wheels outward
  # cannot pass the final validation.
  all_hip_max_deviation = torch.abs(
    robot.data.joint_pos[:, all_hip_ids]
  ).amax(dim=1)
  front_arm_bend_target = torch.tensor(
    FRONT_ARM_BEND_POSE, dtype=robot.data.joint_pos.dtype, device=env.device
  )
  front_arm_bend_error = torch.linalg.vector_norm(
    robot.data.joint_pos[:, front_arm_bend_ids] - front_arm_bend_target, dim=1
  )
  front_wheel_height = robot.data.site_pos_w[:, front_wheel_site_ids, 2].mean(dim=1)
  front_wheel_offset_w = (
    robot.data.site_pos_w[:, front_wheel_site_ids]
    - robot.data.root_link_pos_w[:, None]
  )
  root_quat_w = robot.data.root_link_quat_w[:, None].expand(
    -1, front_wheel_offset_w.shape[1], -1
  )
  front_wheel_pos_b = quat_apply_inverse(root_quat_w, front_wheel_offset_w)
  front_wheel_hanging_target = torch.tensor(
    FRONT_WHEEL_HANGING_POSITIONS_B,
    dtype=front_wheel_pos_b.dtype,
    device=env.device,
  )
  front_wheel_side_error = torch.abs(
    front_wheel_pos_b - front_wheel_hanging_target
  ).mean(dim=(1, 2))
  rear_wheel_offset_w = (
    robot.data.site_pos_w[:, rear_wheel_site_ids]
    - robot.data.root_link_pos_w[:, None]
  )
  rear_wheel_pos_b = quat_apply_inverse(
    root_quat_w, rear_wheel_offset_w
  )
  # RL/RR are the two support wheels.  Their centres must differ only along
  # the lateral body axis; a mismatch in body x or z is the unwanted
  # fore/aft scissor that makes the two rolling axes non-collinear.
  rear_wheel_axis_error = torch.abs(
    rear_wheel_pos_b[:, 0, (0, 2)] - rear_wheel_pos_b[:, 1, (0, 2)]
  ).sum(dim=1)
  rear_target = torch.tensor(
    REAR_LEG_SUPPORT_POSE, dtype=robot.data.joint_pos.dtype, device=env.device
  )
  rear_leg_pose_error = torch.linalg.vector_norm(
    robot.data.joint_pos[:, rear_leg_ids] - rear_target, dim=1
  )
  final_forbidden_history = forbidden_sensor.data.force_history
  if final_forbidden_history is not None:
    final_forbidden_force = torch.linalg.vector_norm(final_forbidden_history, dim=-1)
    final_forbidden_contact = torch.any(final_forbidden_force > 0.5, dim=(1, 2))
  else:
    assert forbidden_sensor.data.force is not None
    final_forbidden_contact = torch.any(
      torch.linalg.vector_norm(forbidden_sensor.data.force, dim=-1) > 0.5, dim=1
    )
  metrics = {
    "upright_rate": upright.float().mean().item(),
    "initial_upright_rate": initial_upright.float().mean().item(),
    "ever_upright_rate": ever_upright.float().mean().item(),
    "mean_first_upright_time": first_upright_time[ever_upright].mean().item()
    if torch.any(ever_upright)
    else float("inf"),
    "survival_rate": survived.float().mean().item(),
    "upright_and_survived_rate": (upright & survived).float().mean().item(),
    "mean_gravity_error": gravity_error.mean().item(),
    "mean_base_height": robot.data.root_link_pos_w[:, 2].mean().item(),
    "mean_front_wheel_height": front_wheel_height.mean().item(),
    "mean_front_wheel_side_error": front_wheel_side_error.mean().item(),
    "mean_rear_wheel_axis_error": rear_wheel_axis_error.mean().item(),
    # Keep these ungated diagnostics so near-upright checkpoints can be
    # improved before they are strict enough to count as a completed stance.
    "mean_front_leg_pose_error_all": front_leg_pose_error.mean().item(),
    "mean_front_hip_parallel_error_all": front_hip_parallel_error.mean().item(),
    "mean_all_hip_max_deviation_all": all_hip_max_deviation.mean().item(),
    "mean_front_arm_bend_error_all": front_arm_bend_error.mean().item(),
    "mean_rear_leg_support_pose_error_all": rear_leg_pose_error.mean().item(),
    "mean_rear_thigh_position": robot.data.joint_pos[:, rear_thigh_ids].mean().item(),
    "mean_rear_calf_position": robot.data.joint_pos[:, rear_calf_ids].mean().item(),
    # The reference-mapped raised stance places the hanging front wheel
    # centres at 43.9 cm.  Allow 4 cm of balance variation while requiring
    # them to be visibly clear of the terrain.
    "upright_front_wheel_clearance_rate": (
      (upright & (front_wheel_height > FRONT_WHEEL_MIN_HEIGHT)).float().mean().item()
    ),
    "upright_front_wheel_side_alignment_rate": (
      (upright & (front_wheel_side_error < 0.04)).float().mean().item()
    ),
    "upright_rear_wheel_axis_alignment_rate": (
      (upright & (rear_wheel_axis_error < 0.02)).float().mean().item()
    ),
    "mean_velocity_abs_error": (velocity_error_sum / num_steps).item(),
    "mean_forward_velocity": (forward_velocity_sum / num_steps).item(),
    "mean_yaw_velocity_abs_error": (yaw_velocity_error_sum / num_steps).item(),
    "mean_yaw_velocity": (yaw_velocity_sum / num_steps).item(),
    "mean_wheel_action_abs": (wheel_action_abs_sum / num_steps).item(),
    "mean_applied_wheel_action_abs": (
      applied_wheel_action_abs_sum / num_steps
    ).item(),
    "mean_pre_upright_forward_abs_velocity": (
      pre_upright_forward_abs_sum / transition_sample_count.clamp_min(1)
    ).item(),
    "mean_front_leg_velocity": (front_leg_velocity_sum / num_steps).item(),
    "mean_forbidden_contact_count": (forbidden_contact_count_sum / num_steps).item(),
    "ever_forbidden_contact_rate": ever_forbidden_contact.float().mean().item(),
    "upright_forbidden_contact_rate": (
      forbidden_contact_after_upright_sum / upright_sample_count.clamp_min(1)
    ).item(),
    "final_forbidden_contact_rate": final_forbidden_contact.float().mean().item(),
    "upright_front_leg_pose_error": front_leg_pose_error[upright].mean().item()
    if torch.any(upright)
    else float("inf"),
    "upright_front_leg_pose_rate": (
      (upright & (front_leg_pose_error < 0.55)).float().mean().item()
    ),
    "upright_front_hip_parallel_error": (
      front_hip_parallel_error[upright].mean().item()
      if torch.any(upright)
      else float("inf")
    ),
    "upright_front_hip_parallel_rate": (
      (upright & (front_hip_parallel_error < 0.20)).float().mean().item()
    ),
    "upright_all_hip_no_splay_rate": (
      (upright & (all_hip_max_deviation < 0.12)).float().mean().item()
    ),
    "upright_front_arm_bend_error": (
      front_arm_bend_error[upright].mean().item()
      if torch.any(upright)
      else float("inf")
    ),
    "upright_front_arm_bend_rate": (
      (upright & (front_arm_bend_error < 0.45)).float().mean().item()
    ),
    "mean_command_x": (command_x_sum / num_steps).item(),
    "mean_action_delta": (action_delta_sum / max(num_steps - 1, 1)).item(),
    "mean_transition_leg_speed": (
      transition_leg_speed_sum / transition_sample_count.clamp_min(1)
    ).item(),
    "mean_transition_action_delta": (
      transition_action_delta_sum / transition_sample_count.clamp_min(1)
    ).item(),
    "num_steps": float(num_steps),
  }
  if torch.any(upright):
    upright_front_pos = robot.data.joint_pos[upright][:, front_leg_ids]
    upright_front_vel = robot.data.joint_vel[upright][:, front_leg_ids]
    for joint_idx, joint_name in enumerate(FRONT_LEG_JOINTS):
      metrics[f"upright_{joint_name}_pos"] = upright_front_pos[:, joint_idx].mean().item()
      metrics[f"upright_{joint_name}_vel_abs"] = (
        upright_front_vel[:, joint_idx].abs().mean().item()
      )
  wrapped_env.close()
  return metrics


def main() -> None:
  metrics = run(tyro.cli(EvalConfig))
  for name, value in metrics.items():
    print(f"{name}: {value:.6f}")


if __name__ == "__main__":
  main()
