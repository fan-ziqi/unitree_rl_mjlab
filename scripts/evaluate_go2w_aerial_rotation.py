"""Headless, per-command validation for Go2W one-shot aerial rotations.

The evaluator starts every trial from the normal four-wheel reset.  It pins one
of the five compact aerial one-hot commands only for this evaluation rollout;
the policy never receives target joint positions or a reference trajectory.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from tensordict import TensorDict

from src.assets.robots.unitree_go2w.go2w_constants import GO2W_LEG_JOINTS

TASK_ID = "Unitree-Go2W-Aerial-Rotation-Flat"
MODE_NAMES = ("front", "back", "left", "right", "yaw")
TARGET_ANGLE = math.tau
MetricDict = dict[str, float | int | str]


@dataclass
class EvalConfig:
    checkpoint_file: Path
    task_id: str = TASK_ID
    mode: int = 0
    all_modes: bool = False
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


def _pin_modes(command_term, modes: torch.Tensor) -> None:
    """Set one one-hot per environment and reset its command event state."""
    if modes.shape != (command_term.num_envs,):
        raise ValueError("modes must contain exactly one index per environment.")
    if torch.any((modes < 0) | (modes >= len(MODE_NAMES))):
        raise ValueError("modes contains an invalid aerial-mode index.")
    command_term.command_buf.zero_()
    command_term.command_buf[
        torch.arange(command_term.num_envs, device=modes.device), modes
    ] = 1.0
    command_term.was_airborne.zero_()
    command_term.has_grounded.zero_()
    command_term._airborne_time.zero_()
    command_term._flight_rotation.zero_()
    command_term._current_flight_qualified.zero_()
    command_term._landing_settle_time.zero_()
    command_term._landing_started.zero_()
    command_term._landing_hold_time.zero_()
    command_term._rotation_progress.zero_()
    command_term._launch_axis_w.zero_()
    # The command term captures the launch axis from the normal reset attitude
    # on its first control step, just as it does during training.
    command_term._new_skill.fill_(True)
    command_term._last_attempt_succeeded.zero_()


def _pin_mode(command_term, mode: int) -> None:
    """Compatibility wrapper for a scalar fixed-mode evaluation."""
    _pin_modes(
        command_term,
        torch.full(
            (command_term.num_envs,), mode, dtype=torch.long, device=command_term.device
        ),
    )


def _fixed_reset_observation(
    base_env: ManagerBasedRlEnv, modes: int | torch.Tensor
) -> TensorDict:
    """Pin one command per environment and rebuild the observation-history window.

    The aerial actor consumes ten consecutive observations.  ``env.reset()``
    necessarily constructs that history using the command sampled by the normal
    reset path; simply overwriting the command afterwards therefore evaluates a
    policy that sees nine stale one-hots.  A stationary pre-roll at the same
    physical reset state gives the requested command its proper 200-ms history
    without changing simulation state or granting privileged information.
    """
    command_term = base_env.command_manager.get_term("trick")
    if isinstance(modes, int):
        _pin_mode(command_term, modes)
    else:
        _pin_modes(command_term, modes)
    base_env.observation_manager._obs_buffer = None
    observations = None
    for _ in range(10):
        observations = base_env.observation_manager.compute(update_history=True)
    assert observations is not None
    base_env.obs_buf = observations
    return TensorDict(observations, batch_size=[base_env.num_envs])


def run(cfg: EvalConfig) -> MetricDict | list[MetricDict]:
    import mjlab.tasks  # noqa: F401

    import src.tasks  # noqa: F401

    if not cfg.checkpoint_file.is_file():
        raise FileNotFoundError(cfg.checkpoint_file)
    if not cfg.all_modes and not 0 <= cfg.mode < len(MODE_NAMES):
        raise ValueError(f"mode must be in [0, {len(MODE_NAMES) - 1}]")
    if cfg.num_envs <= 0 or cfg.duration_s <= 0.0:
        raise ValueError("num_envs and duration_s must be positive")
    if cfg.all_modes and (
        cfg.num_envs < len(MODE_NAMES) or cfg.num_envs % len(MODE_NAMES)
    ):
        raise ValueError(
            "all_modes requires num_envs to be a positive multiple of five."
        )

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
    if cfg.all_modes:
        num_per_mode = cfg.num_envs // len(MODE_NAMES)
        mode_indices = (
            torch.arange(cfg.num_envs, device=base_env.device) // num_per_mode
        )
    else:
        mode_indices = torch.full(
            (cfg.num_envs,), cfg.mode, dtype=torch.long, device=base_env.device
        )
    obs = _fixed_reset_observation(base_env, mode_indices)
    command_term = base_env.command_manager.get_term("trick")
    robot = base_env.scene["robot"]
    wheel_sensor = base_env.scene[command_cfg.sensor_name]
    min_ballistic_time = command_cfg.min_ballistic_time
    normal_gravity = torch.tensor((0.0, 0.0, -1.0), device=base_env.device)
    default_height = robot.data.default_root_state[:, 2]
    leg_joint_ids, _ = robot.find_joints(GO2W_LEG_JOINTS, preserve_order=True)

    trial_open = torch.ones(cfg.num_envs, dtype=torch.bool, device=base_env.device)
    has_grounded = torch.zeros_like(trial_open)
    ever_airborne = torch.zeros_like(trial_open)
    airborne_time = torch.zeros(cfg.num_envs, device=base_env.device)
    peak_airborne_time = torch.zeros(cfg.num_envs, device=base_env.device)
    completed = torch.zeros_like(trial_open)
    attempt_failed = torch.zeros_like(trial_open)
    failed = torch.zeros_like(trial_open)
    peak_progress = torch.zeros(cfg.num_envs, device=base_env.device)
    peak_height_delta = torch.zeros(cfg.num_envs, device=base_env.device)
    peak_axis_rate = torch.zeros(cfg.num_envs, device=base_env.device)
    takeoff_vertical_speed = torch.zeros(cfg.num_envs, device=base_env.device)
    # Report joint excursion descriptively for video/data review.  It is not a
    # success criterion: expressive legs are legal as long as no non-wheel
    # geometry supports on the ground.
    peak_leg_deviation = torch.zeros(cfg.num_envs, device=base_env.device)
    completion_gravity_error = torch.zeros(cfg.num_envs, device=base_env.device)
    completion_linear_speed = torch.zeros(cfg.num_envs, device=base_env.device)
    completion_angular_speed = torch.zeros(cfg.num_envs, device=base_env.device)
    completion_all_wheels = torch.zeros_like(trial_open)
    # Completion is deliberately strict.  Keep a separate audit of the best
    # *actual* four-wheel touchdown after a full turn so a zero completion
    # rate tells us whether the remaining defect is attitude, residual speed,
    # or no wheelward recovery at all.
    full_turn_seen = torch.zeros_like(trial_open)
    post_turn_touchdown = torch.zeros_like(trial_open)
    post_turn_best_gravity_error = torch.full(
        (cfg.num_envs,), float("inf"), device=base_env.device
    )
    post_turn_best_linear_speed = torch.full(
        (cfg.num_envs,), float("inf"), device=base_env.device
    )
    post_turn_best_angular_speed = torch.full(
        (cfg.num_envs,), float("inf"), device=base_env.device
    )
    # Averages of separately minimized speed, attitude, and contact can make
    # different instants of one failed rollout look like a stable landing.
    # Record their simultaneous occurrence and the command term's actual
    # consecutive-settle accumulator instead.
    strict_landing_seen = torch.zeros_like(trial_open)
    peak_command_landing_settle_time = torch.zeros(
        cfg.num_envs, device=base_env.device
    )
    # Keep the physical failure modes separate.  A single ``illegal_reset``
    # number tells us that an attempt failed but cannot distinguish a trunk
    # collision from a compactness violation or a deliberate over-rotation
    # guard.  These are evaluator-only diagnostics: none enters the policy
    # observation or the reward.
    termination_by_term = {
        name: torch.zeros_like(trial_open)
        for name in base_env.termination_manager._term_dones
    }

    num_steps = round(cfg.duration_s / base_env.step_dt)
    with torch.inference_mode():
        for _ in range(num_steps):
            # The term stores signed rotation accumulated while airborne.  Record it
            # before a successful completion deliberately clears the one-shot state.
            peak_progress = torch.maximum(
                peak_progress,
                torch.where(
                    trial_open,
                    command_term._rotation_progress,
                    torch.zeros_like(peak_progress),
                ),
            )
            pre_active = torch.sum(command_term.command, dim=1) > 0.5
            obs, _, dones, _ = env.step(policy(obs))
            contacts = _wheel_contacts(wheel_sensor)
            airborne = ~torch.any(contacts, dim=1)
            has_grounded |= trial_open & torch.all(contacts, dim=1)
            airborne_time = torch.where(
                trial_open & has_grounded & airborne,
                airborne_time + base_env.step_dt,
                torch.zeros_like(airborne_time),
            )
            peak_airborne_time = torch.maximum(peak_airborne_time, airborne_time)
            first_liftoff = (
                trial_open
                & (airborne_time >= min_ballistic_time)
                & ~ever_airborne
            )
            takeoff_vertical_speed[first_liftoff] = torch.clamp(
                robot.data.root_link_lin_vel_w[first_liftoff, 2], min=0.0
            )
            ever_airborne |= trial_open & (airborne_time >= min_ballistic_time)
            height_delta = robot.data.root_link_pos_w[:, 2] - default_height
            peak_height_delta = torch.maximum(
                peak_height_delta,
                torch.where(trial_open, height_delta, torch.zeros_like(height_delta)),
            )
            axis_rate = torch.sum(
                robot.data.root_link_ang_vel_w * command_term._launch_axis_w, dim=1
            )
            peak_axis_rate = torch.maximum(
                peak_axis_rate,
                torch.where(
                    trial_open & airborne, axis_rate, torch.zeros_like(axis_rate)
                ),
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
            post_active = torch.sum(command_term.command, dim=1) > 0.5
            full_turn_seen |= trial_open & (
                command_term._rotation_progress >= TARGET_ANGLE
            )
            post_turn_touchdown_now = (
                trial_open
                & full_turn_seen
                & torch.all(contacts, dim=1)
            )
            if torch.any(post_turn_touchdown_now):
                gravity_error = torch.sum(
                    torch.square(robot.data.projected_gravity_b - normal_gravity),
                    dim=1,
                )
                linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=1)
                angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=1)
                post_turn_touchdown |= post_turn_touchdown_now
                post_turn_best_gravity_error = torch.where(
                    post_turn_touchdown_now,
                    torch.minimum(post_turn_best_gravity_error, gravity_error),
                    post_turn_best_gravity_error,
                )
                post_turn_best_linear_speed = torch.where(
                    post_turn_touchdown_now,
                    torch.minimum(post_turn_best_linear_speed, linear_speed),
                    post_turn_best_linear_speed,
                )
                post_turn_best_angular_speed = torch.where(
                    post_turn_touchdown_now,
                    torch.minimum(post_turn_best_angular_speed, angular_speed),
                    post_turn_best_angular_speed,
                )
            strict_landing_now = (
                trial_open
                & (command_term._rotation_progress >= TARGET_ANGLE)
                & (
                    command_term._rotation_progress
                    <= TARGET_ANGLE + command_cfg.max_overrotation
                )
                & torch.all(contacts, dim=1)
                & (
                    torch.sum(
                        torch.square(robot.data.projected_gravity_b - normal_gravity),
                        dim=1,
                    )
                    < command_cfg.landing_gravity_error_limit
                )
                & (
                    torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=1)
                    < command_cfg.landing_linear_velocity_limit
                )
                & (
                    torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=1)
                    < command_cfg.landing_angular_velocity_limit
                )
            )
            strict_landing_seen |= strict_landing_now
            peak_command_landing_settle_time = torch.maximum(
                peak_command_landing_settle_time,
                command_term._landing_settle_time,
            )
            # A one-hot now clears when its single attempt ends, successful or
            # failed.  Use the command term's explicit strict outcome bit so a
            # return to idle is never reported as a completed aerial maneuver.
            attempt_finished_now = trial_open & pre_active & ~post_active
            completed_now = (
                attempt_finished_now
                & command_term._last_attempt_succeeded
                & ~dones.bool()
            )
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
                completion_all_wheels[completed_now] = torch.all(
                    contacts[completed_now], dim=1
                )
            completed |= completed_now
            for name, term_done in base_env.termination_manager._term_dones.items():
                termination_by_term[name] |= (
                    trial_open & dones.bool() & term_done
                )
            attempt_failed |= trial_open & attempt_finished_now & ~completed_now
            failed |= trial_open & dones.bool() & ~completed_now
            trial_open &= ~(attempt_finished_now | dones.bool())

    peak_progress = torch.maximum(peak_progress, command_term._rotation_progress)

    def summarize(mode_index: int, mask: torch.Tensor) -> MetricDict:
        completed_mask = completed & mask
        completed_count = completed_mask.sum().item()
        post_turn_touchdown_mask = post_turn_touchdown & mask
        post_turn_touchdown_count = post_turn_touchdown_mask.sum().item()
        metrics: MetricDict = {
            "mode": MODE_NAMES[mode_index],
            "mode_index": mode_index,
            "num_envs": int(mask.sum().item()),
            "duration_s": cfg.duration_s,
            "airborne_rate": ever_airborne[mask].float().mean().item(),
            "full_turn_rate": (peak_progress[mask] >= TARGET_ANGLE)
            .float()
            .mean()
            .item(),
            "completion_rate": completed[mask].float().mean().item(),
            "attempt_failure_rate": attempt_failed[mask].float().mean().item(),
            "illegal_reset_rate": failed[mask].float().mean().item(),
            "unfinished_rate": trial_open[mask].float().mean().item(),
            "mean_peak_rotation_rad": peak_progress[mask].mean().item(),
            "mean_peak_rotation_turns": (peak_progress[mask] / TARGET_ANGLE)
            .mean()
            .item(),
            "mean_peak_height_delta_m": peak_height_delta[mask].mean().item(),
            "mean_peak_axis_rate_rad_s": peak_axis_rate[mask].mean().item(),
            "mean_takeoff_vertical_speed_m_s": takeoff_vertical_speed[mask]
            .mean()
            .item(),
            "mean_peak_airborne_time_s": peak_airborne_time[mask].mean().item(),
            "mean_peak_leg_deviation_rad": peak_leg_deviation[mask].mean().item(),
            "p95_peak_leg_deviation_rad": torch.quantile(
                peak_leg_deviation[mask], 0.95
            ).item(),
            "max_peak_leg_deviation_rad": peak_leg_deviation[mask].max().item(),
            "completion_four_wheel_contact_rate": completion_all_wheels[completed_mask]
            .float()
            .sum()
            .item()
            / max(completed_count, 1),
            "completion_mean_gravity_error": completion_gravity_error[completed_mask]
            .mean()
            .item()
            if completed_count
            else float("inf"),
            "completion_mean_linear_speed": completion_linear_speed[completed_mask]
            .mean()
            .item()
            if completed_count
            else float("inf"),
            "completion_mean_angular_speed": completion_angular_speed[completed_mask]
            .mean()
            .item()
            if completed_count
            else float("inf"),
            "post_turn_four_wheel_touchdown_rate": post_turn_touchdown_mask.float()
            .mean()
            .item(),
            "strict_landing_state_rate": strict_landing_seen[mask].float()
            .mean()
            .item(),
            "mean_peak_command_landing_settle_time_s": peak_command_landing_settle_time[
                mask
            ]
            .mean()
            .item(),
            "max_peak_command_landing_settle_time_s": peak_command_landing_settle_time[
                mask
            ]
            .max()
            .item(),
            "post_turn_best_gravity_error": post_turn_best_gravity_error[
                post_turn_touchdown_mask
            ]
            .mean()
            .item()
            if post_turn_touchdown_count
            else float("inf"),
            "post_turn_best_linear_speed": post_turn_best_linear_speed[
                post_turn_touchdown_mask
            ]
            .mean()
            .item()
            if post_turn_touchdown_count
            else float("inf"),
            "post_turn_best_angular_speed": post_turn_best_angular_speed[
                post_turn_touchdown_mask
            ]
            .mean()
            .item()
            if post_turn_touchdown_count
            else float("inf"),
        }
        metrics.update(
            {
                f"termination_{name}_rate": termination_by_term[name][mask]
                .float()
                .mean()
                .item()
                for name in termination_by_term
            }
        )
        return metrics

    metrics: MetricDict | list[MetricDict]
    if cfg.all_modes:
        metrics = [
            summarize(mode_index, mode_indices == mode_index)
            for mode_index in range(len(MODE_NAMES))
        ]
    else:
        metrics = summarize(cfg.mode, torch.ones_like(mode_indices, dtype=torch.bool))
    env.close()
    return metrics


def main() -> None:
    cfg = tyro.cli(EvalConfig)
    metrics = run(cfg)
    if cfg.emit_metrics:
        print(json.dumps(metrics, sort_keys=True))
    elif not cfg.quiet:
        if isinstance(metrics, list):
            for mode_metrics in metrics:
                print(
                    " ".join(f"{name}: {value}" for name, value in mode_metrics.items())
                )
        else:
            for name, value in metrics.items():
                print(f"{name}: {value}")


if __name__ == "__main__":
    main()
