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
# This is a *visual-review* threshold, not the reward or task target.  A
# rendered maneuver can read as a complete revolution slightly before its
# integrated angle reaches the exact 2π engineering threshold.  Keep the two
# reports separate so a useful video is not mislabeled as no progress, while
# PPO still trains the full-turn objective.
VISUAL_TURN_FRACTION = 0.85
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
    # Require a visible stop after the one-shot landing, rather than a long
    # engineering hold that hides otherwise successful demonstrations.
    post_idle_settle_time: float = 0.30
    # Usually this is left unset and the task's standard network is used.
    # Explicit dimensions make this evaluator compatible with a deliberately
    # wider training run without changing its fixed-command physics.
    actor_hidden_dims: tuple[int, ...] | None = None
    critic_hidden_dims: tuple[int, ...] | None = None
    # Kept optional so ordinary MLP checkpoints remain the default, while a
    # recurrent PPO checkpoint can be reconstructed for the same fixed-command
    # physical evaluation.
    actor_class_name: str | None = None
    critic_class_name: str | None = None
    observation_history_length: int | None = None
    emit_metrics: bool = False
    output_path: Path | None = None
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
    command_term._landing_started.zero_()
    command_term._rotation_progress.zero_()
    command_term._launch_axis_w.zero_()
    command_term._launch_root_quat_w.zero_()
    # The command term captures the launch axis from the normal reset attitude
    # on its first control step, just as it does during training.
    command_term._new_skill.fill_(True)
    # Fixed one-hots must replace the reset-time sampled delayed event.
    command_term._pending_trigger.fill_(False)
    command_term._trigger_time.zero_()


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
    """Schedule fixed events while preserving the trained all-zero history.

    Aerial training holds the public command at zero for ``trigger_idle_time``
    after reset and only then exposes a one-hot.  Repeating a requested one-hot
    into every history slot, as the former evaluator did, created a command
    history the actor never saw while training.  Schedule the exact same
    pending event and rebuild the history from the genuine all-zero command;
    ``run`` advances that physical idle interval before scoring the maneuver.
    """
    command_term = base_env.command_manager.get_term("trick")
    if isinstance(modes, int):
        modes = torch.full(
            (command_term.num_envs,), modes, dtype=torch.long, device=command_term.device
        )
    if modes.shape != (command_term.num_envs,):
        raise ValueError("modes must contain exactly one index per environment.")
    if torch.any((modes < 0) | (modes >= len(MODE_NAMES))):
        raise ValueError("modes contains an invalid aerial-mode index.")
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
    command_term._pending_mode.copy_(modes)
    command_term._pending_trigger.fill_(True)
    command_term._trigger_time.zero_()
    base_env.observation_manager._obs_buffer = None
    observations = None
    history_length = base_env.cfg.observations["actor"].history_length or 1
    for _ in range(history_length):
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
    if (
        cfg.num_envs <= 0
        or cfg.duration_s <= 0.0
        or cfg.post_idle_settle_time <= 0.0
    ):
        raise ValueError("num_envs, duration_s, and post_idle_settle_time must be positive")
    if cfg.all_modes and (
        cfg.num_envs < len(MODE_NAMES) or cfg.num_envs % len(MODE_NAMES)
    ):
        raise ValueError(
            "all_modes requires num_envs to be a positive multiple of five."
        )

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    env_cfg = load_env_cfg(cfg.task_id, play=True)
    # The event/startup randomizers draw through the environment's own seed,
    # not only PyTorch's global RNG.  Without this assignment two CLI runs
    # with the same ``--seed`` can evaluate different contact/friction draws
    # and make a strict one-shot landing rate look non-reproducible.
    env_cfg.seed = cfg.seed
    env_cfg.scene.num_envs = cfg.num_envs
    if cfg.observation_history_length is not None:
        if cfg.observation_history_length <= 0:
            raise ValueError("observation_history_length must be positive")
        for group_name in ("actor", "critic"):
            env_cfg.observations[group_name].history_length = cfg.observation_history_length
    # A trial's desired mode must not be replaced by the normal periodic command
    # sampler before its evaluation interval has elapsed.
    command_cfg = env_cfg.commands["trick"]
    command_cfg.idle_probability = 0.0
    command_cfg.resampling_time_range = (
        cfg.duration_s + command_cfg.trigger_idle_time + 1.0,
        cfg.duration_s + command_cfg.trigger_idle_time + 1.0,
    )
    env_cfg.episode_length_s = cfg.duration_s + command_cfg.trigger_idle_time + 0.5
    agent_cfg = load_rl_cfg(cfg.task_id)
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
    # Reproduce the regular controller boundary: all-zero history/default
    # action first, then the requested one-hot becomes the newest history
    # element.  This is intentionally physical rather than a buffer-only
    # prefill, so the first maneuver action sees the same state/history pair
    # as PPO did during training.
    pre_trigger_steps = math.ceil(command_cfg.trigger_idle_time / base_env.step_dt)
    with torch.inference_mode():
        for _ in range(pre_trigger_steps):
            obs, _, dones, _ = env.step(policy(obs))
            if torch.any(dones):
                raise RuntimeError("Aerial fixed-command idle pre-roll terminated.")
    if not torch.all(torch.sum(command_term.command, dim=1) > 0.5):
        raise RuntimeError("Aerial fixed-command event did not trigger after idle pre-roll.")
    robot = base_env.scene["robot"]
    wheel_sensor = base_env.scene[command_cfg.sensor_name]
    min_ballistic_time = command_cfg.min_ballistic_time
    normal_gravity = torch.tensor((0.0, 0.0, -1.0), device=base_env.device)
    default_height = robot.data.default_root_state[:, 2]
    leg_joint_ids, _ = robot.find_joints(GO2W_LEG_JOINTS, preserve_order=True)
    wheel_site_ids, _ = robot.find_sites(("FL", "FR", "RL", "RR"), preserve_order=True)

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
    peak_airborne_wheel_root_distance = torch.zeros(
        cfg.num_envs, device=base_env.device
    )
    completion_gravity_error = torch.zeros(cfg.num_envs, device=base_env.device)
    completion_orientation_similarity = torch.zeros(
        cfg.num_envs, device=base_env.device
    )
    completion_linear_speed = torch.zeros(cfg.num_envs, device=base_env.device)
    completion_angular_speed = torch.zeros(cfg.num_envs, device=base_env.device)
    completion_all_wheels = torch.zeros_like(trial_open)
    # Completion is deliberately strict.  Keep a separate audit of the best
    # *actual* four-wheel touchdown after a full turn so a zero completion
    # rate tells us whether the remaining defect is attitude, residual speed,
    # or no wheelward recovery at all.
    full_turn_seen = torch.zeros_like(trial_open)
    near_full_turn_seen = torch.zeros_like(trial_open)
    post_turn_touchdown = torch.zeros_like(trial_open)
    visual_turn_touchdown = torch.zeros_like(trial_open)
    post_turn_best_gravity_error = torch.full(
        (cfg.num_envs,), float("inf"), device=base_env.device
    )
    post_turn_best_orientation_error = torch.full(
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
    # Record their simultaneous occurrence and our measured idle-settle
    # window instead.
    strict_landing_seen = torch.zeros_like(trial_open)
    idle_stable_time = torch.zeros(
        cfg.num_envs, device=base_env.device
    )
    peak_idle_stable_time = torch.zeros_like(idle_stable_time)
    # A non-zero command is one event, not permission to keep hopping until a
    # lucky landing.  The physical event closes at its *first contact*, not
    # when the brief command-verdict window later clears the public one-hot.
    # This catches a rebound that starts inside that window as a second flight.
    landing_seen = torch.zeros_like(trial_open)
    event_closed = torch.zeros_like(trial_open)
    post_event_airborne_time = torch.zeros(cfg.num_envs, device=base_env.device)
    post_event_relaunch = torch.zeros_like(trial_open)
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
            landing_seen |= trial_open & command_term._landing_started
            # The public one-hot clears at first contact.  Keep the physical
            # trial open through the following idle settle window so a quiet
            # landing, not just a command transition, determines success.
            post_landing_window = landing_seen & trial_open
            post_event_airborne_time = torch.where(
                post_landing_window & ~dones.bool() & airborne,
                post_event_airborne_time + base_env.step_dt,
                torch.zeros_like(post_event_airborne_time),
            )
            post_event_relaunch |= (
                post_landing_window
                & ~dones.bool()
                & (post_event_airborne_time >= min_ballistic_time)
            )
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
            wheel_root_distance = torch.max(
                torch.linalg.vector_norm(
                    robot.data.site_pos_w[:, wheel_site_ids]
                    - robot.data.root_link_pos_w.unsqueeze(1),
                    dim=2,
                ),
                dim=1,
            ).values
            peak_airborne_wheel_root_distance = torch.maximum(
                peak_airborne_wheel_root_distance,
                torch.where(
                    trial_open & airborne,
                    wheel_root_distance,
                    torch.zeros_like(peak_airborne_wheel_root_distance),
                ),
            )
            post_active = torch.sum(command_term.command, dim=1) > 0.5
            full_turn_seen |= trial_open & (
                command_term._rotation_progress >= TARGET_ANGLE
            )
            near_full_turn_seen |= trial_open & (
                command_term._rotation_progress
                >= VISUAL_TURN_FRACTION * TARGET_ANGLE
            )
            # A visible pass is intentionally less exact than the strict
            # completion state below: near one full airborne turn and a real
            # four-wheel landing without a reset.  It never feeds reward,
            # termination, or training selection.
            visual_turn_touchdown |= (
                trial_open
                & near_full_turn_seen
                & ~dones.bool()
                & torch.all(contacts, dim=1)
            )
            post_turn_touchdown_now = (
                trial_open
                & full_turn_seen
                & ~dones.bool()
                & torch.all(contacts, dim=1)
            )
            if torch.any(post_turn_touchdown_now):
                gravity_error = torch.sum(
                    torch.square(robot.data.projected_gravity_b - normal_gravity),
                    dim=1,
                )
                orientation_similarity = torch.abs(
                    torch.sum(
                        robot.data.root_link_quat_w
                        * command_term._launch_root_quat_w,
                        dim=1,
                    )
                )
                linear_speed = torch.linalg.vector_norm(robot.data.root_link_lin_vel_w, dim=1)
                angular_speed = torch.linalg.vector_norm(robot.data.root_link_ang_vel_w, dim=1)
                post_turn_touchdown |= post_turn_touchdown_now
                post_turn_best_gravity_error = torch.where(
                    post_turn_touchdown_now,
                    torch.minimum(post_turn_best_gravity_error, gravity_error),
                    post_turn_best_gravity_error,
                )
                post_turn_best_orientation_error = torch.where(
                    post_turn_touchdown_now,
                    torch.minimum(
                        post_turn_best_orientation_error,
                        1.0 - orientation_similarity,
                    ),
                    post_turn_best_orientation_error,
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
                    torch.abs(
                        torch.sum(
                            robot.data.root_link_quat_w
                            * command_term._launch_root_quat_w,
                            dim=1,
                        )
                    )
                    >= command_cfg.landing_orientation_dot_min
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
            idle_stable_now = event_closed & strict_landing_now
            idle_stable_time = torch.where(
                idle_stable_now,
                idle_stable_time + base_env.step_dt,
                torch.zeros_like(idle_stable_time),
            )
            peak_idle_stable_time = torch.maximum(
                peak_idle_stable_time, idle_stable_time
            )
            attempt_finished_now = trial_open & pre_active & ~post_active
            completed_now = (
                trial_open
                & event_closed
                & strict_landing_now
                & (
                    idle_stable_time + 0.5 * base_env.step_dt
                    >= cfg.post_idle_settle_time
                )
            )
            if torch.any(completed_now):
                gravity_error = torch.sum(
                    torch.square(robot.data.projected_gravity_b - normal_gravity), dim=1
                )
                completion_gravity_error[completed_now] = gravity_error[completed_now]
                completion_orientation_similarity[completed_now] = torch.abs(
                    torch.sum(
                        robot.data.root_link_quat_w[completed_now]
                        * command_term._launch_root_quat_w[completed_now],
                        dim=1,
                    )
                )
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
            attempt_failed |= trial_open & dones.bool() & ~completed_now
            failed |= trial_open & dones.bool() & ~completed_now
            event_closed |= attempt_finished_now
            trial_open &= ~(completed_now | dones.bool())

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
            "visual_near_turn_rate": (
                peak_progress[mask] >= VISUAL_TURN_FRACTION * TARGET_ANGLE
            )
            .float()
            .mean()
            .item(),
            "visual_four_wheel_landing_rate": (
                visual_turn_touchdown[mask].float().mean().item()
            ),
            "completion_rate": completed[mask].float().mean().item(),
            "attempt_failure_rate": attempt_failed[mask].float().mean().item(),
            "illegal_reset_rate": failed[mask].float().mean().item(),
            "unfinished_rate": trial_open[mask].float().mean().item(),
            "post_event_relaunch_rate": post_event_relaunch[mask].float()
            .mean()
            .item(),
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
            "mean_peak_airborne_wheel_root_distance_m": peak_airborne_wheel_root_distance[
                mask
            ]
            .mean()
            .item(),
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
            "completion_mean_orientation_similarity": completion_orientation_similarity[
                completed_mask
            ]
            .mean()
            .item()
            if completed_count
            else 0.0,
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
            "mean_peak_command_landing_settle_time_s": peak_idle_stable_time[
                mask
            ]
            .mean()
            .item(),
            "max_peak_command_landing_settle_time_s": peak_idle_stable_time[
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
            "post_turn_best_orientation_error": post_turn_best_orientation_error[
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
    if cfg.output_path is not None:
        cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_path.write_text(json.dumps(metrics, sort_keys=True) + "\n")
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
