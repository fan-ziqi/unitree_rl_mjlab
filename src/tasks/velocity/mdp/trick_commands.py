"""Compact commands for the Go2W trick environments.

The actor only receives the returned tensors.  Task-specific targets such as
support contacts, gravity directions, and aerial rotation axes remain inside
the reward functions so that the policy interface stays proprioceptive.

The aerial command is an *event*, not a persistent velocity request: a
non-zero one-hot is held for one attempt and is cleared automatically only
after a complete turn and stable landing.  A failed partial hop therefore
cannot turn itself into an idle command to farm recovery reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply, quat_from_euler_xyz


class StanceSpinCommand(CommandTerm):
  """Sample ``[stand, front, rear, left, right, spin_rate]`` commands."""

  cfg: "StanceSpinCommandCfg"

  def __init__(self, cfg: "StanceSpinCommandCfg", env):
    super().__init__(cfg, env)
    if len(cfg.mode_probabilities) != 5:
      raise ValueError("mode_probabilities must contain stand/front/rear/left/right.")
    if any(probability < 0.0 for probability in cfg.mode_probabilities):
      raise ValueError("mode_probabilities cannot contain negative values.")
    if sum(cfg.mode_probabilities) <= 0.0:
      raise ValueError("mode_probabilities must have positive total mass.")
    if not 0.0 <= cfg.spin_idle_probability <= 1.0:
      raise ValueError("spin_idle_probability must be in [0, 1].")
    if cfg.spin_rate_range[0] <= 0.0 or cfg.spin_rate_range[0] > cfg.spin_rate_range[1]:
      raise ValueError("spin_rate_range must be a positive ordered magnitude range.")
    if cfg.spin_rate_ramp_rate <= 0.0:
      raise ValueError("spin_rate_ramp_rate must be positive.")

    self.command_buf = torch.zeros(self.num_envs, 6, device=self.device)
    self._target_spin_rate = torch.zeros(self.num_envs, device=self.device)
    self._mode_probabilities = torch.tensor(
      cfg.mode_probabilities, dtype=torch.float32, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    return self.command_buf

  def _update_metrics(self) -> None:
    # Metrics are defined by the reward terms because the command does not
    # prescribe an orientation or a contact mask to the actor.
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    if count == 0:
      return

    modes = torch.multinomial(self._mode_probabilities, count, replacement=True)
    self.command_buf[env_ids] = 0.0
    self._target_spin_rate[env_ids] = 0.0
    self.command_buf[env_ids, modes] = 1.0

    # Stand uses a non-zero rate for the support-changing Thomas-like orbit.
    # Front and rear support use it for a fixed-pair handstand spin.  Left and
    # right support remain static because their wheel geometry does not provide
    # the demonstrated spin skill.
    spin_ids = env_ids[modes <= 2]
    if len(spin_ids) == 0:
      return
    moving = torch.rand(len(spin_ids), device=self.device) > self.cfg.spin_idle_probability
    moving_ids = spin_ids[moving]
    if len(moving_ids) == 0:
      return
    magnitude = torch.empty(len(moving_ids), device=self.device).uniform_(
      *self.cfg.spin_rate_range
    )
    sign = torch.where(
      torch.rand(len(moving_ids), device=self.device) < 0.5,
      -torch.ones_like(magnitude),
      torch.ones_like(magnitude),
    )
    self._target_spin_rate[moving_ids] = sign * magnitude

  def _update_command(self) -> None:
    """Ramp only the continuous rate channel; stance one-hots switch directly."""
    current_rate = self.command_buf[:, 5]
    max_delta = self.cfg.spin_rate_ramp_rate * self._env.step_dt
    delta = torch.clamp(self._target_spin_rate - current_rate, -max_delta, max_delta)
    self.command_buf[:, 5] = current_rate + delta

  def set_curriculum(
    self,
    *,
    mode_probabilities: tuple[float, float, float, float, float] | None = None,
    spin_idle_probability: float | None = None,
    spin_rate_range: tuple[float, float] | None = None,
  ) -> None:
    """Update sampling difficulty without changing the actor command layout."""
    if mode_probabilities is not None:
      if len(mode_probabilities) != 5 or any(p < 0.0 for p in mode_probabilities):
        raise ValueError("mode_probabilities must contain five non-negative values.")
      if sum(mode_probabilities) <= 0.0:
        raise ValueError("mode_probabilities must have positive total mass.")
      self.cfg.mode_probabilities = mode_probabilities
      self._mode_probabilities = torch.tensor(
        mode_probabilities, dtype=torch.float32, device=self.device
      )
    if spin_idle_probability is not None:
      if not 0.0 <= spin_idle_probability <= 1.0:
        raise ValueError("spin_idle_probability must be in [0, 1].")
      self.cfg.spin_idle_probability = spin_idle_probability
    if spin_rate_range is not None:
      if spin_rate_range[0] <= 0.0 or spin_rate_range[0] > spin_rate_range[1]:
        raise ValueError("spin_rate_range must be a positive ordered magnitude range.")
      self.cfg.spin_rate_range = spin_rate_range


@dataclass(kw_only=True)
class StanceSpinCommandCfg(CommandTermCfg):
  """Configuration for the compact continuous-contact trick command."""

  entity_name: str
  mode_probabilities: tuple[float, float, float, float, float] = (
    0.45,
    0.15,
    0.15,
    0.125,
    0.125,
  )
  spin_idle_probability: float = 0.55
  spin_rate_range: tuple[float, float] = (1.0, 6.0)
  spin_rate_ramp_rate: float = 2.0

  def build(self, env) -> StanceSpinCommand:
    return StanceSpinCommand(self, env)


class StanceLocomotionCommand(CommandTerm):
  """Sample ``[stand, front, rear, lin_vel_x, yaw_rate]`` commands.

  Lateral velocity is intentionally absent from the command and is always
  rewarded toward zero.  The same planar x/yaw intent therefore applies to a
  normal four-wheel pose and both two-wheel upright poses.
  """

  cfg: "StanceLocomotionCommandCfg"

  def __init__(self, cfg: "StanceLocomotionCommandCfg", env):
    super().__init__(cfg, env)
    if len(cfg.mode_probabilities) != 3:
      raise ValueError("mode_probabilities must contain stand/front/rear.")
    if any(probability < 0.0 for probability in cfg.mode_probabilities):
      raise ValueError("mode_probabilities cannot contain negative values.")
    if sum(cfg.mode_probabilities) <= 0.0:
      raise ValueError("mode_probabilities must have positive total mass.")
    if not 0.0 <= cfg.idle_probability <= 1.0:
      raise ValueError("idle_probability must be in [0, 1].")
    self._validate_range("lin_vel_x_range", cfg.lin_vel_x_range)
    self._validate_range("yaw_rate_range", cfg.yaw_rate_range)

    self.command_buf = torch.zeros(self.num_envs, 5, device=self.device)
    self._mode_probabilities = torch.tensor(
      cfg.mode_probabilities, dtype=torch.float32, device=self.device
    )

  @staticmethod
  def _validate_range(name: str, value: tuple[float, float]) -> None:
    if value[0] > value[1]:
      raise ValueError(f"{name} must be ordered.")

  @property
  def command(self) -> torch.Tensor:
    return self.command_buf

  def _update_metrics(self) -> None:
    pass

  def _sample_active_axis(
    self, value_range: tuple[float, float], count: int
  ) -> torch.Tensor:
    """Sample a deliberately non-zero velocity without changing its command API.

    Exact one-point ranges are used by fixed-command evaluation and must be
    preserved verbatim.  For a training interval spanning zero, draw from the
    portions outside the command deadband instead of sampling near zero and
    silently erasing most moving requests afterwards.
    """
    lower, upper = value_range
    values = torch.zeros(count, dtype=self.command_buf.dtype, device=self.device)
    if count == 0 or lower == upper:
      if count > 0:
        values.fill_(lower)
      return values

    deadband = self.cfg.command_deadband
    negative_span = max(0.0, -deadband - lower)
    positive_span = max(0.0, upper - deadband)
    total_span = negative_span + positive_span
    if total_span <= 0.0:
      return values
    if negative_span <= 0.0:
      return values.uniform_(deadband, upper)
    if positive_span <= 0.0:
      return values.uniform_(lower, -deadband)

    positive = torch.rand(count, device=self.device) < positive_span / total_span
    positive_count = int(positive.sum().item())
    if positive_count > 0:
      values[positive] = torch.empty(
        positive_count, dtype=values.dtype, device=self.device
      ).uniform_(deadband, upper)
    negative_count = count - positive_count
    if negative_count > 0:
      values[~positive] = torch.empty(
        negative_count, dtype=values.dtype, device=self.device
      ).uniform_(lower, -deadband)
    return values

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    if count == 0:
      return
    modes = torch.multinomial(self._mode_probabilities, count, replacement=True)
    self.command_buf[env_ids] = 0.0
    self.command_buf[env_ids, modes] = 1.0
    idle = torch.rand(count, device=self.device) < self.cfg.idle_probability
    # A plain uniform draw made the actor see x and yaw together almost all
    # the time, while the deadband erased many individual axes.  It could
    # therefore learn a coupled wheel motion without learning either control
    # dimension in isolation.  Keep the external command exactly
    # ``one-hot + x + yaw``, but give PPO an explicit and balanced internal
    # distribution of x-only, yaw-only, and combined requests.
    moving_ids = env_ids[~idle]
    if len(moving_ids) > 0:
      request_type = torch.randint(0, 3, (len(moving_ids),), device=self.device)
      x_active = request_type != 1  # x-only or combined.
      yaw_active = request_type != 0  # yaw-only or combined.
      x_ids = moving_ids[x_active]
      yaw_ids = moving_ids[yaw_active]
      if len(x_ids) > 0:
        self.command_buf[x_ids, 3] = self._sample_active_axis(
          self.cfg.lin_vel_x_range, len(x_ids)
        )
      if len(yaw_ids) > 0:
        self.command_buf[yaw_ids, 4] = self._sample_active_axis(
          self.cfg.yaw_rate_range, len(yaw_ids)
        )

    # A front/rear two-wheel task begins near the commanded body orientation
    # rather than requiring a sparse 90-degree transition from four-wheel
    # standing.  It deliberately writes only the floating-base state: neither
    # reset nor action space specifies any leg configuration.
    if self.cfg.initialize_stance_on_reset:
      first_command = self.command_counter[env_ids] == 0
      initial_ids = env_ids[first_command]
      if len(initial_ids) > 0:
        self._initialize_stance_reset(initial_ids, modes[first_command])

  def _initialize_stance_reset(
    self, env_ids: torch.Tensor, modes: torch.Tensor
  ) -> None:
    """Place front/rear commands near their physical two-wheel support pose."""
    stance = modes > 0
    stance_ids = env_ids[stance]
    if len(stance_ids) == 0:
      return

    robot = self._env.scene[self.cfg.entity_name]
    # Pitch +pi/2 puts the front wheels below the trunk; -pi/2 does the same
    # for the rear wheels.  The low root height is obtained from the actual
    # wheel geometry, so both support wheels begin at the floor rather than
    # the policy having to discover a jump/fall transition first.
    pitch = torch.where(
      modes[stance] == 1,
      torch.full((len(stance_ids),), torch.pi / 2.0, device=self.device),
      torch.full((len(stance_ids),), -torch.pi / 2.0, device=self.device),
    )
    pitch += torch.empty_like(pitch).uniform_(
      -self.cfg.stance_reset_pitch_noise, self.cfg.stance_reset_pitch_noise
    )
    yaw = torch.empty_like(pitch).uniform_(-0.20, 0.20)
    zeros = torch.zeros_like(pitch)
    orientation = quat_from_euler_xyz(zeros, pitch, yaw)

    pose = robot.data.default_root_state[stance_ids, :7].clone()
    pose[:, :2] += self._env.scene.env_origins[stance_ids, :2]
    pose[:, :2] += torch.empty(
      len(stance_ids), 2, device=self.device, dtype=pose.dtype
    ).uniform_(-0.25, 0.25)
    height = torch.where(
      modes[stance] == 1,
      torch.full((len(stance_ids),), self.cfg.front_reset_height, device=self.device),
      torch.full((len(stance_ids),), self.cfg.rear_reset_height, device=self.device),
    )
    pose[:, 2] = height + torch.empty_like(height).uniform_(
      -self.cfg.stance_reset_height_noise, self.cfg.stance_reset_height_noise
    )
    pose[:, 3:7] = orientation
    robot.write_root_link_pose_to_sim(pose, env_ids=stance_ids)
    robot.write_root_link_velocity_to_sim(
      torch.zeros(len(stance_ids), 6, device=self.device, dtype=pose.dtype),
      env_ids=stance_ids,
    )
  def _update_command(self) -> None:
    pass

  def set_curriculum(
    self,
    *,
    mode_probabilities: tuple[float, float, float] | None = None,
    idle_probability: float | None = None,
    lin_vel_x_range: tuple[float, float] | None = None,
    yaw_rate_range: tuple[float, float] | None = None,
  ) -> None:
    if mode_probabilities is not None:
      if len(mode_probabilities) != 3 or any(p < 0.0 for p in mode_probabilities):
        raise ValueError("mode_probabilities must contain three non-negative values.")
      if sum(mode_probabilities) <= 0.0:
        raise ValueError("mode_probabilities must have positive total mass.")
      self.cfg.mode_probabilities = mode_probabilities
      self._mode_probabilities = torch.tensor(
        mode_probabilities, dtype=torch.float32, device=self.device
      )
    if idle_probability is not None:
      if not 0.0 <= idle_probability <= 1.0:
        raise ValueError("idle_probability must be in [0, 1].")
      self.cfg.idle_probability = idle_probability
    if lin_vel_x_range is not None:
      self._validate_range("lin_vel_x_range", lin_vel_x_range)
      self.cfg.lin_vel_x_range = lin_vel_x_range
    if yaw_rate_range is not None:
      self._validate_range("yaw_rate_range", yaw_rate_range)
      self.cfg.yaw_rate_range = yaw_rate_range


@dataclass(kw_only=True)
class StanceLocomotionCommandCfg(CommandTermCfg):
  """Configuration for normal/front/rear x-yaw wheeled locomotion."""

  entity_name: str
  mode_probabilities: tuple[float, float, float] = (0.40, 0.30, 0.30)
  idle_probability: float = 0.20
  lin_vel_x_range: tuple[float, float] = (-0.4, 0.4)
  yaw_rate_range: tuple[float, float] = (-0.4, 0.4)
  command_deadband: float = 0.05
  initialize_stance_on_reset: bool = True
  # These heights are calibrated against the actual MuJoCo collision geometry
  # together with the folded front/rear support-leg poses above.  They put the
  # support wheel centres at their 8.6 cm radius without a thigh/calf contact.
  front_reset_height: float = 0.392
  rear_reset_height: float = 0.340
  stance_reset_pitch_noise: float = 0.02
  stance_reset_height_noise: float = 0.003

  def build(self, env) -> StanceLocomotionCommand:
    return StanceLocomotionCommand(self, env)


class AerialRotationCommand(CommandTerm):
  """Sample one aerial trick one-hot, with the all-zero vector as idle."""

  cfg: "AerialRotationCommandCfg"

  def __init__(self, cfg: "AerialRotationCommandCfg", env):
    super().__init__(cfg, env)
    if len(cfg.mode_probabilities) != 5:
      raise ValueError("mode_probabilities must contain five aerial tricks.")
    if any(probability < 0.0 for probability in cfg.mode_probabilities):
      raise ValueError("mode_probabilities cannot contain negative values.")
    if sum(cfg.mode_probabilities) <= 0.0:
      raise ValueError("mode_probabilities must have positive total mass.")
    if not 0.0 <= cfg.idle_probability <= 1.0:
      raise ValueError("idle_probability must be in [0, 1].")
    if len(cfg.axes) != 5:
      raise ValueError("axes must contain front/back/left/right/yaw directions.")
    if cfg.target_angle <= 0.0 or cfg.max_overrotation < 0.0:
      raise ValueError("target_angle must be positive and max_overrotation non-negative.")
    if (
      cfg.rotation_progress_clearance_start < 0.0
      or cfg.rotation_progress_clearance_full
      <= cfg.rotation_progress_clearance_start
      or cfg.rotation_rate_clearance_start < 0.0
      or cfg.rotation_rate_clearance_full <= cfg.rotation_rate_clearance_start
    ):
      raise ValueError("Aerial rotation clearance gates must be ordered positive ranges.")
    if (
      cfg.landing_linear_velocity_limit <= 0.0
      or cfg.landing_angular_velocity_limit <= 0.0
      or cfg.min_ballistic_time <= 0.0
    ):
      raise ValueError("landing limits and min_ballistic_time must be positive.")

    self.command_buf = torch.zeros(self.num_envs, 5, device=self.device)
    self._mode_probabilities = torch.tensor(
      cfg.mode_probabilities, dtype=torch.float32, device=self.device
    )
    self.was_airborne = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    # Contact sensors can report no match on the first simulation step after
    # reset.  A maneuver cannot have taken off until it has first been seen on
    # its ordinary four-wheel support, otherwise an unchanging idle robot can
    # be incorrectly labelled as a landed aerial attempt.
    self.has_grounded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._airborne_time = torch.zeros(self.num_envs, device=self.device)
    self._landing_settle_time = torch.zeros(self.num_envs, device=self.device)
    self._rotation_progress = torch.zeros(self.num_envs, device=self.device)
    self._launch_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
    self._new_skill = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.command_buf

  def _update_metrics(self) -> None:
    pass

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    if count == 0:
      return
    self.command_buf[env_ids] = 0.0
    self.was_airborne[env_ids] = False
    self.has_grounded[env_ids] = False
    self._airborne_time[env_ids] = 0.0
    self._landing_settle_time[env_ids] = 0.0
    self._rotation_progress[env_ids] = 0.0
    self._launch_axis_w[env_ids] = 0.0
    self._new_skill[env_ids] = False
    active = torch.rand(count, device=self.device) > self.cfg.idle_probability
    active_ids = env_ids[active]
    if len(active_ids) == 0:
      return
    modes = torch.multinomial(self._mode_probabilities, len(active_ids), replacement=True)
    self.command_buf[active_ids, modes] = 1.0
    self._new_skill[active_ids] = True

  def _update_command(self) -> None:
    """Finish a one-shot request after a full turn, landing, and settling.

    CommandManager calls this after rewards.  Therefore the terminal landing
    reward still sees the requested one-hot, while the next actor observation
    correctly returns to the all-zero idle command.
    """
    active = torch.sum(self.command_buf, dim=1) > 0.5
    if not torch.any(active):
      self.was_airborne[~active] = False
      self.has_grounded[~active] = False
      self._airborne_time[~active] = 0.0
      self._landing_settle_time[~active] = 0.0
      self._rotation_progress[~active] = 0.0
      self._new_skill[~active] = False
      return

    asset = self._env.scene[self.cfg.entity_name]
    mode = torch.argmax(self.command_buf, dim=1)
    axes_b = torch.tensor(
      self.cfg.axes, dtype=asset.data.root_link_quat_w.dtype, device=self.device
    )[mode]
    self._launch_axis_w[self._new_skill] = quat_apply(
      asset.data.root_link_quat_w[self._new_skill], axes_b[self._new_skill]
    )
    self._new_skill[active] = False

    sensor: ContactSensor = self._env.scene[self.cfg.sensor_name]
    found = sensor.data.found
    assert found is not None
    contacts = (found.reshape(self.num_envs, found.shape[1], -1) > 0).any(dim=-1)
    airborne = ~torch.any(contacts, dim=1)
    self.has_grounded |= active & torch.all(contacts, dim=1)
    # A single contact-sensor gap can occur while a wheel rolls over a contact
    # edge or while the body is already colliding with the floor.  It is not a
    # jump.  A maneuver becomes airborne only after a short *continuous*
    # wheel-free interval; this is a physical validity condition, not a pose
    # or reference-trajectory target.
    flight_step = active & self.has_grounded & airborne
    self._airborne_time = torch.where(
      flight_step,
      self._airborne_time + self._env.step_dt,
      torch.zeros_like(self._airborne_time),
    )
    self.was_airborne |= self._airborne_time >= self.cfg.min_ballistic_time

    axis_rate = torch.sum(
      asset.data.root_link_ang_vel_w * self._launch_axis_w, dim=1
    )
    signed_delta = active * airborne * self.was_airborne * axis_rate * self._env.step_dt
    self._rotation_progress = torch.clamp_min(
      self._rotation_progress + signed_delta, 0.0
    )

    normal_gravity = torch.tensor(
      (0.0, 0.0, -1.0),
      dtype=asset.data.projected_gravity_b.dtype,
      device=self.device,
    )
    gravity_error = torch.sum(
      torch.square(asset.data.projected_gravity_b - normal_gravity), dim=1
    )
    linear_speed = torch.linalg.vector_norm(asset.data.root_link_lin_vel_w, dim=1)
    angular_speed = torch.linalg.vector_norm(asset.data.root_link_ang_vel_w, dim=1)
    stable_landing = (
      active
      & self.was_airborne
      & torch.all(contacts, dim=1)
      & (gravity_error < self.cfg.landing_gravity_error_limit)
      & (linear_speed < self.cfg.landing_linear_velocity_limit)
      & (angular_speed < self.cfg.landing_angular_velocity_limit)
    )
    self._landing_settle_time = torch.where(
      stable_landing,
      self._landing_settle_time + self._env.step_dt,
      torch.zeros_like(self._landing_settle_time),
    )
    complete_turn = (
      (self._rotation_progress >= self.cfg.target_angle)
      & (
        self._rotation_progress
        <= self.cfg.target_angle + self.cfg.max_overrotation
      )
    )
    finished = stable_landing & complete_turn & (
      self._landing_settle_time >= self.cfg.landing_settle_time
    )
    self.command_buf[finished] = 0.0
    self.was_airborne[finished] = False
    self._landing_settle_time[finished] = 0.0
    self._rotation_progress[finished] = 0.0

  def set_curriculum(
    self,
    *,
    idle_probability: float | None = None,
    mode_probabilities: tuple[float, float, float, float, float] | None = None,
    rotation_progress_clearance_start: float | None = None,
    rotation_progress_clearance_full: float | None = None,
    rotation_rate_clearance_start: float | None = None,
    rotation_rate_clearance_full: float | None = None,
  ) -> None:
    """Update hidden reward difficulty without adding an actor input."""
    if idle_probability is not None:
      if not 0.0 <= idle_probability <= 1.0:
        raise ValueError("idle_probability must be in [0, 1].")
      self.cfg.idle_probability = idle_probability
    if mode_probabilities is not None:
      if len(mode_probabilities) != 5 or any(p < 0.0 for p in mode_probabilities):
        raise ValueError("mode_probabilities must contain five non-negative values.")
      if sum(mode_probabilities) <= 0.0:
        raise ValueError("mode_probabilities must have positive total mass.")
      self.cfg.mode_probabilities = mode_probabilities
      self._mode_probabilities = torch.tensor(
        mode_probabilities, dtype=torch.float32, device=self.device
      )
    progress_pair = (rotation_progress_clearance_start, rotation_progress_clearance_full)
    rate_pair = (rotation_rate_clearance_start, rotation_rate_clearance_full)
    if (progress_pair[0] is None) != (progress_pair[1] is None):
      raise ValueError("Rotation-progress clearance limits must be updated together.")
    if (rate_pair[0] is None) != (rate_pair[1] is None):
      raise ValueError("Rotation-rate clearance limits must be updated together.")
    if progress_pair[0] is not None and progress_pair[1] is not None:
      if progress_pair[0] < 0.0 or progress_pair[1] <= progress_pair[0]:
        raise ValueError("Rotation-progress clearance limits must be ordered.")
      self.cfg.rotation_progress_clearance_start = progress_pair[0]
      self.cfg.rotation_progress_clearance_full = progress_pair[1]
    if rate_pair[0] is not None and rate_pair[1] is not None:
      if rate_pair[0] < 0.0 or rate_pair[1] <= rate_pair[0]:
        raise ValueError("Rotation-rate clearance limits must be ordered.")
      self.cfg.rotation_rate_clearance_start = rate_pair[0]
      self.cfg.rotation_rate_clearance_full = rate_pair[1]


@dataclass(kw_only=True)
class AerialRotationCommandCfg(CommandTermCfg):
  """Configuration for one-shot front/back/side/yaw aerial commands."""

  entity_name: str
  axes: tuple[tuple[float, float, float], ...] = (
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
  )
  mode_probabilities: tuple[float, float, float, float, float] = (
    0.2,
    0.2,
    0.2,
    0.2,
    0.2,
  )
  idle_probability: float = 0.15
  sensor_name: str = "wheel_ground_contact"
  landing_settle_time: float = 0.10
  landing_gravity_error_limit: float = 0.30
  landing_linear_velocity_limit: float = 0.75
  landing_angular_velocity_limit: float = 1.5
  # Four controller steps at the 50-Hz policy rate.  This removes the
  # observed ground-pivot exploit while allowing an ordinary compact jump to
  # accrue its full turn from the first genuine ballistic interval.
  min_ballistic_time: float = 0.08
  target_angle: float = math.tau
  max_overrotation: float = 0.75
  # Reward gates only; neither is exposed to the policy as a phase, pose, or
  # additional command.  The completed maneuver remains one full normal-wheel
  # landing at ``target_angle`` throughout training.
  rotation_progress_clearance_start: float = 0.12
  rotation_progress_clearance_full: float = 0.34
  # Angular rate is a momentum-shaping signal.  The stricter progress gate
  # above decides when angle itself is worth rewarding, so rate must begin
  # earlier in the ballistic arc to avoid asking PPO to accelerate only at
  # the apex.
  rotation_rate_clearance_start: float = 0.08
  rotation_rate_clearance_full: float = 0.24

  def build(self, env) -> AerialRotationCommand:
    return AerialRotationCommand(self, env)
