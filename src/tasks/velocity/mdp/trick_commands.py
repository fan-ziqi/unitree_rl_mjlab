"""Compact commands for the Go2W trick environments.

The actor only receives the returned tensors.  Task-specific targets such as
support contacts, gravity directions, and aerial rotation axes remain inside
the reward functions so that the policy interface stays proprioceptive.

The aerial command is an *event*, not a persistent velocity request: a
non-zero one-hot is held for one attempt and is cleared after its first
landing decision window, whether that landing succeeds or fails.  A failed
partial hop therefore cannot retry the maneuver or keep the one-hot alive to
farm recovery reward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply


class StanceSpinCommand(CommandTerm):
  """Sample the complete five-one-hot continuous-contact trick command.

  The public layout is ``[normal, front, rear, left, right, spin_rate]``.
  Its literal all-zero value is the normal four-wheel idle.  A nonzero normal
  rate requests a local world-down rotation in one of the five contact modes.
  ``normal`` with zero rate is ordinary four-wheel idle; with a nonzero rate
  it requests the compact, level four-wheel common-axle pivot.  Its sign
  controls only world-down rotation direction.  Front/rear two-wheel modes
  retain that signed rotation request; left/right are static side supports and
  explicitly ignore it.  The stance itself—and the leg reshaping needed to
  make a local pivot possible—is
  discovered by the policy rather than encoded in the command.
  No command carries a pose, phase, or limb target.
  """

  cfg: StanceSpinCommandCfg

  def __init__(self, cfg: StanceSpinCommandCfg, env):
    super().__init__(cfg, env)
    if len(cfg.mode_probabilities) != 5:
      raise ValueError("mode_probabilities must contain stand/front/rear/left/right.")
    if any(probability < 0.0 for probability in cfg.mode_probabilities):
      raise ValueError("mode_probabilities cannot contain negative values.")
    if sum(cfg.mode_probabilities) <= 0.0:
      raise ValueError("mode_probabilities must have positive total mass.")
    if not 0.0 <= cfg.spin_idle_probability <= 1.0:
      raise ValueError("spin_idle_probability must be in [0, 1].")
    if not 0.0 <= cfg.upright_static_probability <= 1.0:
      raise ValueError("upright_static_probability must be in [0, 1].")
    if cfg.spin_rate_range[0] <= 0.0 or cfg.spin_rate_range[0] > cfg.spin_rate_range[1]:
      raise ValueError("spin_rate_range must be a positive ordered magnitude range.")
    if cfg.spin_rate_ramp_rate <= 0.0:
      raise ValueError("spin_rate_ramp_rate must be positive.")
    if cfg.transition_active_time <= 0.0:
      raise ValueError("spin transition duration must be positive.")
    if not 0.0 <= cfg.direct_switch_probability <= 1.0:
      raise ValueError("direct_switch_probability must be in [0, 1].")

    self.command_buf = torch.zeros(self.num_envs, 6, device=self.device)
    self._target_spin_rate = torch.zeros(self.num_envs, device=self.device)
    self._scheduled_command = torch.zeros_like(self.command_buf)
    self._scheduled_spin_rate = torch.zeros(self.num_envs, device=self.device)
    self._next_scheduled_command = torch.zeros_like(self.command_buf)
    self._next_scheduled_spin_rate = torch.zeros(self.num_envs, device=self.device)
    self._transition_phase = torch.zeros(
      self.num_envs, dtype=torch.int8, device=self.device
    )
    self._transition_time = torch.zeros(self.num_envs, device=self.device)
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
    next_probabilities = self._mode_probabilities.expand(count, -1).clone()
    next_probabilities[torch.arange(count, device=self.device), modes] = 0.0
    no_alternative = next_probabilities.sum(dim=1) == 0.0
    next_probabilities[no_alternative] = self._mode_probabilities
    next_modes = torch.multinomial(next_probabilities, 1).squeeze(1)
    # A direct A -> B change is the final public behaviour, but making every
    # early sample solve both a new support and a mode switch diluted the
    # two-wheel discovery return.  This sampler-only curriculum never changes
    # the public command tensor or inserts a hidden target: a zero probability
    # merely holds the sampled one-hot until its next resample.
    switch = torch.rand(count, device=self.device) < self.cfg.direct_switch_probability
    next_modes = torch.where(switch, next_modes, modes)
    self.command_buf[env_ids] = 0.0
    self._target_spin_rate[env_ids] = 0.0
    self._scheduled_command[env_ids] = 0.0
    self._scheduled_spin_rate[env_ids] = 0.0
    self._next_scheduled_command[env_ids] = 0.0
    self._next_scheduled_spin_rate[env_ids] = 0.0
    self._transition_phase[env_ids] = 3
    self._transition_time[env_ids] = 0.0
    non_idle = (
      torch.rand(count, device=self.device) > self.cfg.spin_idle_probability
    )
    # Every non-idle command has a public stance one-hot.  The literal
    # all-zero command remains default four-wheel idle.
    active_ids = env_ids[non_idle]
    if len(active_ids) > 0:
      self._scheduled_command[active_ids, modes[non_idle]] = 1.0
      self._next_scheduled_command[active_ids, next_modes[non_idle]] = 1.0

    # A static named-mode sample is useful only while PPO is discovering the
    # corresponding two-wheel support.  It must be a self-contained hold: an
    # old sampler could create ``normal@+r -> front@0`` midway through one
    # event, which asks the actor to brake exactly because the one-hot
    # changes.  Static holds apply to every named support (front/rear/left/
    # right), while front/rear and normal can retain one signed rate through a
    # direct switch.  Side supports stay static by task definition.
    static_named_hold = torch.zeros(count, dtype=torch.bool, device=self.device)
    if self.cfg.upright_static_probability > 0.0:
      first_named = non_idle & (modes != 0)
      static_named_hold = first_named & (
        torch.rand(count, device=self.device) < self.cfg.upright_static_probability
      )
      static_ids = env_ids[static_named_hold]
      if len(static_ids) > 0:
        self._next_scheduled_command[static_ids] = 0.0
        self._next_scheduled_command[static_ids, modes[static_named_hold]] = 1.0

    dynamic = non_idle
    dynamic_ids = env_ids[dynamic]
    if len(dynamic_ids) > 0:
      magnitude = torch.empty(len(dynamic_ids), device=self.device).uniform_(
        *self.cfg.spin_rate_range
      )
      sign = torch.where(
        torch.rand(len(dynamic_ids), device=self.device) < 0.5,
        -torch.ones_like(magnitude),
        torch.ones_like(magnitude),
      )
      signed_rate = sign * magnitude
      # The lateral one-hots are the static left/right two-wheel supports
      # shown in the AS2W manoeuvre.  They share the public rate slot for
      # interface compatibility, but that slot must remain zero: the spin
      # reward deliberately ignores lateral rates and the support reward
      # treats a nonzero value as a dynamic request.  Sampling a random rate
      # here therefore silently removed most useful side-support rollouts.
      side_dynamic = modes[non_idle] >= 3
      signed_rate = torch.where(side_dynamic, torch.zeros_like(signed_rate), signed_rate)
      # ``static_named_hold`` may have replaced the initially sampled next
      # one-hot.  Read the scheduled command rather than the stale sample so
      # both rate assignments describe the public command that will actually
      # be emitted.
      self._scheduled_spin_rate[dynamic_ids] = signed_rate
      self._next_scheduled_spin_rate[dynamic_ids] = signed_rate
      # A zero-rate front/rear one-hot is a temporary support-discovery
      # command.  It is emitted only as the self-contained hold constructed
      # above, never as one side of a mode change.  Normal deliberately stays
      # dynamic: default four-wheel idle remains the literal all-zero command.
      static_dynamic = static_named_hold[dynamic]
      self._scheduled_spin_rate[dynamic_ids[static_dynamic]] = 0.0
      self._next_scheduled_spin_rate[dynamic_ids[static_dynamic]] = 0.0

    if len(active_ids) > 0:
      # The reference manoeuvre has no side-support pivot.  Keep the public
      # six-vector unchanged, but emit a literal zero rate whenever its
      # selected one-hot is left/right.  This is command semantics, not a
      # posture target: PPO remains free to find the two-wheel support.
      scheduled_modes = torch.argmax(self._scheduled_command[active_ids, :5], dim=1)
      next_scheduled_modes = torch.argmax(
        self._next_scheduled_command[active_ids, :5], dim=1
      )
      self._scheduled_spin_rate[active_ids[scheduled_modes >= 3]] = 0.0
      self._next_scheduled_spin_rate[active_ids[next_scheduled_modes >= 3]] = 0.0

    # A physical reset is already the literal four-wheel idle.  Do not insert
    # an uncommanded idle interval before an externally valid one-hot: it
    # would train a different distribution from a caller that requests normal
    # immediately after reset.  Start the first command at once, then make the
    # second one-hot a direct A->B switch; the normal rate ramp still begins
    # from zero, so the initial acceleration remains physically smooth.
    if len(active_ids) > 0:
      self.command_buf[active_ids] = self._scheduled_command[active_ids]
      self._target_spin_rate[active_ids] = self._scheduled_spin_rate[active_ids]
      self._transition_phase[active_ids] = 1

  def _update_command(self) -> None:
    """Hold a command, directly switch once, then hold until resampling.

    A persistent external one-hot is the public spin interface.  Returning to
    idle before the sampler's 6-s command interval had elapsed trained a
    distribution that never occurs under fixed-command evaluation or the
    requested continuous AS2W pivot.  The all-zero command remains available
    when the sampler explicitly emits it through ``spin_idle_probability``.
    """
    pending = self._transition_phase < 3
    self._transition_time[pending] += self._env.step_dt
    second_start = (
      (self._transition_phase == 1)
      & (self._transition_time >= 0.5 * self.cfg.transition_active_time)
    )
    if torch.any(second_start):
      self.command_buf[second_start] = self._next_scheduled_command[second_start]
      self._target_spin_rate[second_start] = self._next_scheduled_spin_rate[
        second_start
      ]
      self._transition_phase[second_start] = 2
    finish = (
      (self._transition_phase == 2)
      & (self._transition_time >= self.cfg.transition_active_time)
    )
    if torch.any(finish):
      # Keep the second public one-hot and its signed rate through the next
      # resample.  This makes stage-0 normal a sustained pivot and makes a
      # direct dynamic A -> B switch continuous instead of ending in a brake.
      self._transition_phase[finish] = 3

    current_rate = self.command_buf[:, 5]
    max_delta = self.cfg.spin_rate_ramp_rate * self._env.step_dt
    delta = torch.clamp(self._target_spin_rate - current_rate, -max_delta, max_delta)
    self.command_buf[:, 5] = current_rate + delta

  def set_curriculum(
    self,
    *,
    mode_probabilities: tuple[float, float, float, float, float] | None = None,
    spin_idle_probability: float | None = None,
    upright_static_probability: float | None = None,
    direct_switch_probability: float | None = None,
    spin_rate_range: tuple[float, float] | None = None,
    resampling_time_range: tuple[float, float] | None = None,
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
    if upright_static_probability is not None:
      if not 0.0 <= upright_static_probability <= 1.0:
        raise ValueError("upright_static_probability must be in [0, 1].")
      self.cfg.upright_static_probability = upright_static_probability
    if direct_switch_probability is not None:
      if not 0.0 <= direct_switch_probability <= 1.0:
        raise ValueError("direct_switch_probability must be in [0, 1].")
      self.cfg.direct_switch_probability = direct_switch_probability
    if spin_rate_range is not None:
      if spin_rate_range[0] <= 0.0 or spin_rate_range[0] > spin_rate_range[1]:
        raise ValueError("spin_rate_range must be a positive ordered magnitude range.")
      self.cfg.spin_rate_range = spin_rate_range
    if resampling_time_range is not None:
      if (
        resampling_time_range[0] <= 0.0
        or resampling_time_range[0] > resampling_time_range[1]
      ):
        raise ValueError("resampling_time_range must be positive and ordered.")
      self.cfg.resampling_time_range = resampling_time_range


@dataclass(kw_only=True)
class StanceSpinCommandCfg(CommandTermCfg):
  """Configuration for the compact five-mode ground-trick command."""

  entity_name: str
  mode_probabilities: tuple[float, float, float, float, float] = (
    0.24,
    0.28,
    0.28,
    0.10,
    0.10,
  )
  spin_idle_probability: float = 0.55
  # A zero rate on any active named one-hot asks for that static two-wheel
  # support.  It is a curriculum sampling probability for a *held* support,
  # not another command component.  It must never turn one half of a direct
  # dynamic one-hot switch into a stop: final dynamic-switch stages set it to
  # zero altogether.
  upright_static_probability: float = 0.0
  # Sampling-only curriculum knob.  The actor always receives the same
  # persistent one-hot plus signed rate; early support discovery can hold one
  # command, while final stages restore direct A -> B changes.
  direct_switch_probability: float = 1.0
  spin_rate_range: tuple[float, float] = (5.0, 9.0)
  spin_rate_ramp_rate: float = 12.0
  transition_active_time: float = 4.2

  def build(self, env) -> StanceSpinCommand:
    return StanceSpinCommand(self, env)


class StanceLocomotionCommand(CommandTerm):
  """Sample ``[stand, front, rear, lin_vel_x, yaw_rate]`` commands.

  Lateral velocity is intentionally absent from the command and is always
  rewarded toward zero.  The same planar x/yaw intent therefore applies to a
  normal four-wheel pose and both two-wheel upright poses.
  """

  cfg: StanceLocomotionCommandCfg

  def __init__(self, cfg: StanceLocomotionCommandCfg, env):
    super().__init__(cfg, env)
    if len(cfg.mode_probabilities) != 3:
      raise ValueError("mode_probabilities must contain stand/front/rear.")
    if any(probability < 0.0 for probability in cfg.mode_probabilities):
      raise ValueError("mode_probabilities cannot contain negative values.")
    if sum(cfg.mode_probabilities) <= 0.0:
      raise ValueError("mode_probabilities must have positive total mass.")
    if not 0.0 <= cfg.idle_probability <= 1.0:
      raise ValueError("idle_probability must be in [0, 1].")
    if cfg.mode_idle_probabilities is not None:
      self._validate_mode_idle_probabilities(cfg.mode_idle_probabilities)
    self._validate_range("lin_vel_x_range", cfg.lin_vel_x_range)
    self._validate_range("yaw_rate_range", cfg.yaw_rate_range)
    if cfg.transition_active_time <= 0.0:
      raise ValueError("locomotion transition duration must be positive.")
    if not 0.0 <= cfg.direct_switch_probability <= 1.0:
      raise ValueError("direct_switch_probability must be in [0, 1].")

    self.command_buf = torch.zeros(self.num_envs, 5, device=self.device)
    self._scheduled_command = torch.zeros_like(self.command_buf)
    self._next_scheduled_command = torch.zeros_like(self.command_buf)
    self._transition_phase = torch.zeros(
      self.num_envs, dtype=torch.int8, device=self.device
    )
    self._transition_time = torch.zeros(self.num_envs, device=self.device)
    self._mode_probabilities = torch.tensor(
      cfg.mode_probabilities, dtype=torch.float32, device=self.device
    )
    self._mode_idle_probabilities = self._make_mode_idle_probabilities(
      cfg.mode_idle_probabilities
    )

  @staticmethod
  def _validate_range(name: str, value: tuple[float, float]) -> None:
    if value[0] > value[1]:
      raise ValueError(f"{name} must be ordered.")

  @staticmethod
  def _validate_mode_idle_probabilities(value: tuple[float, float, float]) -> None:
    if len(value) != 3 or any(not 0.0 <= probability <= 1.0 for probability in value):
      raise ValueError("mode_idle_probabilities must contain three values in [0, 1].")

  def _make_mode_idle_probabilities(
    self, value: tuple[float, float, float] | None
  ) -> torch.Tensor:
    if value is None:
      value = (self.cfg.idle_probability,) * 3
    return torch.tensor(value, dtype=torch.float32, device=self.device)

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
    next_probabilities = self._mode_probabilities.expand(count, -1).clone()
    next_probabilities[torch.arange(count, device=self.device), modes] = 0.0
    no_alternative = next_probabilities.sum(dim=1) == 0.0
    next_probabilities[no_alternative] = self._mode_probabilities
    next_modes = torch.multinomial(next_probabilities, 1).squeeze(1)
    switch = torch.rand(count, device=self.device) < self.cfg.direct_switch_probability
    next_modes = torch.where(switch, next_modes, modes)
    sampled = torch.zeros(count, 5, device=self.device)
    sampled[torch.arange(count, device=self.device), modes] = 1.0
    # The fused actor needs normal rolling examples from its first update, but
    # front/rear commands first need to discover legal static two-wheel
    # balance.  Sampling that distinction here changes neither the command
    # layout nor the policy target: it only avoids withholding normal motion
    # until after the stance curriculum has already consumed most training.
    idle = torch.rand(count, device=self.device) < self._mode_idle_probabilities[modes]
    # A plain uniform draw made the actor see x and yaw together almost all
    # the time, while the deadband erased many individual axes.  It could
    # therefore learn a coupled wheel motion without learning either control
    # dimension in isolation.  Keep the external command exactly
    # ``one-hot + x + yaw``, but give PPO an explicit and balanced internal
    # distribution of x-only, yaw-only, and combined requests.
    moving_rows = torch.nonzero(~idle, as_tuple=False).squeeze(1)
    if len(moving_rows) > 0:
      request_type = torch.randint(0, 3, (len(moving_rows),), device=self.device)
      x_active = request_type != 1  # x-only or combined.
      yaw_active = request_type != 0  # yaw-only or combined.
      x_rows = moving_rows[x_active]
      yaw_rows = moving_rows[yaw_active]
      if len(x_rows) > 0:
        sampled[x_rows, 3] = self._sample_active_axis(
          self.cfg.lin_vel_x_range, len(x_rows)
        )
      if len(yaw_rows) > 0:
        sampled[yaw_rows, 4] = self._sample_active_axis(
          self.cfg.yaw_rate_range, len(yaw_rows)
        )
    next_sampled = sampled.clone()
    next_sampled[:, :3] = 0.0
    next_sampled[torch.arange(count, device=self.device), next_modes] = 1.0
    # Early static support discovery sets the probability above to zero.  A
    # later static curriculum stage raises it and trains normal <-> front/rear
    # changes before x/yaw locomotion starts, without adding a hidden target.
    # The reset state is physically normal four-wheel idle, so training can
    # issue its first public one-hot immediately.  This matches a controller
    # that switches directly from normal driving to a front/rear stance.
    self.command_buf[env_ids] = sampled
    self._scheduled_command[env_ids] = sampled
    self._next_scheduled_command[env_ids] = next_sampled
    self._transition_phase[env_ids] = 1
    self._transition_time[env_ids] = 0.0

  def _update_command(self) -> None:
    """Hold one mode, make a direct switch, then hold the new command.

    A public one-hot remains meaningful after a mode change.  Clearing it to
    normal at 4.2 seconds used to teach the policy an unrequested recovery in
    the final third of every rollout, while fixed-command validation and the
    intended controller both hold the selected mode.  The all-zero/default
    state remains available to callers; this sampler simply stops injecting
    it between two active user commands.
    """
    pending = self._transition_phase < 3
    self._transition_time[pending] += self._env.step_dt
    second_start = (
      (self._transition_phase == 1)
      & (self._transition_time >= 0.5 * self.cfg.transition_active_time)
    )
    if torch.any(second_start):
      self.command_buf[second_start] = self._next_scheduled_command[second_start]
      self._transition_phase[second_start] = 2
    finish = (
      (self._transition_phase == 2)
      & (self._transition_time >= self.cfg.transition_active_time)
    )
    if torch.any(finish):
      # Preserve the second one-hot plus its x/yaw request until the next
      # resample.  This makes the generated A -> B segment match a persistent
      # external command rather than appending a normal-only cooldown.
      self._transition_phase[finish] = 3

  def set_curriculum(
    self,
    *,
    mode_probabilities: tuple[float, float, float] | None = None,
    idle_probability: float | None = None,
    mode_idle_probabilities: tuple[float, float, float] | None = None,
    direct_switch_probability: float | None = None,
    lin_vel_x_range: tuple[float, float] | None = None,
    yaw_rate_range: tuple[float, float] | None = None,
    resampling_time_range: tuple[float, float] | None = None,
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
      if mode_idle_probabilities is None:
        self.cfg.mode_idle_probabilities = None
        self._mode_idle_probabilities = self._make_mode_idle_probabilities(None)
    if mode_idle_probabilities is not None:
      self._validate_mode_idle_probabilities(mode_idle_probabilities)
      self.cfg.mode_idle_probabilities = mode_idle_probabilities
      self._mode_idle_probabilities = self._make_mode_idle_probabilities(
        mode_idle_probabilities
      )
    if direct_switch_probability is not None:
      if not 0.0 <= direct_switch_probability <= 1.0:
        raise ValueError("direct_switch_probability must be in [0, 1].")
      self.cfg.direct_switch_probability = direct_switch_probability
    if lin_vel_x_range is not None:
      self._validate_range("lin_vel_x_range", lin_vel_x_range)
      self.cfg.lin_vel_x_range = lin_vel_x_range
    if yaw_rate_range is not None:
      self._validate_range("yaw_rate_range", yaw_rate_range)
      self.cfg.yaw_rate_range = yaw_rate_range
    if resampling_time_range is not None:
      self._validate_range("resampling_time_range", resampling_time_range)
      if resampling_time_range[0] <= 0.0:
        raise ValueError("resampling_time_range must be positive.")
      self.cfg.resampling_time_range = resampling_time_range


@dataclass(kw_only=True)
class StanceLocomotionCommandCfg(CommandTermCfg):
  """Configuration for normal/front/rear x-yaw wheeled locomotion."""

  entity_name: str
  mode_probabilities: tuple[float, float, float] = (0.40, 0.30, 0.30)
  idle_probability: float = 0.20
  # ``None`` preserves the scalar idle probability for callers that do not
  # need a curriculum distinction between normal, front, and rear commands.
  mode_idle_probabilities: tuple[float, float, float] | None = None
  lin_vel_x_range: tuple[float, float] = (-0.4, 0.4)
  yaw_rate_range: tuple[float, float] = (-0.4, 0.4)
  command_deadband: float = 0.05
  transition_active_time: float = 4.2
  direct_switch_probability: float = 1.0

  def build(self, env) -> StanceLocomotionCommand:
    return StanceLocomotionCommand(self, env)


class AerialRotationCommand(CommandTerm):
  """Sample one aerial trick event, with the all-zero vector as idle.

  A nonzero one-hot permits one qualified ballistic interval only.  The first
  wheel contact after that interval closes the attempt; after a short landing
  decision window the public command returns to zero whether the landing
  passed or failed.  This prevents one command from asking the policy to keep
  launching new flips until it eventually gets a lucky landing.
  """

  cfg: AerialRotationCommandCfg

  def __init__(self, cfg: AerialRotationCommandCfg, env):
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
      raise ValueError(
        "target_angle must be positive and max_overrotation non-negative."
      )
    if (
      cfg.landing_linear_velocity_limit <= 0.0
      or cfg.landing_angular_velocity_limit <= 0.0
      or cfg.min_ballistic_time <= 0.0
      or cfg.trigger_idle_time <= 0.0
      or cfg.landing_control_time <= 0.0
    ):
      raise ValueError("aerial event durations and limits must be positive.")

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
    # Accumulate the complete continuous wheel-free interval.  We only commit
    # it to the scored turn after it survives ``min_ballistic_time``; this
    # rejects contact glitches without silently discarding the real initial
    # part of a fast flip.
    self._flight_rotation = torch.zeros(self.num_envs, device=self.device)
    self._current_flight_qualified = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._landing_started = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    # Keep the public one-hot briefly after the first wheel contact.  The
    # policy then still knows which aerial axis it is braking, while the event
    # remains physically closed to a second flight by ``_landing_started``.
    self._landing_control_time = torch.zeros(self.num_envs, device=self.device)
    self._rotation_progress = torch.zeros(self.num_envs, device=self.device)
    self._launch_axis_w = torch.zeros(self.num_envs, 3, device=self.device)
    # A 2π aerial turn is only complete when the whole base frame—not merely
    # its gravity vector—returns to the frame present at command onset.  The
    # initial orientation is private event bookkeeping, just like the world
    # turn axis: it never enters the policy observation.
    self._launch_root_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
    self._new_skill = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self._pending_mode = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    self._pending_trigger = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )
    self._trigger_time = torch.zeros(self.num_envs, device=self.device)

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
    self._flight_rotation[env_ids] = 0.0
    self._current_flight_qualified[env_ids] = False
    self._landing_started[env_ids] = False
    self._landing_control_time[env_ids] = 0.0
    self._rotation_progress[env_ids] = 0.0
    self._launch_axis_w[env_ids] = 0.0
    self._launch_root_quat_w[env_ids] = 0.0
    self._new_skill[env_ids] = False
    self._pending_mode[env_ids] = 0
    self._pending_trigger[env_ids] = False
    self._trigger_time[env_ids] = 0.0
    active = torch.rand(count, device=self.device) > self.cfg.idle_probability
    active_ids = env_ids[active]
    if len(active_ids) == 0:
      return
    modes = torch.multinomial(
      self._mode_probabilities, len(active_ids), replacement=True
    )
    # Populate the public history with a stable one-hot before the first
    # ballistic action.  The policy has no recurrent state beyond its normal
    # 10-frame observation window; without this short default-pose warm-up it
    # sees a reset history full of zeros and collapses to no jump.  This is a
    # command-observation consistency delay, not a pose or trajectory target.
    self._pending_mode[active_ids] = modes
    self._pending_trigger[active_ids] = True

  def _update_command(self) -> None:
    """Finish exactly one aerial attempt, then return to idle.

    CommandManager calls this after rewards.  Once the first wheel contact
    closes flight, retain the same public one-hot for a short landing-control
    window so PPO can use the still-known flip axis to dissipate contact
    angular momentum and bring the other wheels down.  The post-landing
    relaunch termination begins at that first contact, so the retained one-hot
    can never authorize a second aerial attempt.  The command then becomes
    literal idle for the ordinary default-pose settling check.
    """
    pending = self._pending_trigger
    self._trigger_time[pending] += self._env.step_dt
    trigger = pending & (self._trigger_time >= self.cfg.trigger_idle_time)
    if torch.any(trigger):
      self.command_buf[trigger] = 0.0
      self.command_buf[
        trigger, self._pending_mode[trigger]
      ] = 1.0
      self._new_skill[trigger] = True
      self._pending_trigger[trigger] = False

    active = torch.sum(self.command_buf, dim=1) > 0.5
    if not torch.any(active):
      # Preserve the final attempt state until the next resample.  The fixed
      # evaluator uses the outcome bit to distinguish a failed event from a
      # real completed maneuver after the public one-hot has gone idle.
      return

    asset = self._env.scene[self.cfg.entity_name]
    mode = torch.argmax(self.command_buf, dim=1)
    axes_b = torch.tensor(
      self.cfg.axes, dtype=asset.data.root_link_quat_w.dtype, device=self.device
    )[mode]
    self._launch_axis_w[self._new_skill] = quat_apply(
      asset.data.root_link_quat_w[self._new_skill], axes_b[self._new_skill]
    )
    self._launch_root_quat_w[self._new_skill] = asset.data.root_link_quat_w[
      self._new_skill
    ]
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
    # Once the first post-flight contact has happened, this event is closed:
    # no later wheel-free gap may be credited as a second jump.
    flight_step = active & (~self._landing_started) & self.has_grounded & airborne
    self._airborne_time = torch.where(
      flight_step,
      self._airborne_time + self._env.step_dt,
      torch.zeros_like(self._airborne_time),
    )

    axis_rate = torch.sum(asset.data.root_link_ang_vel_w * self._launch_axis_w, dim=1)
    raw_flight_delta = flight_step.to(axis_rate.dtype) * axis_rate * self._env.step_dt
    self._flight_rotation = torch.where(
      flight_step,
      self._flight_rotation + raw_flight_delta,
      torch.zeros_like(self._flight_rotation),
    )
    newly_qualified = (
      flight_step
      & (~self._current_flight_qualified)
      & (self._airborne_time >= self.cfg.min_ballistic_time)
    )
    self._current_flight_qualified = torch.where(
      flight_step,
      self._current_flight_qualified
      | (self._airborne_time >= self.cfg.min_ballistic_time),
      torch.zeros_like(self._current_flight_qualified),
    )
    self.was_airborne |= self._current_flight_qualified
    # A short gap earns nothing.  Once the interval is confirmed ballistic,
    # credit every measured radian from liftoff exactly once.  Later airborne
    # frames then contribute their ordinary per-step rotation.
    signed_delta = torch.where(
      newly_qualified,
      self._flight_rotation,
      torch.where(
        flight_step & self._current_flight_qualified,
        raw_flight_delta,
        torch.zeros_like(axis_rate),
      ),
    )
    self._rotation_progress = torch.clamp_min(
      self._rotation_progress + signed_delta, 0.0
    )

    first_landing = (
      active
      & self.was_airborne
      & (~self._landing_started)
      & torch.any(contacts, dim=1)
    )
    self._landing_started |= first_landing
    self._landing_control_time = torch.where(
      self._landing_started,
      self._landing_control_time + self._env.step_dt,
      torch.zeros_like(self._landing_control_time),
    )
    # A partial first contact may rebound and needs the retained one-hot to
    # complete its contact recovery.  Once all four wheels are simultaneously
    # grounded, however, the aerial event has a physically usable default
    # support: retaining a flip one-hot for the rest of the generic window
    # taught m1000 yaw to launch again from that valid landing.  Hand off to
    # the literal idle controller immediately at this real four-wheel
    # boundary; the time window remains the fallback for partial landings.
    four_wheel_landing = self._landing_started & torch.all(contacts, dim=1)
    finish_landing_control = self._landing_started & active & (
      (self._landing_control_time >= self.cfg.landing_control_time)
      | four_wheel_landing
    )
    self.command_buf[finish_landing_control] = 0.0

  def set_curriculum(
    self,
    *,
    idle_probability: float | None = None,
    mode_probabilities: tuple[float, float, float, float, float] | None = None,
  ) -> None:
    """Update only command sampling without adding an actor input."""
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
  # ``abs(dot(q_land, q_launch))`` is invariant to the quaternion sign.  A
  # 0.985 threshold permits about 20 degrees of residual whole-body
  # rotation, while rejecting the yaw drift that projected gravity cannot
  # observe after a front/back/side flip.
  landing_orientation_dot_min: float = 0.985
  landing_linear_velocity_limit: float = 0.75
  landing_angular_velocity_limit: float = 1.5
  # Four controller steps at the 50-Hz policy rate.  This removes the
  # observed ground-pivot exploit while allowing an ordinary compact jump to
  # accrue its full turn from the first genuine ballistic interval.
  min_ballistic_time: float = 0.08
  target_angle: float = math.tau
  max_overrotation: float = 0.75
  trigger_idle_time: float = 0.5
  # Keep the original one-hot long enough after initial wheel touchdown for
  # the actor to apply a direction-aware contact brake, then expose literal
  # idle for the same default-pose settling criterion as before.
  landing_control_time: float = 0.24

  def build(self, env) -> AerialRotationCommand:
    return AerialRotationCommand(self, env)
