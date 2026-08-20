"""Action terms specific to the velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
  JointPositionAction,
  JointPositionActionCfg,
  JointVelocityAction,
  JointVelocityActionCfg,
)


@dataclass(kw_only=True)
class UprightGatedJointVelocityActionCfg(JointVelocityActionCfg):
  """Rear-wheel velocity action released only after front-wheel lift-off.

  The Go2W reset is deliberately a four-wheel quadruped, so its rear tyres are
  held at zero while either front wheel supports the body.  Once both front
  wheels are clear, the robot is a rear-wheel pendulum and needs rear-motor
  torque to complete and stabilize the rise.  This is an actuator-availability
  constraint based on current contact state, not a get-up reward or trajectory.
  """

  front_wheel_sensor_name: str = ""
  front_release_force_threshold: float = 1.0

  def build(self, env: ManagerBasedRlEnv) -> UprightGatedJointVelocityAction:
    return UprightGatedJointVelocityAction(self, env)


class UprightGatedJointVelocityAction(JointVelocityAction):
  """Apply rear-wheel commands only in a rear-wheel support transition."""

  cfg: UprightGatedJointVelocityActionCfg

  def __init__(
    self, cfg: UprightGatedJointVelocityActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    if not cfg.front_wheel_sensor_name:
      raise ValueError("front_wheel_sensor_name must be specified.")
    if cfg.front_release_force_threshold <= 0.0:
      raise ValueError("front_release_force_threshold must be positive.")
    super().__init__(cfg, env)
    self._front_wheel_sensor = env.scene[cfg.front_wheel_sensor_name]

  def process_actions(self, actions: torch.Tensor) -> None:
    # Use force history when it is available so a single zero-match substep
    # cannot enable a wheel while a front tyre is still supporting the body.
    force_history = self._front_wheel_sensor.data.force_history
    if force_history is not None:
      force_norm = torch.linalg.vector_norm(force_history, dim=-1)
      front_wheel_contact = torch.any(
        force_norm > self.cfg.front_release_force_threshold, dim=(1, 2)
      )
    else:
      force = self._front_wheel_sensor.data.force
      if force is None:
        raise RuntimeError("front-wheel contact sensor does not provide force data.")
      front_wheel_contact = torch.any(
        torch.linalg.vector_norm(force, dim=-1)
        > self.cfg.front_release_force_threshold,
        dim=1,
    )
    rear_support = ~front_wheel_contact
    super().process_actions(actions)
    # This is a hard availability constraint, not an action filter: physical
    # rear-wheel commands are exactly the policy output after front lift-off,
    # and exactly zero while the robot is still on four wheels.
    if isinstance(self._offset, torch.Tensor):
      self._processed_actions[~rear_support] = self._offset[~rear_support]
    else:
      self._processed_actions[~rear_support] = self._offset


@dataclass(kw_only=True)
class DefaultIdleGatedJointPositionActionCfg(JointPositionActionCfg):
  """Hold literal model-default positions for an untriggered trick command.

  A normal stationary one-hot, or an all-zero aerial event command, is a
  public controller state rather than a skill that PPO needs to rediscover.
  The gate is inactive for every triggered command and does not encode a
  two-wheel posture, a takeoff pulse, or a motion reference.
  """

  command_name: str = "trick"
  idle_mode_index: int | None = 0
  stationary_command_start_index: int = 0
  command_deadband: float = 0.05
  idle_contact_sensor_name: str = ""
  idle_gravity_alignment: float = 0.98

  def build(self, env: ManagerBasedRlEnv) -> DefaultIdleGatedJointPositionAction:
    return DefaultIdleGatedJointPositionAction(self, env)


@dataclass(kw_only=True)
class DefaultIdleGatedJointVelocityActionCfg(JointVelocityActionCfg):
  """Hold literal zero wheel velocity for an untriggered trick command."""

  command_name: str = "trick"
  idle_mode_index: int | None = 0
  stationary_command_start_index: int = 0
  command_deadband: float = 0.05
  idle_contact_sensor_name: str = ""
  idle_gravity_alignment: float = 0.98

  def build(self, env: ManagerBasedRlEnv) -> DefaultIdleGatedJointVelocityAction:
    return DefaultIdleGatedJointVelocityAction(self, env)


class _DefaultIdleGate:
  """Lock default control only after an idle request has physically landed."""

  cfg: DefaultIdleGatedJointPositionActionCfg | DefaultIdleGatedJointVelocityActionCfg

  def _configure_default_idle_gate(self) -> None:
    if self.cfg.command_deadband < 0.0:
      raise ValueError("command_deadband must be non-negative.")
    if not 0.0 < self.cfg.idle_gravity_alignment <= 1.0:
      raise ValueError("idle_gravity_alignment must be in (0, 1].")
    self._idle_command = self._env.command_manager.get_term(self.cfg.command_name)
    self._idle_contact_sensor = (
      self._env.scene[self.cfg.idle_contact_sensor_name]
      if self.cfg.idle_contact_sensor_name
      else None
    )

  def _default_idle_mask(self) -> torch.Tensor:
    command = self._idle_command.command
    if self.cfg.idle_mode_index is None:
      # Aerial uses an all-zero event command as idle.  Its one-hot events
      # have norm one, so the same scalar deadband remains unambiguous.
      return torch.linalg.vector_norm(command, dim=1) <= self.cfg.command_deadband
    if not 0 <= self.cfg.idle_mode_index < command.shape[1]:
      raise ValueError("idle_mode_index is outside the command vector.")
    if not 0 <= self.cfg.stationary_command_start_index <= command.shape[1]:
      raise ValueError("stationary_command_start_index is outside the command vector.")
    stationary = torch.linalg.vector_norm(
      command[:, self.cfg.stationary_command_start_index :], dim=1
    ) <= self.cfg.command_deadband
    return (command[:, self.cfg.idle_mode_index] > 0.5) & stationary

  def _physical_idle_mask(self) -> torch.Tensor:
    """Require upright four-wheel support before freezing an idle action.

    A normal zero command is literal default control while already idle.  But
    it is also the requested destination after a one-hot changes from a
    two-wheel stance or an aerial event.  Let PPO use its ordinary actions
    until it has put all wheels down and returned upright; otherwise a command
    gate would make a controlled return physically impossible.
    """
    if self._idle_contact_sensor is None:
      return torch.ones(self._env.num_envs, dtype=torch.bool, device=self._env.device)
    found = self._idle_contact_sensor.data.found
    if found is None:
      raise RuntimeError("idle contact sensor does not provide contact matches.")
    contacts = (found.reshape(self._env.num_envs, found.shape[1], -1) > 0).any(dim=-1)
    four_wheel_support = torch.all(contacts, dim=1)
    gravity = torch.nn.functional.normalize(self._entity.data.projected_gravity_b, dim=1)
    upright = -gravity[:, 2] >= self.cfg.idle_gravity_alignment
    return four_wheel_support & upright

  def _apply_default_idle_target(self) -> None:
    idle = self._default_idle_mask() & self._physical_idle_mask()
    if isinstance(self._offset, torch.Tensor):
      self._processed_actions[idle] = self._offset[idle]
    else:
      self._processed_actions[idle] = self._offset


class DefaultIdleGatedJointPositionAction(
  _DefaultIdleGate, JointPositionAction
):
  """Position residuals are disabled only while the public idle is active."""

  cfg: DefaultIdleGatedJointPositionActionCfg

  def __init__(
    self, cfg: DefaultIdleGatedJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self._configure_default_idle_gate()

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._apply_default_idle_target()


class DefaultIdleGatedJointVelocityAction(
  _DefaultIdleGate, JointVelocityAction
):
  """Wheel residuals are disabled only while the public idle is active."""

  cfg: DefaultIdleGatedJointVelocityActionCfg

  def __init__(
    self, cfg: DefaultIdleGatedJointVelocityActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    self._configure_default_idle_gate()

  def process_actions(self, actions: torch.Tensor) -> None:
    super().process_actions(actions)
    self._apply_default_idle_target()
