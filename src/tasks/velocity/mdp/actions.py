"""Action terms specific to the velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
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
