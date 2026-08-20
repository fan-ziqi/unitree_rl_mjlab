"""Action terms specific to the velocity tasks."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
  JointEffortAction,
  JointEffortActionCfg,
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
class JointImpedanceEffortActionCfg(JointEffortActionCfg):
  """Torque residual action around the robot's reset joint posture.

  The physical actuator continues to provide a deliberately low-gain
  impedance to the normal wheel-standing configuration.  Policy actions are
  then direct feed-forward joint torques, which lets a maneuver use brief,
  compliant pulses instead of moving a high-stiffness position target and
  holding each leg rigidly at that target.  It is not a posture controller or
  trajectory source: the only position target is the static reset posture.
  """

  hold_default_position: bool = True
  soft_limit: float | None = None
  """Symmetric joint deviation where the passive safety stop begins."""

  hard_limit: float | None = None
  """Symmetric joint deviation where the passive safety stop reaches full force."""

  limit_stiffness: float = 0.0
  """Maximum passive restoring torque at ``hard_limit``."""

  limit_damping: float = 0.0
  """Velocity damping enabled progressively inside the soft stop."""

  def build(self, env: ManagerBasedRlEnv) -> JointImpedanceEffortAction:
    return JointImpedanceEffortAction(self, env)


class JointImpedanceEffortAction(JointEffortAction):
  """Apply direct torque while retaining a soft neutral impedance."""

  cfg: JointImpedanceEffortActionCfg

  def __init__(
    self, cfg: JointImpedanceEffortActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    if (cfg.soft_limit is None) != (cfg.hard_limit is None):
      raise ValueError("soft_limit and hard_limit must be configured together.")
    if cfg.soft_limit is not None and cfg.hard_limit <= cfg.soft_limit:
      raise ValueError("hard_limit must be greater than soft_limit.")
    super().__init__(cfg, env)
    self._default_position_target = self._entity.data.default_joint_pos[
      :, self._target_ids
    ].clone()

  def apply_actions(self) -> None:
    if self.cfg.hold_default_position:
      # Match ``JointPositionAction``'s encoder-bias convention.  Keeping
      # this neutral target current on every physics substep avoids a reset
      # transient while leaving all maneuver-specific motion to the policy's
      # torque residual.
      encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
      self._entity.set_joint_position_target(
        self._default_position_target - encoder_bias, joint_ids=self._target_ids
      )
    effort = self._processed_actions
    if self.cfg.soft_limit is not None:
      assert self.cfg.hard_limit is not None
      # This is a passive joint-space safety stop, not a motion reference:
      # the policy has unconstrained torque authority in the central compact
      # range and encounters a progressively stronger restoring force only
      # near the same measured excursion terminal used by the task.  It keeps
      # random initial torque samples from repeatedly crashing through that
      # bound while retaining compliant, visibly mobile legs in normal use.
      deviation = self._entity.data.joint_pos[:, self._target_ids] - self._default_position_target
      normalized_excess = torch.clamp(
        (torch.abs(deviation) - self.cfg.soft_limit)
        / (self.cfg.hard_limit - self.cfg.soft_limit),
        min=0.0,
        max=1.0,
      )
      safety_effort = -torch.sign(deviation) * self.cfg.limit_stiffness * torch.square(
        normalized_excess
      )
      safety_effort -= (
        self.cfg.limit_damping
        * normalized_excess
        * self._entity.data.joint_vel[:, self._target_ids]
      )
      effort = effort + safety_effort
    self._entity.set_joint_effort_target(effort, joint_ids=self._target_ids)
