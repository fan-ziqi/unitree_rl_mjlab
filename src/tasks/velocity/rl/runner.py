import os

import torch
from mjlab.entity import Entity
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.rl.runner import MjlabOnPolicyRunner

import wandb
from src.tasks.velocity.mdp.actions import JointImpedanceEffortAction


def _get_velocity_policy_metadata(env, run_name: str) -> dict[str, list | str | float]:
  """Export standard metadata for position and compliant-effort policies."""
  joint_action = env.action_manager.get_term("joint_pos")
  if not isinstance(joint_action, JointImpedanceEffortAction):
    return get_base_metadata(env, run_name)

  robot: Entity = env.scene["robot"]
  joint_name_to_ctrl_id = {
    actuator.target.split("/")[-1]: actuator.id for actuator in robot.spec.actuators
  }
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[joint_name]
    for joint_name in robot.joint_names
    if joint_name in joint_name_to_ctrl_id
  ]
  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  return {
    "run_path": run_name,
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
    "command_names": list(env.command_manager.active_terms),
    "observation_names": env.observation_manager.active_terms["actor"],
    "action_scale": joint_action._scale[0].cpu().tolist()
    if isinstance(joint_action._scale, torch.Tensor)
    else joint_action._scale,
    "joint_action_type": "soft_impedance_effort_residual",
  }


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    super().save(path, infos)
    policy_path = path.split("model")[0]
    filename = "policy.onnx"
    self.export_policy_to_onnx(policy_path, filename)
    run_name: str = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )  # type: ignore[assignment]
    onnx_path = os.path.join(policy_path, filename)
    metadata = _get_velocity_policy_metadata(self.env.unwrapped, run_name)
    attach_metadata_to_onnx(onnx_path, metadata)
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
