"""Register the Unitree Go2W upright walking task."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import unitree_go2w_upright_flat_env_cfg
from .rl_cfg import unitree_go2w_upright_ppo_runner_cfg


register_mjlab_task(
  task_id="Unitree-Go2W-Upright-Flat",
  env_cfg=unitree_go2w_upright_flat_env_cfg(),
  play_env_cfg=unitree_go2w_upright_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_upright_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
