"""Register independent Go2W support, locomotion, and aerial environments."""

from mjlab.tasks.registry import register_mjlab_task

from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .aerial_env_cfg import unitree_go2w_aerial_rotation_flat_env_cfg
from .env_cfgs import (
  unitree_go2w_spin_stance_flat_env_cfg,
  unitree_go2w_stance_locomotion_flat_env_cfg,
)
from .rl_cfg import (
  unitree_go2w_aerial_rotation_ppo_runner_cfg,
  unitree_go2w_spin_stance_ppo_runner_cfg,
  unitree_go2w_stance_locomotion_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Spin-Stance-Flat",
  env_cfg=unitree_go2w_spin_stance_flat_env_cfg(),
  play_env_cfg=unitree_go2w_spin_stance_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_spin_stance_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Stance-Locomotion-Flat",
  env_cfg=unitree_go2w_stance_locomotion_flat_env_cfg(),
  play_env_cfg=unitree_go2w_stance_locomotion_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_stance_locomotion_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-Go2W-Aerial-Rotation-Flat",
  env_cfg=unitree_go2w_aerial_rotation_flat_env_cfg(),
  play_env_cfg=unitree_go2w_aerial_rotation_flat_env_cfg(play=True),
  rl_cfg=unitree_go2w_aerial_rotation_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
