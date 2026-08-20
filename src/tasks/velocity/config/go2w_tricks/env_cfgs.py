"""Compatibility imports for the compact Go2W trick configuration modules.

The active environments live in small task-specific files so that reward and
termination choices are inspectable without paging through historical trials.
"""

from .aerial_env_cfg import unitree_go2w_aerial_rotation_flat_env_cfg
from .ground_env_cfg import (
  unitree_go2w_spin_stance_flat_env_cfg,
  unitree_go2w_stance_locomotion_flat_env_cfg,
)

__all__ = (
  "unitree_go2w_aerial_rotation_flat_env_cfg",
  "unitree_go2w_spin_stance_flat_env_cfg",
  "unitree_go2w_stance_locomotion_flat_env_cfg",
)
