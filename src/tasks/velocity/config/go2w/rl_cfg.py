"""RL configuration for the Go2W upright-walking task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def unitree_go2w_upright_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO configuration for Go2W upright walking."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      # Match the reference Go2W handstand policy configuration.
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        # The reference task uses unit-variance initial exploration.  With a
        # 0.25-rad direct action scale, this is required to discover a rise
        # from the four-wheel reset; it has no runtime effect on the deployed
        # deterministic policy and is not an action filter.
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      # Keep the rollout/return horizon aligned with the proven Go2 upright
      # configuration. Smoothness remains enforced by direct task costs, not
      # by a temporal action filter or a long return horizon.
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="go2w_upright_velocity",
    # Direct, deployment-identical action saturation.  The policy never sees
    # a temporally filtered target: each output is simply bounded to the
    # reference-compatible residual range before the per-joint affine scale is
    # applied.  This is a direct, stateless saturation at deployment too.
    clip_actions=10.0,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10000,
  )
