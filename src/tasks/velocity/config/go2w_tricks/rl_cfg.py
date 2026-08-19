"""PPO settings for the compact Go2W trick tasks."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


def _trick_runner_cfg(
  experiment_name: str,
  *,
  num_steps_per_env: int,
  gamma: float,
  hidden_dims: tuple[int, ...] = (512, 256, 128),
  lam: float = 0.95,
  init_std: float = 1.0,
  entropy_coef: float = 0.01,
  learning_rate: float = 1.0e-3,
  desired_kl: float | None = 0.01,
  schedule: str = "adaptive",
  num_learning_epochs: int = 5,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=hidden_dims,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": init_std,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=hidden_dims, activation="elu", obs_normalization=True
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=entropy_coef,
      num_learning_epochs=num_learning_epochs,
      num_mini_batches=4,
      learning_rate=learning_rate,
      schedule=schedule,
      gamma=gamma,
      lam=lam,
      desired_kl=desired_kl,
      max_grad_norm=1.0,
    ),
    experiment_name=experiment_name,
    save_interval=100,
    num_steps_per_env=num_steps_per_env,
    max_iterations=10000,
  )


def unitree_go2w_spin_stance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _trick_runner_cfg(
    "go2w_spin_stance", num_steps_per_env=64, gamma=0.995
  )


def unitree_go2w_stance_locomotion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _trick_runner_cfg(
    # Keep frequent policy updates: PPO bootstraps values at rollout boundaries,
    # so a 96-step rollout did not expose a missing terminal signal.  A modestly
    # longer GAE trace retains more short recovery credit without throttling
    # iteration throughput fourfold.
    "go2w_stance_locomotion",
    num_steps_per_env=24,
    gamma=0.997,
    lam=0.98,
    # The wider capacity ablation did not improve signed command following and
    # learned the rear transition more slowly.  Retain the compact shared MLP
    # so this task spends its samples on the reward/command problem instead.
    hidden_dims=(512, 256, 128),
    # This leaves leg residuals small (their action scales are 0.125/0.25 rad)
    # while allowing the separately scaled velocity wheels to explore enough
    # balancing torque to prevent immediate collapse.
    init_std=0.5,
    # Preserve the deliberately broad initial Gaussian exploration (std=0.5),
    # but do not keep injecting entropy after all three stances have been
    # discovered.  The previous 0.002 coefficient let the scalar action std
    # grow late in training and produced checkpoint-to-checkpoint collapse of
    # a previously legal front/rear support.  A small residual coefficient is
    # enough for the x/yaw-conditioned policy without destabilising it.
    entropy_coef=0.0005,
  )


def unitree_go2w_aerial_rotation_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _trick_runner_cfg(
    "go2w_aerial_rotation", num_steps_per_env=96, gamma=0.997
  )
