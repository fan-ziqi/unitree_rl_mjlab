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
  distribution_class: str = "GaussianDistribution",
  std_type: str = "scalar",
  distribution_params: dict[str, float] | None = None,
  entropy_coef: float = 0.01,
  learning_rate: float = 1.0e-3,
  desired_kl: float | None = 0.01,
  schedule: str = "adaptive",
  num_learning_epochs: int = 5,
  clip_actions: float | None = None,
) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=hidden_dims,
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": distribution_class,
        "init_std": init_std,
        "std_type": std_type,
      }
      | (distribution_params or {}),
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
    # This must be set in the runner, rather than only documented next to an
    # action scale: RSL-RL's Gaussian policy otherwise emits unbounded samples.
    # The vector wrapper applies this bound identically in train, evaluator,
    # and video recorder.
    clip_actions=clip_actions,
  )


def unitree_go2w_spin_stance_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _trick_runner_cfg(
    "go2w_spin_stance",
    # A support-pair change needs more than the former short rollout, but this
    # remains far below an episode and keeps updates fast on the flat scene.
    num_steps_per_env=64,
    gamma=0.995,
    # Five visibly distinct contact outcomes share one command-conditioned
    # policy.  Match the aerial policy capacity so the easy normal branch
    # cannot exhaust a narrow latent before front/rear/side supports form.
    hidden_dims=(512, 512, 256),
    # The normal pivot previously collapsed shared Gaussian exploration to
    # 0.11, while RSL's unbounded heteroscedastic form later exploded to 11.
    # Use the same public observation to condition exploration by one-hot,
    # but keep its physical action spread within 0.15--0.65.  No new policy
    # input, posture target, or controller is introduced.
    init_std=0.60,
    distribution_class=(
      "src.tasks.velocity.mdp.trick_distributions:"
      "BoundedHeteroscedasticGaussianDistribution"
    ),
    std_type="log",
    distribution_params={"min_std": 0.15, "max_std": 0.65},
    entropy_coef=0.005,
    clip_actions=1.0,
  )


def unitree_go2w_stance_locomotion_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _trick_runner_cfg(
    "go2w_stance_locomotion",
    num_steps_per_env=64,
    gamma=0.995,
    # Normal/front/rear transitions and their x/yaw responses are a genuinely
    # multimodal mapping from a ten-frame proprioceptive history.  The wider
    # MLP is still inexpensive compared with 8,192 parallel physics worlds.
    hidden_dims=(512, 512, 256),
    # 0.60/0.002 collapsed before a lift was discovered, whereas 1.0/0.006
    # grew past 1.8 and reduced the task to random leg flailing.  Keep the
    # middle bounded exploration level needed to discover a physical support
    # without prescribing posture or trajectory.
    init_std=0.8,
    entropy_coef=0.0035,
    clip_actions=1.0,
  )


def unitree_go2w_aerial_rotation_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return _trick_runner_cfg(
    "go2w_aerial_rotation",
    # A maneuver's measured ballistic interval is about 0.5--1.0 s.  Forty-
    # eight control steps keep a complete landing transition in a rollout
    # while avoiding the 96-step collection stall (with 8,192 environments,
    # a 393k-sample PPO batch remains ample).
    num_steps_per_env=48,
    gamma=0.997,
    # The comparable m300 audit falsified the larger 1024/1024/512 network:
    # with every other setting unchanged it regressed the learned front full
    # turn from 96% to 0%.  Keep the compact shared representation, which
    # learns physical launch coordination sooner; a later capacity change
    # needs evidence that it improves all five modes, not just a hypothesis
    # about mode interference.
    hidden_dims=(512, 512, 256),
    # Position-action scales below are intended as a compact mechanical
    # envelope.  Make +/- one a real bound so exploration cannot turn a
    # nominal 0.55-rad calf residual into a multi-radian joint target.
    # The actuator torque limits remain the physical limit.  A 1.5 residual
    # range merely lets the position servo sustain that limit through the
    # launch, whereas +/- 1.0 repeatedly plateaued at safe 0.2--0.5-turn
    # hops before a full ballistic turn was ever sampled.
    clip_actions=1.5,
    # The m800 1.0/0.002 restart never recovered a qualified takeoff: broad
    # clipped samples prevented the actor from consolidating even the first
    # ballistic launch.  The former 0.60/0.001 setting did reliably discover
    # real flight; keep that physical launch prior while the rebalanced mode
    # sampler and lighter compactness cost supply the missing full-turn route.
    init_std=0.60,
    entropy_coef=0.001,
    # The large batch already produces a low-variance PPO gradient.  A
    # smaller actor step prevents a rare successful mode from overwriting
    # still-exploring command branches between checkpoints.
    learning_rate=5.0e-4,
  )
