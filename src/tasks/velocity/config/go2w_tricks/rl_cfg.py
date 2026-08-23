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
  clip_actions: float | None = None,
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
    init_std=0.60,
    # Front support otherwise becomes deterministic long before rear/side
    # have sampled a legal transition from the common four-wheel reset.  Keep
    # exploration alive across the five one-hots; the environment's generic
    # action-rate cost still rejects high-frequency free-leg flailing.
    entropy_coef=0.010,
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
    init_std=0.60,
    # The former entropy pressure grew the scalar Gaussian standard deviation
    # past 1.7 despite the clipped action envelope.  That continually knocked
    # a discovered rear support out of balance and prevented x/yaw response
    # from consolidating.  Initial exploration is unchanged; this only lets
    # PPO reduce noise once a physical stance is found.
    entropy_coef=0.002,
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
    # Five physically distinct launch/landing mappings share this policy.  A
    # wider middle layer is inexpensive relative to MuJoCo collection at 8192
    # environments and prevents the easy yaw branch from monopolising a small
    # latent representation.
    hidden_dims=(512, 512, 256),
    # This leaves initial exploration inside the physically tested compact
    # position-residual envelope while still exposing coordinated launch
    # pulses in the large parallel batch.
    # V85/V86 both discover full rotations but their scalar Gaussian standard
    # deviation grows past 1.5 late in training, destroying the short
    # all-wheel landing window before the completion event can be reinforced.
    # Start with enough residual exploration to jump, then use a low entropy
    # pressure so PPO can consolidate a discovered landing rather than keep
    # widening every limb action.  This changes no command, reward, contact
    # criterion, or reference trajectory.
    # V95's correct one-shot/safe-landing objective collapsed to std=0.27
    # before any full turn was sampled, yielding only small safe hops.  The
    # action clamp still bounds every joint residual; retain a wider Gaussian
    # long enough for the rare complete ballistic event to be discovered.
    init_std=0.45,
    # Position-action scales below are intended as a compact mechanical
    # envelope.  Make +/- one a real bound so exploration cannot turn a
    # nominal 0.55-rad calf residual into a multi-radian joint target.
    # The actuator torque limits remain the physical limit.  A 1.5 residual
    # range merely lets the position servo sustain that limit through the
    # launch, whereas +/- 1.0 repeatedly plateaued at safe 0.2--0.5-turn
    # hops before a full ballistic turn was ever sampled.
    clip_actions=1.5,
    # V140's outcome reward is physically well behaved but its scalar action
    # standard deviation contracts from 0.45 to about 0.26 before pitch/roll
    # has sampled a full event.  A middle entropy pressure preserves
    # coordinated launch exploration across all five one-hots without
    # returning to the high-variance, permanently noisy landing regime.
    entropy_coef=0.003,
  )
