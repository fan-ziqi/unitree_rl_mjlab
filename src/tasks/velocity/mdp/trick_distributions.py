"""Small PPO exploration distribution used by the fused Go2W spin policy."""

from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import HeteroscedasticGaussianDistribution
from torch.distributions import Normal


class BoundedHeteroscedasticGaussianDistribution(
  HeteroscedasticGaussianDistribution
):
  """Command-conditioned Gaussian exploration with a finite safe range.

  Normal/front/rear spin commands require different discovery actions.  A
  single learnable global standard deviation either collapses after normal
  succeeds or stays too noisy for a static upright support.  RSL's supplied
  heteroscedastic variant conditions its scale on the existing observation,
  but leaves the log scale unbounded.  This tiny wrapper keeps that useful
  conditioning while constraining sampled standard deviation to the explicit
  safe interval.  It has no effect on deterministic inference means.
  """

  def __init__(
    self,
    output_dim: int,
    init_std: float = 0.60,
    std_type: str = "log",
    min_std: float = 0.20,
    max_std: float = 0.90,
  ) -> None:
    if std_type != "log":
      raise ValueError("bounded heteroscedastic exploration requires log std.")
    if min_std <= 0.0 or max_std < min_std:
      raise ValueError("standard-deviation bounds must be positive and ordered.")
    if not min_std <= init_std <= max_std:
      raise ValueError("init_std must lie inside the configured bounds.")
    super().__init__(output_dim, init_std=init_std, std_type=std_type)
    self._min_log_std = math.log(min_std)
    self._max_log_std = math.log(max_std)

  def update(self, mlp_output: torch.Tensor) -> None:
    mean, log_std = torch.unbind(mlp_output, dim=-2)
    std = torch.exp(torch.clamp(log_std, self._min_log_std, self._max_log_std))
    self._distribution = Normal(mean, std)
