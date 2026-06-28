from dataclasses import dataclass

@dataclass
class ARDAEConfig:
  sigma_min: float | None = 0.001
  sigma_max: float | None = 0.5
  use_log_scale: bool | None = True