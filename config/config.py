from dataclasses import dataclass

@dataclass
class ARDAEConfig:
  sigma_min: float | None = 0.001
  sigma_max: float | None = 0.5
  use_log_scale: bool | None = True
  
  input_dim: int =2
  h_dim: int =1000
  noise_param: float = 0.1
  noise_min: float = 0.001
  noise_max: float = 0.5
  num_hidden_layers: int =1
  nonlinearity: str ='silu' # relu  / elu / tanh / softplus / csoftplus / leaky_relu / silu or switch
  noise_type='gaussian' # gaussian / poisson / gamma
  use_metric = False