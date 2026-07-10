from models.ardae.ardae_base import ARDAE
from models.ardae.ardae_gamma import GammaARDAE
from models.ardae.ardae_gaussian import GaussianARDAE
from models.ardae.ardae_poisson import PoissonARDAE


ARDAE._distribution_classes = {
    "gaussian": GaussianARDAE,
    "poisson": PoissonARDAE,
    "gamma": GammaARDAE,
}


__all__ = [
    "ARDAE",
    "GaussianARDAE",
    "PoissonARDAE",
    "GammaARDAE",
]
