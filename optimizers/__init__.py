#from optimizer import Optimizer
from .bayesbybackprop import BayesByBackprop
from .bayesbybackprop import BayesByBackprop as BBB

from .noisyadam import NoisyAdam
from .noisyadam import NoisyAdam as NA

from .blrvi import VariationalOnlineGuassNewton
from .blrvi import VariationalOnlineGuassNewton as VOGN

from .sgd import StochasticGradientDescent
from .sgd import StochasticGradientDescent as SGD

from .swag import StochasticWeightAveragingGaussian
from .swag import StochasticWeightAveragingGaussian as SWAG

from .hmc import HamiltonianMonteCarlo
from .hmc import HamiltonianMonteCarlo as HMC

from .adam import Adam
