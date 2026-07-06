"""Generic environment package (pillar: base): the stateless ``FuncEnv`` core + the stateful
``GymVecEnvWrapper`` facade + state/obs types + components."""
from .component import EnvComponent, FieldSpec
from .core import EnvObs, EnvState, FuncEnv, StepOutput, make_observation_space
from .env import GymVecEnvWrapper

__all__ = ["FuncEnv", "GymVecEnvWrapper", "EnvObs", "EnvState", "StepOutput",
           "make_observation_space", "EnvComponent", "FieldSpec"]
