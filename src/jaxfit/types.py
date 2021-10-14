from typing import Callable, Dict, Set, Tuple, Union

import jaxlib.xla_extension
from jax.interpreters.partial_eval import DynamicJaxprTracer

Array = jaxlib.xla_extension.DeviceArray
TracerOrArray = Union[DynamicJaxprTracer, Array]
DTree = Dict[str, TracerOrArray]

JaxPairTuple = Set[Tuple[TracerOrArray, TracerOrArray]]

DynamicJaxFunction = Callable[[TracerOrArray], TracerOrArray]

NewtonStateTuple = Tuple[
    TracerOrArray,
    TracerOrArray,
    TracerOrArray,
    TracerOrArray,
    TracerOrArray,
]

MigradStateTuple = Tuple[
    TracerOrArray,
    TracerOrArray,
    TracerOrArray,
    TracerOrArray,
]

SolverFunction = Callable[
    [
        TracerOrArray,
        TracerOrArray,
        TracerOrArray,
    ],
    TracerOrArray,
]


# TODO: abc


class Distribution:
    @property
    def parameters(self) -> Set[str]:
        raise NotImplementedError

    @property
    def observables(self) -> Set[str]:
        raise NotImplementedError

    def log_prob(
        self, observables: Set[str], parameters: Set[str]
    ) -> Callable[[DTree, DTree], TracerOrArray]:
        """Get the statistical model function

        i.e. ln f(x,theta), which can either be interepreted as
        - ln P(x|theta), the probability of x given theta; or
        - ln L(theta|x), the likelihood of theta given observation x
        """
        raise NotImplementedError

    # TODO: def sample(self, rng, observables, parameters, asimov=False)


class Function:
    def parameters(self) -> Set[str]:
        raise NotImplementedError

    def value(self, parameters: Set[str]) -> Callable[[DTree], TracerOrArray]:
        raise NotImplementedError


class Parameter:
    @property
    def name(self):
        raise NotImplementedError

    def value(self, parameters: Set[str]) -> Callable[[DTree], TracerOrArray]:
        raise NotImplementedError
