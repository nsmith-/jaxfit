from typing import Callable, Set, Tuple, Union

from jax.interpreters.partial_eval import DynamicJaxprTracer

TracerOrArray = Union[DynamicJaxprTracer]

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
