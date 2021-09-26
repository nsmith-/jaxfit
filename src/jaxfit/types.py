from typing import Callable, Set, Tuple, Union

import jaxlib.xla_extension
from jax.interpreters.partial_eval import DynamicJaxprTracer

Array = jaxlib.xla_extension.DeviceArray
TracerOrArray = Union[DynamicJaxprTracer, Array]

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
