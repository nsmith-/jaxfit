from typing import Callable, Set, Tuple

from jax.interpreters.partial_eval import DynamicJaxprTracer

JaxPairTuple = Set[Tuple[DynamicJaxprTracer, DynamicJaxprTracer]]

DynamicJaxFunction = Callable[[DynamicJaxprTracer], DynamicJaxprTracer]

NewtonStateTuple = Tuple[
    DynamicJaxprTracer,
    DynamicJaxprTracer,
    DynamicJaxprTracer,
    DynamicJaxprTracer,
    DynamicJaxprTracer,
]

MigradStateTuple = Tuple[
    DynamicJaxprTracer,
    DynamicJaxprTracer,
    DynamicJaxprTracer,
    DynamicJaxprTracer,
]

SolverFunction = Callable[
    [
        DynamicJaxFunction,
        DynamicJaxprTracer,
        DynamicJaxprTracer,
    ],
    DynamicJaxprTracer,
]
