from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from jaxfit.typing import DynamicJaxFunction, DynamicJaxprTracer, JaxPairTuple


def romberg(
    f: DynamicJaxFunction,
    a: DynamicJaxprTracer,
    b: DynamicJaxprTracer,
    steps: int = 5,
    tol: float = 1e-4,
    debug: bool = False,
) -> DynamicJaxprTracer:
    """1D numerical integration using Romberg method

    If steps=None, eagerly evaluate until error estimate
    is below tol, and return the appropriate step count
    rather than the integral.

    https://en.wikipedia.org/wiki/Romberg%27s_method
    """

    def h(n: DynamicJaxprTracer) -> DynamicJaxprTracer:
        return (b - a) * (2 ** -n)

    memo: JaxPairTuple = {}

    def r(n: DynamicJaxprTracer, m: DynamicJaxprTracer) -> DynamicJaxprTracer:
        if (n, m) in memo:
            return memo[(n, m)]
        elif n == 0:
            out = h(1) * (f(a) + f(b))
        elif m == 0:
            pts = a + (2 * jnp.arange(2 ** (n - 1)) + 1) * h(n)
            out = 0.5 * r(n - 1, 0) + h(n) * jnp.sum(f(pts))
        else:
            out = (4 ** m * r(n, m - 1) - r(n - 1, m - 1)) / (4 ** m - 1)
        if debug:
            print(f"Romberg n={n} m={m} out={out}")  # noqa
        memo[(n, m)] = out
        return out

    if steps is None:
        for i in range(1, 10):
            if abs(r(i, i) - r(i, i - 1)) < tol:
                return i
    if isinstance(steps, tuple):
        return r(*steps)
    return r(steps, steps)


default_integrator = partial(romberg, steps=3)


def piecewise(
    f: DynamicJaxFunction,
    edges: DynamicJaxprTracer,
    integrator: Any = default_integrator,
) -> DynamicJaxprTracer:
    """Compute piecewise integral

    Returns an array of size len(edges)-1 corresponding to the
    integral of f between each respective edge.
    """

    def F(a: DynamicJaxprTracer, b: DynamicJaxprTracer) -> DynamicJaxprTracer:
        return integrator(f, a, b)

    return jax.vmap(F)(edges[:-1], edges[1:])
