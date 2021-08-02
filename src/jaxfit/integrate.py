from functools import partial

import jax
import jax.numpy as jnp


def romberg(f, a, b, steps=5, tol=1e-4, debug=False):
    """1D numerical integration using Romberg method

    If steps=None, eagerly evaluate until error estimate
    is below tol, and return the appropriate step count
    rather than the integral.

    https://en.wikipedia.org/wiki/Romberg%27s_method
    """

    def h(n):
        return (b - a) * (2 ** -n)

    memo = {}

    def r(n, m):
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
            print(f"Romberg {n=} {m=} {out=}")
        memo[(n, m)] = out
        return out

    if steps is None:
        for i in range(1, 10):
            if abs(r(i, i) - r(i, i - 1)) < tol:
                return i
    if isinstance(steps, tuple):
        return r(*steps)
    return r(steps, steps)


def piecewise(f, edges, integrator=partial(romberg, steps=3)):
    """Compute piecewise integral

    Returns an array of size len(edges)-1 corresponding to the
    integral of f between each respective edge.
    """

    def F(a, b):
        return integrator(f, a, b)

    return jax.vmap(F)(edges[:-1], edges[1:])
