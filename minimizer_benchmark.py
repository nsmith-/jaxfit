from jaxfit.minimize import newtons_method, newton_hessinv, migrad
import jax
import jax.flatten_util
import jax.scipy.optimize
import jax.numpy as jnp
import scipy.stats
import time
import pandas as pd
from functools import partial


def random_quadratic():
    # no Wishart in jax.scipy.stats :(
    M = jnp.array(scipy.stats.wishart.rvs(df=4, scale=jnp.eye(4)))

    def fun(x):
        return x.T @ M @ x

    x0 = jnp.array([0.0, 1.0, 2.0, 3.0])
    return fun, x0


def random_multiprocess(bins=1000, asimov=False):
    # TODO: set seed
    processes = 5
    normuncs = 5
    nuisval = jnp.array(
        scipy.stats.norm.rvs(loc=1, scale=0.2, size=(normuncs, processes))
    )
    templates = jnp.array(scipy.stats.poisson.rvs(10, size=(processes, bins)))
    parameters = (jnp.ones(1), jnp.zeros(normuncs * processes), jnp.ones(bins))
    x0, unravel = jax.flatten_util.ravel_pytree(parameters)
    bbsum = templates.sum(axis=0)

    def expectation(param):
        r, nuis, bb = unravel(param)
        norm = jnp.concatenate([r, jnp.ones(processes - 1)])
        norm = norm * jnp.power(nuisval, nuis.reshape(normuncs, processes)).prod(axis=0)
        return bb * (norm @ templates)

    if asimov:
        counts = jnp.array(expectation(x0))
    else:
        counts = jnp.array(scipy.stats.poisson.rvs(expectation(x0)))

    def fun(param):
        _, nuis, bb = unravel(param)
        nll = -jnp.sum(jax.scipy.stats.poisson.logpmf(counts, expectation(param)))
        nll = nll - jnp.sum(jax.scipy.stats.poisson.logpmf(bbsum, bb * bbsum))
        nll = nll + 0.5 * nuis @ nuis
        return nll

    return fun, x0


def timeit(fun):
    tic = time.monotonic()
    x = fun()
    toc = time.monotonic()
    return toc - tic, x


class iminuitwrap:
    def __init__(self, fun, x0):
        import iminuit

        self.minuit = iminuit.Minuit(
            jax.jit(fun), list(x0), grad=jax.jit(jax.grad(fun))
        )
        self.minuit.strategy = 1
        self.minuit.print_level = 0

    def __call__(self):
        self.minuit.migrad()
        v = jnp.array(self.minuit.values)
        self.minuit.reset()
        return v


class jitwrap:
    def __init__(self, fun, x0):
        self.jfun = jax.jit(fun)
        self.x0 = x0

    def __call__(self):
        return self.jfun(self.x0)


if __name__ == "__main__":

    def run(bins):
        fun, x0 = random_multiprocess(bins)
        minimizers = {
            "mymigrad": jitwrap(partial(migrad, fun), x0),
            "newton_hessinv": jitwrap(
                partial(newtons_method, fun, quad_solver=newton_hessinv), x0
            ),
            "newton_cg": jitwrap(partial(newtons_method, fun), x0),
            "bfgs": jitwrap(
                lambda x0: jax.scipy.optimize.minimize(fun, x0, method="bfgs").x, x0
            ),
            "iminuit": iminuitwrap(fun, x0),
        }

        def bench(minimizer):
            out = []
            for _ in range(5):
                time, xmin = timeit(minimizer)
                out.append({"time": time, "xmin": xmin, "fmin": fun(xmin)})
            out = pd.DataFrame(out)
            out.index.name = "run"
            return out

        bpoints = pd.concat(
            map(bench, minimizers.values()), keys=minimizers.keys(), names=["minimizer"]
        )
        return bpoints

    bins = jnp.geomspace(10, 1000, 5, dtype="int32")
    df = pd.concat(map(run, bins), keys=bins, names=["bins"])
    print(df)
    df.to_pickle("minimizer_benchmark.pkl")
