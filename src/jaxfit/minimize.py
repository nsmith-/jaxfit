"""JIT-compilable second-order minimizers

See also https://github.com/google/jaxopt
TODO implicit differentiation using jaxopt.implicit_diff.custom_fixed_point
TODO test suite with https://en.wikipedia.org/wiki/Test_functions_for_optimization ?
"""
import jax
import jax.numpy as jnp


def hvp(f, x, v):
    """Hessian-vector product function"""
    return jax.grad(lambda y: jnp.vdot(jax.grad(f)(y), v))(x)


def newton_mfree(f, x, g):
    """Compute the Newton direction using a matrix-free algorithm

    Any matrix-free linear solver could be substituted. cg is
    supposed to be the fastest but only applies to symmetric, positive-definite systems

    A popular improvement is to truncate the cg algorithm.
    Another option is to regularize using hvp(f, x, y) + lambda * y
    per http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf
    or more generally https://en.wikipedia.org/wiki/Trust_region
    """
    y, status = jax.scipy.sparse.linalg.cg(lambda y: hvp(f, x, y), -g)
    # TODO: status not yet implemented, but if it were we should check it for issues
    return y


def newton_hessinv(f, x, g):
    """Find newton direciton by directly inverting hessian

    This is the most expensive option
    """
    return jnp.linalg.inv(jax.hessian(f)(x)) @ -g


def newtons_method(f, x0, edm_goal=1e-3, maxiter=1000, quad_solver=newton_mfree):
    """Basic Newton's method

    https://en.wikipedia.org/wiki/Newton%27s_method_in_optimization
    Terminated using the same estimated distance to minimum critera as migrad
    """

    def cond(state):
        niter, _, _, _, edm = state
        return (edm >= edm_goal) & (niter < maxiter)

    def step(state):
        niter, x, g, cg, edm = state
        x_next = x + cg
        g_next = jax.grad(f)(x_next)
        cg_next = quad_solver(f, x_next, g_next)
        edm_next = -g_next @ cg_next / 2.0
        return niter + 1, x_next, g_next, cg_next, edm_next

    g0 = jax.grad(f)(x0)
    cg0 = quad_solver(f, x0, g0)
    state = (0, x0, g0, cg0, -g0 @ cg0 / 2.0)
    state = jax.lax.while_loop(cond, step, state)
    niter, x, g, cg, edm = state
    return x


def migrad(f, x0, edm_goal=1e-3, maxiter=1000, debug=0):
    """Something similar to MINUIT migrad algorithm

    Uses the DFP algorithm [1] to update an approximate covariance matrix
    each step, and then applies a line search along the Newton direction.

    [1] https://en.wikipedia.org/wiki/Davidon%E2%80%93Fletcher%E2%80%93Powell_formula
    """

    def edm(state):
        _, x, g, cov = state
        return jnp.einsum("i,ij,j->", g, cov, g) / 2.0

    def cond(state):
        niter = state[0]
        return (edm(state) >= edm_goal) & (niter < maxiter)

    def dfp_update(cov, y, s):
        ycovy = jnp.einsum("i,ij,j->", y, cov, y)
        ys = y @ s
        if debug >= 2:
            print(f"DFP update {y=} {s=}")
            # https://root.cern.ch/doc/master/DavidonErrorUpdator_8cxx_source.html#l00025
            print(f"gvg={ycovy}")
            print(f"delgam={ys}")
        #         if (delgam > gvg) {
        #           // use rank 1 formula
        #           vUpd += gvg * Outer_product(MnAlgebraicVector(dx / delgam - vg / gvg));
        #         }
        return (
            cov
            - jnp.einsum("ij,j,k,kl->il", cov, y, y, cov)
            / jnp.einsum("i,ij,j->", y, cov, y)
            + jnp.einsum("i,j->ij", s, s) / (y @ s)
        )

    def step(state):
        # https://root.cern.ch/doc/master/VariableMetricBuilder_8cxx_source.html#l00242
        niter, x, g, cov = state
        step = -(cov @ g)
        # in real migrad this would be a line search
        # https://root.cern.ch/doc/master/MnLineSearch_8cxx_source.html#l00046
        scale = (-g @ step) / (step @ hvp(f, x, step))
        if debug >= 2:
            print(f"Scaling {step=} by {scale=}")
        step = step * scale
        x_next = x + step
        g_next = jax.grad(f)(x_next)
        y = g_next - g
        cov_next = dfp_update(cov, y, step)
        return niter + 1, x_next, g_next, cov_next

    # https://root.cern.ch/doc/master/MnSeedGenerator_8cxx_source.html#l00070
    # sadly in autodiff the diagonal of the hessian is just as expensive as the whole thing
    # https://github.com/google/jax/issues/3801
    cov_init = jnp.diag(1.0 / jnp.diag(jax.hessian(f)(x0)))
    state = (0, x0, jax.grad(f)(x0), cov_init)
    if debug:
        print(f"f={f(state[1])} edm={edm(state)}")
        if debug >= 2:
            print("{state=}")
        while cond(state):
            state = step(state)
            print(f"f={f(state[1])} edm={edm(state)}")
            if debug >= 2:
                print("{state=}")
    else:
        state = jax.lax.while_loop(cond, step, state)
    niter, x, g, cov = state
    return x
