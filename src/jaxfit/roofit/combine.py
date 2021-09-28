"""Custom RooFit objects found in CMS combine
"""
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Set

import jax.numpy as jnp
import jax.scipy.special as special
import jax.scipy.stats as stats

from jaxfit.roofit.common import (
    RooCategory,
    RooConstVar,
    RooGaussian,
    RooProduct,
    RooRealVar,
)
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace
from jaxfit.types import Array, Distribution, Function


def _fasthisto2array(h):
    return jnp.array([h[i] for i in range(h.size())])


@RooWorkspace.register
@dataclass
class RooSimultaneousOpt(Model, Distribution):
    """This is a product of pdfs"""

    indexCat: RooCategory
    pdfs: Dict[str, Model]

    @classmethod
    def readobj(cls, obj, recursor):
        cat = obj.indexCat()
        return cls(
            indexCat=recursor(cat),
            pdfs={label: recursor(obj.getPdf(label)) for label, idx in cat.states()},
            # obj.extraConstraints()
            # obj.channelMasks()
        )

    @property
    def observables(self):
        out = reduce(set.union, (pdf.observables for pdf in self.pdfs.items()), set())
        if out & {self.indexCat.name}:
            raise RuntimeError("gotta think")
        return out | {self.indexCat.name}

    @property
    def parameters(self) -> Set[str]:
        return reduce(set.union, (pdf.parameters for pdf in self.pdfs.items()), set())

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in pdf {self.name}")
        missing = {self.observable.name} - observables
        if missing:
            raise RuntimeError(f"Missing observables: {missing} in pdf {self.name}")


def _asym_interpolation(θ, δp, δm):
    """A function that is C^2 continuous in theta

    delta_p, delta_m should both be signed quantities
    """
    morph = jnp.where(
        θ > 1,
        δp * θ,
        jnp.where(
            θ < -1,
            δm * θ,
            0.5
            * (
                (δp + δm) * θ + (δp - δm) * (3 * θ ** 6 - 10 * θ ** 4 + 15 * θ ** 2) / 8
            ),
        ),
    )
    return morph


@RooWorkspace.register
@dataclass
class ProcessNormalization(Model, Function):
    nominal: float
    symParams: List[Model]  # TODO: abstract parameter?
    symLogKappa: Array  # 1d
    asymParams: List[Model]
    asymLogKappa: Array  # (param, lo/hi)
    additional: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(
            nominal=obj.getNominalValue(),
            symParams=[recursor(p) for p in obj.getSymErrorParameters()],
            symLogKappa=jnp.array(list(obj.getSymLogKappa())),
            asymParams=[recursor(p) for p in obj.getAsymErrorParameters()],
            asymLogKappa=jnp.array([[lo, hi] for lo, hi in obj.getAsymLogKappa()]),
            additional=[recursor(p) for p in obj.getAdditionalModifiers()],
        )

    @property
    def parameters(self):
        return (
            set(p.name for p in self.symParams)
            | set(p.name for p in self.asymParams)
            | set(p.name for p in self.additional)
        )

    def value(self, parameters):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in function {self.name}")

        def val(param):
            symTheta = jnp.array([param[p.name] for p in self.symParams])
            symShift = jnp.sum(self.symLogKappa * symTheta, axis=-1)
            asymTheta = jnp.array([param[p.name] for p in self.asymParams])
            asymShift = jnp.sum(
                _asym_interpolation(
                    asymTheta, self.asymLogKappa[:, 1], self.asymLogKappa[:, 0]
                ),
                axis=-1,
            )
            addFactor = jnp.prod([param[p.name] for p in self.additional])
            return self.nominal * jnp.exp(symShift + asymShift) * addFactor

        return val


def _bbparse(param):
    """Guess based on structure if scaled poisson or gaussian"""
    if isinstance(param, RooProduct) and len(param.components) == 2:
        realparam, scale = param.components
        if not isinstance(scale, RooConstVar):
            raise RuntimeError("unexpected type while parsing barlow beeston")
        return realparam, scale.val
    elif isinstance(param, RooRealVar):
        return param, 0.0
    raise RuntimeError("unexpected type while parsing barlow beeston")


@RooWorkspace.register
@dataclass
class CMSHistErrorPropagator(Model, Distribution):
    # FIXME: subclass RooRealSumPdf?
    x: Model
    functions: List[Model]
    coefficients: List[Model]
    bbpars: List[Model]
    bbscale: Array

    @classmethod
    def readobj(cls, obj, recursor):
        bbpars = []
        bbscale = []
        for p in obj.binparsList():
            realp, scale = _bbparse(recursor(p))
            bbpars.append(realp)
            bbscale.append(scale)
        out = cls(
            x=recursor(obj.getX()),
            functions=[recursor(f) for f in obj.funcList()],
            coefficients=[recursor(c) for c in obj.coefList()],
            bbpars=bbpars,
            bbscale=jnp.array(bbscale),
        )
        assert all(isinstance(x, CMSHistFunc) for x in out.functions)
        assert all(isinstance(x, ProcessNormalization) for x in out.coefficients)
        assert len(out.bbpars) == out.x.binning.n
        return out

    @property
    def observables(self):
        return {self.x.name}

    @property
    def parameters(self):
        fpars = reduce(set.union, (x.parameters for x in self.functions), set())
        cpars = reduce(set.union, (x.parameters for x in self.coefficients), set())
        bpars = set(p.name for p in self.bbpars)
        return fpars | cpars | bpars

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in pdf {self.name}")
        missing = {self.observable.name} - observables
        if missing:
            raise RuntimeError(f"Missing observables: {missing} in pdf {self.name}")
        bberrors = jnp.sqrt(sum(f.bberrors ** 2 for f in self.functions))

        def logp(data, param):
            if len(data["weight"]) != len(self.bbpars):
                # FIXME: would be better to make a binned data indicator?
                # in principle we can do a binned fit to unbinned data via jnp.searchsorted, jnp.bincount
                raise RuntimeError("not correctly binned data?")
            observed_total = jnp.sum(data["weight"], axis=-1)
            observed_diff = data["weight"] / observed_total
            # FIXME: can f be an integral of a unbinned pdf?
            expected_extended = sum(
                f(parameters) * c(parameters)
                for f, c in zip(self.functions, self.coefficients)
            )
            # here we can do analytic Balow-Beeston in principle
            bbparams = jnp.array([param[p.name] for p in self.bbparams])
            expected_extended = expected_extended * jnp.where(
                self.bbscale > 0.0, bbparams, 1.0
            ) + jnp.where(self.bbscale == 0.0, bberrors * bbparams, 0.0)
            expected_total = jnp.sum(expected_extended)
            expected_diff = expected_extended / expected_total
            # Poisson normalization term
            nterm = stats.poisson.logpmf(observed_total, expected_total)
            # Multinomial distribution term
            dterm = special.gammaln(observed_total + 1) + jnp.sum(
                special.xlogy(observed_diff, expected_diff)
                - special.gammaln(observed_diff + 1),
                axis=-1,
            )
            return nterm + dterm

        return logp


@RooWorkspace.register
@dataclass
class CMSHistFunc(Model, Function):
    x: Model
    verticalParams: List[Model]
    verticalMorphsLo: Array  # 2d: (bin, param)
    verticalMorphsHi: Array  # 2d: (bin, param)
    bberrors: Array  # 1d
    nominal: Array  # 1d

    @classmethod
    def readobj(cls, obj, recursor):
        if len(obj.getHorizontalMorphs()):
            raise NotImplementedError("horizontal morphs from CMSHistFunc")
        morphs = [
            {
                "param": recursor(p),
                "lo": _fasthisto2array(obj.getShape(0, 0, i + 1, 0)),
                "hi": _fasthisto2array(obj.getShape(0, 0, i + 1, 1)),
            }
            for i, p in enumerate(obj.getVerticalMorphs())
        ]
        return cls(
            x=recursor(obj.getXVar()),
            verticalParams=[m["param"] for m in morphs],
            verticalMorphsLo=jnp.array([m["lo"] for m in morphs]).T,
            verticalMorphsHi=jnp.array([m["hi"] for m in morphs]).T,
            bberrors=_fasthisto2array(obj.errors()),
            nominal=_fasthisto2array(obj.getShape(0, 0, 0, 0)),
        )

    @property
    def parameters(self):
        return (
            set(p.name for p in self.symParams)
            | set(p.name for p in self.asymParams)
            | set(p.name for p in self.additional)
        )

    def value(self, parameters):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in function {self.name}")

        def val(param):
            vertp = jnp.array([param[p.name] for p in self.verticalParams])
            # TODO: log kappa. vs delta
            return self.nominal + jnp.sum(
                _asym_interpolation(
                    vertp, self.verticalMorphsLo, self.verticalMorphsHi
                ),
                axis=-1,
            )

        return val


@RooWorkspace.register
@dataclass
class SimpleGaussianConstraint(RooGaussian):
    # combine implements a fast logpdf for this, hence the specializtion
    pass
