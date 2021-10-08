"""Custom RooFit objects found in CMS combine
"""
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Set, Union

import jax.numpy as jnp
import jax.scipy.stats as stats

from jaxfit.roofit.common import (
    RooCategory,
    RooConstVar,
    RooGaussian,
    RooPoisson,
    RooProdPdf,
    RooProduct,
)
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace
from jaxfit.types import Array, Distribution, Function


def _fasthisto2array(h):
    return jnp.array([h[i] for i in range(h.size())])


def _factorize(pdf):
    if isinstance(pdf, RooProdPdf):
        for p in pdf.pdfs:
            yield from _factorize(p)
    else:
        yield pdf


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
        out = reduce(set.union, (pdf.observables for pdf in self.pdfs.values()), set())
        if out & {self.indexCat.name}:
            raise RuntimeError("gotta think")
        return out | {self.indexCat.name}

    @property
    def parameters(self) -> Set[str]:
        return reduce(set.union, (pdf.parameters for pdf in self.pdfs.values()), set())

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        """Combine-specific log_prob

        We can do more optimization because we expect a certain structure, based on
        how combine builds the RooFit model.
        """
        generic = {}
        gaus_constr = {}
        pois_constr = {}
        for cat, catp in self.pdfs.items():
            for pdf in _factorize(catp):
                obs = list(pdf.observables)
                if len(obs) != 1:
                    raise RuntimeError(
                        f"Unfactorized pdf: {pdf.name}, observables {obs}"
                    )
                obs = obs[0]
                if isinstance(pdf, RooGaussian):
                    if not obs.endswith("_In"):
                        raise RuntimeError(
                            f"Expected {pdf.name} to have a global observable-like name, instead have {obs}"
                        )
                    if obs in gaus_constr:
                        # check same constraint?
                        continue
                    if not (pdf.mean.name in parameters or pdf.mean.const):
                        raise RuntimeError(
                            f"Constraint depends on non-const missing parameter {pdf.mean.name}"
                        )
                    if not (pdf.sigma.name in parameters or pdf.sigma.const):
                        raise RuntimeError(
                            f"Constraint depends on non-const missing parameter {pdf.sigma.name}"
                        )
                    gaus_constr[obs] = (
                        pdf.mean.val if pdf.mean.const else pdf.mean.name,
                        pdf.sigma.val if pdf.sigma.const else pdf.sigma.name,
                    )
                elif isinstance(pdf, RooPoisson):
                    if not obs.endswith("_In"):
                        raise RuntimeError(
                            f"Expected {pdf.name} to have a global observable-like name, instead have {obs}"
                        )
                    if obs in pois_constr:
                        # check same constraint?
                        continue
                    if not (pdf.mean.name in parameters or pdf.mean.const):
                        raise RuntimeError(
                            f"Constraint depends on non-const missing parameter {pdf.mean.name}"
                        )
                    pois_constr[obs] = pdf.mean.val if pdf.mean.const else pdf.mean.name
                else:
                    # this pdf has observables that may depend on cat
                    # and anyway we delegate construction to it
                    # probably better to fix combine structure in the beginning..
                    if cat in generic:
                        raise RuntimeError("did we over-factorize?")
                    generic[cat] = pdf.log_prob(observables, parameters)

        def logp(data, param):
            gausx = jnp.array([data[p] for p in gaus_constr])
            gausm = jnp.array(
                [
                    p if isinstance(p, float) else param[p]
                    for p, _ in gaus_constr.values()
                ]
            )
            gauss = jnp.array(
                [
                    p if isinstance(p, float) else param[p]
                    for _, p in gaus_constr.values()
                ]
            )
            gaus_prob = stats.norm.logpdf(gausx, loc=gausm, scale=gauss)
            poisx = jnp.array([data[p] for p in pois_constr])
            poism = jnp.array(
                [p if isinstance(p, float) else param[p] for p in pois_constr.values()]
            )
            pois_prob = stats.poisson.logpmf(poisx, mu=poism)
            return (
                sum(generic[cat](data[cat], param) for cat in self.pdfs)
                + jnp.sum(gaus_prob)
                + jnp.sum(pois_prob)
            )

        return logp


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
    asymLogKappaLo: Array  # 1d
    asymLogKappaHi: Array  # 1d
    additional: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        asympar = [[-lo, hi] for lo, hi in obj.getAsymLogKappa()]
        return cls(
            nominal=obj.getNominalValue(),
            symParams=[recursor(p) for p in obj.getSymErrorParameters()],
            symLogKappa=jnp.array(list(obj.getSymLogKappa())),
            asymParams=[recursor(p) for p in obj.getAsymErrorParameters()],
            asymLogKappaLo=jnp.array([p[0] for p in asympar]),
            asymLogKappaHi=jnp.array([p[1] for p in asympar]),
            additional=[recursor(p) for p in obj.getAdditionalModifiers()],
        )

    @property
    def parameters(self):
        return reduce(
            set.union,
            (p.parameters for p in self.symParams + self.asymParams + self.additional),
            set(),
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
                    asymTheta, self.asymLogKappaHi, self.asymLogKappaLo
                ),
                axis=-1,
            )
            addFactor = jnp.prod(jnp.array([param[p.name] for p in self.additional]))
            return self.nominal * jnp.exp(symShift + asymShift) * addFactor

        return val


def _bbparse(obj, functions, recursor):
    bbpars = []
    bbscale = []
    pariter = (recursor(p) for p in obj.binparsList())
    nch = len(functions)
    for bintype in obj.binTypes():
        if len(bintype) == 1 and bintype[0] == 0:
            # No MC stat
            bbpars.append([0.0] * len(functions))
            bbscale.append([-2.0] * len(functions))
        elif len(bintype) == 1 and bintype[0] == 1:
            # BB-lite (single gaussian)
            bbpars.append([next(pariter)] + [0.0] * (nch - 1))
            bbscale.append([-1.0] + [-2.0] * (nch - 1))
        else:
            procpar = []
            procscale = []
            for proc, binproctype in zip(functions, bintype):
                if binproctype == 2:
                    # Full BB, Poisson
                    param = next(pariter)
                    if not isinstance(param, RooProduct):
                        raise RuntimeError(
                            "unexpected type while parsing barlow beeston"
                        )
                    realparam, scale = param.components
                    if not isinstance(scale, RooConstVar):
                        raise RuntimeError(
                            "unexpected type while parsing barlow beeston"
                        )
                    procpar.append(realparam)
                    procscale.append(scale.val)
                elif binproctype == 3:
                    # Full BB, Gaussian
                    procpar.append(next(pariter))
                    procscale.append(0.0)
                else:
                    # This process doesn't contribute
                    procpar.append(0.0)
                    procscale.append(-2.0)
            bbpars.append(procpar)
            bbscale.append(procscale)

    return bbpars, jnp.array(bbscale)


@RooWorkspace.register
@dataclass
class CMSHistErrorPropagator(Model, Distribution):
    # FIXME: subclass RooRealSumPdf?
    x: Model
    functions: List[Model]
    coefficients: List[Model]
    bbpars: List[List[Union[Model, float]]]
    bbscale: Array  # 2d: bin, proc

    @classmethod
    def readobj(cls, obj, recursor):
        functions = [recursor(f) for f in obj.funcList()]
        bbpars, bbscale = _bbparse(obj, functions, recursor)
        out = cls(
            x=recursor(obj.getX()),
            functions=functions,
            coefficients=[recursor(c) for c in obj.coefList()],
            bbpars=bbpars,
            bbscale=bbscale,
        )
        assert all(isinstance(x, CMSHistFunc) for x in out.functions)
        assert all(
            isinstance(x, ProcessNormalization) or x.name == "RooRealVar:ZERO"
            for x in out.coefficients
        )
        assert len(out.bbpars) == out.x.binning.n
        return out

    @property
    def observables(self):
        return {self.x.name}

    @property
    def parameters(self):
        fpars = reduce(set.union, (x.parameters for x in self.functions), set())
        cpars = reduce(set.union, (x.parameters for x in self.coefficients), set())
        bpars = reduce(
            set.union,
            (
                p.parameters
                for procpars in self.bbpars
                for p in procpars
                if not isinstance(p, float)
            ),
            set(),
        )
        return fpars | cpars | bpars

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in pdf {self.name}")
        missing = self.observables - observables
        if missing:
            raise RuntimeError(f"Missing observables: {missing} in pdf {self.name}")

        chvals = [
            (f.value(parameters), c.value(parameters))
            for f, c in zip(self.functions, self.coefficients)
        ]
        bb_errors = jnp.array([f.bberrors for f in self.functions]).T
        bblite_errors = jnp.sqrt(jnp.sum(bb_errors ** 2, axis=1))

        def logp(data, param):
            if len(data["weight"]) != len(self.bbpars):
                # FIXME: would be better to make a binned data indicator?
                # in principle we can do a binned fit to unbinned data via jnp.searchsorted, jnp.bincount
                raise RuntimeError("not correctly binned data?")
            observed = data["weight"]
            expected_perchannel = jnp.array([f(param) * c(param) for f, c in chvals]).T
            # here we can do analytic Balow-Beeston in principle
            bbparams = jnp.array(
                [
                    [p if isinstance(p, float) else param[p.name] for p in procpars]
                    for procpars in self.bbpars
                ]
            )
            expected_perchannel = expected_perchannel + jnp.where(
                self.bbscale > 0.0,
                (bbparams * self.bbscale - 1) * expected_perchannel,
                jnp.where(
                    self.bbscale == 0.0,
                    bb_errors * bbparams,
                    jnp.where(
                        self.bbscale == -1.0, bblite_errors[:, None] * bbparams, 0.0
                    ),
                ),
            )
            expected = jnp.sum(expected_perchannel, axis=1)
            return jnp.sum(stats.poisson.logpmf(observed, expected))
            # Multinomial distribution term
            # dterm = special.gammaln(observed_total + 1) + jnp.sum(
            #     special.xlogy(observed_diff, expected_diff)
            #     - special.gammaln(observed_diff + 1),
            #     axis=-1,
            # )

        return logp


@RooWorkspace.register
@dataclass
class CMSHistFunc(Model, Function):
    x: Model
    verticalParams: List[Model]
    verticalMorphsLo: Array  # 2d: (param, bin)
    verticalMorphsHi: Array  # 2d: (param, bin)
    verticalType: int
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
            verticalMorphsLo=jnp.array([m["lo"] for m in morphs]),
            verticalMorphsHi=jnp.array([m["hi"] for m in morphs]),
            verticalType=obj.getVerticalType(),
            bberrors=_fasthisto2array(obj.errors()),
            nominal=_fasthisto2array(obj.getShape(0, 0, 0, 0)),
        )

    @property
    def parameters(self):
        return reduce(set.union, (p.parameters for p in self.verticalParams), set())

    def value(self, parameters):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in function {self.name}")

        if not len(self.verticalParams):
            return lambda param: self.nominal

        def val(param):
            vertp = jnp.array([param[p.name] for p in self.verticalParams])
            if self.verticalType == 0:
                # QuadLinear
                vshift = _asym_interpolation(
                    vertp,
                    self.verticalMorphsHi - self.nominal,
                    self.verticalMorphsLo - self.nominal,
                )
                # TODO: why 3x!!
                return self.nominal + 3 * jnp.sum(vshift, axis=0)
            elif self.verticalType == 1:
                # LogQuadLinear
                vshift = _asym_interpolation(
                    vertp,
                    jnp.log(self.verticalMorphsHi / self.nominal),
                    jnp.log(self.verticalMorphsLo / self.nominal),
                )
                return self.nominal * jnp.exp(jnp.sum(vshift, axis=0))

        return val


@RooWorkspace.register
@dataclass
class SimpleGaussianConstraint(RooGaussian):
    # combine implements a fast logpdf for this, hence the specializtion
    @classmethod
    def readobj(cls, obj, recursor):
        out = cls(
            # bug in combine switches the aux data and the param
            x=recursor(obj.getMean()),
            mean=recursor(obj.getX()),
            sigma=recursor(obj.getSigma()),
        )
        if out.mean.name.endswith("_In"):
            raise RuntimeError()
        if not out.x.name.endswith("_In"):
            raise RuntimeError()

        return out
