"""Custom RooFit objects found in CMS combine
"""
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Union

import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

from jaxfit.roofit._util import DataPack, DataSlice, ParameterPack
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

# debug/validation flags
VECTORIZE_CMSHistFunc = True
VECTORIZE_ProcessNormalization = True


def _fasthisto2array(h):
    return jnp.array([h[i] for i in range(h.size())])


def _factorize_prodpdf(pdf):
    if isinstance(pdf, RooProdPdf):
        for p in pdf.pdfs:
            yield from _factorize_prodpdf(p)
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
    def parameters(self):
        return reduce(set.union, (pdf.parameters for pdf in self.pdfs.values()), set())

    def log_prob(self, observables: DataPack, parameters: ParameterPack):
        """Combine-specific log_prob

        We can do more optimization because we expect a certain structure, based on
        how combine builds the RooFit model.
        """
        generic = {}
        gaus_constr = {}
        pois_constr = {}
        for cat, catp in self.pdfs.items():
            for pdf in _factorize_prodpdf(catp):
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
                    pois_constr[obs] = pdf.mean.val if pdf.mean.const else pdf.mean.name
                else:
                    # this pdf has observables that may depend on cat
                    # and anyway we delegate construction to it
                    # probably better to fix combine structure in the beginning..
                    if cat in generic:
                        raise RuntimeError("did we over-factorize?")
                    generic[cat] = pdf.log_prob(observables.slice(cat), parameters)

        gaus_constr = sorted((x, mu, sigma) for x, (mu, sigma) in gaus_constr)
        gausx = observables.arrayof([p for p, _, _ in gaus_constr])
        gausm = parameters.arrayof([p for _, p, _ in gaus_constr])
        gauss = parameters.arrayof([p for _, _, p in gaus_constr])
        pois_constr = sorted((x, mu) for x, mu in pois_constr)
        poisx = observables.arrayof([p for p, _ in pois_constr])
        poism = parameters.arrayof([p for _, p in pois_constr])

        def logp(data, param):
            gaus_prob = stats.norm.logpdf(
                gausx(data), loc=gausm(param), scale=gauss(param)
            )
            pois_prob = stats.poisson.logpmf(poisx(data), mu=poism(param))
            return (
                reduce(jnp.add, (generic[cat](data, param) for cat in self.pdfs))
                + jnp.sum(gaus_prob)
                + jnp.sum(pois_prob)
            )

        return logp


_asym_poly = jnp.array([3.0, -10.0, 15.0, 0.0]) / 8.0


def _asym_interpolation(x, dx_sum, dx_diff):
    """A function that is C^2 continuous in theta

    dx_sum is the sum of positive and negative relative shifts,
    dx_diff is the difference
    """
    ax = abs(x)
    morph = 0.5 * (
        dx_sum * x
        + dx_diff
        * jnp.where(
            ax > 1.0,
            ax,
            jnp.polyval(_asym_poly, x * x),
        )
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
    additional: Optional[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        asympar = [[-lo, hi] for lo, hi in obj.getAsymLogKappa()]
        addpar = [recursor(p) for p in obj.getAdditionalModifiers()]
        if len(addpar) > 1:
            addpar = RooProduct(components=addpar)
        elif len(addpar) == 0:
            addpar = None
        else:
            addpar = addpar[0]
        return cls(
            nominal=obj.getNominalValue(),
            symParams=[recursor(p) for p in obj.getSymErrorParameters()],
            symLogKappa=jnp.array(list(obj.getSymLogKappa())),
            asymParams=[recursor(p) for p in obj.getAsymErrorParameters()],
            asymLogKappaLo=jnp.array([p[0] for p in asympar]),
            asymLogKappaHi=jnp.array([p[1] for p in asympar]),
            additional=addpar,
        )

    @property
    def parameters(self):
        return reduce(
            set.union,
            (p.parameters for p in self.symParams + self.asymParams),
            self.additional.parameters if self.additional else set(),
        )

    @classmethod
    def vectorize(cls, items: List["ProcessNormalization"], parameters: ParameterPack):
        nominal = jnp.array([c.nominal for c in items])
        symParams = sorted(
            reduce(
                set.union,
                (set(p.name for p in c.symParams) for c in items),
                set(),
            )
        )
        symLogKappa = np.zeros(shape=(len(items), len(symParams)))
        for i, c in enumerate(items):
            for p, val in zip(c.symParams, c.symLogKappa):
                try:
                    pos = symParams.index(p.name)
                except ValueError:
                    continue
                symLogKappa[i, pos] = val
        symLogKappa = jnp.array(symLogKappa)
        symParams = parameters.arrayof(symParams)

        asymParams = sorted(
            reduce(
                set.union,
                (set(p.name for p in c.asymParams) for c in items),
                set(),
            )
        )
        asymLogKappaSum = np.zeros(shape=(len(items), len(asymParams)))
        asymLogKappaDiff = np.zeros(shape=(len(items), len(asymParams)))
        for i, c in enumerate(items):
            for p, lo, hi in zip(c.asymParams, c.asymLogKappaLo, c.asymLogKappaHi):
                try:
                    pos = asymParams.index(p.name)
                except ValueError:
                    continue
                asymLogKappaSum[i, pos] = hi + lo
                asymLogKappaDiff[i, pos] = hi - lo
        asymLogKappaSum = jnp.array(asymLogKappaSum)
        asymLogKappaDiff = jnp.array(asymLogKappaDiff)
        asymParams = parameters.arrayof(asymParams)

        additional = [
            c.additional.value(parameters) if c.additional else None for c in items
        ]

        def val(param):
            symShift = jnp.sum(symLogKappa * symParams(param), axis=1)
            asymShift = jnp.sum(
                _asym_interpolation(
                    asymParams(param),
                    asymLogKappaSum,
                    asymLogKappaDiff,
                ),
                axis=1,
            )
            addParam = jnp.array([p(param) if p else 1.0 for p in additional])
            return nominal * jnp.exp(symShift + asymShift) * addParam

        return val

    def value(self, parameters: ParameterPack):
        symTheta = parameters.arrayof([p.name for p in self.symParams])
        asymTheta = parameters.arrayof([p.name for p in self.asymParams])
        asymSum = self.asymLogKappaHi + self.asymLogKappaLo
        asymDiff = self.asymLogKappaHi - self.asymLogKappaLo
        if self.additional:
            addParam = self.additional.value(parameters)
        else:
            addParam = None

        def val(param):
            symShift = jnp.sum(self.symLogKappa * symTheta(param), axis=-1)
            asymShift = jnp.sum(
                _asym_interpolation(asymTheta(param), asymSum, asymDiff),
                axis=-1,
            )
            out = self.nominal * jnp.exp(symShift + asymShift)
            if addParam is not None:
                out = out * addParam(param)
            return out

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
        if any(x.name == "RooRealVar:ZERO" for x in out.coefficients):
            # we should just trim the process from this model
            raise NotImplementedError("model where one coefficient is fixed to zero")
        assert all(isinstance(x, CMSHistFunc) for x in out.functions)
        assert all(isinstance(x, ProcessNormalization) for x in out.coefficients)
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

    def log_prob(self, observables: DataSlice, parameters: ParameterPack):
        if VECTORIZE_CMSHistFunc:
            procvals = CMSHistFunc.vectorize(self.functions, parameters)
        else:
            procvals = [f.value(parameters) for f in self.functions]
        if VECTORIZE_ProcessNormalization:
            procnorms = ProcessNormalization.vectorize(self.coefficients, parameters)
        else:
            procnorms = [c.value(parameters) for c in self.coefficients]

        bb_errors = jnp.array([f.bberrors for f in self.functions]).T
        bblite_errors = jnp.sqrt(jnp.sum(bb_errors ** 2, axis=1))
        bbparams = parameters.arrayof(
            [
                p if isinstance(p, float) else p.name
                for procpars in self.bbpars
                for p in procpars
            ]
        )
        # TODO: still can't figure out how combine ensures correct binning
        get_observed = observables.arrayof(self.x.binning.edges[: len(self.bbpars) + 1])

        def logp(data, param):
            observed = get_observed(data)
            if VECTORIZE_ProcessNormalization:
                norm = procnorms(param)
            else:
                norm = jnp.array([c(param) for c in procnorms])
            if VECTORIZE_CMSHistFunc:
                process_expected = norm * procvals(param).T
            else:
                process_expected = norm * jnp.array([f(param) for f in procvals]).T
            # here we can do analytic Balow-Beeston in principle
            bb = bbparams(param).reshape(self.bbscale.shape)
            process_expected = process_expected + jnp.where(
                self.bbscale > 0.0,
                (bb * self.bbscale - 1) * process_expected,
                jnp.where(
                    self.bbscale == 0.0,
                    bb_errors * bb,
                    jnp.where(self.bbscale == -1.0, bblite_errors[:, None] * bb, 0.0),
                ),
            )
            expected = jnp.sum(process_expected, axis=1)
            return jnp.sum(stats.poisson.logpmf(observed, expected))

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
        out = cls(
            x=recursor(obj.getXVar()),
            verticalParams=[m["param"] for m in morphs],
            verticalMorphsLo=jnp.array([m["lo"] for m in morphs]),
            verticalMorphsHi=jnp.array([m["hi"] for m in morphs]),
            verticalType=obj.getVerticalType(),
            bberrors=_fasthisto2array(obj.errors()),
            nominal=_fasthisto2array(obj.getShape(0, 0, 0, 0)),
        )
        if len(out.bberrors) != len(out.nominal):
            # assume nominal has correct number of bins always
            out.bberrors = out.bberrors[: len(out.nominal)]
        assert len(out.verticalMorphsHi) == 0 or out.verticalMorphsHi.shape[1] == len(
            out.nominal
        )
        assert len(out.verticalMorphsLo) == 0 or out.verticalMorphsLo.shape[1] == len(
            out.nominal
        )
        return out

    @property
    def parameters(self):
        return reduce(set.union, (p.parameters for p in self.verticalParams), set())

    @classmethod
    def vectorize(cls, items: List["CMSHistFunc"], parameters: ParameterPack):
        nominal = jnp.array([c.nominal for c in items])
        nbins = nominal.shape[-1]
        verticalType = jnp.array([c.verticalType for c in items])
        verticalParams = sorted(
            reduce(
                set.union,
                (set(p.name for p in c.verticalParams) for c in items),
                set(),
            )
        )
        posmap = {n: i for i, n in enumerate(verticalParams)}
        asymSum = np.zeros(shape=(len(items), len(verticalParams), nbins))
        asymDiff = np.zeros(shape=(len(items), len(verticalParams), nbins))
        for i, c in enumerate(items):
            nom = c.nominal
            for p, lo, hi in zip(
                c.verticalParams, c.verticalMorphsLo, c.verticalMorphsHi
            ):
                try:
                    pos = posmap[p.name]
                except KeyError:
                    continue
                if c.verticalType == 0:
                    asymSum[i, pos, :] = hi + lo - 2 * nom
                    asymDiff[i, pos, :] = hi - lo
                else:
                    asymSum[i, pos, :] = np.log(hi / nom)
                    asymDiff[i, pos, :] = np.log(lo / nom)
        asymSum = jnp.array(asymSum)
        asymDiff = jnp.array(asymDiff)
        verticalParams = parameters.arrayof(verticalParams)

        def val(param):
            vshift = jnp.sum(
                _asym_interpolation(
                    verticalParams(param)[None, :, None], asymSum, asymDiff
                ),
                axis=1,
            )
            return jnp.where(
                (verticalType == 0)[:, None],
                nominal + 3 * vshift,
                nominal * jnp.exp(vshift),
            )

        return val

    def value(self, parameters: ParameterPack):
        if not len(self.verticalParams):
            return lambda param: self.nominal

        vertp = parameters.arrayof([p.name for p in self.verticalParams])
        if self.verticalType == 0:
            asymSum = self.verticalMorphsHi + self.verticalMorphsLo - 2 * self.nominal
            asymDiff = self.verticalMorphsHi - self.verticalMorphsLo
        elif self.verticalType == 1:
            asymSum = jnp.log(self.verticalMorphsHi / self.nominal) + jnp.log(
                self.verticalMorphsLo / self.nominal
            )
            asymDiff = jnp.log(self.verticalMorphsHi / self.nominal) - jnp.log(
                self.verticalMorphsLo / self.nominal
            )
        else:
            raise NotImplementedError(f"vertical type {self.verticalType}")

        def val(param):
            vshift = jnp.sum(
                _asym_interpolation(vertp(param)[:, None], asymSum, asymDiff), axis=0
            )
            if self.verticalType == 0:
                # QuadLinear
                # TODO: why 3x!!
                return self.nominal + 3 * vshift
            elif self.verticalType == 1:
                # LogQuadLinear
                return self.nominal * jnp.exp(vshift)

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
