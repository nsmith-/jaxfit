"""Custom RooFit objects found in CMS combine
"""
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Union

import jax
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
    RooRealSumPdf,
    RooRealVar,
)
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace
from jaxfit.types import Array, Distribution, Function


def _fasthisto2array(h):
    return jnp.array([h[i] for i in range(h.size())])


def _factorize_prodpdf(pdf):
    if isinstance(pdf, RooProdPdf):
        for p in pdf.pdfs:
            yield from _factorize_prodpdf(p)
    elif (
        isinstance(pdf, RooRealSumPdf)
        and len(pdf.coefficients) == 1
        and pdf.coefficients[0].const
        and pdf.coefficients[0].val == 1.0
    ):
        # combine adds this extra layer for some reason
        yield pdf.functions[0]
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
        cms_hists = {}
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
                elif isinstance(pdf, CMSHistErrorPropagator):
                    cms_hists[cat] = pdf
                else:
                    # this pdf has observables that may depend on cat
                    # and anyway we delegate construction to it
                    # probably better to fix combine structure in the beginning..
                    if cat in generic:
                        raise RuntimeError("did we over-factorize?")
                    generic[cat] = pdf.log_prob(observables.slice(cat), parameters)

        gaus_constr = sorted((x, mu, sigma) for x, (mu, sigma) in gaus_constr.items())
        gausx = observables.arrayof([p for p, _, _ in gaus_constr])
        gausm = parameters.arrayof([p for _, p, _ in gaus_constr])
        gauss = parameters.arrayof([p for _, _, p in gaus_constr])
        pois_constr = sorted((x, mu) for x, mu in pois_constr.items())
        poisx = observables.arrayof([p for p, _ in pois_constr])
        poism = parameters.arrayof([p for _, p in pois_constr])

        hist_lp = CMSHistErrorPropagator.vectorize(cms_hists, observables, parameters)

        def logp(data, param):
            out = jnp.sum(
                stats.norm.logpdf(gausx(data), loc=gausm(param), scale=gauss(param))
            )
            out = out + jnp.sum(stats.poisson.logpmf(poisx(data), mu=poism(param)))
            if generic:
                out = out + reduce(
                    jnp.add, (lp(data, param) for lp in generic.values())
                )
            if cms_hists:
                out = out + hist_lp(data, param)
            return out

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
    def vectorize(
        cls, items: List[List["ProcessNormalization"]], parameters: ParameterPack
    ):
        nch = len(items)
        nproc = max(len(ch) for ch in items)
        nominal = np.zeros(shape=(nch, nproc))
        symParams = sorted(
            reduce(
                set.union,
                (set(p.name for p in proc.symParams) for ch in items for proc in ch),
                set(),
            )
        )
        symLogKappa = np.zeros(shape=(nch, nproc, len(symParams)))
        print(
            f"ProcessNormalization vectorize symmetric modifier shape: {symLogKappa.shape}"
        )

        posmap = {n: i for i, n in enumerate(symParams)}
        for ich, ch in enumerate(items):
            for iproc, proc in enumerate(ch):
                nominal[ich, iproc] = proc.nominal
                for p, val in zip(proc.symParams, proc.symLogKappa):
                    try:
                        pos = posmap[p.name]
                    except KeyError:
                        continue
                    symLogKappa[ich, iproc, pos] = val

        symParams = parameters.arrayof(symParams)

        asymParams = sorted(
            reduce(
                set.union,
                (set(p.name for p in proc.asymParams) for ch in items for proc in ch),
                set(),
            )
        )
        asymLogKappaSum = np.zeros(shape=(nch, nproc, len(asymParams)))
        asymLogKappaDiff = np.zeros(shape=(nch, nproc, len(asymParams)))
        posmap = {n: i for i, n in enumerate(asymParams)}
        for ich, ch in enumerate(items):
            for iproc, proc in enumerate(ch):
                for p, lo, hi in zip(
                    proc.asymParams, proc.asymLogKappaLo, proc.asymLogKappaHi
                ):
                    try:
                        pos = posmap[p.name]
                    except KeyError:
                        continue
                    asymLogKappaSum[ich, iproc, pos] = hi + lo
                    asymLogKappaDiff[ich, iproc, pos] = hi - lo

        asymParams = parameters.arrayof(asymParams)
        print(
            f"ProcessNormalization vectorize asymmetric modifier shape: {asymLogKappaSum.shape}"
        )

        addparam = []
        addpos0 = []
        addpos1 = []
        for ich, ch in enumerate(items):
            for iproc, proc in enumerate(ch):
                if proc.additional is not None:
                    addparam.append(proc.additional.value(parameters))
                    addpos0.append(ich)
                    addpos1.append(iproc)

        addpos0 = jnp.array(addpos0)
        addpos1 = jnp.array(addpos1)
        print(
            f"ProcessNormalization vectorize additional params: {len(addparam)} ({len(addparam)*100/nch/nproc:.0f}%)"
        )

        def val(param):
            symShift = jnp.sum(symLogKappa * symParams(param), axis=2)
            asymShift = jnp.sum(
                _asym_interpolation(
                    asymParams(param),
                    asymLogKappaSum,
                    asymLogKappaDiff,
                ),
                axis=2,
            )
            addFactor = (
                jnp.ones_like(nominal)
                .at[addpos0, addpos1]
                .set(jnp.array([p(param) for p in addparam]))
            )
            return nominal * jnp.exp(symShift + asymShift) * addFactor

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

    # transpose (bin, proc) -> (proc, bin)
    bbpars = list(list(bp) for bp in np.array(bbpars).T)
    bbscale = np.array(bbscale).T
    return bbpars, jnp.array(bbscale)


@RooWorkspace.register
@dataclass
class CMSHistErrorPropagator(Model, Distribution):
    # FIXME: subclass RooRealSumPdf?
    x: Model
    functions: List[Model]
    coefficients: List[Model]
    bbpars: List[List[Union[Model, float]]]
    bbscale: Array  # 2d: proc, bin

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
        if any(
            x.name in ("RooRealVar:ZERO", "RooRealVar:ONE") for x in out.coefficients
        ):
            # TODO: we should just trim the process from this model
            raise NotImplementedError(
                "model where one coefficient is fixed to zero or one"
            )
        assert all(isinstance(x, CMSHistFunc) for x in out.functions)
        for i, c in enumerate(out.coefficients):
            if isinstance(c, ProcessNormalization):
                pass
            elif isinstance(c, RooProduct) and all(
                isinstance(x, (ProcessNormalization, AsymPow)) for x in c.components
            ):
                # TODO: handle RooProduct of ProcessNormalization and several AsymPow
                raise NotImplementedError(
                    "Try running text2workspace with --X-pack-asympows"
                )
            elif isinstance(c, RooRealVar) and c.const:
                out.coefficients[i] = ProcessNormalization(
                    nominal=c.val,
                    symParams=[],
                    symLogKappa=jnp.array([]),
                    asymParams=[],
                    asymLogKappaLo=jnp.array([]),
                    asymLogKappaHi=jnp.array([]),
                    additional=None,
                )
            else:
                import pdb

                pdb.set_trace()
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

    @classmethod
    def vectorize(
        cls,
        channels: Dict[str, "CMSHistErrorPropagator"],
        observables: DataSlice,
        parameters: ParameterPack,
    ):
        # (ch, proc, bin)
        procvals, mask, bb_errors = CMSHistFunc.vectorize(
            [ch.functions for ch in channels.values()], parameters
        )
        procnorms = ProcessNormalization.vectorize(
            [ch.coefficients for ch in channels.values()], parameters
        )
        bblite_errors = np.sqrt(np.sum(bb_errors ** 2, axis=1))
        bbparams = np.full(shape=bb_errors.shape, fill_value=0.0, dtype=object)
        bbscale = np.full(shape=bb_errors.shape, fill_value=-2.0)
        for ich, ch in enumerate(channels.values()):
            nproc, nbin = ch.bbscale.shape
            bbscale[ich, :nproc, :nbin] = ch.bbscale
            for iproc, proc in enumerate(ch.bbpars):
                for ibin, bpar in enumerate(proc):
                    if isinstance(bpar, float):
                        bbparams[ich, iproc, ibin] = bpar
                    else:
                        bbparams[ich, iproc, ibin] = bpar.name
        bbparams = parameters.arrayof(list(bbparams.flatten()))

        # TODO: better gathering of observed yields
        obs_stack = [
            observables.slice(cat).arrayof(ch.x.binning.edges)
            for cat, ch in channels.items()
        ]

        def logp(data, param):
            observed = jnp.array([get(data) for get in obs_stack])
            process_expected = procnorms(param)[:, :, None] * procvals(param)
            # here we can do analytic Balow-Beeston in principle
            bb = bbparams(param).reshape(bb_errors.shape)
            process_expected = process_expected + jnp.where(
                bbscale > 0.0,
                (bb * bbscale - 1) * process_expected,
                jnp.where(
                    bbscale == 0.0,
                    bb_errors * bb,
                    jnp.where(bbscale == -1.0, bblite_errors[:, None, :] * bb, 0.0),
                ),
            )
            expected = jnp.sum(process_expected, axis=1)
            return jnp.sum(
                jnp.where(mask[:, 0, :], stats.poisson.logpmf(observed, expected), 0.0)
            )

        return logp

    def log_prob(self, observables: DataSlice, parameters: ParameterPack):
        raise NotImplementedError("This should be vectorized")


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
    def vshape(self, items: List["CMSHistFunc"]):
        nbins = len(items[0].nominal)
        verticalParams = sorted(
            reduce(
                set.union,
                (set(p.name for p in c.verticalParams) for c in items),
                set(),
            )
        )
        return (len(items), len(verticalParams), nbins)

    @classmethod
    def vectorize(cls, items: List[List["CMSHistFunc"]], parameters: ParameterPack):
        """Vectorize a (channel, process) array of CMSHistFunc

        Returns a function that evaluates to the (channel, process, bin) array
        and a mask array that is True for valid bins
        """
        nch = len(items)
        nproc = max(len(ch) for ch in items)
        nbins = max(max(len(p.nominal) for p in ch) for ch in items)
        verticalParams = sorted(
            reduce(
                set.union,
                (
                    set(p.name for p in proc.verticalParams)
                    for ch in items
                    for proc in ch
                ),
                set(),
            )
        )
        nsyst = len(verticalParams)
        print(
            f"CMSHistFunc vectorize shape (ch, proc, bin, syst): {nch}, {nproc}, {nbins}, {nsyst}"
        )
        nominal = np.zeros(shape=(nch, nproc, nbins))
        verticalType = np.zeros(shape=(nch, nproc))
        asymSum = np.zeros(shape=(nch, nproc, nbins, nsyst))
        asymDiff = np.zeros(shape=(nch, nproc, nbins, nsyst))
        mask = np.zeros(shape=(nch, nproc, nbins), dtype=bool)
        bb_errors = np.zeros(shape=(nch, nproc, nbins))

        posmap = {n: i for i, n in enumerate(verticalParams)}
        for ich, ch in enumerate(items):
            for iproc, proc in enumerate(ch):
                n = len(proc.nominal)
                nominal[ich, iproc, :n] = proc.nominal
                verticalType[ich, iproc] = proc.verticalType
                mask[ich, iproc, :n] = True
                bb_errors[ich, iproc, :n] = proc.bberrors
                for p, lo, hi in zip(
                    proc.verticalParams, proc.verticalMorphsLo, proc.verticalMorphsHi
                ):
                    try:
                        pos = posmap[p.name]
                    except KeyError:
                        continue
                    if proc.verticalType == 0:
                        asymSum[ich, iproc, :n, pos] = hi + lo - 2 * proc.nominal
                        asymDiff[ich, iproc, :n, pos] = hi - lo
                    else:
                        asymSum[ich, iproc, :n, pos] = np.log(hi / proc.nominal)
                        asymDiff[ich, iproc, :n, pos] = np.log(lo / proc.nominal)

        verticalParams = parameters.arrayof(verticalParams)

        def val(param):
            vshift = jnp.sum(
                _asym_interpolation(
                    verticalParams(param)[None, None, None, :], asymSum, asymDiff
                ),
                axis=3,
            )
            return jnp.where(
                (verticalType == 0)[:, :, None],
                nominal + 3 * vshift,
                nominal * jnp.exp(vshift),
            )

        return val, mask, bb_errors

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


@RooWorkspace.register
@dataclass
class AsymPow(Model, Function):
    kappaLo: Model
    kappaHi: Model
    theta: Model

    @classmethod
    def readobj(cls, obj, recursor):
        # FIXME: in ROOT 6.24 we get proxy accessors (getProxy/numProxies)
        items = [recursor(x) for x in obj.servers()]
        if len(items) == 3 and (items[0].const and items[1].const):
            kappaLo, kappaHi, theta = items
        elif len(items) == 2 and items[0].const:
            kappaLo, theta = items
            kappaHi = kappaLo
        else:
            raise RuntimeError("Could not parse AsymPow")
        out = cls(
            kappaLo=kappaLo,
            kappaHi=kappaHi,
            theta=theta,
        )
        return out


# hgg TODO
# CMSHggFormulaA2
# CMSHggFormulaB2
# CMSHggFormulaC1
# CMSHggFormulaD2
# RooBernsteinFast<1>
# RooBernsteinFast<2>
# RooBernsteinFast<3>
# RooBernsteinFast<4>
# RooBernsteinFast<5>
# RooBernsteinFast<6>
# RooCheapProduct
# RooExponential
# RooFormulaVar
# RooMultiPdf
# RooPower
# RooRecursiveFraction
# hzz TODO
# RooFormulaVar
# RooBernstein
# VerticalInterpPdf
# RooGenericPdf
# RooHistPdf
# RooDoubleCBFast
