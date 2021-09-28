"""Built-in RooFit objects
"""
from dataclasses import dataclass
from typing import Any, List

import jax.numpy as jnp

from jaxfit.roofit._util import _importROOT
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace
from jaxfit.types import Array


@RooWorkspace.register
@dataclass
class _Unknown(Model):
    children: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(children=[recursor(child) for child in obj.servers()])


@RooWorkspace.register
@dataclass
class RooProdPdf(Model):
    pdfs: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        assert set(obj.pdfList()) == set(obj.servers())
        return cls(
            pdfs=[recursor(pdf) for pdf in obj.pdfList()],
        )


@RooWorkspace.register
@dataclass
class RooCategory(Model):
    labels: List[str]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(
            labels=[label for label, idx in obj.states()],
        )


@RooWorkspace.register
@dataclass
class RooProduct(Model):
    components: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(components=[recursor(x) for x in obj.components()])


@RooWorkspace.register
@dataclass
class RooConstVar(Model):
    val: float

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(val=obj.getVal())


@RooWorkspace.register
@dataclass
class RooRealSumPdf(Model):
    functions: List[Model]
    coefficients: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(
            functions=[recursor(f) for f in obj.funcList()],
            coefficients=[recursor(c) for c in obj.coefList()],
        )


@RooWorkspace.register
@dataclass
class RooPoisson(Model):
    x: Model
    mean: Model

    @classmethod
    def readobj(cls, obj, recursor):
        # FIXME: in ROOT 6.24 we get proxy accessors (getProxy/numProxies)
        # For now assume servers always in correct order
        x, mean = obj.servers()
        return cls(x=recursor(x), mean=recursor(mean))


@RooWorkspace.register
@dataclass
class RooGaussian(Model):
    x: Model
    mean: Model
    sigma: Model

    @classmethod
    def readobj(cls, obj, recursor):
        # FIXME: in ROOT 6.24 we get proxy accessors (getProxy/numProxies)
        x, mean, sigma = obj.servers()
        return cls(
            x=recursor(obj.getX()),
            mean=recursor(obj.getMean()),
            sigma=recursor(obj.getSigma()),
        )


def _parse_binning(binning, recursor):
    n = binning.numBins()
    if binning.isUniform():
        return {
            "type": "uniform",
            "edges": jnp.array([binning.lowBound(), binning.highBound()]),
            "n": n,
        }
    elif binning.isParameterized():
        raise NotImplementedError("RooParamBinning")
    return {
        "type": "edges",
        "edges": jnp.array(
            [binning.binLow(i) for i in range(n)] + [binning.binHigh(n - 1)]
        ),
        "n": n,
    }


@RooWorkspace.register
@dataclass
class RooRealVar(Model):
    val: float
    min: float
    max: float
    const: bool
    binning: Any = None  # FIXME: proper classes

    @classmethod
    def readobj(cls, obj, recursor):
        out = cls(
            val=obj.getVal(),
            min=obj.getMin(),
            max=obj.getMax(),
            const=obj.getAttribute("Constant"),
        )
        bnames = list(obj.getBinningNames())
        if bnames == [""]:
            out.binning = _parse_binning(obj.getBinning(""), recursor)
        else:
            # mostly because I don't know the use case
            raise NotImplementedError("Multiple binnings for RooRealVar")
        return out


@RooWorkspace.register
@dataclass
class RooDataSet(Model):
    observables: List[Model]
    points: Array  # 2d
    weights: Array = None  # 1d

    @classmethod
    def readobj(cls, obj, recursor):
        ROOT = _importROOT()
        data = obj.store()
        if not isinstance(data, ROOT.RooVectorDataStore):
            raise NotImplementedError("Non-memory data stores")
        out = cls(
            observables=[recursor(p) for p in data.row()],
            points=jnp.array([list(p) for p in data.getBatch(0, data.size())]),
        )
        if data.isWeighted():
            # if w2 == w we can assume, at least for combine, that this is binned poisson (or asimov)
            # check that data = bin centers
            # if w2 != w we might need to do some apprixmation a la RooFit.SumW2Error(True) minimizer option
            out.weights = jnp.array(list(data.getWeightBatch(0, data.size())))
        return out
