"""Built-in RooFit objects
"""
from dataclasses import dataclass
from functools import reduce
from typing import Any, List, Set

import jax.numpy as jnp
import jax.scipy.stats as stats

from jaxfit.roofit._util import _importROOT
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace
from jaxfit.types import Array, Distribution, Function


@RooWorkspace.register
@dataclass
class _Unknown(Model):
    children: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(children=[recursor(child) for child in obj.servers()])


@RooWorkspace.register
@dataclass
class RooProdPdf(Model, Distribution):
    pdfs: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        assert set(obj.pdfList()) == set(obj.servers())
        return cls(
            pdfs=[recursor(pdf) for pdf in obj.pdfList()],
        )

    @property
    def observables(self):
        return reduce(set.union, (pdf.observables for pdf in self.pdfs), set())

    @property
    def parameters(self) -> Set[str]:
        return reduce(set.union, (pdf.parameters for pdf in self.pdfs), set())

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        # TODO: we don't really want to skip pdfs with missing observables?
        # right now its a hack to skip duplicate constraints on global observables
        vals = [
            pdf.log_prob(observables, parameters)
            for pdf in self.pdfs
            if pdf.observables & observables
        ]

        def logp(data, param):
            return sum(val(data, param) for val in vals)

        return logp


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
class RooProduct(Model, Function):
    components: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(components=[recursor(x) for x in obj.components()])

    @property
    def parameters(self) -> Set[str]:
        return reduce(set.union, (x.parameters for x in self.components), set())


@RooWorkspace.register
@dataclass
class RooConstVar(Model):
    val: float

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(val=obj.getVal())

    @property
    def parameters(self):
        return set()

    def value(self, parameters):
        return lambda param: self.val


@RooWorkspace.register
@dataclass
class RooRealSumPdf(Model, Distribution):
    functions: List[Model]
    coefficients: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        out = cls(
            functions=[recursor(f) for f in obj.funcList()],
            coefficients=[recursor(c) for c in obj.coefList()],
        )
        # TODO: Is this always true?
        assert all(isinstance(f, Distribution) for f in out.functions)
        return out

    @property
    def observables(self):
        return reduce(set.union, (x.observables for x in self.functions), set())

    @property
    def parameters(self) -> Set[str]:
        fpars = reduce(set.union, (x.parameters for x in self.functions), set())
        cpars = reduce(set.union, (x.parameters for x in self.coefficients), set())
        return fpars | cpars

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        funcs = [f.log_prob(observables, parameters) for f in self.functions]
        coefs = [c.value(parameters) for c in self.coefficients]

        def logp(data, param):
            # TODO: normalize by sum(c)?
            return sum(f(data, param) * c(param) for f, c in zip(funcs, coefs))

        return logp


@RooWorkspace.register
@dataclass
class RooPoisson(Model, Distribution):
    x: Model
    mean: Model

    @classmethod
    def readobj(cls, obj, recursor):
        # FIXME: in ROOT 6.24 we get proxy accessors (getProxy/numProxies)
        # For now assume servers always in correct order
        x, mean = map(recursor, obj.servers())
        return cls(x=x, mean=mean)

    @property
    def observables(self):
        return self.x.parameters

    @property
    def parameters(self) -> Set[str]:
        return self.mean.parameters

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in function {self.name}")
        missing = self.observables - observables
        if missing:
            raise RuntimeError(
                f"Missing observables: {missing} in function {self.name}"
            )

        def logp(data, param):
            return stats.poisson.logpmf(data[self.x.name], mu=param[self.mean.name])

        return logp


@RooWorkspace.register
@dataclass
class RooGaussian(Model):
    x: Model
    mean: Model
    sigma: Model

    @classmethod
    def readobj(cls, obj, recursor):
        # FIXME: in ROOT 6.24 we get proxy accessors (getProxy/numProxies)
        x, mean, sigma = map(recursor, obj.servers())
        return cls(
            x=x,
            mean=mean,
            sigma=sigma,
        )

    @property
    def observables(self):
        return self.x.parameters

    @property
    def parameters(self) -> Set[str]:
        return self.mean.parameters | self.sigma.parameters

    def log_prob(self, observables: Set[str], parameters: Set[str]):
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in function {self.name}")
        missing = self.observables - observables
        if missing:
            raise RuntimeError(
                f"Missing observables: {missing} in function {self.name}"
            )

        def logp(data, param):
            return stats.norm.logpdf(
                data[self.x.name],
                loc=param[self.mean.name],
                scale=param[self.sigma.name],
            )

        return logp


@RooWorkspace.register
@dataclass
class RooBinning(Model):
    edges: Array

    @property
    def n(self):
        return len(self.edges) - 1

    @classmethod
    def readobj(cls, obj, recursor):
        n = obj.numBins()
        return cls(
            edges=jnp.array([obj.binLow(i) for i in range(n)] + [obj.binHigh(n - 1)])
        )


@RooWorkspace.register
@dataclass
class RooUniformBinning(Model):
    n: int
    lo: float
    hi: float

    @property
    def edges(self):
        return jnp.linspace(self.lo, self.hi, self.n + 1)

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(
            n=obj.numBins(),
            lo=obj.lowBound(),
            hi=obj.highBound(),
        )


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
            out.binning = recursor(obj.getBinning(""))
        else:
            # mostly because I don't know the use case
            raise NotImplementedError("Multiple binnings for RooRealVar")
        return out

    @property
    def parameters(self):
        return set() if self.const else {self.name}

    def value(self, parameters):
        if self.const:
            return lambda param: self.val
        missing = self.parameters - parameters
        if missing:
            raise RuntimeError(f"Missing parameters: {missing} in var {self.name}")
        return lambda param: param[self.name]


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
