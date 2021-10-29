"""Built-in RooFit objects
"""
from dataclasses import dataclass
from functools import reduce
from typing import List, Tuple, Dict, Any, Literal

import jax.numpy as jnp
import jax.scipy.stats as stats
import numpy as np

from jaxfit.roofit._util import ParameterPack, _importROOT
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace
from jaxfit.types import Array, Distribution, Function, Parameter

BREAK_UNKNOWN = True


class DataPack:
    def __init__(self):
        self._auxdata = ParameterPack()
        self._binned = {}
        self._unbinned = {}

    def arrayof(self, obs: Dict[str, Any], kind: Literal["aux", "binned", "unbinned"]):
        """Return function to fetch data array from input"""
        if kind == "aux":
            unpack = self._auxdata.arrayof(list(obs))
            return lambda data: unpack(data["_aux_"])
        key = hex(id(obs))  # arbitrary
        if kind == "binned":
            self._binned[key] = obs
        else:
            self._unbinned[key] = obs
        return lambda data: data[key]

    def ravel(
        self, data: "RooDataSet", auxdata: Dict[str, float]
    ) -> Dict[Tuple[str], Array]:
        out = {"_aux_": self._auxdata.ravel(auxdata)}
        obsmap = {obs.name: (i, obs) for i, obs in enumerate(data.observables)}
        for key, axes in self._binned.items():
            # TODO this is trash
            catinfo = None
            bininfo = None
            for name, vals in axes.items():
                idx, obs = obsmap[name]
                if isinstance(obs, RooCategory):
                    if catinfo is not None:
                        raise RuntimeError("only supports one category")
                    catinfo = vals, idx, obs
                elif isinstance(obs, RooRealVar):
                    if bininfo is not None:
                        raise RuntimeError("only supports one binned axis")
                    bininfo = vals, idx, obs

            if catinfo is None or bininfo is None:
                raise RuntimeError(
                    "all binned data needs a category or binning for now"
                )

            catvals, catidx, catobs = catinfo
            binedges, binidx, _ = bininfo
            binvals = 0.5 * (binedges[1:] + binedges[:-1])
            arr = []
            for val in catvals:
                pts = data.points[:, data.points[catidx, :] == catobs.labels.index(val)]
                if not jnp.all(pts[binidx, :] == binvals):
                    raise RuntimeError("binning mismatch")
                # _weight_ index always last
                arr.append(pts[-1, :])
            out[key] = jnp.stack(arr)

        for key, axes in self._unbinned.items():
            catinfo = None
            cols = []
            for name, val in axes.items():
                idx, obs = obsmap[name]
                if isinstance(obs, RooCategory):
                    if catinfo is not None:
                        raise RuntimeError("only supports one category")
                    catinfo = val, idx, obs
                elif isinstance(obs, RooRealVar):
                    cols.append(idx)

            if catinfo is None:
                raise RuntimeError("all unbinned data needs a category")

            catval, catidx, catobs = catinfo
            cut = data.points[catidx, :] == catobs.labels.index(catval)
            out[key] = data.points[:, cut][cols, :]

        return out


@RooWorkspace.register
@dataclass
class _Unknown(Model):
    children: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        if BREAK_UNKNOWN:
            import pdb

            pdb.set_trace()
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
    def parameters(self):
        return reduce(set.union, (pdf.parameters for pdf in self.pdfs), set())

    def log_prob(self, observables: DataPack, parameters: ParameterPack):
        vals = [pdf.log_prob(observables, parameters) for pdf in self.pdfs]

        def logp(data, param):
            return sum(val(data, param) for val in vals)

        return logp


@RooWorkspace.register
@dataclass
class RooCategory(Model):
    labels: List[str]

    @classmethod
    def readobj(cls, obj, recursor):
        items = list(tuple(s) for s in obj.states())
        items.sort(key=lambda s: s[1])
        labels, indices = map(list, zip(*items))
        if not all(i == j for i, j in enumerate(indices)):
            raise RuntimeError("Category indices have gaps")
        return cls(labels=labels)


def _factorize_prod(fcn):
    if isinstance(fcn, list):
        for p in fcn:
            yield from _factorize_prod(p)
    elif isinstance(fcn, RooProduct):
        for p in fcn.components:
            yield from _factorize_prod(p)
    else:
        yield fcn


@RooWorkspace.register
@dataclass
class RooProduct(Model, Function):
    components: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(components=[recursor(x) for x in obj.components()])

    @property
    def parameters(self):
        return reduce(set.union, (x.parameters for x in self.components), set())

    @classmethod
    def vectorize(cls, items: List["RooProduct"], parameters: ParameterPack):
        factors = [
            sorted(_factorize_prod(item), key=lambda p: p.name) for item in items
        ]
        n = len(items)
        parents = [i for i, vals in enumerate(factors) for _ in vals]
        factors = [p for vals in factors for p in vals]
        canVectorize = [isinstance(p, RooRealVar) for p in factors]
        nvect = sum(canVectorize)
        if nvect > 0:
            vIdx, vParam = zip(
                *(
                    (parent, p.val if p.const else p.name)
                    for parent, p, v in zip(parents, factors, canVectorize)
                    if v
                )
            )
            vIdx = jnp.array(vIdx)
            vParam = parameters.arrayof(vParam)

        if nvect < len(factors):
            addIdx, addParam = zip(
                *(
                    (parent, p.value(parameters))
                    for parent, p, v in zip(parents, factors, canVectorize)
                    if not v
                )
            )
            addIdx = jnp.array(addIdx)

        def val(param):
            out = jnp.ones(n)
            if nvect > 0:
                out = out.at[vIdx].multiply(vParam(param))
            if nvect < len(factors):
                out = out.at[addIdx].multiply(jnp.array([p(param) for p in addParam]))
            return out

        return val

    def value(self, parameters):
        addParam = [p.value(parameters) for p in self.components]
        return lambda param: reduce(jnp.multiply, (v(param) for v in addParam))


@RooWorkspace.register
@dataclass
class RooFormulaVar(Model, Function):
    formula: str
    parameters: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(
            formula=obj.formula().formulaString(),
            parameters=[recursor(x) for x in obj.servers()],
        )

    @property
    def parameters(self):
        return reduce(set.union, (x.parameters for x in self.parameters), set())

    @classmethod
    def vectorize(cls, items: List["RooFormulaVar"], parameters: ParameterPack):
        raise NotImplementedError

    def value(self, parameters):
        raise NotImplementedError


@RooWorkspace.register
@dataclass
class RooAddPdf(Model, Distribution):
    pdfs: List[Model]
    coefficients: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        out = cls(
            pdfs=[recursor(f) for f in obj.pdfList()],
            coefficients=[recursor(c) for c in obj.coefList()],
        )
        return out


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
    def parameters(self):
        fpars = reduce(set.union, (x.parameters for x in self.functions), set())
        cpars = reduce(set.union, (x.parameters for x in self.coefficients), set())
        return fpars | cpars

    def log_prob(self, observables: DataPack, parameters: ParameterPack):
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
        return self.x.observables

    @property
    def parameters(self):
        return self.mean.parameters

    def log_prob(self, observables: DataPack, parameters: ParameterPack):
        x = observables.arrayof([self.x.name])
        mu = parameters.arrayof([self.mean.name])

        def logp(data, param):
            return stats.poisson.logpmf(x(data), mu=mu(param))

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
        return self.x.observables

    @property
    def parameters(self):
        return self.mean.parameters | self.sigma.parameters

    def log_prob(self, observables: DataPack, parameters: ParameterPack):
        raise RuntimeError("unvectorized gaussian")
        x = observables.arrayof([self.x.name])
        loc = parameters.arrayof([self.mean.name])
        scale = parameters.arrayof([self.sigma.name])

        def logp(data, param):
            return stats.norm.logpdf(
                x(data),
                loc=loc(param),
                scale=scale(param),
            )

        return logp


@RooWorkspace.register
@dataclass
class RooUniform(Model):
    x: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        # FIXME: in ROOT 6.24 we get proxy accessors (getProxy/numProxies)
        x = [recursor(x) for x in obj.servers()]
        return cls(x=x)

    @property
    def observables(self):
        return set()

    @property
    def parameters(self):
        return reduce(set.union, (x.parameters for x in self.x), set())

    def log_prob(self, observables: DataPack, parameters: ParameterPack):
        def logp(data, param):
            return 1.0

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
class RooRealVar(Model, Parameter):
    val: float
    min: float
    max: float
    const: bool
    binning: Model

    @classmethod
    def readobj(cls, obj, recursor):
        bnames = list(obj.getBinningNames())
        if "" in bnames:
            binning = recursor(obj.getBinning(""))
        else:
            # mostly because I don't know the use case
            raise NotImplementedError(
                f"No or multiple binnings for {obj.GetName()}: {bnames}"
            )
        # TODO: why do all vars have a binning? how to tell default from real?
        out = cls(
            val=obj.getVal(),
            min=obj.getMin(),
            max=obj.getMax(),
            const=obj.getAttribute("Constant"),
            binning=binning,
        )
        return out

    @property
    def observables(self):
        return {self.name}

    @property
    def parameters(self):
        return set() if self.const else {self.name}

    def value(self, parameters):
        if self.const:
            return lambda param: self.val

        return parameters.arrayof([self.name], scalar=True)


@RooWorkspace.register
@dataclass
class RooConstVar(Model, Parameter):
    val: float

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(val=obj.getVal())

    @property
    def observables(self):
        return {self.name}

    @property
    def parameters(self):
        return set()

    @property
    def const(self):
        return True

    def value(self, parameters):
        return lambda param: self.val


@RooWorkspace.register
@dataclass
class RooDataSet(Model):
    observables: List[Model]
    points: Array  # (obs, evt)

    @classmethod
    def readobj(cls, obj, recursor):
        ROOT = _importROOT()
        data = obj.store()
        if not isinstance(data, ROOT.RooVectorDataStore):
            raise NotImplementedError("Non-memory data stores")

        n = data.numEntries()
        if n == 0:
            raise RuntimeError("Empty dataset!")

        observables = [recursor(p) for p in data.row()]
        weighted = data.isWeighted()
        if (observables[-1].name == "RooRealVar:_weight_") != weighted:
            raise RuntimeError(
                "Presence of _weight_ observable is inconsistent with isWeighted"
            )
        points = np.empty(shape=(len(observables), n))
        for j in range(n):
            for i, (obs, x) in enumerate(zip(observables, data.get(j))):
                # NB: int to float conversion for category index
                points[i, j] = (
                    x.getVal() if isinstance(obs, RooRealVar) else x.getIndex()
                )
            if weighted:
                points[-1, j] = data.weight()
        out = cls(
            observables=observables,
            points=jnp.array(points),
        )
        return out

    def categories(self) -> List[Tuple[int, str]]:
        return [
            (i, p) for i, p in enumerate(self.observables) if isinstance(p, RooCategory)
        ]

    def variables(self) -> List[Tuple[int, str]]:
        return [
            (i, p) for i, p in enumerate(self.observables) if isinstance(p, RooRealVar)
        ]
