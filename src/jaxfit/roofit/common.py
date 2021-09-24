"""Built-in RooFit objects
"""
from jaxfit.roofit._util import _importROOT
from jaxfit.roofit.workspace import RooWorkspace


@RooWorkspace.register
class RooProdPdf:
    @classmethod
    def readobj(cls, obj, recursor):
        assert set(obj.pdfList()) == set(obj.servers())
        return {
            "pdfs": [recursor(pdf) for pdf in obj.pdfList()],
        }


@RooWorkspace.register
class RooCategory:
    @classmethod
    def readobj(cls, obj, recursor):
        return {
            "labels": [label for label, idx in obj.states()],
        }


@RooWorkspace.register
class RooProduct:
    @classmethod
    def readobj(cls, obj, recursor):
        return {"components": [recursor(x) for x in obj.components()]}


@RooWorkspace.register
class RooConstVar:
    @classmethod
    def readobj(cls, obj, recursor):
        return {"val": obj.getVal()}


@RooWorkspace.register
class RooRealSumPdf:
    @classmethod
    def readobj(cls, obj, recursor):
        return {
            "functions": [recursor(f) for f in obj.funcList()],
            "coefficients": [recursor(c) for c in obj.coefList()],
        }


@RooWorkspace.register
class RooPoisson:
    @classmethod
    def readobj(cls, obj, recursor):
        # FIXME: in ROOT 6.24 we get proxy accessors (getProxy/numProxies)
        # For now assume servers always in correct order
        x, mean = obj.servers()
        return {"x": recursor(x), "mean": recursor(mean)}


def _parse_binning(binning, recursor):
    n = binning.numBins()
    if binning.isUniform():
        return {
            "type": "uniform",
            "lo": binning.lowBound(),
            "hi": binning.highBound(),
            "n": n,
        }
    elif binning.isParameterized():
        raise NotImplementedError("RooParamBinning")
    return {
        "type": "edges",
        "edges": [binning.binLow(i) for i in range(n)] + [binning.binHigh(n - 1)],
    }


@RooWorkspace.register
class RooRealVar:
    @classmethod
    def readobj(cls, obj, recursor):
        assert len(obj.servers()) == 0
        out = {
            "val": obj.getVal(),
            "min": obj.getMin(),
            "max": obj.getMax(),
            "const": obj.getAttribute("Constant"),
        }
        bnames = list(obj.getBinningNames())
        if bnames == [""]:
            out["binning"] = _parse_binning(obj.getBinning(""), recursor)
        else:
            # mostly because I don't know the use case
            raise NotImplementedError("Multiple binnings for RooRealVar")
        return out


@RooWorkspace.register
class RooDataSet:
    @classmethod
    def readobj(cls, obj, recursor):
        ROOT = _importROOT()
        data = obj.store()
        if not isinstance(data, ROOT.RooVectorDataStore):
            raise NotImplementedError("Non-memory data stores")
        out = {
            "observables": [recursor(p) for p in data.row()],
            "points": [list(p) for p in data.getBatch(0, data.size())],
        }
        if data.isWeighted():
            out["weights"] = (list(data.getWeightBatch(0, data.size())),)
        return out
