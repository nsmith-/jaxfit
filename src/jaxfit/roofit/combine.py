"""Custom RooFit objects found in CMS combine
"""
from dataclasses import dataclass
from typing import Any, Dict, List

from jaxfit.roofit.common import RooCategory, RooGaussian
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace


def _fasthisto2list(h):
    return [h[i] for i in range(h.size())]


@RooWorkspace.register
@dataclass
class RooSimultaneousOpt(Model):
    indexCat: RooCategory
    components: Dict[str, Any]

    @classmethod
    def readobj(cls, obj, recursor):
        cat = obj.indexCat()
        return cls(
            indexCat=recursor(cat),
            components={
                label: recursor(obj.getPdf(label)) for label, idx in cat.states()
            },
            # obj.extraConstraints()
            # obj.channelMasks()
        )


@RooWorkspace.register
@dataclass
class ProcessNormalization(Model):
    nominal: float
    symParams: List[Model]  # TODO: abstract parameter?
    symLogKappa: List[float]
    asymParams: List[Model]
    asymLogKappa: List[float]
    additional: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(
            nominal=obj.getNominalValue(),
            symParams=[recursor(p) for p in obj.getSymErrorParameters()],
            symLogKappa=list(obj.getSymLogKappa()),
            asymParams=[recursor(p) for p in obj.getAsymErrorParameters()],
            asymLogKappa=[(lo, hi) for lo, hi in obj.getAsymLogKappa()],
            additional=[recursor(p) for p in obj.getAdditionalModifiers()],
        )


@RooWorkspace.register
@dataclass
class CMSHistErrorPropagator(Model):
    # FIXME: subclass RooRealSumPdf?
    x: Model
    functions: List[Model]
    coefficients: List[Model]
    binpars: List[Model]

    @classmethod
    def readobj(cls, obj, recursor):
        return cls(
            x=recursor(obj.getX()),
            functions=[recursor(f) for f in obj.funcList()],
            coefficients=[recursor(c) for c in obj.coefList()],
            binpars=[recursor(p) for p in obj.binparsList()],  # TODO: optional?
        )


@RooWorkspace.register
@dataclass
class CMSHistFunc(Model):
    x: Model
    verticalParams: List[Model]
    verticalMorphsLo: List[List[float]]
    verticalMorphsHi: List[List[float]]
    bberrors: List[float]
    nominal: List[float]

    @classmethod
    def readobj(cls, obj, recursor):
        if len(obj.getHorizontalMorphs()):
            raise NotImplementedError("horizontal morphs from CMSHistFunc")
        morphs = [
            {
                "param": recursor(p),
                "lo": _fasthisto2list(obj.getShape(0, 0, i + 1, 0)),
                "hi": _fasthisto2list(obj.getShape(0, 0, i + 1, 1)),
            }
            for i, p in enumerate(obj.getVerticalMorphs())
        ]
        return cls(
            x=recursor(obj.getXVar()),
            verticalParams=[m["param"] for m in morphs],
            verticalMorphsLo=[m["lo"] for m in morphs],
            verticalMorphsHi=[m["hi"] for m in morphs],
            bberrors=_fasthisto2list(obj.errors()),
            nominal=_fasthisto2list(obj.getShape(0, 0, 0, 0)),
        )


@RooWorkspace.register
@dataclass
class SimpleGaussianConstraint(RooGaussian):
    blah: int = 1
    pass
    # combine implements a fast logpdf for this, hence the specializtion
