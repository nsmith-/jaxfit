"""Custom RooFit objects found in CMS combine
"""
from dataclasses import dataclass
from typing import Dict, List

import jax.numpy as jnp

from jaxfit.roofit.common import RooCategory, RooGaussian
from jaxfit.roofit.model import Model
from jaxfit.roofit.workspace import RooWorkspace
from jaxfit.types import Array


def _fasthisto2array(h):
    return jnp.array([h[i] for i in range(h.size())])


@RooWorkspace.register
@dataclass
class RooSimultaneousOpt(Model):
    indexCat: RooCategory
    components: Dict[str, Model]

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
    verticalMorphsLo: Array  # 2d: (param, bin)
    verticalMorphsHi: Array  # 2d: (param, bin)
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
            bberrors=_fasthisto2array(obj.errors()),
            nominal=_fasthisto2array(obj.getShape(0, 0, 0, 0)),
        )


@RooWorkspace.register
@dataclass
class SimpleGaussianConstraint(RooGaussian):
    # combine implements a fast logpdf for this, hence the specializtion
    pass
