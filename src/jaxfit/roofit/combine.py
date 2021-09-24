"""Custom RooFit objects found in CMS combine
"""
from jaxfit.roofit.workspace import RooWorkspace


def _fasthisto2list(h):
    return [h[i] for i in range(h.size())]


@RooWorkspace.register
class RooSimultaneousOpt:
    @classmethod
    def readobj(cls, obj, recursor):
        cat = obj.indexCat()
        return {
            "indexCat": recursor(cat),
            "components": {
                label: recursor(obj.getPdf(label)) for label, idx in cat.states()
            },
            # obj.extraConstraints()
            # obj.channelMasks()
        }


@RooWorkspace.register
class ProcessNormalization:
    @classmethod
    def readobj(cls, obj, recursor):
        return {
            "nominal": obj.getNominalValue(),
            "symParams": [recursor(p) for p in obj.getSymErrorParameters()],
            "symLogKappa": list(obj.getSymLogKappa()),
            "asymParams": [recursor(p) for p in obj.getAsymErrorParameters()],
            "asymLogKappa": [(lo, hi) for lo, hi in obj.getAsymLogKappa()],
            "additional": [recursor(p) for p in obj.getAdditionalModifiers()],
        }


@RooWorkspace.register
class CMSHistErrorPropagator:
    @classmethod
    def readobj(cls, obj, recursor):
        return {
            "x": recursor(obj.getX()),  # probably optional?
            "functions": [recursor(f) for f in obj.funcList()],
            "coefficients": [recursor(c) for c in obj.coefList()],
            "binpars": [recursor(p) for p in obj.binparsList()],  # TODO: optional?
        }


@RooWorkspace.register
class CMSHistFunc:
    @classmethod
    def readobj(cls, obj, recursor):
        if len(obj.getHorizontalMorphs()):
            raise NotImplementedError("horizontal morphs from CMSHistFunc")
        return {
            "x": recursor(obj.getXVar()),
            "vertical": [
                {
                    "param": recursor(p),
                    "effect_lo": _fasthisto2list(obj.getShape(0, 0, i + 1, 0)),
                    "effect_hi": _fasthisto2list(obj.getShape(0, 0, i + 1, 1)),
                }
                for i, p in enumerate(obj.getVerticalMorphs())
            ],
            "bberrors": _fasthisto2list(obj.errors()),
            "nominal": _fasthisto2list(obj.getShape(0, 0, 0, 0)),
        }


@RooWorkspace.register
class SimpleGaussianConstraint:
    @classmethod
    def readobj(cls, obj, recursor):
        return {
            "x": recursor(obj.getX()),
            "mean": recursor(obj.getMean()),
            "sigma": recursor(obj.getSigma()),
        }
