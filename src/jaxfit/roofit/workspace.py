import gzip
import json
from typing import Any, Type

from jaxfit.roofit.model import Model


class RooWorkspace:
    models = {}

    @classmethod
    def register(cls, model: Type):
        cls.models[model.__name__] = model
        return model

    def __init__(self, models=None, roots=None):
        self._inputobj = {}
        self._out = {} if models is None else models
        self._roots = [] if roots is None else roots
        self._unknown_classes = set()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_inputobj")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def getref(self, obj: Any) -> str:
        """
        Get a guaranteed unique name to reference this object by
        """
        name = obj.Class().GetName() + ":" + obj.GetName()
        # JSONPointer escape
        name = name.replace("~", "~0").replace("/", "~1")
        if name in self._inputobj:
            cand = self._inputobj[name]
            if cand is obj:
                return name
            elif isinstance(cand, list):
                if obj is cand[0]:
                    return name
                for i, o in enumerate(cand[1:]):
                    if o is obj:
                        return f"{name};{i+1}"
                cand.append(obj)
                return f"{name};{i+1}"
            else:
                self._inputobj[name] = [cand]
                return name
        self._inputobj[name] = obj
        return name

    def readobj(self, obj: Any):
        self._roots.append(self._readobj(obj))

    @classmethod
    def from_file(cls, fname: str):
        if fname.endswith(".gz"):
            with gzip.open(fname, "rt") as fin:
                data = json.load(fin)
        else:
            with open(fname, "r") as fin:
                data = json.load(fin)

        roots = data.pop("_roots")
        out = {}

        def deref(name):
            try:
                return out[name]
            except KeyError:
                pass
            clsname = name.split(":")[0]
            try:
                objclass = cls.models[clsname]
            except KeyError:
                raise RuntimeError(f"Unknown Model class {clsname} in file {fname}")

            model = objclass.from_dict(data.pop(name), deref)
            model.name = name
            out[name] = model
            return model

        for name in roots:
            deref(name)
        return cls(models=out, roots=[out[name] for name in roots])

    def to_file(self, fname: str):
        data = {name: model.to_dict() for name, model in self._out.items()}
        data["_roots"] = [r.name for r in self._roots]
        if fname.endswith(".gz"):
            with gzip.open(fname, "wt") as fout:
                json.dump(data, fout)
        else:
            with open(fname, "w") as fout:
                json.dump(data, fout)

    def _readobj(self, obj: Any) -> Model:
        name = self.getref(obj)
        try:
            return self._out[name]
        except KeyError:
            pass
        objclass = obj.Class().GetName()

        try:
            model = self.models[objclass].readobj(obj, self._readobj)
        except KeyError:
            self._unknown_classes.add(objclass)
            model = self.models["_Unknown"].readobj(obj, self._readobj)

        model.name = name
        self._out[name] = model
        return model
