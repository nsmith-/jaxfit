import json
from typing import Any, Type

from jaxfit.roofit.model import Model


class RooWorkspace:
    models = {}

    @classmethod
    def register(cls, model: Type):
        cls.models[model.__name__] = model
        return model

    def __init__(self):
        self._inputobj = {}
        self._out = {}
        self._roots = []
        self._unknown_classes = set()

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

    def to_file(self, fname: str):
        with open(fname, "w") as fout:
            json.dump(
                {
                    name: json.loads(model.to_json())
                    for name, model in self._out.items()
                },
                fout,
            )

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
