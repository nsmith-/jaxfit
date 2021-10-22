import gzip
import json
from typing import Any, Dict, List, Type

from jaxfit.roofit.model import Model


def _getref(seen: Dict[str, Any], obj: Any) -> str:
    """
    Get a guaranteed unique name to reference this object by
    """
    name = obj.Class().GetName() + ":" + obj.GetName()
    # JSONPointer escape
    name = name.replace("~", "~0").replace("/", "~1")
    if name in seen:
        cand = seen[name]
        if cand is obj:
            return name
        elif isinstance(cand, list):
            if obj is cand[0]:
                return name
            for i, o in enumerate(cand[1:]):
                if o is obj:
                    return f"{name};{i+1}"
            cand.append(obj)
            return f"{name};{len(cand)}"
        else:
            seen[name] = [cand]
            return name
    seen[name] = obj
    return name


class RooWorkspace:
    models = {}

    @classmethod
    def register(cls, model: Type):
        cls.models[model.__name__] = model
        return model

    def __init__(self, roots: Dict[str, Model] = None):
        self._roots = {} if roots is None else roots

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_inputobj")
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __getitem__(self, key):
        return self._roots[key]

    @classmethod
    def from_root(cls, items: List[Any]):
        """Import several items from a common ROOT RooWorkspace object"""
        roots, inputs, known = {}, {}, {}
        unknown = set()

        def readobj(obj: Any) -> Model:
            name = _getref(inputs, obj)
            try:
                return known[name]
            except KeyError:
                pass
            objclass = obj.Class().GetName()
            try:
                modelclass = cls.models[objclass]
            except KeyError:
                unknown.add(objclass)
                modelclass = cls.models["_Unknown"]

            model = modelclass.readobj(obj, readobj)
            model.name = name
            known[name] = model
            return model

        for item in items:
            model = readobj(item)
            roots[model.name] = model

        print(f"Unknown class types: {unknown}")
        return cls(roots=roots)

    @classmethod
    def from_file(cls, fname: str):
        if fname.endswith(".gz"):
            with gzip.open(fname, "rt") as fin:
                data = json.load(fin)
        else:
            with open(fname, "r") as fin:
                data = json.load(fin)

        roots = data.pop("_roots")
        allobj = {}

        def deref(name):
            name = name[1:]  # strip leading slash
            try:
                return allobj[name]
            except KeyError:
                pass
            clsname = name.split(":")[0]
            try:
                objclass = cls.models[clsname]
            except KeyError:
                raise RuntimeError(f"Unknown Model class {clsname} in file {fname}")

            model = objclass.from_dict(data.pop(name), deref)
            model.name = name
            allobj[name] = model
            return model

        out = {}
        for item in roots:
            obj = deref(item["$ref"])
            out[obj.name] = obj
        return cls(roots=out)

    def to_file(self, fname: str):
        out = {}

        def ref(obj):
            if obj.name not in out:
                out[obj.name] = obj.to_dict(ref)
            return f"/{obj.name}"

        out["_roots"] = [{"$ref": ref(obj)} for obj in self._roots.values()]
        if fname.endswith(".gz"):
            with gzip.open(fname, "wt") as fout:
                json.dump(out, fout)
        else:
            with open(fname, "w") as fout:
                json.dump(out, fout)
