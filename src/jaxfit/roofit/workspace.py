from typing import Any, Type


class RooWorkspace:
    models = {}

    @classmethod
    def register(cls, model: Type):
        cls.models[model.__name__] = model

    def __init__(self):
        self._inputobj = {}
        self._out = {
            "_roots": [],
        }
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

    def readobj(self, obj):
        ref = self._readobj(obj)
        self._out["_roots"].append(ref)

    def to_file(self, fname: str):
        import json

        with open(fname, "w") as fout:
            json.dump(self._out, fout)

    def _readobj(self, obj):
        name = self.getref(obj)
        try:
            return self._out[name]
        except KeyError:
            pass
        objclass = obj.Class().GetName()

        try:
            out = self.models[objclass].readobj(obj, self._readobj)
        except KeyError:
            self._unknown_classes.add(objclass)
            out = {"children": []}
            for child in obj.servers():
                out["children"].append(self._readobj(child))

        out["class"] = objclass
        self._out[name] = out
        return {"$ref": f"/{name}"}
