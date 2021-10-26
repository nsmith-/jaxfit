from dataclasses import dataclass, fields

import numpy as np
import jax.numpy as jnp

from jaxfit.types import Array


def _node_visitor(obj, prefix):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _node_visitor(v, f"{prefix}['{k}']")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _node_visitor(v, f"{prefix}[{i}]")
    elif isinstance(obj, Model):
        for k in fields(obj):
            yield from _node_visitor(getattr(obj, k.name), f"{prefix}.{k.name}")
        yield prefix, obj


def _fromdict(obj, deref):
    """Turn dict into Model"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, dict) and not ("$ref" in obj or "$array" in obj):
        return {k: _fromdict(v, deref) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fromdict(v, deref) for v in obj]
    elif isinstance(obj, dict) and "$ref" in obj:
        return deref(obj["$ref"])
    elif isinstance(obj, dict) and "$array" in obj:
        # slightly faster to have numpy convert first
        return jnp.array(np.array(obj["$array"]))
    raise RuntimeError(f"Unrecognized object in serialization: {obj}")


def _asdict(obj, ref):
    """Turn Model into a dict

    This is used rather than using dataclasses.asdict
    because we want child models to be referenced rather
    than recursed.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    elif isinstance(obj, dict):
        return {k: _asdict(v, ref) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_asdict(v, ref) for v in obj]
    elif isinstance(obj, Model):
        return {"$ref": ref(obj)}
    elif isinstance(obj, Array):
        return {"$array": obj.tolist()}
    raise RuntimeError(f"Unrecognized object in serialization: {obj}")


@dataclass
class Model:
    @property
    def name(self):
        try:
            return self._name
        except AttributeError:
            self._name = f"{type(self).__name__}:anon{id(self):x}"
            return self._name

    @name.setter
    def name(self, name):
        if hasattr(self, "_name"):
            raise RuntimeError(f"Object already named! {self.name}")
        self._name = name

    def nodes(self):
        yield from _node_visitor(self, "")

    def to_dict(self, ref):
        return {k.name: _asdict(getattr(self, k.name), ref) for k in fields(self)}

    @classmethod
    def from_dict(cls, d, deref):
        out = cls(**_fromdict(d, deref))
        return out
