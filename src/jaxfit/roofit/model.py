from dataclasses import dataclass, fields

import jax.numpy as jnp

from jaxfit.types import Array


def _fromdict(obj, deref):
    """Turn dict into Model"""
    if isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict) and not ("$ref" in obj or "$array" in obj):
        return {k: _fromdict(v, deref) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_fromdict(v, deref) for v in obj]
    elif isinstance(obj, dict) and "$ref" in obj:
        return deref(obj["$ref"][1:])
    elif isinstance(obj, dict) and "$array" in obj:
        return jnp.array(obj["$array"])
    raise RuntimeError(f"Unrecognized object in serialization: {obj}")


def _asdict(obj):
    """Turn Model into a dict

    This is used rather than using dataclasses.asdict
    because we want child models to be referenced rather
    than recursed.
    """
    if isinstance(obj, (str, int, float, bool)):
        return obj
    elif isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_asdict(v) for v in obj]
    elif isinstance(obj, Model):
        return {"$ref": f"/{obj.name}"}
    elif isinstance(obj, Array):
        return {"$array": obj.tolist()}
    raise RuntimeError(f"Unrecognized object in serialization: {obj}")


@dataclass
class Model:
    # The name is set by the containing RooWorkspace
    # name: str = None

    def to_dict(self):
        return {k.name: _asdict(getattr(self, k.name)) for k in fields(self)}

    @classmethod
    def from_dict(cls, d, deref):
        out = cls(**_fromdict(d, deref))
        return out
