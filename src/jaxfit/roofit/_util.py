from functools import partial
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp

from jaxfit.types import Array


def _importROOT():
    try:
        import ROOT
    except ImportError:
        raise RuntimeError("This code path requires ROOT to be available")
    # TODO: check v6.22 at least
    return ROOT


class ParameterPack:
    def __init__(self):
        self._all = set()
        self._calls = {}
        self._finalized = False

    def arrayof(self, params: List[Union[float, str]]):
        if self._finalized:
            raise RuntimeError("you missed the bus")
        if not len(params):
            return lambda param: jnp.array([])
        if not all(isinstance(p, (float, str)) for p in params):
            raise RuntimeError
        if not all(p.startswith("RooRealVar:") for p in params if isinstance(p, str)):
            raise RuntimeError
        ref = id(params)
        self._calls[ref] = params
        self._all |= {p for p in params if not isinstance(p, float)}
        return partial(self._get, ref)

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        # TODO: use self._calls to make a smart choice here?
        self._flat = sorted(self._all)
        for ref in self._calls:
            old = self._calls[ref]
            idx = jnp.array(
                [-1 if isinstance(p, float) else self._flat.index(p) for p in old]
            )
            consts = jnp.array([p if isinstance(p, float) else 0.0 for p in old])
            self._calls[ref] = (idx, consts)

    def _get(self, ref, param):
        self.finalize()
        idx, consts = self._calls[ref]
        return jnp.where(idx >= 0, param[idx], consts)

    def ravel(self, params: Dict[str, float]):
        self.finalize()
        return jnp.array([params[p] for p in self._flat])

    def unravel(self, params: Array):
        self.finalize()
        return dict(zip(self._flat, params))

    def index(self, param: str):
        self.finalize()
        return self._flat.index(param)


class DataSlice:
    def __init__(self, parent: "DataSlice", cat: str):
        self._parent = parent
        self._cat = cat

    def slice(self, cat: str):
        return DataSlice(self, cat)

    def _array(self, obs: Union[List[str], Array], path: Tuple[str]):
        return self._parent._array(obs, (self._cat,) + path)

    def arrayof(self, obs: Union[List[str], Array]):
        """Return function to fetch data array from input

        obs can either be a list of observables (which will be columns in the
        return array), or it can be an array which encodes the edges of a 1D binning
        """
        return self._parent._array(obs, (self._cat,))


class DataPack(DataSlice):
    def __init__(self):
        self._auxdata = ParameterPack()
        self._slices = {}

    def _array(self, obs: Union[List[str], Array], path: Tuple[str]):
        if path in self._slices:
            raise RuntimeError(f"Data used twice! path: {path}")
        if path[0] == "aux":
            raise RuntimeError("aux is a protected name, please change category name")
        self._slices[path] = obs
        return lambda data: data[path]

    def arrayof(self, obs: List[str]):
        """Return function to fetch data array from input

        For DataPack (a top-level object) this can only be
        a list of observables, which will be assumed to be auxiliary
        data (i.e. they have one point) and the
        callable will return a single row vector
        """
        unpack = self._auxdata.arrayof(obs)
        return lambda data: unpack(data[("_aux_",)])

    def ravel(
        self, data: Dict[Tuple[str], Array], auxdata: Dict[str, float]
    ) -> Dict[Tuple[str], Array]:
        out = {("_aux_",): self._auxdata.ravel(auxdata)}
        # TODO check binnning? how to check that self._slices[path] matches
        # the observables in data? if binned, then actually we just want _weight_
        for k, v in self._slices.items():
            if isinstance(v, list):
                raise NotImplementedError("unbinned data")
            else:
                # truncate data to the necessary number of bins
                # TODO is this just combine weirdness or what?
                out[k] = data[k][: len(v) - 1]
        return out
