from collections import defaultdict
from functools import partial, reduce
from typing import Dict, List, Tuple, Union

import jax.numpy as jnp
import numpy as np

from jaxfit.types import Array


def _importROOT():
    try:
        import ROOT
    except ImportError:
        raise RuntimeError("This code path requires ROOT to be available")
    # TODO: check v6.22 at least
    return ROOT


def _parameter_sorter(allvars, calls):
    """greedy traveling salesman-ish"""
    presort = {n: i for i, n in enumerate(allvars)}
    adjacency = np.zeros(shape=(len(presort), len(presort)))
    for ref, count in calls.items():
        if isinstance(ref, str):
            continue
        pars = [p for p in ref if isinstance(p, str)]
        for i, j in zip(pars[:-1], pars[1:]):
            adjacency[presort[i], presort[j]] += count
    flat = []
    unvisited = set(range(len(presort)))
    while unvisited:
        i = min(unvisited)
        unvisited.remove(i)
        p = allvars[i]
        flat.append(p)
        while i >= 0:
            for inext in np.argsort(adjacency[i])[::-1]:
                if adjacency[i, inext] > 0 and inext in unvisited:
                    p = allvars[inext]
                    flat.append(p)
                    unvisited.remove(inext)
                    i = inext
                elif adjacency[i, inext] > 0:
                    continue
                else:
                    i = -1
                break
    return flat


class ParameterPack:
    def __init__(self):
        self._all = set()
        self._calls = {}
        self._finalized = False

    def arrayof(self, params: List[Union[float, str]], scalar=False):
        if self._finalized:
            raise RuntimeError("you missed the bus")
        if not len(params):
            return lambda param: jnp.array([])
        if scalar and len(params) > 1:
            raise ValueError("Can only be scalar if one parameter is fetched")
        if all(isinstance(p, float) for p in params):
            consts = params[0] if scalar else jnp.array(params)
            return lambda param: consts
        if not all(isinstance(p, (float, str)) for p in params):
            raise RuntimeError
        if not all(p.startswith("RooRealVar:") for p in params if isinstance(p, str)):
            raise RuntimeError
        ref = params[0] if scalar else tuple(params)
        if ref in self._calls:
            self._calls[ref] += 1
        else:
            self._calls[ref] = 1
            self._all |= {p for p in params if not isinstance(p, float)}
        return partial(self._get, ref)

    def finalize(self):
        if self._finalized:
            return
        self._finalized = True
        self._flat = sorted(self._all)
        if False:
            # seems slower in some cases?
            self._flat = _parameter_sorter(self._flat, self._calls)

        stats = defaultdict(int)
        for ref in self._calls:
            if isinstance(ref, str):
                self._calls[ref] = self._flat.index(ref)
                stats["scalar"] += 1
                continue
            take = jnp.array([self._flat.index(p) for p in ref if isinstance(p, str)])
            if jnp.all(jnp.diff(take) == 1):
                take = slice(int(take[0]), int(take[-1]) + 1)
                stats["contiguous take"] += 1
            if all(isinstance(p, str) for p in ref):
                self._calls[ref] = (take,)
                stats["skip put"] += 1
                continue
            put = jnp.array([i for i, p in enumerate(ref) if isinstance(p, str)])
            if jnp.all(jnp.diff(put) == 1):
                put = slice(int(put[0]), int(put[-1]) + 1)
                stats["contiguous put"] += 1
            consts = jnp.array([p if isinstance(p, float) else 0.0 for p in ref])
            self._calls[ref] = (take, put, consts)
            stats["full"] += 1
        print(f"ParameterPack stats: {dict(stats)} ({len(self._calls)} calls)")

    def _get(self, ref, param):
        self.finalize()
        op = self._calls[ref]
        if isinstance(op, int):
            return param[op]
        elif len(op) == 1:
            (take,) = op
            return param[take]
        else:
            take, put, consts = op
            return consts.at[put].set(param[take])

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
        if path[0] == "_aux_":
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

    def ravel(self, data, auxdata: Dict[str, float]) -> Dict[Tuple[str], Array]:
        # NB data is RooDataSet but circular import.. need to move some of this
        out = {("_aux_",): self._auxdata.ravel(auxdata)}
        categories = data.categories()
        for k, v in self._slices.items():
            # TODO check that k matches the observables in data
            if len(k) != len(categories):
                raise RuntimeError("Expected data to have same categories as slice")
            cut = reduce(
                jnp.logical_and,
                (
                    data.points[col, :] == cat.labels.index(key)
                    for (col, cat), key in zip(categories, k)
                ),
            )
            catdata = data.points[:, cut]
            if isinstance(v, list):
                cols = []
                for colname in v:
                    for i, col in data.variables():
                        if col.name == colname:
                            cols.append(i)
                if len(cols) != len(v):
                    raise RuntimeError("missing columns")
                out[k] = catdata[tuple(cols), :]
            else:
                if len(data.variables()) != 2:
                    raise NotImplementedError("Binned data in multiple dimensions?")
                col, var = data.variables()[0]
                # at least for combine this ends up being bin centers
                # and for some reason isn't truncated
                centers = catdata[col, : len(v) - 1]
                if not jnp.all(centers == 0.5 * (v[1:] + v[:-1])):
                    raise RuntimeError("Data centers do not match expected binning")
                assert data.variables()[-1][1].name == "RooRealVar:_weight_"
                out[k] = catdata[-1, : len(v) - 1]
        return out
