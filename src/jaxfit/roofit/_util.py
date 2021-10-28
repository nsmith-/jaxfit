from collections import defaultdict
from functools import partial
from typing import Dict, List, Union

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
