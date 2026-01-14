import math
from itertools import combinations, combinations_with_replacement

import numpy as np
import tqdm


class VariantsGenerator:
    def __init__(self, lengths):
        self._lengths = lengths

    def _for_future(self, r, display_tqdm=False):
        total = math.comb(len(self._lengths) + r - 1, r)
        result = set()
        for c in tqdm.tqdm(combinations_with_replacement(self._lengths, r), total=total, disable=not display_tqdm):
            for s in np.cumsum(c):
                result.add(int(s))
        return result

    def generate_v_set(self, r, w, display_tqdm=False):
        variants = self._for_future(r)
        v_set = set()

        for c in tqdm.tqdm(combinations(variants, w), disable=not display_tqdm):
            v_set.add(tuple(sorted(c)))

        return v_set

    def get_baseline_set(self, n, j=1):
        def _get_j_set(j):
            if j == 0:
                return {0}

            res = set()

            for l_r in _get_j_set(j - 1):
                for l_base in self._lengths:
                    res.add(l_base + l_r)

            return sorted(list(res))

        result = set()
        for k in range(1, j + 1):
            result = result.union(_get_j_set(k))

        if len(result) < n:
            return self.get_baseline_set(n, j + 1)
        return tuple(sorted(result)[:n])
