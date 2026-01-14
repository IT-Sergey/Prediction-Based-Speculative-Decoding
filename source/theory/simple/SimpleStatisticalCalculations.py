from functools import cache

import numpy as np

from source.theory.simple.SimpleProbabilisticScheme import SimpleProbabilisticScheme


class SimpleStatisticalCalculations:
    def __init__(self, scheme: SimpleProbabilisticScheme):
        self._probabilistic_scheme: SimpleProbabilisticScheme = scheme
    def score(self, w, l_vec):
        return self.expectation_sorted(w, tuple(sorted(l_vec)))

    @cache
    def expectation_sorted(self, w, l_vec):
        if w == 0: return 0
        res = 0

        for i in range(w):
            new_l_vec = tuple((np.array(l_vec) - l_vec[i])[i + 1:])
            res += self._probabilistic_scheme.p(l_vec[i]) * (1 + self.expectation_sorted(w - (i + 1), new_l_vec))
        return res

    @cache
    def delta_sorted(self, w, l_vec):
        if w == 1:
            return self.expectation_sorted(1, l_vec)

        res = self._probabilistic_scheme.p(l_vec[w - 1])
        for i in range(w - 1):
            new_l_vec = tuple((np.array(l_vec) - l_vec[i])[i + 1:])
            tmp = self.delta_sorted(w - (i + 1), new_l_vec)
            res += self._probabilistic_scheme.p(l_vec[i]) * tmp
        return res