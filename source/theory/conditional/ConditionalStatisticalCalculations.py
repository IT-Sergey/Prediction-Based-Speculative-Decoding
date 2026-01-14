from functools import cache

import numpy as np

from source.theory.conditional.ConditionalProbabilisticScheme import ConditionalProbabilisticScheme


class ConditionalStatisticalCalculations:
    def __init__(self, scheme: ConditionalProbabilisticScheme):
        self._probabilistic_scheme: ConditionalProbabilisticScheme = scheme

    def score(self, w, l_vec):
        return self.complete_expectation_sorted(w, tuple(sorted(l_vec)))

    @cache
    def expectation_sorted(self, w, l_vec, context):
        if w == 0: return 0

        res = 0
        for i in range(w):
            new_l = np.array(l_vec) - l_vec[i]
            new_context = [*context[1:], l_vec[i]]
            new_l = new_l[new_l>0]
            res += self._probabilistic_scheme.p(l_vec[i], tuple(context)) * (
                        1 + self.expectation_sorted(w - i - 1, tuple(new_l), tuple(new_context)))

        return res

    def complete_expectation_sorted(self, w, l_vec):
        res = 0
        for context, p in self._probabilistic_scheme.get_contexts_and_probabilities():
            res += p * self.expectation_sorted(w, l_vec, context)

        return res

    def complete_expectation_sorted_with_g(self, w, g):
        res = 0
        for context, p in self._probabilistic_scheme.get_contexts_and_probabilities():
            res += p * self.expectation_sorted(w, g(context), tuple(context))

        return res