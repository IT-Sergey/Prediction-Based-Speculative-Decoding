from collections import defaultdict

import tqdm

from source.theory.KGramModel import KGramModel


class ConditionalProbabilisticScheme:
    def __init__(self, k_gram: KGramModel):
        self._model: KGramModel = k_gram

        self._map = defaultdict(lambda: defaultdict(float))
        _context_count = defaultdict(int)
        self._context_probability = defaultdict(float)

        total_contexts = 0

        for k_gram, frequency in k_gram.frequencies.items():
            self._map[k_gram[:-1]][k_gram[-1]] += frequency
            _context_count[k_gram[:-1]] += frequency
            total_contexts += frequency

        self._map = dict(self._map)

        for context in self._map:
            for outcome in self._map[context]:
                self._map[context][outcome] /= _context_count[context]
            self._context_probability[context] = _context_count[context] / total_contexts

    def p(self, outcome, context):
        if context in self._map:
            return self._map[context][outcome]
        return 0

    def p_context(self, context):
        return self._context_probability[context]

    def get_contexts_and_probabilities(self):
        return self._context_probability.items()

    def most_frequent_outcomes(self, n):
        return self._model.most_frequent_outcomes(n)

    def most_frequent_outcomes_on_context(self, n, context):
        return sorted(self._map[context].items(), key=lambda x: x[1], reverse=True)[:n]

    def build_assignment_table(self, n: int):
        outcomes = self.most_frequent_outcomes(self._model.number_of_outcomes)
        weighted_map_p = 0
        assignment_table = {}
        for context in tqdm.tqdm(self._map):
            mfo = self.most_frequent_outcomes_on_context(n, context)
            vec = [r[0] for r in mfo]
            weighted_map_p += max([r[1] for r in mfo])*self._context_probability[context]
            if len(vec) < n:
                for outcome in outcomes:
                    if outcome not in vec:
                        vec.append(outcome)

                    if len(vec) == n:
                        break
            assignment_table[tuple(context)] = tuple(sorted(vec))
        return assignment_table

