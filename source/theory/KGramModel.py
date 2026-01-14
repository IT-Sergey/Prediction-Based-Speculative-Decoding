from collections import defaultdict
from typing import Optional, List

import tqdm


class KGramModel:
    def __init__(self, k: int):
        self.k: int = k
        self._frequencies = defaultdict(int)
        self._context: List[Optional[int]] = [None for _ in range(k)]
        self._is_ok = False
        self.alphabet = set()

        self._outcome_frequencies = defaultdict(int)

    @property
    def number_of_outcomes(self):
        return len(self._outcome_frequencies)

    def feed(self, symbol: int):
        self._context.pop(0)
        self._context.append(symbol)

        self._outcome_frequencies[symbol] += 1

        self.alphabet.add(symbol)

        if self._is_ok:
            self._frequencies[tuple(self._context)] += 1
        else:
            if None not in self._context:
                self._is_ok = True

    @property
    def frequencies(self):
        return self._frequencies

    def train(self, sample: List[int]):
        for s in tqdm.tqdm(sample):
            self.feed(s)

    def most_frequent_outcomes(self, n):
        return [r[0] for r in sorted(self._outcome_frequencies.items(), key=lambda x: x[1], reverse=True)[:n]]