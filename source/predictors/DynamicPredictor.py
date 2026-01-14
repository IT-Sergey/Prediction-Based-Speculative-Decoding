from collections import defaultdict
from typing import Dict, Counter

from source.huffman.Huffman import HuffmanCode
from source.predictors.BasePredictor import BasePredictor


class DynamicPredictor(BasePredictor):
    def __init__(self, assignment_table, initial_context, backup_variant):
        self._prediction_table = assignment_table
        self._context = initial_context
        self._backup_variant = sorted(backup_variant)

    def feed(self, current):
        self._context = tuple([*self._context[1:], current])

    @property
    def cardinality(self):
        return len(self._prediction_table)

    @property
    def name(self) -> str:
        return f"DynamicPredictor"

    def implicitly_predict(self, n):
        if self._context in self._prediction_table:
            return tuple(self._prediction_table[self._context][:n])
        return self._backup_variant[:n]

    def predict(self, previous, n):
        self.feed(previous)
        return self.implicitly_predict(n)
