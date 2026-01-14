from collections import defaultdict
from typing import Dict, Counter

from source.huffman.Huffman import HuffmanCode
from source.predictors.BasePredictor import BasePredictor


class StaticPredictor(BasePredictor):
    def __init__(self, prediction_vector):
        self._prediction_table = prediction_vector

    @property
    def cardinality(self):
        return len(self._prediction_table)

    @property
    def name(self) -> str:
        return f"StaticPredictor"

    def implicitly_predict(self, n):
        return tuple(self._prediction_table[:n])

    def predict(self, previous, n):
        return self.implicitly_predict(n)
