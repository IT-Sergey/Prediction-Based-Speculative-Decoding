from collections import defaultdict
from typing import Dict, Counter

from source.huffman.Huffman import HuffmanCode
from source.predictors.BasePredictor import BasePredictor


class ZeroOrderProbabilitiesReconstructor(BasePredictor):
    def __init__(self):
        self._codes: Dict[int, int] = {}
        self._prediction_table = []

        self._prepare()

    @property
    def cardinality(self):
        return len(self._prediction_table)

    @property
    def name(self) -> str:
        return f"ZeroOrderProbabilitiesReconstructor"

    def reconstruct(self, data):
        frequencies = Counter(data)
        huffman_code = HuffmanCode.generate(frequencies)
        _, length_map, _ = HuffmanCode.get_bit_lengths(data, huffman_code)

        self._codes: Dict[int, int] = Counter(length_map.values())
        self._prediction_table = []

        self._prepare()

    def _prepare(self):
        self._form_prediction()

    @property
    def probabilistic_scheme(self):
        probabilistic_scheme = self._reconstruct_length_probabilities().items()
        probabilistic_scheme = [(l, p) for l, p in probabilistic_scheme]
        probabilistic_scheme = sorted(probabilistic_scheme, key=lambda x: x[1], reverse=True)
        return probabilistic_scheme

    def implicitly_predict(self, n):
        return self._prediction_table[:n]

    def predict(self, previous, n):
        return self.implicitly_predict(n)

    def _reconstruct_length_probabilities(self):
        probabilities = defaultdict(float)
        for length in self._codes:
            probabilities[length] += 2 ** (-length) * self._codes[length]

        return probabilities

    def _form_prediction(self):
        self._prediction_table = [r[0] for r in self.probabilistic_scheme]
