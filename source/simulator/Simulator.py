from typing import Counter

import numpy as np

from source.decoders.SpeculativeDecoder import SpeculativeDecoder
from source.huffman.Huffman import HuffmanCode
from source.predictors.BasePredictor import BasePredictor
from source.readers.BinaryReader import BinaryReader
from source.simulator.SimulationResult import SimulationResult
from source.simulator.SimulationSettings import SimulationSettings


class Simulator:
    def __init__(self, predictor: BasePredictor, decoder_type, settings: SimulationSettings):
        self.predictor: BasePredictor = predictor
        self.decoder_type = decoder_type
        self.settings: SimulationSettings = settings

        self.full_dataset = self.settings.test_files + self.settings.training_files

    def _get_train_dataset(self):
        return self.settings.training_files if not self.settings.use_same_dataset else self.full_dataset

    def _get_test_dataset(self):
        return self.settings.test_files if not self.settings.use_same_dataset else self.full_dataset

    def read_dataset(self, files, size_limit):
        return BinaryReader(files).read(size_limit)

    def get_huffman_data(self, dataset, size_limit):
        frequencies, content = self.read_dataset(dataset, size_limit)
        huffman_code = HuffmanCode.generate(frequencies)

        return content, huffman_code

    def train_predictor(self, dataset, size_limit):
        content, huffman_code = self.get_huffman_data(dataset, size_limit)

        lengths, length_map, max_len = HuffmanCode.get_bit_lengths(content, huffman_code)

        self.predictor.train_on_data(lengths)

    def simulate_with_train(self):
        self.train_predictor(self._get_train_dataset(), self.settings.training_dataset_size)

        content, huffman_code = self.get_huffman_data(self.settings.test_files, self.settings.test_dataset_size)

        encoded_data = HuffmanCode.encode(content, huffman_code)
        decoding_tree = HuffmanCode.build_decoding_tree(huffman_code)

        sd = SpeculativeDecoder(decoding_tree,
                                speculation_width=self.settings.speculation_width,
                                predictor=self.predictor,
                                chain_length_limit=self.settings.chain_length_limit)
        return sd.decode(encoded_data)

    def simulate_without_train(self):
        content, huffman_code = self.get_huffman_data(self._get_test_dataset(), self.settings.test_dataset_size)
        self.predictor.reconstruct(content)

        encoded_data = HuffmanCode.encode(content, huffman_code)
        decoding_tree = HuffmanCode.build_decoding_tree(huffman_code)

        sd = SpeculativeDecoder(decoding_tree,
                                speculation_width=self.settings.speculation_width,
                                predictor=self.predictor,
                                chain_length_limit=self.settings.chain_length_limit)
        return sd.decode(encoded_data)

    def simulate(self) -> SimulationResult:
        if self.predictor.requires_training:
            result = self.simulate_with_train()
        else:
            result = self.simulate_without_train()

        _, (successes, tries), commited_codewords, truly_guessed = result

        return SimulationResult(
            successful_decodes=successes,
            total_attempts=tries,
            mean_codewords=np.mean(commited_codewords),
            min_codewords=np.min(commited_codewords),
            max_codewords=np.max(commited_codewords),
            std_codewords=np.std(commited_codewords),
            codeword_throughput_counts=Counter(commited_codewords),
            speculation_width=self.settings.speculation_width,
            predictor_name=self.predictor.name,
            mean_truly_guesses=np.mean(truly_guessed),
            predictor_cardinality=self.predictor.cardinality,
        )
