import math

import numpy as np

from source.decoders.SpeculativeDecoder import SpeculativeDecoder
from source.huffman.Huffman import HuffmanCode
from source.predictors.DynamicPredictor import DynamicPredictor
from source.readers.BinaryReader import BinaryReader
from source.theory.KGramModel import KGramModel
from source.theory.conditional.ConditionalProbabilisticScheme import ConditionalProbabilisticScheme
from source.theory.conditional.ConditionalStatisticalCalculations import ConditionalStatisticalCalculations

if __name__ == '__main__':
    files = [
        "../../datasets/books/anna_karenina.txt"
    ]
    speculation_width = 2
    model_order = 3

    precision = 5

    print("=" * 50)
    print("READING INPUT FILES")
    print("=" * 50)

    frequencies, data = BinaryReader(files).read()
    huffman_codes = HuffmanCode.generate(frequencies)

    lengths, length_map, max_len = HuffmanCode.get_bit_lengths(data, huffman_codes)

    k_plus_1_gram_model = KGramModel(model_order + 1)
    k_plus_1_gram_model.train(lengths)
    probabilistic_scheme = ConditionalProbabilisticScheme(k_plus_1_gram_model)
    calculator = ConditionalStatisticalCalculations(probabilistic_scheme)

    initial_context = tuple(probabilistic_scheme.most_frequent_outcomes(model_order))
    backup_variant = sorted(tuple(probabilistic_scheme.most_frequent_outcomes(speculation_width)))

    assignment_table = probabilistic_scheme.build_assignment_table(speculation_width)

    expected_score = calculator.complete_expectation_sorted_with_g(speculation_width, lambda c: assignment_table[c])
    expected_total_score = 1 + expected_score

    print("-" * 50)
    print(f"{len(files)} files are processed")
    print(f"w = {speculation_width}")
    print(f"expected total rate     = {round(expected_total_score, precision)}")

    print("=" * 50)
    print("SIMULATION STARTED")
    print("=" * 50)

    encoded_data = HuffmanCode.encode(data, huffman_codes)
    decoding_tree = HuffmanCode.build_decoding_tree(huffman_codes)

    print(f"\noriginal size           = {len(data)}")
    print(f"compressed size         = {math.ceil(len(encoded_data) / 8)} (bytes)\t{len(encoded_data)} (bits)")
    print(f"compression ratio       = {round(8 * len(data) / len(encoded_data), precision)}", flush=True)

    predictor = DynamicPredictor(assignment_table, initial_context, backup_variant)

    sd = SpeculativeDecoder(decoding_tree,
                            speculation_width=speculation_width,
                            predictor=predictor,
                            chain_length_limit=None)

    decoded_data, (successes, tries), commited_codewords, truly_guessed = sd.decode(encoded_data)
    real_score = np.mean(commited_codewords)

    print(f"decoded correctly       = {decoded_data == data}")
    print(f"real decoding rate      = {round(real_score, precision)}")
