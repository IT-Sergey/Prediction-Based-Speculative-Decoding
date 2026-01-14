from collections import defaultdict, Counter

import numpy as np

from source.decoders.SpeculativeDecoder import SpeculativeDecoder
from source.huffman.Huffman import HuffmanCode
from source.optimizers.SimpleOptimizer import SimpleOptimizer
from source.optimizers.VariantsGenerator import VariantsGenerator
from source.predictors.StaticPredictor import StaticPredictor
from source.readers.BinaryReader import BinaryReader
from source.theory.KGramModel import KGramModel
from source.theory.conditional.ConditionalProbabilisticScheme import ConditionalProbabilisticScheme
from source.theory.conditional.ConditionalStatisticalCalculations import ConditionalStatisticalCalculations
from source.theory.simple.SimpleProbabilisticScheme import SimpleProbabilisticScheme
from source.theory.simple.SimpleStatisticalCalculations import SimpleStatisticalCalculations

if __name__ == '__main__':
    files = [
        "../../datasets/books/anna_karenina.txt"
    ]
    speculation_width = 2  # From 1 to 5
    use_real_probabilities = True  # True and False. In NOT used by baseline.
    use_baseline = True  # True and False

    future_r = 1 # r-parameter for the `future prediction`.

    precision = 4

    print("=" * 50)
    print("READING INPUT FILES")
    print("=" * 50)

    frequencies, data = BinaryReader(files).read()
    huffman_codes = HuffmanCode.generate(frequencies)

    lengths, length_map, max_len = HuffmanCode.get_bit_lengths(data, huffman_codes)

    real_length_probabilities = Counter(lengths)
    for l in real_length_probabilities:
        real_length_probabilities[l] /= len(data)

    # Estimating probabilities from code (if needed).
    length_probabilities = defaultdict(float)

    if use_real_probabilities:
        length_probabilities = real_length_probabilities
    else:
        for c in huffman_codes:
            length_probabilities[len(huffman_codes[c])] += 2 ** (-len(huffman_codes[c]))

    length_probabilities = sorted(length_probabilities.items(), key=lambda x: x[1], reverse=True)
    codeword_lengths = [r[0] for r in length_probabilities]
    codeword_lengths_probabilities = [r[1] for r in length_probabilities]

    probabilistic_scheme = SimpleProbabilisticScheme(codeword_lengths, codeword_lengths_probabilities)
    stat_calculations = SimpleStatisticalCalculations(probabilistic_scheme)

    k_plus_1_gram_model = KGramModel(1 + 1)
    k_plus_1_gram_model.train(lengths)
    probabilistic_scheme_k = ConditionalProbabilisticScheme(k_plus_1_gram_model)
    k_calculator = ConditionalStatisticalCalculations(probabilistic_scheme_k)

    variant_generator = VariantsGenerator(codeword_lengths)
    # variant_generator = VariantsGenerator(codeword_lengths[:10) # Not all, but good part

    if use_baseline:
        vector = variant_generator.get_baseline_set(speculation_width)
        expected_speculation_rate = stat_calculations.expectation_sorted(speculation_width, vector)
    else:

        print("=" * 50)
        print("FINDING OPTIMAL L-VECTOR")
        print("=" * 50)

        optimizer = SimpleOptimizer(probabilistic_scheme)

        variants = variant_generator.generate_v_set(future_r, speculation_width, display_tqdm=False)
        expected_speculation_rate, vector = optimizer.optimize(speculation_width, variants, display_tqdm=False)

    expected_full_rate = expected_speculation_rate + 1
    expected_speculation_rate_at_k_model = 1 + k_calculator.complete_expectation_sorted(speculation_width, vector)

    print("-" * 50)
    print(f"{len(files)} files are processed")
    print(f"use real probabilities  = {use_real_probabilities}")
    print(f"use baseline approach   = {use_baseline}")

    print("-" * 50)
    print(f"selected L = {vector}")
    print(f"expected total rate     = {round(expected_full_rate, precision)}")
    print(f"expected total rate [k] = {round(expected_speculation_rate_at_k_model, precision)}")

    encoded_data = HuffmanCode.encode(data, huffman_codes)
    decoding_tree = HuffmanCode.build_decoding_tree(huffman_codes)

    print(f"\ncompression ratio       = {round(8 * len(data) / len(encoded_data), precision)}", flush=True)

    predictor = StaticPredictor(vector)

    sd = SpeculativeDecoder(decoding_tree,
                            speculation_width=speculation_width,
                            predictor=predictor,
                            chain_length_limit=None)

    decoded_data, (successes, tries), commited_codewords, truly_guessed = sd.decode(encoded_data)
    real_decoding_rate = np.mean(commited_codewords)

    efficiency = successes / tries

    print(f"decoded correctly       = {decoded_data == data}")
    print(f"real decoding rate      = {round(real_decoding_rate, precision)}")
