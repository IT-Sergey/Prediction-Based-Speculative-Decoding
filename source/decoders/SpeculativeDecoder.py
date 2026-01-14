from typing import List, Dict, Tuple, Optional

import numpy as np
import tqdm

from source.decoders.SingleDecoder import SingleDecoder
from source.decoders.SingleDecodingResult import SingleDecodingResult


class SpeculativeDecoder:
    def __init__(self, decoding_tree, speculation_width, predictor, chain_length_limit: Optional[int] = None):
        self._decoders = [SingleDecoder(decoding_tree) for _ in range(speculation_width + 1)]
        self._speculation_width = speculation_width
        self._predictor = predictor
        self._previous = None
        self._current_position = 0
        self._chain_length_limit: Optional[int] = chain_length_limit

    def flush(self):
        """Reset the decoder state."""
        self._current_position = 0
        self._previous = None

    def speculate(self, data: str, current_position: int) -> Tuple[List[SingleDecodingResult], int]:
        """
        Perform speculative decoding using predicted symbol lengths.

        Parameters
        ----------
        data : str
            The binary string to decode.
        current_position : int
            Current position in the bit string.

        Returns
        -------
            List of successfully decoded symbols in sequence.
            Number of truly guessed starting positions (without saturation effect).
        """
        if current_position >= len(data):
            return [], 0

        # Get predicted lengths for next symbols
        speculated_lengths = self._predictor.implicitly_predict(self._speculation_width)
        lengths = [0, *speculated_lengths]  # Include starting at position 0

        # Try decoding at each predicted offset
        results: Dict[int, SingleDecodingResult] = {}

        for i, length in enumerate(lengths):
            if current_position + length < len(data):
                res = self._decoders[i].decode(data, current_position + length)
                if res.was_decoded:
                    results[length] = res

        # Build sequence of consecutive successful decodings starting from position 0
        decoding_chain: List[SingleDecodingResult] = []
        truly_guessed = 0

        if 0 in results:
            current_offset = 0
            current_sequence = []

            while current_offset in results:
                result = results[current_offset]
                current_sequence.append(result)
                current_offset += result.length

            decoding_chain = current_sequence

            truly_guessed = 1 if results[0].length in speculated_lengths else 0

        if self._chain_length_limit:
            return decoding_chain[:self._chain_length_limit], truly_guessed

        return decoding_chain, truly_guessed

    def decode(self, data: str) -> Tuple[bytes, Tuple[int, int], np.array, np.array]:
        """
        Decode the entire bit string using speculative decoding.

        Parameters
        ----------
        data : str
            The binary string to decode.

        Returns
        -------
        Tuple[bytes, Tuple[int, int], np.array]
            Tuple containing:
            - decoded bytes
            - tuple of (successful_decodes, total_attempts)
            - number of commited codewords per each iteration
            - number of truly guessed starting positions per each iteration (without saturation effect)
        """
        tries = 0
        successes = 0
        decoded_symbols = []
        commited_codewords = []
        truly_predicted = []

        progress_bar = tqdm.tqdm(total=len(data), desc="Decoding")

        while self._current_position < len(data):
            tries += self._speculation_width + 1

            # Perform speculation at current position
            results, truly_predicted_local = self.speculate(data, self._current_position)

            if results:
                # We found a sequence of symbols through speculation
                successes += len(results)
                commited_codewords.append(len(results))
                truly_predicted.append(truly_predicted_local)

                # Update previous symbol for predictor (use first symbol)
                if results[0].symbol is not None:
                    self._previous = results[0].symbol

                # Add all symbols from the sequence to output
                for result in results:
                    if result.symbol is not None:
                        decoded_symbols.append(result.symbol)
                        self._predictor.feed(result.length)

                # Advance position by the total length of all symbols in sequence
                total_length = sum(result.length for result in results)
                self._current_position += total_length

                progress_bar.update(total_length)
            else:
                break

        return bytes(decoded_symbols), (successes, tries), commited_codewords, truly_predicted

    @property
    def current_position(self) -> int:
        """Get the current decoding position."""
        return self._current_position

    @property
    def speculation_width(self) -> int:
        """Get the speculation width."""
        return self._speculation_width
