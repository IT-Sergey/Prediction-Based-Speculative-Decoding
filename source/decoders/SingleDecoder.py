from typing import Dict, Any

from source.decoders.DecodingStatistics import DecodingStatistics
from source.decoders.SingleDecodingResult import SingleDecodingResult


class SingleDecoder:
    def __init__(self, decoding_tree):
        """
        Initialize the SingleDecoder with a Huffman decoding tree.

        Parameters
        ----------
        decoding_tree : Node
            The root node of the Huffman tree used for decoding.
        """
        self.decoding_tree = decoding_tree
        self.current_node = decoding_tree
        self._single_symbol = decoding_tree.symbol if decoding_tree and decoding_tree.symbol is not None else None
        self._stats = DecodingStatistics()

    def decode(self, data: str, position: int = 0) -> SingleDecodingResult:
        """
        Decode a single symbol from the bit string (stateless operation).

        This method resets the internal state after each call, making it
        suitable for random access decoding.

        Parameters
        ----------
        data : str
            The binary string containing Huffman-encoded data.
        position : int, optional
            The starting position in the bit string to begin decoding. Default is 0.

        Returns
        -------
        SingleDecodingResult
            The result of the decoding operation.
        """
        decoded_result = self._decode(data, position)

        self._reset_state()
        self._update_stats(decoded_result)
        return decoded_result

    def _decode(self, data: str, position: int = 0) -> SingleDecodingResult:
        """
        Attempt to decode a single symbol from the bit string starting at the given position.

        Parameters
        ----------
        data : str
            The binary string containing Huffman-encoded data.
        position : int, optional
            The starting position in the bit string to begin decoding. Default is 0.

        Returns
        -------
        SingleDecodingResult
            A dataclass containing:
            - was_decoded: True if a symbol was successfully decoded, False otherwise
            - symbol: The decoded symbol if successful, None otherwise
            - length: The number of bits consumed (0 if no decoding occurred)
        """
        if not data or position >= len(data) or self.decoding_tree is None:
            return SingleDecodingResult(was_decoded=False, symbol=None, length=0)

        # Handle single symbol tree case
        if self._single_symbol is not None:
            if position < len(data):
                return SingleDecodingResult(
                    was_decoded=True,
                    symbol=self._single_symbol,
                    length=1
                )
            else:
                return SingleDecodingResult(was_decoded=False, symbol=None, length=0)

        current_node = self.decoding_tree
        bits_consumed = 0

        # Traverse the tree until we find a leaf node
        for i in range(position, len(data)):
            bit = data[i]
            bits_consumed += 1

            if bit == '0':
                if current_node.left is None:
                    return SingleDecodingResult(was_decoded=False, symbol=None, length=bits_consumed)
                current_node = current_node.left
            elif bit == '1':
                if current_node.right is None:
                    return SingleDecodingResult(was_decoded=False, symbol=None, length=bits_consumed)
                current_node = current_node.right
            else:
                # Invalid bit character
                return SingleDecodingResult(was_decoded=False, symbol=None, length=bits_consumed)

            # If we reached a leaf node, return success
            if current_node.symbol is not None:
                return SingleDecodingResult(
                    was_decoded=True,
                    symbol=current_node.symbol,
                    length=bits_consumed
                )

        # Reached end of data without finding a complete symbol
        return SingleDecodingResult(was_decoded=False, symbol=None, length=bits_consumed)

    def _reset_state(self) -> None:
        """
        Reset the decoder state to the initial tree root.
        """
        self.current_node = self.decoding_tree

    def _update_stats(self, result: SingleDecodingResult) -> None:
        """
        Update statistics based on decoding result.

        Parameters
        ----------
        result : SingleDecodingResult
            The result of the decoding operation.
        """
        self._stats.total_decodes += 1
        self._stats.bits_processed += result.length

        if result.was_decoded:
            self._stats.successful_decodes += 1
            if result.length == 1 and self._single_symbol is not None:
                self._stats.single_symbol_decodes += 1
        else:
            self._stats.failed_decodes += 1

    @property
    def statistics(self) -> DecodingStatistics:
        """
        Get a copy of the current decoding statistics.

        Returns
        -------
        DecodingStatistics
            Current statistics for decoding operations.
        """
        return DecodingStatistics(
            total_decodes=self._stats.total_decodes,
            successful_decodes=self._stats.successful_decodes,
            failed_decodes=self._stats.failed_decodes,
            bits_processed=self._stats.bits_processed,
            single_symbol_decodes=self._stats.single_symbol_decodes,
        )

    def reset_statistics(self) -> None:
        """
        Reset all decoding statistics to zero.
        """
        self._stats.reset()

    def get_statistics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of decoding statistics as a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing statistical summary.
        """
        return self._stats.to_dict()
