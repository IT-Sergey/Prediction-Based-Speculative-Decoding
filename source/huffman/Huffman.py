import heapq
from typing import Any, Dict, List, Optional, Tuple


class Node:
    """
    A node in the Huffman tree.

    This class represents a node in the Huffman coding tree, which can be either
    a leaf node (containing a symbol) or an internal node (containing references
    to child nodes).

    Attributes
    ----------
    symbol : Any, optional
        The symbol stored in this node. None for internal nodes.
    frequency : int
        The frequency of the symbol (for leaf nodes) or the sum of frequencies
        of all symbols in the subtree (for internal nodes).
    left : Node, optional
        Left child node in the Huffman tree.
    right : Node, optional
        Right child node in the Huffman tree.
    """
    __slots__ = ('symbol', 'frequency', 'left', 'right')

    def __init__(self, symbol: Optional[Any], frequency: int) -> None:
        """
        Initialize a new Node.

        Parameters
        ----------
        symbol : Any, optional
            The symbol to store in this node. Use None for internal nodes.
        frequency : int
            The frequency count for this symbol or subtree.
        """
        self.symbol = symbol
        self.frequency = frequency
        self.left: Optional['Node'] = None
        self.right: Optional['Node'] = None

    def __lt__(self, other: 'Node') -> bool:
        """
        Compare nodes by frequency for heapq.

        This allows Node objects to be compared based on their frequency,
        enabling them to be used in a min-heap priority queue.

        Parameters
        ----------
        other : Node
            The other node to compare against.

        Returns
        -------
        bool
            True if this node's frequency is less than the other node's frequency.

        """
        return self.frequency < other.frequency


class HuffmanCode:
    """
    A class implementing Huffman coding compression algorithm.

    Huffman coding is a lossless data compression algorithm that assigns
    variable-length codes to input symbols, with shorter codes assigned to
    more frequent symbols.

    This class provides static methods for building Huffman trees, generating
    codes, encoding and decoding data, and analyzing compression performance.

    References
    ----------
    Huffman, D. A. (1952). "A Method for the Construction of Minimum-Redundancy Codes".

    """

    @staticmethod
    def build_huffman_tree(frequency_table: Dict[Any, int]) -> Optional[Node]:
        """
        Build a Huffman tree from a frequency table.

        The Huffman tree is constructed by repeatedly combining the two nodes
        with the lowest frequencies until only one node remains (the root).

        Parameters
        ----------
        frequency_table : Dict[Any, int]
            A dictionary mapping symbols to their frequencies.

        Returns
        -------
        Node or None
            The root node of the Huffman tree, or None if the frequency table is empty.

        Raises
        ------
        TypeError
            If frequency_table is not a dictionary.

        """
        if not isinstance(frequency_table, dict):
            raise TypeError("frequency_table must be a dictionary")

        if not frequency_table:
            return None
        if len(frequency_table) == 1:
            symbol = next(iter(frequency_table))
            return Node(symbol, frequency_table[symbol])

        # Build heap directly without intermediate list
        heap: List[Node] = [Node(sym, freq) for sym, freq in frequency_table.items()]
        heapq.heapify(heap)

        # Build Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)

            merged = Node(None, left.frequency + right.frequency)
            merged.left = left
            merged.right = right
            heapq.heappush(heap, merged)

        return heap[0]

    @staticmethod
    def generate_huffman_codes(
            node: Optional[Node],
            current_code: str = "",
            huffman_codes: Optional[Dict[Any, str]] = None
    ) -> Dict[Any, str]:
        """
        Generate Huffman codes by traversing the Huffman tree.

        This method performs a depth-first traversal of the Huffman tree to
        assign binary codes to each symbol. Left branches add '0' to the code,
        right branches add '1' to the code.

        Parameters
        ----------
        node : Node, optional
            The current node being processed in the tree traversal.
        current_code : str, optional
            The binary code accumulated so far in the traversal. Default is "".
        huffman_codes : Dict[Any, str], optional
            Dictionary to store the generated codes. If None, a new dict is created.

        Returns
        -------
        Dict[Any, str]
            A dictionary mapping symbols to their Huffman codes.

        """
        if huffman_codes is None:
            huffman_codes = {}

        if node is None:
            return huffman_codes

        # Leaf node - store the code
        if node.symbol is not None:
            huffman_codes[node.symbol] = current_code
            return huffman_codes

        # Internal node - traverse both children
        if node.left:
            HuffmanCode.generate_huffman_codes(node.left, current_code + "0", huffman_codes)
        if node.right:
            HuffmanCode.generate_huffman_codes(node.right, current_code + "1", huffman_codes)

        return huffman_codes

    @staticmethod
    def generate(symbol_frequencies: Dict[Any, int]) -> Dict[Any, str]:
        """
        Generate Huffman codes for given symbol frequencies.

        This is a convenience method that combines building the Huffman tree
        and generating the codes in one step.

        Parameters
        ----------
        symbol_frequencies : Dict[Any, int]
            A dictionary mapping symbols to their frequencies.

        Returns
        -------
        Dict[Any, str]
            A dictionary mapping symbols to their Huffman codes.

        """
        if not symbol_frequencies:
            return {}

        root = HuffmanCode.build_huffman_tree(symbol_frequencies)
        return HuffmanCode.generate_huffman_codes(root, "0" if len(symbol_frequencies) == 1 else "")

    @staticmethod
    def get_bit_lengths(
            data: List[Any],
            code: Dict[Any, str]
    ) -> Tuple[List[int], Dict[Any, int], int]:
        """
        Get bit lengths for each symbol in data based on Huffman codes.

        Parameters
        ----------
        data : List[Any]
            The input data sequence.
        code : Dict[Any, str]
            Huffman codes mapping symbols to binary strings.

        Returns
        -------
        Tuple[List[int], Dict[Any, int], int]
            A tuple containing:
            - List[int]: Bit lengths for each symbol in the input data
            - Dict[Any, int]: Mapping of symbols to their code lengths
            - int: Maximum code length found

        """
        if not data or not code:
            return [], {}, 0

        length_map: Dict[Any, int] = {}
        max_len = 0

        for char in data:
            if char in code:
                code_len = len(code[char])
                length_map[char] = code_len
                max_len = max(max_len, code_len)

        return [length_map.get(char, 0) for char in data], length_map, max_len

    @staticmethod
    def encode(data: List[Any], huffman_codes: Dict[Any, str]) -> str:
        """
        Encode data using Huffman codes.

        Parameters
        ----------
        data : List[Any]
            The input data to encode.
        huffman_codes : Dict[Any, str]
            Huffman codes mapping symbols to binary strings.

        Returns
        -------
        str
            The encoded binary string.

        Raises
        ------
        ValueError
            If a symbol in the data is not found in the Huffman codes.

        """
        if not data or not huffman_codes:
            return ""

        encoded_bits: List[str] = []
        for symbol in data:
            if symbol in huffman_codes:
                encoded_bits.append(huffman_codes[symbol])
            else:
                raise ValueError(f"Symbol '{symbol}' not found in Huffman codes")

        return "".join(encoded_bits)

    @staticmethod
    def decode(encoded_bits: str, huffman_tree: Optional[Node]) -> List[Any]:
        """
        Decode Huffman-encoded bits back to original symbols.

        Parameters
        ----------
        encoded_bits : str
            The binary string to decode.
        huffman_tree : Node, optional
            The root node of the Huffman tree used for decoding.

        Returns
        -------
        List[Any]
            The decoded symbols.

        """
        if not encoded_bits or huffman_tree is None:
            return []

        # Handle single symbol case
        if huffman_tree.symbol is not None:
            return [huffman_tree.symbol] * len(encoded_bits)

        decoded: List[Any] = []
        current_node: Node = huffman_tree

        for bit in encoded_bits:
            # Traverse the tree based on the bit
            if bit == '0':
                if current_node.left is None:
                    raise ValueError("Invalid encoded bits: cannot traverse left from current node")
                current_node = current_node.left
            else:  # bit == '1'
                if current_node.right is None:
                    raise ValueError("Invalid encoded bits: cannot traverse right from current node")
                current_node = current_node.right

            # If we reached a leaf node, add symbol and reset
            if current_node.symbol is not None:
                decoded.append(current_node.symbol)
                current_node = huffman_tree

        return decoded

    @staticmethod
    def build_decoding_tree(huffman_codes: Dict[Any, str]) -> Optional[Node]:
        """
        Build a decoding tree from Huffman codes.

        This method provides an alternative way to build a decoding tree
        when only the Huffman codes are available, not the original frequency table.

        Parameters
        ----------
        huffman_codes : Dict[Any, str]
            Huffman codes mapping symbols to binary strings.

        Returns
        -------
        Node or None
            The root node of the decoding tree, or None if no codes are provided.

        """
        if not huffman_codes:
            return None

        root = Node(None, 0)

        for symbol, code in huffman_codes.items():
            current: Node = root
            for bit in code:
                if bit == '0':
                    if current.left is None:
                        current.left = Node(None, 0)
                    current = current.left
                else:  # bit == '1'
                    if current.right is None:
                        current.right = Node(None, 0)
                    current = current.right
            current.symbol = symbol

        return root

    @staticmethod
    def get_compression_ratio(
            original_data: List[Any],
            encoded_bits: str,
            char_size: int = 8
    ) -> float:
        """
        Calculate compression ratio.

        The compression ratio is calculated as original_size / compressed_size.
        A higher ratio indicates better compression.

        Parameters
        ----------
        original_data : List[Any]
            The original uncompressed data.
        encoded_bits : str
            The encoded binary string.
        char_size : int, optional
            The assumed bit size of each character in the original data.
            Default is 8 (standard ASCII/UTF-8 character size).

        Returns
        -------
        float
            The compression ratio (original_size / compressed_size).

        """
        if not original_data or not encoded_bits:
            return 0.0

        original_bits = len(original_data) * char_size
        compressed_bits = len(encoded_bits)

        return original_bits / compressed_bits if compressed_bits > 0 else 0.0
