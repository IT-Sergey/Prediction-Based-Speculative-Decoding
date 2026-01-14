from collections import Counter
from dataclasses import dataclass


@dataclass
class SimulationResult:
    speculation_width: int
    predictor_name: str
    successful_decodes: int
    mean_truly_guesses: float
    total_attempts: int
    mean_codewords: int
    min_codewords: int
    max_codewords: int
    std_codewords: float
    codeword_throughput_counts: Counter
    predictor_cardinality: int
