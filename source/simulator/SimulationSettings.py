from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class SimulationSettings:
    speculation_width: int
    training_files: List[Path | str]
    test_files: List[Path | str]
    training_dataset_size: Optional[int]
    test_dataset_size: Optional[int]
    chain_length_limit: Optional[int]
    use_same_dataset: bool
