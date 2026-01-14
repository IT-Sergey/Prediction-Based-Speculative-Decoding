from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SingleDecodingResult:
    was_decoded: bool
    symbol: Optional[int]
    length: int
