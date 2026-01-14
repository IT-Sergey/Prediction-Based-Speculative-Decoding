from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DecodingStatistics:
    """Statistics for decoding operations."""
    total_decodes: int = 0
    successful_decodes: int = 0
    failed_decodes: int = 0
    bits_processed: int = 0
    single_symbol_decodes: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate the success rate of decoding operations."""
        if self.total_decodes == 0:
            return 0.0
        return self.successful_decodes / self.total_decodes

    @property
    def average_bits_per_decode(self) -> float:
        """Calculate average bits processed per decode operation."""
        if self.total_decodes == 0:
            return 0.0
        return self.bits_processed / self.total_decodes

    def reset(self) -> None:
        """Reset all statistics to zero."""
        self.total_decodes = 0
        self.successful_decodes = 0
        self.failed_decodes = 0
        self.bits_processed = 0
        self.single_symbol_decodes = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary format."""
        return {
            'total_decodes': self.total_decodes,
            'successful_decodes': self.successful_decodes,
            'failed_decodes': self.failed_decodes,
            'bits_processed': self.bits_processed,
            'single_symbol_decodes': self.single_symbol_decodes,
            'success_rate': self.success_rate,
            'average_bits_per_decode': self.average_bits_per_decode,
        }

    def __str__(self) -> str:
        """String representation of statistics."""
        return (f"DecodingStatistics("
                f"success_rate={self.success_rate:.1%}, "
                f"total={self.total_decodes}, "
                f"avg_bits={self.average_bits_per_decode:.2f}")
