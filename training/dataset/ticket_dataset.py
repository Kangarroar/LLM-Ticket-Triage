"""Stub dataset wrapper for IT ticket fine-tuning data."""

from typing import Any, Dict, List


class TicketDataset:
    """Placeholder dataset class for training examples.

    Each sample is expected to map raw ticket text to structured JSON.
    """

    def __init__(self, examples: List[Dict[str, Any]] | None = None) -> None:
        self.examples = examples or []

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:  # pragma: no cover - trivial
        return self.examples[idx]


