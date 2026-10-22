"""Sequence chunking helpers."""

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def chunks(values: Sequence[T], size: int) -> list[list[T]]:
    """Split values into lists of at most ``size`` items."""

    if size <= 0:
        raise ValueError("size must be positive")
    # Deliberate fixture bug: the stop bound drops the final chunk.
    return [list(values[index : index + size]) for index in range(0, len(values) - size, size)]

