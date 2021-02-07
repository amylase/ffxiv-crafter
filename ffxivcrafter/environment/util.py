from typing import TypeVar

T = TypeVar('T')


def clip(low: T, x: T, high: T) -> T:
    return min(high, max(low, x))