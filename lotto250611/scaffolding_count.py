from collections import Counter
from typing import Iterable, Dict


def build_number_counts(draws: Iterable[Iterable[int]], max_number: int = 42) -> Dict[int, int]:
    """Build frequency counts for lottery numbers.

    Args:
        draws: Iterable of draws, each containing lottery numbers.
        max_number: Highest possible lottery number.

    Returns:
        Dictionary mapping each number to its occurrence count across all draws.
    """
    counter: Counter[int] = Counter()
    for draw in draws:
        counter.update(draw)
    return {num: counter.get(num, 0) for num in range(1, max_number + 1)}
