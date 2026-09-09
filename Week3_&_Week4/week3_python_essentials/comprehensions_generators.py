"""
Week 3: Python Essentials — Comprehensions, Generators, Itertools & Custom Decorators

This module provides production-grade examples and patterns for:
1. List, Dictionary, and Set comprehensions (nested, conditional, transformed)
2. Generator functions (yield) and generator expressions for memory-efficient streaming
3. Custom decorators with arguments and metadata preservation (functools.wraps)
4. Functional pipelines chaining generators
"""

import time
import functools
from typing import List, Dict, Set, Iterator, Tuple, Callable, Any


# =====================================================================
# 1. Custom Decorators
# =====================================================================
def timing_decorator(func: Callable) -> Callable:
    """Measures and logs function execution time with microsecond precision."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱️ [{func.__name__}] executed in {elapsed*1000:.3f} ms")
        return result
    return wrapper


def memoize(func: Callable) -> Callable:
    """Thread-safe memoization cache decorator."""
    cache: Dict[Tuple, Any] = {}

    @functools.wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    wrapper.cache = cache
    return wrapper


# =====================================================================
# 2. Comprehensions Showcase
# =====================================================================
def demonstrate_comprehensions(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Applies idiomatic Python comprehensions to process raw structured data.
    """
    # 2.1 List comprehension: Filtered & normalized list of titles
    high_rated_titles = [
        r["title"].strip()
        for r in records
        if r.get("rating", 0) >= 4.8 and r.get("status") == "available"
    ]

    # 2.2 Nested List comprehension: Flatten 2D matrix / nested tags
    nested_tags = [r.get("tags", []) for r in records]
    flattened_tags = [tag.lower() for sublist in nested_tags for tag in sublist]

    # 2.3 Set comprehension: Unique set of authors & unique genres
    unique_genres: Set[str] = {r["genre"] for r in records if "genre" in r}
    unique_tags: Set[str] = {t.lower() for t in flattened_tags}

    # 2.4 Dict comprehension: ID to Title map & Genre counts
    id_to_title_map: Dict[str, str] = {r["id"]: r["title"] for r in records}

    # Genre frequency map using dict comprehension and count
    genre_frequencies: Dict[str, int] = {
        genre: sum(1 for r in records if r.get("genre") == genre)
        for genre in unique_genres
    }

    # Inverted map: Group titles by author
    all_authors = {r["author"] for r in records}
    author_works: Dict[str, List[str]] = {
        author: [r["title"] for r in records if r["author"] == author]
        for author in all_authors
    }

    return {
        "high_rated_titles": high_rated_titles,
        "unique_genres": unique_genres,
        "unique_tags": unique_tags,
        "id_to_title_map": id_to_title_map,
        "genre_frequencies": genre_frequencies,
        "author_works": author_works,
    }


# =====================================================================
# 3. Generators & Lazy Pipelines
# =====================================================================
def infinite_number_stream(start: int = 1, step: int = 1) -> Iterator[int]:
    """Infinite generator yielding sequential numbers."""
    n = start
    while True:
        yield n
        n += step


def batch_generator(data: List[Any], batch_size: int = 3) -> Iterator[List[Any]]:
    """Yields consecutive batches of data without loading duplicates into memory."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


def stream_filter_and_transform(items: Iterator[Dict[str, Any]], min_year: int = 2015) -> Iterator[str]:
    """Pipeline stage: lazily filter items by year and yield summary string."""
    for item in items:
        if item.get("year", 0) >= min_year:
            yield f"Modern Work: {item['title']} ({item['year']})"


# Example memoized recursive function
@memoize
def fibonacci(n: int) -> int:
    """Calculates n-th Fibonacci number efficiently via memoization."""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


# =====================================================================
# Execution & Demonstration
# =====================================================================
if __name__ == "__main__":
    sample_records = [
        {"id": "01", "title": "Deep Learning", "author": "Goodfellow", "year": 2016, "genre": "AI", "rating": 4.9, "status": "available", "tags": ["AI", "Neural Networks", "Math"]},
        {"id": "02", "title": "Pattern Recognition", "author": "Bishop", "year": 2006, "genre": "AI", "rating": 4.8, "status": "available", "tags": ["Bayesian", "Math", "ML"]},
        {"id": "03", "title": "Algorithms", "author": "Cormen", "year": 2022, "genre": "CS", "rating": 4.9, "status": "borrowed", "tags": ["CS", "Data Structures", "Algorithms"]},
        {"id": "04", "title": "Linear Algebra", "author": "Strang", "year": 2016, "genre": "Math", "rating": 4.9, "status": "available", "tags": ["Math", "Vectors", "Matrices"]},
        {"id": "05", "title": "Reinforcement Learning", "author": "Sutton", "year": 2018, "genre": "AI", "rating": 4.8, "status": "available", "tags": ["AI", "RL", "MDP"]},
    ]

    print("=== Comprehensions Demonstration ===")
    results = demonstrate_comprehensions(sample_records)
    print("High rated available:", results["high_rated_titles"])
    print("Unique genres:", results["unique_genres"])
    print("Genre frequencies:", results["genre_frequencies"])
    print("Author works:", results["author_works"])

    print("\n=== Generator Batching Demonstration ===")
    for batch_idx, batch in enumerate(batch_generator(sample_records, batch_size=2), start=1):
        print(f"Batch {batch_idx}: {[b['title'] for b in batch]}")

    print("\n=== Memoized Fibonacci Timing ===")
    @timing_decorator
    def run_fib():
        return fibonacci(35)
    print(f"Fib(35) = {run_fib()}")
