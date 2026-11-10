#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ЛР 4. MergeSort, CountingSort, RadixSort и HeapSort.

Вариант 10. Реализация, верификация, бенчмарк и графики.

Запуск:
    python lab04-mergesort-linear-heapsort.py --variant 10 --out results
"""
from __future__ import annotations

import argparse
import collections
import csv
import math
from pathlib import Path
import random
import statistics
import sys
import time
import traceback
from typing import Callable, TypeVar


T = TypeVar("T")


# ---------------------------------------------------------------------------
# 1. Сортировки
# ---------------------------------------------------------------------------


def merge(
    left: list[T], right: list[T], key: Callable[[T], object]
) -> list[T]:
    """Стабильно слить два отсортированных списка за O(n)."""
    result: list[T] = []
    i = 0
    j = 0

    while i < len(left) and j < len(right):
        # При равных ключах первым идёт элемент из левой половины.
        if key(left[i]) <= key(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result


def merge_sort(a: list[T], key: Callable[[T], object] = lambda x: x) -> list[T]:
    """Стабильный MergeSort: O(n log n) времени и O(n) доп. памяти."""
    if len(a) < 2:
        return list(a)
    middle = len(a) // 2
    left = merge_sort(a[:middle], key)
    right = merge_sort(a[middle:], key)
    return merge(left, right, key)


def _stable_counting_sort_items(
    items: list[T],
    key: Callable[[T], int],
    key_min: int,
    key_max: int,
) -> list[T]:
    """Общий устойчивый CountingSort по целочисленному ключу."""
    if not items:
        return []
    if key_max < key_min:
        raise ValueError("key_max должен быть не меньше key_min")

    width = key_max - key_min + 1
    counts = [0] * width
    for item in items:
        item_key = key(item)
        if item_key < key_min or item_key > key_max:
            raise ValueError(
                f"ключ {item_key} вне диапазона [{key_min}, {key_max}]"
            )
        counts[item_key - key_min] += 1

    # counts[i] станет числом элементов с ключом не больше i.
    for i in range(1, width):
        counts[i] += counts[i - 1]

    result: list[T] = [items[0]] * len(items)
    # Обход с конца сохраняет порядок равных ключей.
    for item in reversed(items):
        index = key(item) - key_min
        counts[index] -= 1
        result[counts[index]] = item
    return result


def counting_sort(a: list[int], key_max: int) -> list[int]:
    """Устойчивый CountingSort для ключей из [0, key_max].

    Время O(n + k), память O(n + k), где k = key_max + 1.
    """
    if key_max < 0:
        if a:
            raise ValueError("key_max не может быть отрицательным")
        return []
    return _stable_counting_sort_items(a, lambda value: value, 0, key_max)


def _radix_digit_pass(
    items: list[T], exp: int, value: Callable[[T], int]
) -> list[T]:
    """Устойчиво отсортировать элементы по одной цифре."""
    counts = [0] * 10
    for item in items:
        digit = (value(item) // exp) % 10
        counts[digit] += 1
    for i in range(1, 10):
        counts[i] += counts[i - 1]

    if not items:
        return []
    result: list[T] = [items[0]] * len(items)
    for item in reversed(items):
        digit = (value(item) // exp) % 10
        counts[digit] -= 1
        result[counts[digit]] = item
    return result


def radix_sort_lsd(a: list[int]) -> list[int]:
    """LSD RadixSort неотрицательных целых.

    Время O(d(n + b)), память O(n + b), b = 10.
    """
    if not a:
        return []
    if any(value < 0 for value in a):
        raise ValueError(
            "radix_sort_lsd принимает только неотрицательные целые"
        )

    result = list(a)
    maximum = max(result)
    exp = 1
    while maximum // exp > 0:
        result = _radix_digit_pass(result, exp, lambda value: value)
        exp *= 10
    return result


def sift_down(a: list[int], i: int, n: int) -> None:
    """Просеять элемент i вниз в max-куче a[0:n] на месте."""
    while True:
        left = 2 * i + 1
        right = left + 1
        largest = i

        if left < n and a[left] > a[largest]:
            largest = left
        if right < n and a[right] > a[largest]:
            largest = right
        if largest == i:
            return

        a[i], a[largest] = a[largest], a[i]
        i = largest


def build_max_heap(a: list[int]) -> None:
    """Построить max-кучу снизу вверх за O(n)."""
    for i in range(len(a) // 2 - 1, -1, -1):
        sift_down(a, i, len(a))


def heap_sort(a: list[int]) -> None:
    """HeapSort in-place: O(n log n) времени и O(1) доп. памяти."""
    build_max_heap(a)
    for end in range(len(a) - 1, 0, -1):
        a[0], a[end] = a[end], a[0]
        sift_down(a, 0, end)


def quick_sort(a: list[int]) -> list[int]:
    """Рандомизированный трёхчастный QuickSort из ЛР 3."""
    result = list(a)
    if len(result) < 2:
        return result

    stack = [(0, len(result))]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue

        pivot = result[random.randrange(lo, hi)]
        less = lo
        current = lo
        greater = hi
        while current < greater:
            if result[current] < pivot:
                result[less], result[current] = result[current], result[less]
                less += 1
                current += 1
            elif result[current] > pivot:
                greater -= 1
                result[current], result[greater] = (
                    result[greater],
                    result[current],
                )
            else:
                current += 1

        left = (lo, less)
        right = (greater, hi)
        # Меньший отрезок обрабатывается следующим.
        if left[1] - left[0] < right[1] - right[0]:
            stack.append(right)
            stack.append(left)
        else:
            stack.append(left)
            stack.append(right)
    return result


# ---------------------------------------------------------------------------
# 2. Верификация
# ---------------------------------------------------------------------------


class SelfCheckError(Exception):
    """Нарушен контракт или инвариант реализации."""


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise SelfCheckError(message)


def heap_sorted(a: list[int]) -> list[int]:
    result = list(a)
    heap_sort(result)
    return result


SORTS = {
    "merge_sort": lambda a: merge_sort(a),
    "counting_sort": lambda a: counting_sort(a, key_max=999),
    "radix_sort_lsd": lambda a: radix_sort_lsd(a),
    "heap_sort": heap_sorted,
    "quick_sort": quick_sort,
}


def check_sort(fn: Callable[[list[int]], list[int]]) -> None:
    boundary = [
        [],
        [7],
        [0],
        [5, 5, 5],
        [0, 999, 0, 999],
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
    ]
    for values in boundary:
        argument = list(values)
        result = fn(argument)
        expect(argument == values, f"изменён вход {values}: стало {argument}")
        expect(isinstance(result, list), "результат должен быть списком")
        expect(result == sorted(values), f"на {values} получено {result}")

    rng = random.Random(0)
    for _ in range(300):
        values = [rng.randint(0, 999) for _ in range(rng.randint(0, 80))]
        result = fn(list(values))
        expect(
            collections.Counter(result) == collections.Counter(values),
            f"результат не является перестановкой: {values}",
        )
        expect(result == sorted(values), f"расхождение с sorted() на {values}")


def check_merge_stability() -> None:
    rng = random.Random(0)
    pairs = [(rng.randint(0, 5), i) for i in range(100)]
    result = merge_sort(pairs, key=lambda pair: pair[0])
    expected = sorted(pairs, key=lambda pair: pair[0])
    expect(
        result == expected,
        "merge_sort нарушил порядок равных ключей",
    )


def check_counting_stability() -> None:
    pairs = [(2, "a"), (1, "b"), (2, "c"), (1, "d")]
    result = _stable_counting_sort_items(
        pairs, key=lambda pair: pair[0], key_min=1, key_max=2
    )
    expected = [(1, "b"), (1, "d"), (2, "a"), (2, "c")]
    expect(
        result == expected,
        "counting_sort нарушил порядок равных ключей",
    )


def check_radix_digit_stability() -> None:
    pairs = [(21, "a"), (11, "b"), (22, "c"), (12, "d")]
    result = _radix_digit_pass(pairs, exp=1, value=lambda pair: pair[0])
    expected = [(21, "a"), (11, "b"), (22, "c"), (12, "d")]
    expect(result == expected, "поразрядный проход radix нестабилен")


def check_heap_invariant() -> None:
    rng = random.Random(1)
    for _ in range(100):
        heap = [rng.randint(-1000, 1000) for _ in range(rng.randint(0, 100))]
        build_max_heap(heap)
        for parent in range(len(heap) // 2):
            left = 2 * parent + 1
            right = left + 1
            expect(
                heap[parent] >= heap[left],
                "нарушено свойство max-кучи слева",
            )
            if right < len(heap):
                expect(
                    heap[parent] >= heap[right],
                    "нарушено свойство max-кучи справа",
                )


def check_invalid_inputs() -> None:
    try:
        counting_sort([0, 4], key_max=3)
    except ValueError:
        pass
    else:
        raise SelfCheckError(
            "counting_sort не отклонил ключ вне диапазона"
        )

    try:
        radix_sort_lsd([2, -1, 3])
    except ValueError:
        pass
    else:
        raise SelfCheckError(
            "radix_sort_lsd не отклонил отрицательный ключ"
        )


def self_check() -> None:
    checks = [(name, lambda fn=fn: check_sort(fn)) for name, fn in SORTS.items()]
    checks.extend(
        [
            ("стабильность merge_sort", check_merge_stability),
            ("стабильность counting_sort", check_counting_stability),
            ("устойчивость разряда radix", check_radix_digit_stability),
            ("инвариант max-кучи", check_heap_invariant),
            ("некорректные входы", check_invalid_inputs),
        ]
    )

    failed = 0
    print("self_check:")
    for name, check in checks:
        try:
            check()
        except SelfCheckError as error:
            print(f"  {name}: ОШИБКА — {error}")
            failed += 1
        except Exception:  # noqa: BLE001
            print(f"  {name}: ИСКЛЮЧЕНИЕ")
            traceback.print_exc(file=sys.stdout)
            failed += 1
        else:
            print(f"  {name}: OK")
    if failed:
        raise SystemExit(f"self_check: не пройдено разделов — {failed}")
    print("self_check: OK")


# ---------------------------------------------------------------------------
# 3. Бенчмарк и графики
# ---------------------------------------------------------------------------


SIZES = [1_000, 3_000, 10_000, 30_000, 100_000]
REPEATS = 5
SMALL_KEY_MAX = 100
WIDE_KEY_MAX = 10**9


def make_datasets(rng: random.Random, n: int) -> dict[str, list[int]]:
    almost = list(range(n))
    for _ in range(max(1, n // 100)):
        i, j = rng.randrange(n), rng.randrange(n)
        almost[i], almost[j] = almost[j], almost[i]
    return {
        "random_wide": [rng.randint(0, WIDE_KEY_MAX) for _ in range(n)],
        "random_small": [rng.randint(0, SMALL_KEY_MAX) for _ in range(n)],
        "sorted": list(range(n)),
        "reversed": list(range(n, 0, -1)),
        "almost_sorted": almost,
    }


def bench(fn: Callable[[list[int]], object], data: list[int]) -> float:
    fn(list(data))
    times = []
    for _ in range(REPEATS):
        argument = list(data)
        started = time.perf_counter()
        fn(argument)
        times.append(time.perf_counter() - started)
    return statistics.median(times)


def run_benchmarks(seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    algorithms: list[tuple[str, Callable[[list[int]], object]]] = [
        ("merge_sort", merge_sort),
        ("counting_sort", lambda a: counting_sort(a, max(a, default=0))),
        ("radix_sort_lsd", radix_sort_lsd),
        ("heap_sort", heap_sort),
        ("quick_sort", quick_sort),
        ("sorted", sorted),
    ]
    rows: list[dict[str, object]] = []

    for n in SIZES:
        datasets = make_datasets(rng, n)
        print(f"\nn = {n}:")
        for distribution, data in datasets.items():
            print(f"  {distribution}:")
            for algorithm, fn in algorithms:
                if algorithm == "counting_sort" and distribution == "random_wide":
                    print(f"    {algorithm:16s}: пропуск — диапазон ключей")
                    rows.append(
                        {
                            "n": n,
                            "distribution": distribution,
                            "algorithm": algorithm,
                            "seconds": "",
                            "status": "skipped_wide_key_range",
                        }
                    )
                    continue
                elapsed = bench(fn, data)
                print(f"    {algorithm:16s}: {elapsed:.6f} с")
                rows.append(
                    {
                        "n": n,
                        "distribution": distribution,
                        "algorithm": algorithm,
                        "seconds": elapsed,
                        "status": "ok",
                    }
                )
    return rows


def save_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["n", "distribution", "algorithm", "seconds", "status"],
        )
        writer.writeheader()
        writer.writerows(rows)


def log_log_slope(points: list[tuple[int, float]]) -> float:
    xs = [math.log10(n) for n, _ in points]
    ys = [math.log10(seconds) for _, seconds in points]
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return numerator / denominator


def plot_results(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    distributions = [
        "random_wide",
        "random_small",
        "sorted",
        "reversed",
        "almost_sorted",
    ]
    algorithms = [
        "merge_sort",
        "counting_sort",
        "radix_sort_lsd",
        "heap_sort",
        "quick_sort",
        "sorted",
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    for axis, distribution in zip(axes.flat, distributions):
        for algorithm in algorithms:
            points = [
                (int(row["n"]), float(row["seconds"]))
                for row in rows
                if row["distribution"] == distribution
                and row["algorithm"] == algorithm
                and row["status"] == "ok"
            ]
            if not points:
                continue
            slope = log_log_slope(points)
            axis.plot(
                [n for n, _ in points],
                [seconds for _, seconds in points],
                marker="o",
                label=f"{algorithm}, наклон {slope:.2f}",
            )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(distribution)
        axis.set_xlabel("размер входа n")
        axis.set_ylabel("время, с")
        axis.grid(True, which="both", linewidth=0.3)
        axis.legend(fontsize=7)

    axes[2, 1].axis("off")
    fig.suptitle("ЛР 4, вариант 10: сравнение сортировок", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def print_slopes(rows: list[dict[str, object]]) -> None:
    print("\nНаклоны log-log:")
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = collections.defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            grouped[(str(row["distribution"]), str(row["algorithm"]))].append(
                (int(row["n"]), float(row["seconds"]))
            )
    for (distribution, algorithm), points in sorted(grouped.items()):
        print(f"  {distribution:15s} {algorithm:16s}: {log_log_slope(points):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", type=int, required=True)
    parser.add_argument("--out", type=Path, default=Path("lab04-results"))
    args = parser.parse_args()

    seed = 30 + args.variant
    random.seed(seed)
    args.out.mkdir(parents=True, exist_ok=True)

    self_check()
    rows = run_benchmarks(seed)
    save_csv(rows, args.out / "benchmark_results.csv")
    plot_results(rows, args.out / "benchmark_graphs.png")
    print_slopes(rows)
    print(f"\nРезультаты сохранены в {args.out}")


if __name__ == "__main__":
    main()
