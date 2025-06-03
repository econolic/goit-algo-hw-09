from typing import Dict, List, Sequence, Tuple
import time
import unittest

# -------------------------------------------------------------
# Типи та константи
# -------------------------------------------------------------
CoinDict = Dict[int, int]
__all__ = [
    "ChangeImpossibleError",
    "find_coins_greedy",
    "find_min_coins",
    "compare_algorithms",
]


class ChangeImpossibleError(ValueError):
    """Піднімається, коли неможливо сформувати решту з наявних монет."""


def _prepare_coins(coins: Sequence[int] | None) -> List[int]:
    """
    Повертає копію `coins`, відсортовану за спаданням.

    Args:
        coins: Перелік номіналів.

    Returns:
        Відсортований список номіналів за спаданням.
    """
    default = [50, 25, 10, 5, 2, 1]
    uniq = sorted(set(default if coins is None else coins), reverse=True)
    if uniq[-1] != 1:
        # Немає монети номіналу 1 — зміна будь‑якої суми не гарантується
        raise ChangeImpossibleError(
            "Набір номіналів має містити монету 1, або решта може бути неможливою."
        )
    return uniq


# -------------------------------------------------------------
# Алгоритми видачі решти
# -------------------------------------------------------------

def find_coins_greedy(amount: int, coins: Sequence[int] | None = None) -> CoinDict:
    """Видає решту жадібним методом.

    Args:
        amount: Сума, яку потрібно розміняти (>= 0).
        coins: Перелік доступних номіналів. Якщо *None* — стандартний
            український набір ``[50, 25, 10, 5, 2, 1]``.

    Returns:
        ``dict``: ключ — номінал, значення — кількість монет.

    Raises:
        ValueError: ``amount`` < 0.
        ChangeImpossibleError: неможливо сформувати суму.
    """
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною.")
    if amount == 0:
        return {}

    sorted_coins = _prepare_coins(coins)
    remaining = amount
    result: CoinDict = {}

    for coin in sorted_coins:
        if remaining < coin:
            continue
        count, remaining = divmod(remaining, coin)
        if count:
            result[coin] = count
        if remaining == 0:
            break

    if remaining:
        raise ChangeImpossibleError(
            f"Неможливо сформувати {amount} з номіналів {sorted_coins}."
        )

    return dict(sorted(result.items(), reverse=True))


def find_min_coins(amount: int, coins: Sequence[int] | None = None) -> CoinDict:
    """Видає решту з мінімальною кількістю монет (динамічне програмування).

    Алгоритм будує таблицю ``dp[i]`` — мінімальна кількість монет для суми *i*.
    Часова складність ``O(amount × k)``, де *k* — кількість
    різних номіналів.

    Args:
        amount: Сума, яку потрібно розміняти (>= 0).
        coins: Перелік доступних номіналів. Якщо *None* — стандартний
            набір ``[50, 25, 10, 5, 2, 1]``.

    Returns:
        ``dict``: оптимальний розподіл монет.

    Raises:
        ValueError: ``amount`` < 0.
        ChangeImpossibleError: неможливо сформувати суму.
    """
    if amount < 0:
        raise ValueError("Сума не може бути від'ємною.")
    if amount == 0:
        return {}

    sorted_coins = _prepare_coins(coins)
    k = len(sorted_coins)

    # dp[i] — мінімальна кількість монет для суми i
    dp = [float("inf")] * (amount + 1)
    dp[0] = 0
    # Для відстеження номіналу, використаного для dp[i]
    prev_coin = [-1] * (amount + 1)

    for coin in sorted_coins:
        for i in range(coin, amount + 1):
            if dp[i - coin] + 1 < dp[i]:
                dp[i] = dp[i - coin] + 1
                prev_coin[i] = coin

    if dp[amount] == float("inf"):
        raise ChangeImpossibleError(
            f"Неможливо сформувати {amount} з номіналів {sorted_coins}."
        )

    # Відновлюємо шлях
    result: CoinDict = {}
    i = amount
    while i > 0:
        coin = prev_coin[i]
        result[coin] = result.get(coin, 0) + 1
        i -= coin

    return dict(sorted(result.items(), reverse=True))


# -------------------------------------------------------------
# Бенчмаркінг
# -------------------------------------------------------------

def _timeit(func, *args, repeat: int = 5):
    """Повертає мінімальний час із *repeat* запусків (щоб мінімізувати шум)."""
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter_ns()
        func(*args)
        best = min(best, time.perf_counter_ns() - start)
    return best / 1_000_000  # мс


def compare_algorithms(amounts: Sequence[int], coins: Sequence[int] | None = None) -> None:
    """Друкує порівняння часу та кількості монет для двох алгоритмів."""
    hdr = (
        f"{'Сума':>8} | {'Жадібний (мс)':>14} | {'DP (мс)':>10} | "
        f"{'К-ть (Greedy)':>13} | {'К-ть (DP)':>10}"
    )
    print(hdr)
    print("-" * len(hdr))

    for amount in amounts:
        g_time = _timeit(find_coins_greedy, amount, coins)
        d_time = _timeit(find_min_coins, amount, coins)
        g_cnt = sum(find_coins_greedy(amount, coins).values())
        d_cnt = sum(find_min_coins(amount, coins).values())
        print(
            f"{amount:8d} | {g_time:14.3f} | {d_time:10.3f} | "
            f"{g_cnt:13d} | {d_cnt:10d}"
        )


# -------------------------------------------------------------
# Unit‑тести (можна запускати: `python cash_system.py -t`)
# -------------------------------------------------------------

class _CashSystemTest(unittest.TestCase):
    """Набір базових перевірок для обох алгоритмів."""

    def test_basic(self):
        cases: List[Tuple[int, CoinDict]] = [
            (0, {}),
            (1, {1: 1}),
            (113, {50: 2, 10: 1, 2: 1, 1: 1}),
        ]
        for amount, expected in cases:
            self.assertEqual(find_coins_greedy(amount), expected)
            self.assertEqual(find_min_coins(amount), expected)

    def test_non_canonical(self):
        coins = [1, 4, 5]
        greedy = find_coins_greedy(8, coins)
        optimal = find_min_coins(8, coins)
        self.assertGreater(sum(greedy.values()), sum(optimal.values()))

    def test_error(self):
        with self.assertRaises(ValueError):
            find_coins_greedy(-5)
        with self.assertRaises(ChangeImpossibleError):
            find_coins_greedy(3, coins=[2])


# -------------------------------------------------------------
# CLI інтерфейс модуля
# -------------------------------------------------------------

def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Cash-system module")
    parser.add_argument("amount", nargs="*", type=int, help="Список сум для розміну")
    parser.add_argument("-t", "--test", action="store_true", help="Запустити unit-тести")
    args = parser.parse_args()

    if args.test:
        unittest.main(argv=["__ignored__"], exit=False)
        return

    amounts = args.amount or [113, 999, 10000, 50000]
    compare_algorithms(amounts)


if __name__ == "__main__":
    _main()