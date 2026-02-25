# -*- coding: utf-8 -*-
"""
QSSC / Integer Chaos (Vase operator) verification script with configurable integer types.
FINAL OPTIMIZED VERSION:
- Added flexible configuration for number range and types.
- Users can select different number sequences (primes, random, 3n+1, etc.) in config.
"""

from __future__ import annotations
import math
import sys
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# =========================
#    AUTO-TUNING LOGIC
# =========================
def auto_tune_config(t_center: float):
    """
    Автоматически подбирает NMAX и ALPHA под высоту T.
    Алгоритм из старой версии: N зависит от высоты T, а не от ширины диапазона.
    """
    # 1. Эмпирический коэффициент "запаса мощности"
    effort_factor = 5.0  # 10.0 - баланс

    # Считаем NMAX от центральной точки диапазона
    n_max = int(t_center * effort_factor)

    # Округляем до красивого числа (тысяч)
    # (n_max // 1000 + 1) * 1000 гарантирует, что мы всегда округляем вверх до полной 1000
    n_max = (n_max // 1000 + 1) * 1000

    # 2. Подбираем ALPHA (чтобы хвост обрезался мягко)
    # Формула: alpha * n_max^2 = 5.0
    alpha = 5.0 / (n_max ** 2)

    return n_max, alpha


# =========================
#      CONFIGURATION
# =========================
@dataclass(frozen=True)
class Config:
    # Диапазон чисел (начало диапазона)
    M: int = 1

    # Настройки диапазона анализа T
    T_MIN: float = 5.0
    T_MAX: float = 35.0

    NUMBER_TYPE: str = "primes"  # "integers", "primes", "3n+1", "golden", "random", "3n+2", "gaussian"

    # Остальные настройки
    N_T: int = 100000
    CHUNK_SIZE: int = 16384
    USE_2PI: bool = False
    U: float = 1e-3
    ZETA_DPS: int = 16
    TOL: float = 0.1
    PRINT_DECIMALS: int = 16
    SHOW_WORST_K: int = 100

    # --- Динамические свойства (вычисляются на лету) ---

    @property
    def _tuning_params(self) -> Tuple[int, float]:
        """Внутренний метод для расчета параметров на основе центра диапазона."""
        t_center = (self.T_MIN + self.T_MAX) / 2.0
        return auto_tune_config(t_center)

    @property
    def NMAX(self) -> int:
        """Автоматическая настройка NMAX на основе высоты T (через t_center)."""
        n, _ = self._tuning_params
        return n

    @property
    def ALPHA(self) -> float:
        """Автоматическая настройка ALPHA, связанная с NMAX."""
        _, a = self._tuning_params
        return a


# Инициализация конфига
CFG = Config()


# =========================
#       NUMBER SEQUENCES
# =========================

def is_prime(num: int) -> bool:
    """Быстрая проверка числа на простоту."""
    if num < 2: return False
    if num == 2: return True
    if num % 2 == 0: return False
    # Проверяем нечетные делители до корня из num
    limit = int(math.isqrt(num)) + 1
    for i in range(3, limit, 2):
        if num % i == 0:
            return False
    return True


def generate_numbers(M: int, N: int, number_type: str) -> np.ndarray:
    """Генерация различных типов чисел."""
    if number_type == "integers":
        return np.arange(M, M + N, dtype=np.float64)

    elif number_type == "primes":
        primes = []
        candidate = M
        # Если M=1, начинаем проверку с 1 (которая вернет False),
        # но логичнее искать начиная с candidate.
        while len(primes) < N:
            if is_prime(candidate):
                primes.append(candidate)
            candidate += 1
        return np.array(primes, dtype=np.float64)

    elif number_type == "chi4":
        # Генерируем нечетные числа: 1, 3, 5, 7, 9...
        # L(s, chi4) = 1^-s - 3^-s + 5^-s - 7^-s ...
        return 2 * np.arange(M - 1, M + N, dtype=np.float64) + 1

    elif number_type == "3n+1":
        return 3 * np.arange(M, M + N, dtype=np.float64) + 1
    elif number_type == "3n+2":
        # Последовательность: 5, 8, 11, 14... (при M=1)
        # Это Дзета Гурвица zeta(s, 2/3)
        return 3 * np.arange(M, M + N, dtype=np.float64) + 2
    elif number_type == "golden":
        return (np.arange(M, M + N, dtype=np.float64) * (1 + math.sqrt(5)) / 2)
    elif number_type == "random":
        # Исправленная версия: сначала int, потом float
        # Сортировка (np.sort) делает спектр более физичным (возрастающие уровни)
        raw_ints = np.random.randint(M, M + N, size=N)
        return np.sort(raw_ints).astype(np.float64)
    else:
        raise ValueError(f"Unknown number type: {number_type}")

def get_riemann_zeros_in_range(t_min: float, t_max: float, dps: int = 80) -> np.ndarray:
    """
    Получить мнимые части нулей Римана в диапазоне [t_min, t_max].
    """
    if t_max < t_min:
        t_min, t_max = t_max, t_min

    mp.mp.dps = int(dps)  # Устанавливаем точность (должна быть достаточно высокая)

    # Приблизительная оценка числа нулей в диапазоне
    def estimate_n_zeros(T):
        if T <= 0: return 0
        c = T / (2 * math.pi)
        return c * math.log(c) - c + 0.875

    k_start = max(1, int(estimate_n_zeros(t_min)) - 5)
    zeros: List[float] = []
    k = k_start

    print(f"[zeros] Jumping to estimated start index k={k} for T={t_min}...")

    last_print = 0
    while True:
        z = mp.zetazero(k)  # Получаем k-й ноль Римана
        gamma = float(mp.im(z))  # Извлекаем мнимую часть

        if gamma > t_max + 1e-12:
            break
        if gamma >= t_min - 1e-12:
            zeros.append(gamma)

        if k - last_print >= 50:  # Реже обновляем статус для скорости
            print(f"\r[zeros] fetched k={k} (found={len(zeros)})", end="", flush=True)
            last_print = k
        k += 1

    print(f"\r[zeros] done. checked k up to {k - 1}, found in-range={len(zeros)}      ")
    return np.array(zeros, dtype=np.float64)


# =========================
#     QSSC ENGINE
# =========================
def soft_cutoff_heat(lam: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    """
    A4: g(lambda) = exp(-alpha*lambda^2), with numerical underflow protection.
    """
    exp_arg = -alpha * lam * lam
    exp_min = -745.0
    clipped = int(np.sum(exp_arg < exp_min))
    exp_arg = np.maximum(exp_arg, exp_min)
    g = np.exp(exp_arg)
    return g, clipped

def build_vase_spectrum(n: np.ndarray, use_2pi: bool) -> np.ndarray:
    """lambda_n = n or lambda_n = 2*pi*n."""
    if use_2pi:
        return (2.0 * math.pi) * n
    return n.copy()

def compute_Z_profile_integer_flow(
        t: np.ndarray,
        lam: np.ndarray,
        u: float,
        alpha: float,
        chunk_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Вычисление Z(t) для s=0.5 + i t."""
    g, n_clipped = soft_cutoff_heat(lam, alpha=alpha)
    A = (u * u) + (lam * lam)
    logA = np.log(A)
    phi = 0.5 * logA
    amp = 2.0 * g * np.exp(-0.25 * logA)

    zr = np.zeros_like(t, dtype=np.float64)
    zi = np.zeros_like(t, dtype=np.float64)

    n_modes = lam.size
    n_chunks = int(math.ceil(n_modes / max(1, chunk_size)))
    pb = ProgressBar(total=n_chunks, prefix="[compute] Z-profile")

    chunk_idx = 0
    for i0 in range(0, n_modes, chunk_size):
        i1 = min(i0 + chunk_size, n_modes)
        phi_c = phi[i0:i1]
        amp_c = amp[i0:i1]

        # Основная тяжелая операция: матричное умножение
        phase = np.outer(t, phi_c)
        zr += np.dot(np.cos(phase), amp_c)
        zi -= np.dot(np.sin(phase), amp_c)

        chunk_idx += 1
        pb.update(chunk_idx)

    pb.close()
    Z = zr + 1j * zi
    P = np.abs(Z)
    return Z, P

# =========================
#      MINIMA & ROOTS
# =========================
def compare_library_zeros_to_minima(
        zeros_lib: np.ndarray,
        minima_t: np.ndarray,
        tol: float,
        decimals: int,
        show_worst_k: int,
) -> None:
    """Сравнивает библиотечные нули с найденными минимумами и выводит подробный отчет."""
    if zeros_lib.size == 0:
        print("\n[cmp] No library zeros in range.")
        return
    if minima_t.size == 0:
        print("\n[cmp] No minima detected in profile.")
        return

    def n_t_approx(T):
        if T <= 0: return 0
        return (T / (2 * math.pi)) * math.log(T / (2 * math.pi)) - (T / (2 * math.pi)) + 7 / 8

    t_min_val = min(zeros_lib[0], minima_t[0])
    t_max_val = max(zeros_lib[-1], minima_t[-1])

    theoretical_count = int(round(n_t_approx(t_max_val) - n_t_approx(t_min_val)))

    print(f"\n=== Анализ диапазона T=[{t_min_val:.1f}, {t_max_val:.1f}] ===")
    print(f"[theory] Ожидаемое число нулей (Риман-Мангольдт): ~{theoretical_count}")
    print(f"[actual] Библиотечных нулей (mpmath):             {zeros_lib.size}")
    print(f"[yours]  Найденных минимумов (ваша программа):    {minima_t.size}")

    if zeros_lib.size == minima_t.size:
        print(">>> ИДЕАЛЬНОЕ СОВПАДЕНИЕ ПО КОЛИЧЕСТВУ! <<<")
    else:
        diff = minima_t.size - zeros_lib.size
        print(f">>> РАСХОЖДЕНИЕ: {diff} (если >0, есть лишние; если <0, пропущены)")

    fmt = f"{{:.{decimals}f}}"
    deltas = []
    rows = []

    for g in zeros_lib:
        j = int(np.argmin(np.abs(minima_t - g)))
        t_star = float(minima_t[j])
        d = float(abs(t_star - float(g)))
        status = "OK" if d <= tol else "BAD"
        deltas.append(d)
        rows.append((d, float(g), t_star, status))

    deltas_arr = np.array(deltas, dtype=np.float64)
    wow = int(np.sum(deltas_arr <= tol))

    print(f"\n=== Детальный список всех найденных нулей ({len(rows)} шт.) ===")
    print(f"Tol={tol} -> Совпадений: {wow}/{zeros_lib.size}")
    print(f"Средняя ошибка: {np.mean(deltas_arr):.6f}, Макс. ошибка: {np.max(deltas_arr):.6f}")
    print("-" * 65)
    print(f"{'Zeta Zero (Lib)':<22} | {'Calculated (You)':<22} | {'Delta':<10}")
    print("-" * 65)

    rows_sorted_by_time = sorted(rows, key=lambda x: x[1])

    for d, g, t_star, status in rows_sorted_by_time:
        mark = "!!!" if d > tol else ""
        print(f"{fmt.format(g):<22} | {fmt.format(t_star):<22} | {d:.6f} {mark}")

    print("-" * 65)


def find_local_minima_indices(P: np.ndarray) -> np.ndarray:
    """Индексы строгих локальных минимумов для P(t)."""
    mid = (P[1:-1] < P[:-2]) & (P[1:-1] < P[2:])
    return np.flatnonzero(mid) + 1

def refine_minimum_parabola(t: np.ndarray, P: np.ndarray, i: int) -> float:
    """Уточнение локального минимума с использованием параболы."""
    if i <= 0 or i >= len(t) - 1:
        return float(t[i])
    x = t[i - 1:i + 2]
    y = P[i - 1:i + 2]

    try:
        a, b, _ = np.polyfit(x, y, 2)
    except np.linalg.LinAlgError:
        return float(t[i])

    if a <= 0:
        return float(t[i])

    x_star = -b / (2.0 * a)
    if x_star < x[0] or x_star > x[2]:
        return float(t[i])

    return float(x_star)

def extract_profile_minima(t: np.ndarray, P: np.ndarray, refine: bool = True) -> np.ndarray:
    """Извлекаем точные позиции всех локальных минимумов профиля P(t)=|Z(t)|."""
    idx = find_local_minima_indices(P)
    if idx.size == 0:
        return np.array([], dtype=np.float64)
    if not refine:
        return t[idx].astype(np.float64)

    t_ref = []
    for i in idx:
        t_ref.append(refine_minimum_parabola(t, P, int(i)))

    return np.array(t_ref, dtype=np.float64)

class ProgressBar:
    """Минимальный прогресс-бар без зависимостей для работы в терминале."""

    def __init__(self, total: int, prefix: str = "", width: int = 28) -> None:
        self.total = max(1, int(total))
        self.prefix = prefix.strip()
        self.width = max(10, int(width))
        self.last_pct = -1

    def update(self, current: int) -> None:
        current = max(0, min(self.total, int(current)))
        pct = int(round(100.0 * current / self.total))
        if pct == self.last_pct:
            return
        self.last_pct = pct

        filled = int(round(self.width * current / self.total))
        bar = "#" * filled + "-" * (self.width - filled)
        msg = f"{self.prefix} [{bar}] {pct:3d}%"
        print("\r" + msg, end="", file=sys.stdout, flush=True)

    def close(self) -> None:
        self.update(self.total)
        print("", file=sys.stdout)  # newline

# =========================
#           PLOT
# =========================
def plot_vlines_with_labels(
        ax: plt.Axes,
        xs: np.ndarray,
        y_top: float,
        fmt: str,
        *,
        color: str,
        linestyle: str,
        linewidth: float,
        alpha: float,
        fontsize: int,
        rotation: int,
        prefix: str = "",
        stagger_levels: int = 6,
        y_margin: float = 0.02,
) -> None:
    """Draw vertical lines with labels for the Riemann zeros."""
    if xs.size == 0:
        return
    levels = []
    for k in range(stagger_levels):
        levels.append((1.0 - y_margin) * y_top * (1.0 - 0.06 * k))
    for i, x in enumerate(xs):
        ax.axvline(float(x), color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        y = levels[i % stagger_levels]
        label = prefix + (fmt.format(float(x)) if "{" in fmt else (fmt % float(x)))
        ax.text(
            float(x), y, label,
            rotation=rotation, va="top", ha="right", fontsize=fontsize, color=color, alpha=alpha,
        )


# =========================
#      MAIN FUNCTION
# =========================
def main() -> None:
    print("=== Dirichlet L-function (Chi-4) Test ===")
    print("Searching for the 'missing' Gaussian zeros...")

    t = np.linspace(CFG.T_MIN, CFG.T_MAX, CFG.N_T, dtype=np.float64)

    # L(s, chi4) = 1^-s - 3^-s + 5^-s - 7^-s + ...

    print(f"[1/2] Computing Positive Flow (1, 5, 9...)...")
    # k = 0, 1, 2... -> 4k + 1
    n_plus = 4 * np.arange(0, CFG.NMAX, dtype=np.float64) + 1
    lam_plus = build_vase_spectrum(n_plus, use_2pi=CFG.USE_2PI)
    Z_plus, _ = compute_Z_profile_integer_flow(t, lam_plus, CFG.U, CFG.ALPHA, CFG.CHUNK_SIZE)

    print(f"[2/2] Computing Negative Flow (3, 7, 11...)...")
    # k = 0, 1, 2... -> 4k + 3
    n_minus = 4 * np.arange(0, CFG.NMAX, dtype=np.float64) + 3
    lam_minus = build_vase_spectrum(n_minus, use_2pi=CFG.USE_2PI)
    Z_minus, _ = compute_Z_profile_integer_flow(t, lam_minus, CFG.U, CFG.ALPHA, CFG.CHUNK_SIZE)

    # ВЫЧИТАЕМ!
    Z_total = Z_plus - Z_minus
    P_total = np.abs(Z_total)

    # Ищем минимумы
    minima_t = extract_profile_minima(t, P_total, refine=True)
    print(f"\n[found] Found {minima_t.size} zeros!")

    # Ожидаемые первые нули L(s, chi4):
    expected = [6.02, 10.24, 12.98, 16.34, 18.2, 23.2]
    print(f"Expect to see zeros near: {expected}")
    print(f"First 5 found: {minima_t[:5]}")

    # =========================
    #      VISUALIZATION
    # =========================
    fig = plt.figure(figsize=(14, 6))

    # --- ГРАФИК 1: АМПЛИТУДА ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t, P_total, linewidth=1.5, color='green', label=r"$|L(s, \chi_4)|$")
    ax1.set_title(f"Dirichlet Chi-4 (Gaussian Partner)")
    ax1.set_xlabel("t")
    ax1.set_ylabel("|L(s)|")
    ax1.grid(True, alpha=0.5)

    y_top = float(np.max(P_total))

    # 1. Рисуем нули Римана СЕРЫМ
    zeros_riemann = get_riemann_zeros_in_range(CFG.T_MIN, CFG.T_MAX)
    # ИСПРАВЛЕНИЕ ЗДЕСЬ: Заменили "" на "{:.1f}", теперь ошибки не будет
    plot_vlines_with_labels(
        ax1, zeros_riemann, y_top, "{:.1f}",
        color="gray", linestyle=":", linewidth=0.5, alpha=0.4,
        fontsize=6, rotation=90
    )

    # 2. Рисуем НАШИ найденные нули (КРАСНЫМ)
    print(f"[plot] Marking {len(minima_t)} L-zeros in RED...")

    plot_vlines_with_labels(
        ax1, minima_t, y_top, "{:.2f}",
        color="red", linestyle="--", linewidth=1.0, alpha=0.8,
        fontsize=8, rotation=90, prefix="L="
    )

    y_text_pos = -0.05 * y_top
    for r in minima_t:
        ax1.plot(r, 0, marker="o", color="red", markersize=5)
        ax1.text(r, y_text_pos, f"{r:.2f}", color="red", fontsize=8, rotation=90, ha="center", va="top",
                 fontweight='bold')

    # --- ГРАФИК 2: СПИРАЛЬ ---
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(Z_total.real, Z_total.imag, linewidth=0.8, color='purple', alpha=0.8)

    ax2.axhline(0, color='black', linewidth=1)
    ax2.axvline(0, color='black', linewidth=1)
    ax2.plot(0, 0, marker='+', color='red', markersize=20, markeredgewidth=2, label="ZERO")

    ax2.set_title("Phase Trajectory (Centered!)")
    ax2.axis('equal')
    ax2.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()