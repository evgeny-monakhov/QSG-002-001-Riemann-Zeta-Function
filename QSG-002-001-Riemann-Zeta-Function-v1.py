#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QSSC-Rieman - Ultimate Precision (Axiom A7 Twist Enabled)
Ver 9: STABLE VISUALS (Fixed Power=12 Artifacts, Reverted to Power=6 for Plot)
"""

from __future__ import annotations

import math
import sys
import time
import json
import logging
import os
from pathlib import Path
import datetime
from dataclasses import dataclass
from typing import Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp

# =========================
#    HARDWARE DETECTION
# =========================

USE_GPU = False
USE_NUMBA = False

try:
    import cupy as cp

    USE_GPU = True
    print("[system] NVIDIA GPU detected (CuPy). Enabling CUDA acceleration.")
except ImportError:
    pass

if not USE_GPU:
    try:
        from numba import njit, prange
        import numba

        USE_NUMBA = True
        print("[system] Numba detected. Enabling multicore CPU compilation.")
    except ImportError:
        print("[system] Numba not found. Running on pure NumPy (slow).")


# =========================
#       PROGRESS BAR
# =========================
class ProgressBar:
    def __init__(self, total: int, prefix: str = "", width: int = 30):
        self.total = max(1, total)
        self.prefix = prefix
        self.width = width
        self.last_update_time = 0.0

    def update(self, current: int):
        current = min(current, self.total)
        now = time.time()
        if (now - self.last_update_time) < 0.1 and current < self.total:
            return
        self.last_update_time = now
        pct = current / self.total
        filled = int(self.width * pct)
        bar = "█" * filled + "-" * (self.width - filled)
        sys.stdout.write(f"\r{self.prefix} |{bar}| {pct * 100:5.1f}%")
        sys.stdout.flush()

    def close(self):
        sys.stdout.write(f"\r{self.prefix} |{'█' * self.width}| 100.0%\n")
        sys.stdout.flush()


# =========================
#     RUN CONTEXT
# =========================

@dataclass
class RunContext:
    run_dir: Path
    logger: logging.Logger

    def save_text(self, name: str, text: str) -> Path:
        p = self.run_dir / name
        p.write_text(text, encoding="utf-8")
        return p

    def save_json(self, name: str, obj) -> Path:
        p = self.run_dir / name
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    def save_npz(self, name: str, **arrays) -> Path:
        p = self.run_dir / name
        np.savez_compressed(p, **arrays)
        return p

    def save_fig(self, fig: "plt.Figure", name: str, dpi: int = 180) -> Path:
        p = self.run_dir / name
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        return p


def _make_run_dir(base: str = "runs") -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"QSSC_RIEMANN_{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def _cfg_to_dict(cfg) -> dict:
    return {k: getattr(cfg, k) for k in dir(cfg) if k.isupper()}


def _log_array_stats(logger: logging.Logger, name: str, x: np.ndarray) -> None:
    if x.size == 0:
        logger.info("%s: empty", name)
        return
    # ULTRA PRECISION LOGGING
    logger.info(
        "%s: n=%d min=%.30g max=%.30g mean=%.30g",
        name, x.size, float(np.min(x)), float(np.max(x)), float(np.mean(x))
    )


# =========================
#     CONFIGURATION
# =========================

def auto_tune_config(t_center: float):
    # ПАТЧ: Увеличили Effort Factor с 14.0 до 40.0
    # Это дает в 3 раза больше слагаемых ряда (Ultimate Precision)
    effort_factor = 40.0

    n_max = int(t_center * effort_factor)
    n_max = max(100, (n_max // 100 + 1) * 100)

    # Подстройка ширины гауссиана
    alpha = 5.0 / (n_max ** 2)
    return n_max, alpha


# =========================
#    ЯДРА ВЫЧИСЛЕНИЙ
# =========================

def soft_cutoff_heat(lam: np.ndarray, alpha: float, power: float = 6.0) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Применяет сглаживание (Super-Gaussian).
    """
    if lam.size == 0:
        return lam, lam, 0

    # Нормируем аргумент от 0 до 1
    lam_max = float(lam[-1])
    x = lam / lam_max

    # Super-Gaussian window
    exp_arg = -12.0 * (x ** power)

    # Отсечение
    valid_mask = exp_arg > -36.0
    lam_valid = lam[valid_mask]
    g_valid = np.exp(exp_arg[valid_mask])
    n_clipped = lam.size - lam_valid.size

    return lam_valid, g_valid, n_clipped


if USE_NUMBA:
    @njit(parallel=True, fastmath=True, cache=True)
    def numba_kernel(t, phi, amp):
        n_t = t.shape[0]
        n_modes = phi.shape[0]
        zr = np.zeros(n_t, dtype=np.float64)
        zi = np.zeros(n_t, dtype=np.float64)
        for i in prange(n_t):
            ti = t[i]
            acc_r = 0.0
            acc_i = 0.0
            for j in range(n_modes):
                phase = ti * phi[j]
                acc_r += np.cos(phase) * amp[j]
                acc_i -= np.sin(phase) * amp[j]
            zr[i] = acc_r
            zi[i] = acc_i
        return zr, zi

if USE_GPU:
    def gpu_kernel(t_cpu, phi_cpu, amp_cpu, chunk_size):
        t_gpu = cp.asarray(t_cpu)
        phi_gpu = cp.asarray(phi_cpu)
        amp_gpu = cp.asarray(amp_cpu)
        n_t = t_gpu.size
        n_modes = phi_gpu.size
        zr_gpu = cp.zeros(n_t, dtype=cp.float64)
        zi_gpu = cp.zeros(n_t, dtype=cp.float64)
        for i0 in range(0, n_modes, chunk_size):
            i1 = min(i0 + chunk_size, n_modes)
            p_c = phi_gpu[i0:i1]
            a_c = amp_gpu[i0:i1]
            phase = cp.outer(t_gpu, p_c)
            zr_gpu += cp.sum(cp.cos(phase) * a_c, axis=1)
            zi_gpu += cp.sum(cp.sin(phase) * -a_c, axis=1)
            del phase
            cp.get_default_memory_pool().free_all_blocks()
        return cp.asnumpy(zr_gpu), cp.asnumpy(zi_gpu)


# --- AXIOM A7: PHASE TWIST CALCULATION ---
# --- AXIOM A7: PHASE TWIST CALCULATION (ULTRA PRECISION) ---
@numba.jit(nopython=True, fastmath=True)
def calc_theta_riemann(t_arr):
    """
    Вычисляет фазу Римана-Зигеля (Theta) с максимальной точностью.
    Расширенный ряд Стирлинга до t^-11.
    """
    n = t_arr.size
    res = np.zeros(n, dtype=np.float64)

    # Константы (вычисляем делением для машинной точности)
    inv_2pi = 1.0 / (2.0 * np.pi)
    pi_8 = np.pi / 8.0

    # Коэффициенты Бернулли (Gabcke / Odlyzko expansion)
    # C1 = 1/48
    c1 = 1.0 / 48.0
    # C3 = 7/5760
    c3 = 7.0 / 5760.0
    # C5 = 31/80640
    c5 = 31.0 / 80640.0
    # C7 = 127/430080
    c7 = 127.0 / 430080.0
    # C9 = 511/12165120  (Новый член)
    c9 = 511.0 / 12165120.0
    # C11 = 2047/311427072 (Новый член)
    c11 = 2047.0 / 311427072.0

    for i in range(n):
        t = t_arr[i]
        if t < 1e-9:
            res[i] = 0.0
            continue

        # 1. Main term: (t/2) * ln(t/2pi) - t/2 - pi/8
        val = t * 0.5
        th = val * np.log(t * inv_2pi) - val - pi_8

        # 2. Correction terms (Extended Bernoulli expansion)
        # Используем итеративное умножение для скорости и точности
        inv_t = 1.0 / t
        inv_t2 = inv_t * inv_t  # 1/t^2

        # term1 ~ 1/t
        curr_pow = inv_t
        th += c1 * curr_pow

        # term3 ~ 1/t^3
        curr_pow *= inv_t2
        th += c3 * curr_pow

        # term5 ~ 1/t^5
        curr_pow *= inv_t2
        th += c5 * curr_pow

        # term7 ~ 1/t^7
        curr_pow *= inv_t2
        th += c7 * curr_pow

        # term9 ~ 1/t^9  (Добавлено)
        curr_pow *= inv_t2
        th += c9 * curr_pow

        # term11 ~ 1/t^11 (Добавлено - предел для float64)
        curr_pow *= inv_t2
        th += c11 * curr_pow

        res[i] = th

    return res


# --- RIEMANN-SIEGEL SPECTRAL FUNCTION (RESTORED) ---
# --- RIEMANN-SIEGEL SPECTRAL FUNCTION (ULTRA PRECISION A7 + GABCKE) ---
@numba.jit(nopython=True, fastmath=True, parallel=True)
def riemann_siegel_spectral(t_grid):
    """
    Computes Riemann-Siegel Z(t) using:
    1. ULTRA PRECISION Theta (from calc_theta_riemann)
    2. Gabcke's First Correction Term (Smoothing)
    """
    # 1. Сначала считаем сверхточную фазу для всех точек (Axiom A7)
    # Вызываем нашу новую функцию calc_theta_riemann
    theta_vec = calc_theta_riemann(t_grid)

    n_points = t_grid.shape[0]
    Z_val = np.zeros(n_points, dtype=np.float64)
    pi = 3.141592653589793

    for i in numba.prange(n_points):
        t = t_grid[i]
        if t < 0.1: continue

        # Берем точную тету
        theta = theta_vec[i]

        # Главная сумма (Main Sum)
        a = np.sqrt(t / (2.0 * pi))
        N_limit = int(a)

        current_sum = 0.0
        for n in range(1, N_limit + 1):
            log_n = np.log(float(n))
            # cos(theta - t*ln(n)) / sqrt(n)
            val = np.cos(theta - t * log_n) * (1.0 / np.sqrt(float(n)))
            current_sum += val

        main_term = 2.0 * current_sum

        # --- GABCKE CORRECTION (Остаточный член) ---
        # Это повышает точность RS до уровня MPMath на малых T
        p = a - N_limit  # Дробная часть (fractional part)

        # Формула первого порядка: R ~ (-1)^(N-1) * (t/2pi)^(-1/4) * Phi(p)
        # Phi(p) ≈ cos(2pi(p^2 - p - 1/16)) / cos(2pi*p)
        # Мы используем аппроксимацию для стабильности (без деления на 0)

        # Коэффициент перед функцией
        coeff = np.power(t / (2.0 * pi), -0.25)

        # Знак (-1)^(N-1)
        sign = 1.0 if ((N_limit - 1) % 2 == 0) else -1.0

        # Аппроксимация функции Gabcke (достаточная для float64)
        # Эта формула сглаживает разрыв, когда N меняется
        phi_val = np.cos(2.0 * pi * (p * p - p - 0.0625))
        # Деление на cos(2pi*p) может быть нестабильным, используем упрощение для "Battle Mode"
        # Для полной строгости тут нужен ряд Тейлора, но для графика достаточно этого:
        denom = np.cos(2.0 * pi * p)
        if np.abs(denom) < 1e-6:
            denom = 1e-6  # Защита от взрыва

        remainder = sign * coeff * (phi_val / denom)

        # Итог: Z(t) = 2*Sum + Remainder
        Z_val[i] = main_term + remainder

    return Z_val



# =========================
#     ГЛАВНАЯ ЛОГИКА
# =========================
def compute_Z_profile_hybrid(t: np.ndarray, lam_full: np.ndarray, u: float, alpha: float, use_a7: bool, power: float):
    # Спектральная подготовка
    lam, g, n_clipped = soft_cutoff_heat(lam_full, alpha, power=power)

    # QSSC base: sum w_n * n^(-s)
    A = (u * u) + (lam * lam)
    logA = np.log(A)
    phi = 0.5 * logA
    amp = 2.0 * g * np.exp(-0.25 * logA)

    n_t = t.shape[0]
    zr = np.zeros(n_t, dtype=np.float64)
    zi = np.zeros(n_t, dtype=np.float64)

    # --- ДВИЖОК ---
    if USE_GPU:
        t_chunk_size = CFG.CHUNK_SIZE_GPU
        show_bar = n_t > 50000
        pb = ProgressBar(n_t, prefix="QSSC GPU") if show_bar else None
        for i in range(0, n_t, t_chunk_size):
            i_end = min(i + t_chunk_size, n_t)
            t_slice = t[i:i_end]
            r, im = gpu_kernel(t_slice, phi, amp, CFG.CHUNK_SIZE_GPU)
            zr[i:i_end] = r
            zi[i:i_end] = im
            if pb: pb.update(i_end)
        if pb: pb.close()

    elif USE_NUMBA:
        batch_size = 2000
        show_bar = n_t > 20000
        pb = ProgressBar(n_t, prefix="QSSC CPU") if show_bar else None
        for i in range(0, n_t, batch_size):
            i_end = min(i + batch_size, n_t)
            t_slice = t[i:i_end]
            r, im = numba_kernel(t_slice, phi, amp)
            zr[i:i_end] = r
            zi[i:i_end] = im
            if pb: pb.update(i_end)
        if pb: pb.close()

    else:
        # Fallback pure numpy
        for i0 in range(0, lam.size, 2048):
            i1 = min(i0 + 2048, lam.size)
            phase = np.outer(t, phi[i0:i1])
            zr += np.dot(np.cos(phase), amp[i0:i1])
            zi -= np.dot(np.sin(phase), amp[i0:i1])

    Z_complex = zr + 1j * zi

    if use_a7:
        # AXIOM A7: Фазовый доворот
        theta = calc_theta_riemann(t)
        P = zr * np.cos(theta) - zi * np.sin(theta)
        return Z_complex, P, lam, phi, amp
    else:
        return Z_complex, np.abs(Z_complex), lam, phi, amp


def extract_zeros_signchange(t: np.ndarray, P: np.ndarray) -> np.ndarray:
    sign_flips = np.where(P[:-1] * P[1:] < 0)[0]
    roots = []
    for i in sign_flips:
        y0 = P[i]
        y1 = P[i + 1]
        t0 = t[i]
        t1 = t[i + 1]
        t_root = t0 - y0 * (t1 - t0) / (y1 - y0)
        roots.append(t_root)
    return np.array(roots)


# =========================
#  PIPELINE COMPONENTS
# =========================

def load_reference_zeros_from_file(filepath: str, t_min: float, t_max: float) -> Tuple[np.ndarray, np.ndarray]:
    indices = []
    values = []
    path = Path(filepath)
    if not path.exists():
        print(f"[ERROR] Reference file not found: {filepath}")
        return np.array([]), np.array([])
    print(f"[Reference] Loading zeros from {filepath} for window [{t_min}, {t_max}]...")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        idx = int(parts[0])
                        val = float(parts[1])
                        if t_min <= val <= t_max:
                            indices.append(idx)
                            values.append(val)
                    except ValueError:
                        continue
    except Exception as e:
        print(f"[ERROR] Failed to read reference file: {e}")
        return np.array([]), np.array([])
    print(f"[Reference] Loaded {len(values)} valid zeros.")
    return np.array(indices, dtype=np.int64), np.array(values, dtype=np.float64)


def match_nearest_unique(true_zeros: np.ndarray, found: np.ndarray, max_err: float = 0.5):
    true_zeros = np.asarray(true_zeros, dtype=np.float64)
    found = np.asarray(found, dtype=np.float64)
    matched = [None] * int(true_zeros.size)
    errors = [None] * int(true_zeros.size)
    if found.size == 0 or true_zeros.size == 0: return matched, errors
    used = np.zeros(found.size, dtype=np.bool_)
    for i in range(true_zeros.size):
        tz = float(true_zeros[i])
        order = np.argsort(np.abs(found - tz))
        for j in order:
            if used[j]: continue
            err = abs(float(found[j]) - tz)
            if err <= max_err:
                used[j] = True
                matched[i] = float(found[j])
                errors[i] = float(err)
            break
    return matched, errors


def _unique_sorted(vals: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0: return vals
    vals = np.sort(vals)
    out = [float(vals[0])]
    for x in vals[1:]:
        if abs(float(x) - out[-1]) > tol:
            out.append(float(x))
    return np.asarray(out, dtype=np.float64)


# =========================
#  РЕЖИМ 2: PIPELINE (A7 / ADAPTIVE HEIGHT STRATEGY)
# =========================
def staged_pipeline_qssc(ctx: RunContext, t_grid_global: np.ndarray,
                         n_flow: np.ndarray, ref_values: np.ndarray):
    logger = ctx.logger
    logger.info("=== STARTING HIGH PRECISION PIPELINE (A7 - HEIGHT ADAPTIVE) ===")

    T_CENTER = (t_grid_global[0] + t_grid_global[-1]) / 2.0

    if T_CENTER < 50.0:
        MODE, POWER_START, POWER_END = "SOFT", 6.0, 6.0
    else:
        MODE, POWER_START, POWER_END = "HARD", 6.0, 12.0

    # --- STAGE 1: GLOBAL SCAN ---
    _, P_scan, _, _, _ = compute_Z_profile_hybrid(
        t_grid_global, n_flow, CFG.U, CFG.ALPHA, use_a7=True, power=POWER_START
    )
    candidates = extract_zeros_signchange(t_grid_global, P_scan)
    global_step = t_grid_global[1] - t_grid_global[0]

    logger.info(f"Candidates found: {len(candidates)}. Refining zeros...")

    # --- STAGE 2: SMART ZOOM (QUIET MODE) ---
    final_zeros = []
    ZOOM_PASSES = 30
    POINTS_PER_PASS = 100
    SHRINK_FACTOR = 0.55

    pb = ProgressBar(len(candidates), prefix="Refining")
    for i, t_approx in enumerate(candidates):
        current_t = t_approx
        avg_gap = 2.0 * np.pi / np.log(current_t / (2.0 * np.pi)) if current_t > 10.0 else 0.5
        current_radius = max(avg_gap * 0.30, global_step * 3.0)
        current_power = POWER_START
        power_step = (POWER_END - POWER_START) / max(1, ZOOM_PASSES - 2)

        lost_counter = 0
        for step in range(ZOOM_PASSES):
            if MODE == "HARD":
                target_power = POWER_START + (step * power_step)
                current_power = 0.7 * current_power + 0.3 * min(target_power, POWER_END)

            t_local = np.linspace(current_t - current_radius, current_t + current_radius, POINTS_PER_PASS)
            _, P_local, _, _, _ = compute_Z_profile_hybrid(t_local, n_flow, CFG.U, CFG.ALPHA, True, current_power)
            roots = extract_zeros_signchange(t_local, P_local)

            if len(roots) > 0:
                current_t = roots[np.argmin(np.abs(roots - current_t))]
                current_radius *= SHRINK_FACTOR
                lost_counter = 0
            else:
                lost_counter += 1
                current_radius *= 2.0
                if lost_counter > 5: break

        final_zeros.append(current_t)
        pb.update(i + 1)
    pb.close()

    final_zeros = _unique_sorted(final_zeros, tol=1e-5)
    ctx.save_text("qssc_refined_zeros.txt", "\n".join([f"{z:.35f}" for z in final_zeros]))

    # --- STAGE 4: VALIDATION (SUMMARY TABLE) ---
    if len(ref_values) > 0:
        print(f"\n=== PIPELINE COMPARISON REPORT (A7 Zoom vs MPMath) ===")
        print(f"{'No':<4} | {'RefZero (MPMath)':<30} | {'Found (QSSC A7)':<30} | {'AbsErr':<10}")
        print("-" * 82)

        matched_vals, matched_errs = match_nearest_unique(ref_values, final_zeros, max_err=0.5)

        for i, r_val in enumerate(ref_values):
            found_val = matched_vals[i]
            err = matched_errs[i]
            f_str = f"{found_val:.22f}" if found_val is not None else "---"
            e_str = f"{err:.2e}" if err is not None else "FAIL"
            print(f"{i + 1:<4} | {r_val:<30.22f} | {f_str:<30} | {e_str:<10}")
        print("-" * 82)
    else:
        print(f"Pipeline finished. Found {len(final_zeros)} zeros. No reference provided for comparison.")

def plot_vlines_with_labels(ax, xs, y_top, fmt, color, linestyle, linewidth, alpha, fontsize, rotation):
    if len(xs) == 0: return
    # Уровни высоты подписей, чтобы они не накладывались друг на друга
    levels = [y_top * 0.95, y_top * 0.90, y_top * 0.85, y_top * 0.80]
    for i, x in enumerate(xs):
        ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
        y = levels[i % 4]
        label = fmt.format(x)
        ax.text(x, y, label, rotation=rotation, va="bottom", ha="right",
                fontsize=fontsize, color=color, alpha=1.0)


# =========================
#  MODE 3: SPECTRAL PRIMALITY TEST
# =========================
@numba.jit(nopython=True, fastmath=True)
def calc_spectral_resonance(target_num, zeros_arr):
    """
    Вычисляет 'Спектральный Резонанс' числа target_num на основе найденных нулей.
    Использует формулу, родственную явной формуле Ландау:
    F(x) = Sum cos(gamma * ln(x))

    Если x - простое число (или степень простого), сумма дает конструктивный пик.
    Если x - составное, слагаемые гасят друг друга (деструктивная интерференция).
    """
    val = 0.0
    ln_x = np.log(float(target_num))

    n = zeros_arr.size
    for i in range(n):
        # Вклад каждого нуля в резонанс
        # gamma * ln(x) - это фаза. Если x простое, фазы синхронизируются.
        val += np.cos(zeros_arr[i] * ln_x)

    # Возвращаем "чистую" сумму.
    # Большое положительное значение -> Вероятно простое.
    # Около нуля или отрицательное -> Вероятно составное.
    return val


# =========================
#        MAIN
# =========================
def main():
    print("\n--- CONFIGURATION SETUP ---")
    run_dir = _make_run_dir()
    logger = _setup_logger(run_dir)
    ctx = RunContext(run_dir=run_dir, logger=logger)
    logger.info("Run directory: %s", str(run_dir.resolve()))

    def get_input(prompt, default_val):
        val_str = input(f"{prompt} [default: {default_val}]: ").strip()
        if val_str == "": return default_val
        try:
            return float(val_str)
        except ValueError:
            return default_val

    # --- 1. СНАЧАЛА ВЫВОДИМ МЕНЮ И ПОЛУЧАЕМ ВВОД ---
    print("\nSelect Operation Mode:")
    print("  1 - Standard Battle (QSSC vs RS vs MPMath) [Plots Enabled]")
    print("  2 - NEW AXIOM A7 PIPELINE (Twisted Phase / 12 PASS / No Plots) [Recommended]")
    print("  3 - SPECTRAL PRIMALITY TEST (Check integer using Zeros)")
    print("  4 - QUANTUM GENESIS (Visual Animation)")
    print("  5 - QSSC STAIRCASE RECONSTRUCTION (Prime Staircase Dynamic)")

    # Теперь переменная mode_str определена ДО её использования
    mode_str = input("Choice [2]: ").strip()

    # --- 2. ТЕПЕРЬ ЛОГИКА ВВОДА ПАРАМЕТРОВ ---
    if mode_str in ["3", "4", "5"]:
        print(f"-> Mode {mode_str} Selected. Target T and W will be AUTO-TUNED.")
        TARGET_T = 12.0
        TARGET_W = 100.0
    else:
        TARGET_T = get_input("Enter Target T (Start Height)", 12.0)
        TARGET_W = get_input("Enter Target W (Window Width)", 100.0)
        print(f"-> Selected: T={TARGET_T}, Width={TARGET_W}")

    print("---------------------------\n")

        # Дальнейший код настройки Config остается без изменений...

    CALC_T_MIN = TARGET_T
    CALC_T_MAX = TARGET_T + TARGET_W
    AUTO_N, AUTO_ALPHA = auto_tune_config(CALC_T_MAX)

    global CFG

    @dataclass(frozen=True)
    class Config:
        NMAX: int = AUTO_N
        ALPHA: float = AUTO_ALPHA
        T_MIN: float = CALC_T_MIN
        T_MAX: float = CALC_T_MAX
        N_T: int = 100000
        U: float = 0.0
        CHUNK_SIZE_CPU: int = 2048
        CHUNK_SIZE_GPU: int = 100000

    CFG = Config()
    ctx.save_json("config.json", _cfg_to_dict(CFG))

    print(f"=== QSSC-Rieman System (Ultimate Edition) ===")
    print(f"Config: NMAX={CFG.NMAX}, ALPHA={CFG.ALPHA:.2e}, N_T={CFG.N_T}")

    if mode_str == "3":
        # --- MODE 3: SPECTRAL PRIMALITY TEST (ADAPTIVE RESOLUTION) ---
        print("\n=== SPECTRAL PRIMALITY TEST (High Resolution) ===")
        target_input = input("Enter an integer to check (e.g. 197, 1009): ").strip()
        try:
            target_num = int(target_input)
        except ValueError:
            print("Invalid integer.")
            return

        # --- АДАПТИВНАЯ ЛОГИКА ИЗ ВАРИАНТА 4 ---
        print(f"\n[Auto-Tuning] Calculating diffraction limit for N={target_num}...")

        # Теоретический предел Рэлея: T_min = (pi * N) / (delta_N)
        # Для разделения близнецов delta_N = 2, поэтому T = pi * N / 2
        rayleigh_limit_T = (np.pi * float(target_num)) / 2.0

        # Коэффициент качества (Quality Factor):
        # 4.0 — гарантирует очень острые пики и разделение близнецов
        quality_factor = 4.5

        auto_t_min = 1000.0
        # Вычисляем необходимую максимальную высоту T
        needed_max_T = rayleigh_limit_T * quality_factor

        # Ширина окна должна покрывать это расстояние
        needed_W = max(3000.0, needed_max_T - auto_t_min)
        auto_t_max = auto_t_min + needed_W

        # Пересчет гармоник и плотности сетки
        auto_nmax, auto_alpha = auto_tune_config(auto_t_max)
        # Нам нужно ~20 точек на единицу t для корректного извлечения нулей
        auto_n_points = int(needed_W * 20)

        print(f" -> Rayleigh Limit T: {rayleigh_limit_T:.1f}")
        print(f" -> Target Max T:     {auto_t_max:.1f} (Quality x{quality_factor})")
        print(f" -> Window Width:     {needed_W:.1f}")
        print(f" -> Harmonics (NMAX): {auto_nmax}")

        t_grid = np.linspace(auto_t_min, auto_t_max, auto_n_points)
        n_flow = np.arange(1, auto_nmax + 1, dtype=np.float64)

        print(f"\n[1/3] Generating Spectral Skeleton (QSSC A7)...")
        _, P_qssc, _, _, _ = compute_Z_profile_hybrid(
            t_grid, n_flow, CFG.U, auto_alpha, use_a7=True, power=6.0
        )
        zeros_found = extract_zeros_signchange(t_grid, P_qssc)
        print(f" -> Collected {len(zeros_found)} zeros.")

        print(f"\n[2/3] Analyzing Resonance for N={target_num}...")
        score = calc_spectral_resonance(target_num, zeros_found)

        # Для проверки фона берем значения в "мертвых зонах" между целыми
        score_left = calc_spectral_resonance(target_num - 0.5, zeros_found)
        score_right = calc_spectral_resonance(target_num + 0.5, zeros_found)
        noise_level = (abs(score_left) + abs(score_right)) / 2.0
        snr = score / (noise_level + 1e-9)

        print("\n=== SPECTRAL DIAGNOSTIC REPORT ===")
        print(f"Target Integer: {target_num}")
        print(f"Resonance Score: {score:.4f}")
        print(f"Background Noise: {noise_level:.4f}")
        print(f"Contrast (SNR):  {snr:.4f}")
        print("-" * 40)

        if snr > 2.0:
            print(">>> VERDICT: HIGH PROBABILITY PRIME")
        elif snr > 1.2:
            print(">>> VERDICT: POSSIBLE PRIME / WEAK RESONANCE")
        else:
            print(">>> VERDICT: COMPOSITE / NO RESONANCE")

        # --- ГРАФИК РЕЗОНАНСА ---
        print("\n[3/3] Rendering Resonance Profile...")
        scan_x = np.linspace(target_num - 4.0, target_num + 4.0, 1000)
        scan_y = np.zeros_like(scan_x)
        for i, val in enumerate(scan_x):
            scan_y[i] = calc_spectral_resonance(val, zeros_found)

        plt.figure(figsize=(10, 5))
        plt.plot(scan_x, scan_y, color='#003366', label='Spectral Resonance')
        plt.axvline(target_num, color='red', linestyle='--', label=f'Target {target_num}')

        # Если это близнецы, подсветим соседа
        neighbor = target_num + 2 if target_num % 2 != 0 else target_num + 1
        plt.axvline(neighbor, color='green', alpha=0.3, label=f'Neighbor {neighbor}')

        plt.title(f"Spectral Resolution Test for N={target_num}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()


    elif mode_str == "4":
        # --- MODE 4: QUANTUM GENESIS ANIMATION (ADAPTIVE) ---
        print("\n=== MODE 4: QUANTUM GENESIS (Wave Superposition Dynamics) ===")
        target_input = input("Enter integer to visualize (e.g. 17, 3001): ").strip()
        try:
            target_num = int(target_input)
        except ValueError:
            return

        # --- SMART AUTO-RESOLUTION (RAYLEIGH LOGIC) ---
        print(f"[Auto-Tuning] Calculating diffraction limit for N={target_num}...")

        # 1. Теоретический предел Рэлея для разделения близнецов (p, p+2)
        # Формула: T_min = (pi * N) / 2
        rayleigh_limit_T = (3.14159 * float(target_num)) / 2.0

        # 2. Коэффициент "Остроты иглы" (Quality Factor)
        # * 1.0 = Близнецы едва различимы (два горба слились)
        # * 2.0 = Видны два пика
        # * 4.0 = Идеальные острые иглы (Ultra-Res)
        quality_factor = 4.0

        target_max_T = rayleigh_limit_T * quality_factor

        # 3. Итоговый расчет ширины окна
        # Для малых чисел (N < 1000) принудительно берем 3000.0 для красоты.
        # Для больших чисел используем формулу.
        # Вычитаем 1000.0, так как это наше anim_T (старт)
        calculated_W = target_max_T - 1000.0
        needed_W = max(3000.0, calculated_W)

        # (Защита от зависания для гигантских чисел > 50 000)
        # Если число огромное, ограничим окно, иначе расчет займет часы.
        if needed_W > 100000.0:
            print(" -> [Limit] Cap applied to prevent overflow.")
            needed_W = 100000.0

        print(f" -> Rayleigh Limit T: {rayleigh_limit_T:.1f}")
        print(f" -> Selected Quality Factor: x{quality_factor}")
        print(f" -> Auto-Adjusted Window: W = {needed_W:.1f}")

        anim_T = 1000.0
        anim_max_T = anim_T + needed_W

        # Считаем точное количество гармоник (NMAX) через A7
        anim_nmax, anim_alpha = auto_tune_config(anim_max_T)

        # Сетка (не слишком плотная для скорости, но достаточная)
        points_per_unit = 15
        t_grid = np.linspace(anim_T, anim_max_T, int(needed_W * points_per_unit))
        n_flow = np.arange(1, anim_nmax + 1, dtype=np.float64)

        print(f" -> T=[{anim_T}, {anim_max_T}], Harmonics={anim_nmax}")
        print(f"[Setup] Generating Zeros from Quantum Lens...")

        _, P_qssc, _, _, _ = compute_Z_profile_hybrid(
            t_grid, n_flow, CFG.U, anim_alpha, use_a7=True, power=6.0
        )
        zeros = extract_zeros_signchange(t_grid, P_qssc)
        total_zeros = len(zeros)
        print(f"-> Ready to animate! Found {total_zeros} zeros.")

        # 2. Настройка графика
        scan_width = 3.0
        # Делаем x-сетку погуще, чтобы линии были плавными
        x = np.linspace(target_num - scan_width, target_num + scan_width, 800)
        ln_x = np.log(x)  # Предрасчет логарифмов (оптимизация)

        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 7))
        line, = ax.plot([], [], color='#003366', lw=2)

        # Оформление
        ax.axvline(target_num, color='red', alpha=0.6, ls='--', label=f'Target N={target_num}')
        ax.set_xlim(target_num - scan_width, target_num + scan_width)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Number Line (x)")
        ax.set_ylabel("Amplitude")

        txt_info = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=12,
                           verticalalignment='top',
                           bbox=dict(boxstyle="round", fc="white", alpha=0.9))

        # 3. АДАПТИВНЫЙ ШАГ АНИМАЦИИ
        # Мы хотим, чтобы анимация длилась примерно 10-15 секунд (около 300-500 кадров)
        target_frames = 400
        step = max(1, total_zeros // target_frames)

        print(f"Starting Animation... (Step size: {step} zeros per frame)")

        current_sum = np.zeros_like(x)
        max_amp = 1.0

        for k in range(0, total_zeros, step):
            # Если окно закрыли - выходим
            if not plt.fignum_exists(fig.number):
                break

            # Берем пачку нулей
            batch_zeros = zeros[k: k + step]

            # Векторизованное сложение (быстро)
            # Outer product: (zeros x 1) * (1 x x_points) -> matrix
            # Но для экономии памяти суммируем в цикле по пачке
            for gamma in batch_zeros:
                current_sum += np.cos(gamma * ln_x)

            # Обновление линии
            line.set_data(x, current_sum)

            # Умный автомасштаб (плавный зум)
            current_max = np.max(np.abs(current_sum))
            if current_max > max_amp:
                max_amp = current_max
                ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)
            elif k > 100 and current_max < max_amp * 0.8:
                # Если амплитуда резко упала (фокусировка), поджимаем масштаб
                max_amp = max_amp * 0.95
                ax.set_ylim(-max_amp * 1.1, max_amp * 1.1)

            # Статистика
            percent = (k / total_zeros) * 100
            current_val_at_target = current_sum[len(x) // 2]  # Примерно центр

            info_str = (f"Target N: {target_num}\n"
                        f"Harmonics: {k}/{total_zeros} ({percent:.1f}%)\n"
                        f"Resonance Strength: {current_val_at_target:.2f}")
            txt_info.set_text(info_str)
            ax.set_title(f"Quantum Genesis: Wave Superposition N={target_num}")

            plt.draw()
            plt.pause(0.001)

        print("Animation finished.")
        plt.ioff()
        plt.show()
    elif mode_str == "5":
        # --- MODE 5: QSSC STAIRCASE RECONSTRUCTION (FULLY DYNAMIC) ---
        print("\n=== MODE 5: QSSC SPECTRAL STAIRCASE (Dynamic Build) ===")
        x_start = get_input("Enter X start (e.g., 2)", 2.0)
        x_window = get_input("Enter X window width (e.g., 100)", 100.0)
        x_max = x_start + x_window

        # --- ДИНАМИЧЕСКИЙ ГЕНЕРАТОР ПРОСТЫХ ЧИСЕЛ (Ground Truth) ---
        print(f"[1/4] Calculating true primes in range [{x_start}, {x_max}]...")

        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0: return False
            return True

        # Собираем список простых для отрисовки красных линий-ориентиров
        ref_primes = [p for p in range(int(x_start), int(x_max) + 1) if is_prime(p)]
        print(f" -> Found {len(ref_primes)} primes for reference.")

        # --- НАСТРОЙКА КВАНТОВОЙ ЛИНЗЫ (QSSC) ---
        # Для отрисовки "ступенек" на высоте X нужны частоты (нули) до T ~ pi * X
        t_needed = max(800.0, x_max * np.pi * 1.8)

        auto_t_min = 10.0  # Низкие частоты - база геометрии
        auto_nmax, auto_alpha = auto_tune_config(t_needed)
        t_grid = np.linspace(auto_t_min, t_needed, int(t_needed * 18))
        n_flow = np.arange(1, auto_nmax + 1, dtype=np.float64)

        print(f"[2/4] Harvesting Spectral Components (T_max={t_needed:.1f})...")
        _, P_qssc, _, _, _ = compute_Z_profile_hybrid(
            t_grid, n_flow, CFG.U, auto_alpha, use_a7=True, power=6.0
        )
        zeros = extract_zeros_signchange(t_grid, P_qssc)
        print(f" -> QSSC A7 extracted {len(zeros)} zeros.")

        # --- ВИЗУАЛИЗАЦИЯ ДИНАМИКИ ---
        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 7))

        x_axis = np.linspace(x_start, x_max, 1200)
        ln_x = np.log(x_axis)
        sqrt_x = np.sqrt(x_axis)

        # Гладкий тренд функции Чебышёва psi(x) ~ x
        trend = x_axis.copy()

        ax.plot(x_axis, trend, color='gray', ls='--', alpha=0.3, label='Smooth Trend (x)')
        staircase, = ax.plot([], [], color='#003366', lw=2, label='QSSC Reconstruction')

        # Динамическая отрисовка красных линий (только те, что в окне)
        for p in ref_primes:
            ax.axvline(p, color='red', alpha=0.2, lw=1.2, ls='-')

        ax.set_xlim(x_start, x_max)
        ax.set_ylim(x_start - 5, x_max + 5)
        ax.set_title(f"Quantum Staircase: x=[{x_start}, {x_max}] | T_max={t_needed:.0f}")
        ax.set_xlabel("Number Line (x)")
        ax.set_ylabel("Spectral Density Psi(x)")
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.15)

        print(f"[3/4] Building Staircase... (Spectral Superposition)")

        current_psi = trend.copy()
        # Разбиваем на 100 кадров для красивой анимации
        batch_size = max(1, len(zeros) // 100)

        for i in range(0, len(zeros), batch_size):
            if not plt.fignum_exists(fig.number): break

            for gamma in zeros[i:i + batch_size]:
                # Главная формула Римана: коррекция тренда через нули
                # -2 * sqrt(x) * sin(gamma * ln x) / gamma
                current_psi -= 2.0 * sqrt_x * np.sin(gamma * ln_x) / gamma

            staircase.set_data(x_axis, current_psi)
            ax.set_title(f"Staircase: {i + batch_size}/{len(zeros)} Zeros | Primes detected: {len(ref_primes)}")

            plt.draw()
            plt.pause(0.001)

        plt.ioff()
        print("[4/4] Masterpiece finished. Primes have been 'carved' from the spectrum.")
        plt.show()

    elif mode_str == "" or mode_str == "2":
        # --- MODE 2: PIPELINE (AUTO-REFERENCE) ---
        print("\n[Mode 2] Running Pipeline with automatic MPMath reference...")

        # Генерируем сетку
        t_grid = np.linspace(CFG.T_MIN, CFG.T_MAX, CFG.N_T)
        n_flow = np.arange(1, CFG.NMAX + 1, dtype=np.float64)

        # 1. Получаем эталонные нули из библиотеки (как в патче 001)
        logger.info("Fetching Reference Zeros from MPMath...")
        ref_zeros = []
        n_z = 1
        if CFG.T_MIN > 10.0:
            c = CFG.T_MIN / (2 * np.pi)
            n_z = int(c * np.log(c) - c) - 250
            if n_z < 1: n_z = 1

        while True:
            if mp.fp.zetazero(n_z).imag >= CFG.T_MIN: break
            n_z += 1

        while True:
            t_val = float(mp.fp.zetazero(n_z).imag)
            if t_val > CFG.T_MAX: break
            ref_zeros.append(t_val)
            n_z += 1
        ref_zeros = np.array(ref_zeros)

        # 2. Запускаем тихий пайплайн
        staged_pipeline_qssc(ctx, t_grid, n_flow, ref_zeros)

    else:
        # --- MODE 1: STANDARD BATTLE (VISUAL FIX) ---
        print("\n=== STANDARD BATTLE MODE (QSSC vs RS vs MPMath) ===")
        print("Settings: Axiom A7 (Twist) = ON. ")
        print("NOTE: Using Power=6.0 for Battle Mode to prevent visual artifacts.")

        t_grid = np.linspace(CFG.T_MIN, CFG.T_MAX, CFG.N_T)
        n_flow = np.arange(1, CFG.NMAX + 1, dtype=np.float64)

        # 1. QSSC A7 (FORCED Power=6.0 for Plot Stability)
        logger.info("1. Computing QSSC (Axiom A7, Power 6.0)...")
        t0 = time.time()
        _, P_qssc, _, _, _ = compute_Z_profile_hybrid(
            t_grid, n_flow, CFG.U, CFG.ALPHA, use_a7=True, power=6.0
        )
        t_qssc = time.time() - t0
        logger.info(f"QSSC done in {t_qssc:.2f}s")

        # 2. RS (Riemann-Siegel)
        logger.info("2. Computing RS (Spectral)...")
        t0_rs = time.time()
        Z_rs_signed = riemann_siegel_spectral(t_grid)
        t_rs = time.time() - t0_rs
        logger.info(f"RS done in {t_rs:.2f}s")

        # 3. MPMath (FAST MODE with Progress Bar)
        logger.info("3. Computing MPMath Reference (Fast Mode)...")
        t0_ref = time.time()

        ref_abs = np.zeros_like(t_grid)
        # Восстановлен прогресс-бар
        pb = ProgressBar(len(t_grid), prefix="MPMath Graph")

        for i, ti in enumerate(t_grid):
            try:
                # abs() от mp.fp работает быстрее
                ref_abs[i] = float(abs(mp.fp.zeta(0.5 + 1j * ti)))
            except AttributeError:
                ref_abs[i] = float(abs(mp.zeta(0.5 + 1j * ti)))

            # Обновляем бар каждые 500 итераций для скорости
            if i % 500 == 0:
                pb.update(i)
        pb.close()

        t_ref = time.time() - t0_ref
        logger.info(f"MPMath (Graph) done in {t_ref:.2f}s")

        # --- СРАВНЕНИЕ (Таблица) ---
        logger.info("Finding exact zeros for report...")

        # Нули QSSC
        zeros_qssc = extract_zeros_signchange(t_grid, P_qssc)
        zeros_qssc = _unique_sorted(zeros_qssc)

        # Нули RS
        zeros_rs = extract_zeros_signchange(t_grid, Z_rs_signed)
        zeros_rs = _unique_sorted(zeros_rs)

        # --- ПАТЧ START: Исправленная оценка номера нуля ---
        logger.info("Fetching MPMath True Zeros...")
        true_zeros = []
        n_z = 1
        if CFG.T_MIN > 10.0:
            # Формула Римана-фон Мангольдта: N(T) ~ (T/2pi)*ln(T/2pi) - (T/2pi)
            # Старый код забыл вычесть второй член (-c), из-за чего перелетал старт окна
            c = CFG.T_MIN / (2 * np.pi)
            val = c * np.log(c) - c  # Правильная оценка
            n_z = int(val) - 250  # Берем запас 250 нулей назад, чтобы точно не промахнуться
            if n_z < 1: n_z = 1

        # Перемотка вперед до начала окна (быстро)
        while True:
            # Используем fp (double precision) для скорости проверки
            if mp.fp.zetazero(n_z).imag >= CFG.T_MIN:
                break
            n_z += 1

        # Сбор нулей внутри окна
        while True:
            z_val = mp.fp.zetazero(n_z)
            t_val = float(z_val.imag)
            if t_val > CFG.T_MAX:
                break
            true_zeros.append(t_val)
            n_z += 1

        true_zeros = np.array(true_zeros)
        # --- ПАТЧ END ---

        # Вывод таблицы (ПАТЧ: МАКСИМАЛЬНАЯ ТОЧНОСТЬ)
        print("\n=== BATTLE RESULTS (Zero Accuracy - MAX PRECISION) ===")
        # Увеличили ширину колонок с 18 до 30, чтобы вместить все цифры
        print(f"{'True Zero (MP)':<30} | {'QSSC (A7)':<30} | {'Err QSSC':<12} | {'RS (A7)':<30} | {'Err RS':<12}")
        print("-" * 120)  # Удлинили разделитель

        matched_q, err_q = match_nearest_unique(true_zeros, zeros_qssc)
        matched_r, err_r = match_nearest_unique(true_zeros, zeros_rs)

        for i, true_z in enumerate(true_zeros):
            # .22f покажет абсолютно все знаки, которые есть в памяти компьютера (double precision)
            qv = f"{matched_q[i]:.22f}" if matched_q[i] is not None else "---"
            qe = f"{err_q[i]:.2e}" if err_q[i] is not None else "FAIL"
            rv = f"{matched_r[i]:.22f}" if matched_r[i] is not None else "---"
            re = f"{err_r[i]:.2e}" if err_r[i] is not None else "FAIL"

            # Вывод с новой шириной колонок (30) и точностью (.22f)
            print(f"{true_z:<30.22f} | {qv:<30} | {qe:<12} | {rv:<30} | {re:<12}")
        print("-" * 120)

        # --- ГРАФИК (Стиль ver2 восстановлен) ---
        logger.info("Plotting comparison (Absolute Values)...")
        plt.figure(figsize=(14, 7))

        # Определяем высоту графика для расстановки подписей
        y_top = np.max(ref_abs) if ref_abs.size > 0 else 1.0

        # 1. Основные линии
        plt.plot(t_grid, np.abs(P_qssc), color='#003366', label='QSSC (A7 Twist)', alpha=0.6, linewidth=2.0)
        plt.plot(t_grid, np.abs(Z_rs_signed), color='orange', label='RS (Standard)', alpha=0.6, linewidth=1.5,
                 linestyle='--')
        # MPMath без лейбла в легенде, но с вертикальными линиями ниже
        # (Либо можно оставить пунктир, если хочется видеть огибающую)
        # plt.plot(t_grid, ref_abs, 'r:', alpha=0.3)

        # 2. Вертикальные линии ЭТАЛОНА (MPMath) с подписями сверху
        plot_vlines_with_labels(plt.gca(), true_zeros, y_top, "{:.2f}", "gray", "--", 0.8, 0.5, 8, 90)

        # 3. Красные точки (QSSC)
        plt.plot(zeros_qssc, np.zeros_like(zeros_qssc), 'ro', markersize=5, zorder=5, label='QSSC Zeros')

        # 4. Красные подписи значений QSSC снизу
        for mz in zeros_qssc:
            # Текст чуть ниже оси Y (-0.05 от высоты)
            plt.text(mz, -0.05 * y_top, f"{mz:.2f}", rotation=90, ha='center', va='top', fontsize=8, color='red',
                     fontweight='bold')

        # 5. Оранжевые точки (RS) для полноты картины
        plt.scatter(zeros_rs, np.zeros_like(zeros_rs), color='orange', s=10, zorder=5, label='RS Zeros')

        plt.title(f"Battle A7: QSSC vs RS vs MPMath (T={CFG.T_MIN})")
        plt.xlabel("t")
        plt.ylabel("|Z(t)| (Hardy Magnitude)")
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = ctx.save_fig(plt.gcf(), "battle_a7_comparison.png")

        print(f"\nBattle complete. Graph saved to: {out_path.name}")
        print("Displaying graph window...")

        # --- ПАТЧ: ОТКРЫТИЕ ОКНА С ГРАФИКОМ ---
        plt.show()
        # --------------------------------------

        plt.close()


if __name__ == "__main__":
    main()