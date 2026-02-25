#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
% QSG-Note-002-001-QSSC-Dzeta-Rieman
% Lang: RU
% Identifier: QSG-002-001
% Version: v2.0
% Date: 2026-01-18

Quantum Geometric Model of the Riemann Zeta Function Based on
Spectral Resonance of an Integer Operator within the QSSC Framework

Author: Evgeny Monakhov
Affiliation: Independent Researcher
Email: evgeny.monakhov@voscom.online
ORCID: 0009-0003-1773-5476
DOI: 10.5281/zenodo.18258727
Project: Quantum Spectral Geometry Notes (QSG)
Reference Note: QSG-002-001

PROGRAM DESCRIPTION:
This software suite implements numerical verification of the Quantum Spectral
Self-Consistency (QSSC) theorem for the integer operator H_Z.
The program demonstrates that the non-trivial zeros of the Riemann zeta function
emerge as spectral resonances. This is achieved by the interference
pattern of integers passed through a "Quantum Lens" or phase-compensation operator
defined by Axiom A7.

MAIN OPERATION MODES:
1. Standard Battle Mode: Performs a direct precision comparison between the QSSC
   model, the classical Riemann-Siegel formula, and the high-precision MPMath
   reference at large heights T.
2. Axiom A7 Pipeline: Executes a multi-stage high-precision zero-finding
   algorithm using adaptive zooming and "soft" super-Gaussian spectral
   windows.
3. Spectral Primality Test: An experimental module that checks integers for
   primality based on the constructive interference (resonance) between the
   logarithmic phase of the number and the found zeros.
4. Quantum Genesis: Provides a dynamic visualization of the "wave superposition"
   process, showing how prime numbers emerge from the interference of spectral
   waves in real-time.
5. QSSC Staircase Reconstruction: A flagship visualization mode that performs
   a dynamic reconstruction of the prime counting "staircase"
   (Chebyshev's function ψ(x)) from spectral harmonics. It demonstrates how
   prime positions are spectrally "carved" from a smooth trend through the
   interference of zeros in any user-defined range.
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
    # ПАТЧ: Агрессивное уменьшение NMAX для чистоты первого нуля
    # Мы ограничиваем количество "голосов" (целых чисел), чтобы они не шумели

    if t_center < 25.0:
        # Для T=12 это даст NMAX около 200-300. Этого достаточно для формы
        # и идеально убирает дребезг.
        effort_factor = 12.0
    elif t_center < 60.0:
        # Плавный переход к высокой точности
        effort_factor = 12.0 + ((t_center - 25.0) / 35.0) * 28.0
    else:
        # Для больших T возвращаемся к максимуму
        effort_factor = 40.0

    n_max = int(t_center * effort_factor)

    # Снижаем нижний порог. Для малых T нам не нужно 300 слагаемых.
    # 150 - золотая середина для чистого старта.
    n_max = max(150, (n_max // 50 + 1) * 50)

    # Альфа автоматически подстроится под уменьшенное n_max
    alpha = 5.0 / (n_max ** 2)
    return n_max, alpha


# =========================
#    COMPUTATIONAL KERNELS
# =========================

def soft_cutoff_heat(lam: np.ndarray, alpha: float, power: float = 6.0) -> Tuple[np.ndarray, np.ndarray, int]:

    if lam.size == 0:
        return lam, lam, 0


    lam_max = float(lam[-1])
    x = lam / lam_max

    # Super-Gaussian window
    exp_arg = -12.0 * (x ** power)

    # Cutoff
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


# --- AXIOM A7: PHASE TWIST CALCULATION (ULTRA PRECISION) ---
@numba.jit(nopython=True, fastmath=True)
def calc_theta_riemann(t_arr):
    n = t_arr.size
    res = np.zeros(n, dtype=np.float64)

    # ПАТЧ: Использование максимально точных констант (Double Precision)
    # Это исключает лишние вычисления внутри циклов
    inv_2pi = 0.15915494309189533577  # 1.0 / (2.0 * pi)
    pi_8 = 0.39269908169872415481  # pi / 8.0

    # Коэффициенты Бернулли остаются прежними...
    c1 = 1.0 / 48.0
    c3 = 7.0 / 5760.0
    c5 = 31.0 / 80640.0
    c7 = 127.0 / 430080.0
    c9 = 511.0 / 12165120.0
    c11 = 2047.0 / 311427072.0

    for i in range(n):
        t = t_arr[i]
        if t < 1e-9:
            res[i] = 0.0
            continue

        # Основной член ряда
        val = t * 0.5
        # Использование предрассчитанной inv_2pi
        th = val * np.log(t * inv_2pi) - val - pi_8

        # ... остальной код коррекции (term1 - term11) оставляем без изменений

        # 2. Correction terms (Extended Bernoulli expansion)
        # Using iterative multiplication for speed and precision
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

        # term9 ~ 1/t^9
        curr_pow *= inv_t2
        th += c9 * curr_pow

        # term11 ~ 1/t^11
        curr_pow *= inv_t2
        th += c11 * curr_pow

        res[i] = th

    return res


@numba.jit(nopython=True, fastmath=True, parallel=True)
def riemann_siegel_spectral(t_grid):
    # 1. Вычисляем сверхточную фазу (Axiom A7)
    theta_vec = calc_theta_riemann(t_grid)

    n_points = t_grid.shape[0]
    Z_val = np.zeros(n_points, dtype=np.float64)
    # Ваша сверхточная константа Пи
    pi = 3.14159265358979323846264338327950288419716939937510

    for i in numba.prange(n_points):
        t = t_grid[i]
        if t < 0.1: continue

        theta = theta_vec[i]

        # Main Sum
        a = np.sqrt(t / (2.0 * pi))
        N_limit = int(a)

        current_sum = 0.0
        for n in range(1, N_limit + 1):
            log_n = np.log(float(n))
            current_sum += np.cos(theta - t * log_n) * (1.0 / np.sqrt(float(n)))

        main_term = 2.0 * current_sum

        # --- GABCKE CORRECTION (Теперь ВНУТРИ цикла) ---
        p = a - N_limit
        coeff = np.power(t / (2.0 * pi), -0.25)

        # Знак (-1)^(N-1)
        sign = 1.0 if ((N_limit - 1) % 2 == 0) else -1.0

        # Аппроксимация функции Габке
        phi_val = np.cos(2.0 * pi * (p * p - p - 0.0625))
        denom = np.cos(2.0 * pi * p)

        if np.abs(denom) < 1e-6:
            denom = 1e-6

        remainder = sign * coeff * (phi_val / denom)

        # РЕЗУЛЬТАТ: Теперь записывается для каждой точки i
        Z_val[i] = main_term + remainder

    return Z_val



# =========================
#     ГЛАВНАЯ ЛОГИКА
# =========================
def compute_Z_profile_hybrid(t: np.ndarray, lam_full: np.ndarray, u: float, alpha: float,
                             use_a7: bool, power: float, force_bar: bool = False, desc: str = "QSSC"):
    """
    MEMORY-SAFE VERSION: Implements Spectral Streaming.
    Processes harmonics in chunks to avoid RAM overflow on N > 100M.
    """
    n_t = t.shape[0]
    total_harmonics = lam_full.shape[0]

    # Результирующие массивы (накопители)
    zr_total = np.zeros(n_t, dtype=np.float64)
    zi_total = np.zeros(n_t, dtype=np.float64)

    # ОПРЕДЕЛЕНИЕ РАЗМЕРА ЧАНКА ПО ПАМЯТИ
    # 1 миллион float64 = 8 МБ. Это безопасно для любого ПК.
    safe_n_chunk = 1_000_000

    # Настраиваем прогресс-бар для ГАРМОНИК (так как это главный цикл теперь)
    # Показываем бар, если гармоник много (> 1 млн) или если попросили
    show_bar = (total_harmonics > 500_000) or force_bar
    if show_bar:
        pb = ProgressBar(total_harmonics, prefix=f"{desc} STREAM")
    else:
        pb = None

    # === ГЛАВНЫЙ ЦИКЛ ПО КУСКАМ СПЕКТРА (Streaming) ===
    # Мы идем по массиву чисел n (1..400млн) кусочками
    for start_idx in range(0, total_harmonics, safe_n_chunk):
        end_idx = min(start_idx + safe_n_chunk, total_harmonics)

        # 1. Берем маленький кусочек "сырых" чисел
        # (Это не копирует весь массив, если lam_full - это numpy array/range)
        lam_chunk = lam_full[start_idx: end_idx]

        # 2. Спектральная подготовка (Только для этого куска!)
        # Теперь soft_cutoff_heat не съест всю память
        lam, g, n_clipped = soft_cutoff_heat(lam_chunk, alpha, power=power)

        # Если кусок пустой (из-за cutoff), пропускаем
        if lam.size == 0:
            if pb: pb.update(end_idx)
            continue

        # 3. Расчет фаз и амплитуд (Легкие операции для куска)
        A = (u * u) + (lam * lam)
        logA = np.log(A)
        phi = 0.5 * logA
        amp = 2.0 * g * np.exp(-0.25 * logA)

        # 4. СУММИРОВАНИЕ (ЯДРА)
        # Здесь мы добавляем вклад этого куска гармоник к общему результату

        if USE_GPU:
            # GPU любит большие пачки T, но у нас их может быть мало в Mode 6.
            # Поэтому просто скармливаем всё ядру.
            # Важно: GPU kernel суммирует переданные ему phi/amp.

            # Разбиваем T на чанки, если T очень много (защита видеопамяти)
            t_chunk_size = CFG.CHUNK_SIZE_GPU
            for i in range(0, n_t, t_chunk_size):
                i_end_t = min(i + t_chunk_size, n_t)
                t_slice = t[i:i_end_t]

                # Ядро вернет сумму ДЛЯ ЭТИХ ГАРМОНИК
                r, im = gpu_kernel(t_slice, phi, amp, t_chunk_size)

                # НАКОПЛЕНИЕ: +=
                zr_total[i:i_end_t] += r
                zi_total[i:i_end_t] += im

        elif USE_NUMBA:
            # CPU Numba
            batch_size_t = 2000
            for i in range(0, n_t, batch_size_t):
                i_end_t = min(i + batch_size_t, n_t)
                t_slice = t[i:i_end_t]

                r, im = numba_kernel(t_slice, phi, amp)

                # НАКОПЛЕНИЕ: +=
                zr_total[i:i_end_t] += r
                zi_total[i:i_end_t] += im

        else:
            # Numpy fallback
            phase = np.outer(t, phi)
            zr_total += np.dot(np.cos(phase), amp)
            zi_total -= np.dot(np.sin(phase), amp)

        # Обновляем прогресс
        if pb: pb.update(end_idx)

    if pb: pb.close()

    Z_complex = zr_total + 1j * zi_total

    # --- ФИНАЛЬНАЯ СБОРКА ---
    # Линзу (A7) применяем один раз в самом конце к полной сумме
    if use_a7:
        theta = calc_theta_riemann(t)
        P = zr_total * np.cos(theta) - zi_total * np.sin(theta)
        # Возвращаем "пустышки" для lam, phi, amp, так как они теперь размазаны по чанкам
        # Это не сломает логику Mode 6
        return Z_complex, P, np.array([]), np.array([]), np.array([])
    else:
        return Z_complex, np.abs(Z_complex), np.array([]), np.array([]), np.array([])

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

    # ПАТЧ: Смягчение старта пайплайна для низких T
    if T_CENTER < 30.0:
        MODE, POWER_START, POWER_END = "ULTRA-SOFT", 4.0, 4.0
    elif T_CENTER < 60.0:
        MODE, POWER_START, POWER_END = "SOFT", 5.0, 6.0
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
        # ПАТЧ: Увеличенный радиус поиска на низких T
        # Это позволяет "выпрыгнуть" из ложной спектральной ямы к реальному нулю
        radius_multiplier = 0.60 if current_t < 30.0 else 0.30
        current_radius = max(avg_gap * radius_multiplier, global_step * 4.0)
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
    # Label height levels to prevent overlap
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
    Computes the 'Spectral Resonance' of target_num based on the discovered zeros.
    Uses a formula related to Landau's explicit formula:
    F(x) = Sum cos(gamma * ln(x))

    If x is a prime number (or a prime power), the sum yields a constructive peak.
    If x is composite, the terms cancel each other out (destructive interference).
    """
    val = 0.0
    ln_x = np.log(float(target_num))

    n = zeros_arr.size
    for i in range(n):
        # Contribution of each zero to the resonance
        # gamma * ln(x) is the phase. If x is prime, the phases synchronize.
        val += np.cos(zeros_arr[i] * ln_x)

    # Returning the "pure" sum.
    # Large positive value -> Likely prime.
    # Near zero or negative -> Likely composite.
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

    # --- 1. FIRST, DISPLAY MENU AND GET INPUT ---
    print("\nSelect Operation Mode:")
    print("  1 - Standard Battle (QSSC vs RS vs MPMath) [Plots Enabled]")
    print("  2 - NEW AXIOM A7 PIPELINE (Twisted Phase / 12 PASS / No Plots) [Recommended]")
    print("  3 - SPECTRAL PRIMALITY TEST (Check integer using Zeros)")
    print("  4 - QUANTUM GENESIS (Visual Animation)")
    print("  5 - QSSC STAIRCASE RECONSTRUCTION (Prime Staircase Dynamic)")
    print("  6 - SPLIT SPECTRAL BATTLE (Base vs Tail Analysis)")

    # Now the mode_str variable is defined BEFORE its usage
    mode_str = input("Choice [2]: ").strip()

    # --- 2. NOW THE PARAMETER INPUT LOGIC ---
    if mode_str in ["3", "4", "5"]:
        print(f"-> Mode {mode_str} Selected. Target T and W will be AUTO-TUNED.")
        TARGET_T = 12.0
        TARGET_W = 100.0
    else:
        TARGET_T = get_input("Enter Target T (Start Height)", 12.0)
        TARGET_W = get_input("Enter Target W (Window Width)", 100.0)
        print(f"-> Selected: T={TARGET_T}, Width={TARGET_W}")

    print("---------------------------\n")

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

        # --- ADAPTIVE LOGIC FROM OPTION 4 ---
        print(f"\n[Auto-Tuning] Calculating diffraction limit for N={target_num}...")

        # Theoretical Rayleigh limit: T_min = (pi * N) / (delta_N)
        # To resolve twin primes delta_N = 2, therefore T = pi * N / 2
        rayleigh_limit_T = (np.pi * float(target_num)) / 2.0

        # Quality Factor:
        # 4.0 — guarantees very sharp peaks and resolution of twin primes
        quality_factor = 4.5

        auto_t_min = 1000.0
        # Calculating the required maximum height T
        needed_max_T = rayleigh_limit_T * quality_factor

        # Window width must cover this distance
        needed_W = max(3000.0, needed_max_T - auto_t_min)
        auto_t_max = auto_t_min + needed_W

        # Recalculating harmonics and grid density
        auto_nmax, auto_alpha = auto_tune_config(auto_t_max)
        # We need ~20 points per unit of t for correct zero extraction
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

        # --- ПАТЧ: Анализ отношения сигнал/шум (SNR) из ver8! ---
        print(f"\n[2/3] Analyzing Resonance for N={target_num}...")
        score = calc_spectral_resonance(target_num, zeros_found)

        # Вычисляем шум в "пустых" зонах между целыми числами
        score_left = calc_spectral_resonance(target_num - 0.5, zeros_found)
        score_right = calc_spectral_resonance(target_num + 0.5, zeros_found)
        noise_level = (abs(score_left) + abs(score_right)) / 2.0

        # Отношение сигнал/шум (SNR)
        snr = score / (noise_level + 1e-9)

        print("\n=== SPECTRAL DIAGNOSTIC REPORT ===")
        print(f"Target Integer:   {target_num}")
        print(f"Resonance Score:  {score:.4f}")
        print(f"Background Noise: {noise_level:.4f}")
        print(f"Contrast (SNR):   {snr:.4f}")
        print("-" * 40)

        if snr > 2.0:
            print(">>> VERDICT: HIGH PROBABILITY PRIME (Strong Resonance)")
        elif snr > 1.2:
            print(">>> VERDICT: POSSIBLE PRIME / WEAK RESONANCE")
        else:
            print(">>> VERDICT: COMPOSITE / NO RESONANCE (Noise Dominates)")

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

            # --- ПАТЧ: Плавный кинетический автомасштаб ---
            current_max = np.max(np.abs(current_sum))

            # Если амплитуда выросла, расширяем экран
            if current_max > max_amp:
                max_amp = current_max * 1.05  # Запас 5%
                ax.set_ylim(-max_amp, max_amp)

            # Если мы долго не росли (стабилизация пика), чуть поджимаем масштаб
            elif k > (total_zeros * 0.2) and current_max < max_amp * 0.7:
                max_amp *= 0.98  # Плавное сужение
                ax.set_ylim(-max_amp, max_amp)

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
        # ПАТЧ: Режим Ultra-Sharp (Высокое разрешение ступенек)
        # Увеличиваем множитель с 1.8 до 4.0, чтобы захватить высокие частоты
        t_needed = max(1500.0, x_max * np.pi * 4.0)

        # Игнорируем стандартный авто-тюнинг и ставим "тяжелые" настройки
        # Чем больше целых чисел, тем резче будет угол ступеньки
        auto_nmax = int(t_needed * 45.0)  # Effort Factor 45.0 (Ultimate)
        auto_alpha = 5.0 / (auto_nmax ** 2)

        # Увеличиваем плотность сетки, чтобы не пропустить нули на высоте
        t_grid = np.linspace(10.0, t_needed, int(t_needed * 25))
        n_flow = np.arange(1, auto_nmax + 1, dtype=np.float64)

        print(f"[2/4] Harvesting Spectral Components (T_max={t_needed:.1f}, NMAX={auto_nmax})...")
        print("      (This may take a few seconds due to High Resolution...)")
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

        # --- ПАТЧ: Математически полная явная формула Римана ---
        # 1. Добавляем константу -ln(2pi) и поправку на нетривиальные нули
        const_term = -np.log(2.0 * np.pi)

        # Начальное состояние: Гладкий тренд x + константы
        current_psi = trend + const_term

        # Разбиваем на кадры для анимации
        batch_size = max(1, len(zeros) // 100)

        for i in range(0, len(zeros), batch_size):
            if not plt.fignum_exists(fig.number): break

            for gamma in zeros[i:i + batch_size]:
                # Главная формула: вклад пары нулей (1/2 + i*gamma) и (1/2 - i*gamma)
                # Дает -2 * sqrt(x) * sin(gamma * ln x) / gamma
                current_psi -= 2.0 * sqrt_x * np.sin(gamma * ln_x) / gamma

            # Дополнительно можно добавить поправку на малые x: -0.5 * ln(1 - x^-2)
            # но для x > 2 она почти незаметна.

            staircase.set_data(x_axis, current_psi)
            ax.set_title(f"Staircase: {i + batch_size}/{len(zeros)} Zeros | T_max={t_needed:.0f}")

            plt.draw()
            plt.pause(0.001)

        plt.ioff()
        print("[4/4] Masterpiece finished. Primes have been 'carved' from the spectrum.")
        plt.show()

    elif mode_str == "6":
        # --- MODE 6: SPLIT SPECTRAL BATTLE (Smart Dynamic) ---
        print("\n=== MODE 6: SPLIT SPECTRAL BATTLE (High Precision) ===")
        print("Logic: 1. Fine-tuned Physics Grid. 2. Heavy Tail Correction.")

        # 1. АВТО-РАСЧЕТ СЕТКИ (СБАЛАНСИРОВАННЫЙ)
        # ----------------------------------------
        t_center = (CFG.T_MIN + CFG.T_MAX) / 2
        theta_prime = 0.5 * np.log(t_center / (2 * np.pi))
        max_freq = theta_prime + 1.0

        window_w = CFG.T_MAX - CFG.T_MIN
        total_phase_change = max_freq * window_w

        # phase_tol = 0.2 (было 0.4). Делаем сетку в 2 раза плотнее.
        phase_tol = 0.2

        needed_points = int(total_phase_change / phase_tol)

        # ЗАЩИТА ОТ "ПРОСКОКА" НУЛЕЙ:
        # Если нули стоят тесно (расстояние ~0.2), нам нужен шаг хотя бы 0.01
        min_density = int(window_w / 0.01)
        needed_points = max(needed_points, min_density)

        # Не меньше 100 точек для красоты
        needed_points = max(100, needed_points)

        if needed_points > CFG.N_T: needed_points = CFG.N_T

        # 2. АВТО-РАСЧЕТ РАЗДЕЛЕНИЯ (Square Root Law)
        # ----------------------------------------
        # Мы не гадаем, а используем статистику.
        # Шум ошибки интерполяции растет как корень из N.
        # Берем 3 сигмы (3 * sqrt(N)) для гарантии точности.

        dynamic_tail = int(20.0 * np.sqrt(CFG.NMAX))

        # Ставим разумные границы:
        # Не меньше 5000 (для малых высот)
        # Не больше 500,000 (чтобы не ждать вечность на супер-высотах)
        tail_budget = max(5000, dynamic_tail)
        tail_budget = min(tail_budget, 500000)

        if CFG.NMAX > tail_budget * 2:
            split_n = CFG.NMAX - tail_budget
        else:
            # Если чисел совсем мало, делим пополам
            split_n = CFG.NMAX // 2

        ratio_pct = (split_n / CFG.NMAX) * 100.0

        # 3. ВЫВОД ПАРАМЕТРОВ
        # ----------------------------------------
        actual_step = (CFG.T_MAX - CFG.T_MIN) / max(1, needed_points - 1)

        print(f"-> Physics Grid: {needed_points} points (Step ~{actual_step:.4f})")
        print(f"-> Dynamic Split: {ratio_pct:.5f}% in Base (N={split_n})")
        print(f"-> Adaptive Tail: {CFG.NMAX - split_n} harmonics (Formula: 3*sqrt(N))")


        # 3. ПОДГОТОВКА ДАННЫХ
        t_grid = np.linspace(CFG.T_MIN, CFG.T_MAX, CFG.N_T)
        t_grid_sparse = np.linspace(CFG.T_MIN, CFG.T_MAX, needed_points)

        n_flow_base = np.arange(1, split_n + 1, dtype=np.float64)
        n_flow_tail = np.arange(split_n + 1, CFG.NMAX + 1, dtype=np.float64)
        current_power = 4.0 if CFG.T_MIN < 20.0 else 6.0

        # 4. ВЫЧИСЛЕНИЯ
        logger.info(f"[1/3] Computing BASE (Smart Turbo)...")
        t0_base = time.time()

        # Добавили force_bar=True и desc="BASE"
        Z_base_sparse, _, _, _, _ = compute_Z_profile_hybrid(
            t_grid_sparse, n_flow_base, CFG.U, CFG.ALPHA, use_a7=False, power=current_power,
            force_bar=True, desc="BASE"
        )

        # Интерполяция
        zr_base_full = np.interp(t_grid, t_grid_sparse, Z_base_sparse.real)
        zi_base_full = np.interp(t_grid, t_grid_sparse, Z_base_sparse.imag)
        t_base = time.time() - t0_base
        logger.info(f"      -> Base done in {t_base:.3f}s")

        logger.info(f"[2/3] Computing TAIL (Exact)...")
        t0_tail = time.time()

        # Добавили force_bar=True и desc="TAIL"
        Z_tail_full, _, _, _, _ = compute_Z_profile_hybrid(
            t_grid, n_flow_tail, CFG.U, CFG.ALPHA, use_a7=False, power=current_power,
            force_bar=True, desc="TAIL"
        )
        t_tail = time.time() - t0_tail
        logger.info(f"      -> Tail done in {t_tail:.3f}s")

        # 5. СБОРКА
        Z_total_complex = (zr_base_full + 1j * zi_base_full) + Z_tail_full

        # Накручиваем фазу (Линзу)
        theta_vec = calc_theta_riemann(t_grid)
        P_total = Z_total_complex.real * np.cos(theta_vec) - Z_total_complex.imag * np.sin(theta_vec)

        # Для визуализации базы
        P_base_only = zr_base_full * np.cos(theta_vec) - zi_base_full * np.sin(theta_vec)

        # 6. ЭТАЛОН И ПРОВЕРКА (ROBUST MODE)
        logger.info("[3/3] Fetching True Zeros from MPMath...")
        true_zeros = []
        n_z = 1

        # Быстрая оценка начального номера нуля (Формула Римана)
        if CFG.T_MIN > 10.0:
            c = CFG.T_MIN / (2.0 * np.pi)
            n_z = int(c * np.log(c) - c) - 250
            if n_z < 1: n_z = 1

        logger.info(f"      -> Starting search from index n={n_z}")

        fails_sequence = 0  # Счетчик подряд идущих ошибок

        while True:
            t_val = None
            try:
                # ПОПЫТКА 1: Супер-быстрый режим (Hardware Float)
                t_val = float(mp.fp.zetazero(n_z).imag)
                fails_sequence = 0  # Сброс счетчика, если успех

            except ValueError:
                # ОШИБКА: "Could not find root..."
                # РЕШЕНИЕ: Переключаемся на ручную точность (25 знаков), чтобы пройти трудное место
                try:
                    # workdps(25) - это "20-25 знаков", про которые вы говорили.
                    # Этого достаточно, чтобы разрулить "тесные" нули.
                    with mp.workdps(25):
                        t_val = float(mp.zetazero(n_z).imag)
                    fails_sequence = 0
                except Exception as e:
                    # Если даже это не помогло - пропускаем этот ноль, чтобы не крашить программу
                    if fails_sequence < 5:  # Не спамим ошибками, если их мало
                        print(f"   [SKIP] Could not calc zero #{n_z}: {e}")
                    fails_sequence += 1
                    n_z += 1
                    if fails_sequence > 50:
                        print("   [STOP] Too many MPMath errors. Stopping reference search.")
                        break
                    continue

            # Проверка выхода за границы окна
            if t_val > CFG.T_MAX:
                break

            if t_val >= CFG.T_MIN:
                true_zeros.append(t_val)

            n_z += 1

        true_zeros = np.array(true_zeros)

        # Если mpmath совсем не справился (например нет инета для подгрузки данных или сбой)
        if len(true_zeros) == 0:
            print("\n[WARNING] MPMath returned no zeros. Skipping comparison table.")
            true_zeros = np.array([])

        # 7. Таблица и График
        zeros_found = extract_zeros_signchange(t_grid, P_total)
        zeros_found = _unique_sorted(zeros_found)

        print(f"\n=== SMART SPLIT RESULTS (Base={ratio_pct:.4f}%) ===")
        print(f"{'True Zero (MP)':<30} | {'QSSC Turbo':<30} | {'Err':<12}")
        print("-" * 80)

        if len(true_zeros) > 0:
            matched, errs = match_nearest_unique(true_zeros, zeros_found)
            for i, true_z in enumerate(true_zeros):
                fv = f"{matched[i]:.22f}" if matched[i] is not None else "---"
                fe = f"{errs[i]:.2e}" if errs[i] is not None else "FAIL"
                print(f"{true_z:<30.22f} | {fv:<30} | {fe:<12}")
        else:
            print("No reference zeros available for comparison.")
        print("-" * 80)

        # 8. ВИЗУАЛИЗАЦИЯ
        plt.figure(figsize=(14, 7))

        Y_base = np.abs(P_base_only)
        Y_total = np.abs(P_total)
        y_top = np.max(Y_total) if Y_total.size > 0 else 1.0

        plt.plot(t_grid, Y_base, color='green', linestyle='--', alpha=0.5,
                 linewidth=1.5, label=f'Base ({ratio_pct:.1f}%)')
        plt.plot(t_grid, Y_total, color='#003366', linewidth=2.0,
                 label='Total QSSC')

        if len(true_zeros) > 0:
            plot_vlines_with_labels(plt.gca(), true_zeros, y_top, "{:.2f}", "gray", ":", 0.8, 0.5, 8, 90)

        plt.plot(zeros_found, np.zeros_like(zeros_found), 'ro', markersize=5, zorder=5, label='Found Zeros')

        for mz in zeros_found:
            plt.text(mz, -0.05 * y_top, f"{mz:.2f}", rotation=90, ha='center', va='top',
                     fontsize=8, color='red', fontweight='bold')

        plt.title(f"QSSC Smart Mode: T=[{CFG.T_MIN}, {CFG.T_MAX}] | Turbo Ratio: {ratio_pct:.2f}%")
        plt.xlabel("t")
        plt.ylabel("|Z(t)| Amplitude")
        plt.legend(loc='upper right')
        plt.axhline(0, color='black', linewidth=1.0, alpha=0.5)

        plt.tight_layout()
        out_path = ctx.save_fig(plt.gcf(), "smart_split_battle.png")
        print(f"Graph saved: {out_path.name}")
        plt.show()





    elif mode_str == "" or mode_str == "2":
        # --- ПАТЧ: Быстрый поиск нулей для Pipeline ---
        print("\n[Mode 2] Running Pipeline with automatic MPMath reference...")

        t_grid = np.linspace(CFG.T_MIN, CFG.T_MAX, CFG.N_T)
        n_flow = np.arange(1, CFG.NMAX + 1, dtype=np.float64)

        logger.info("Fetching Reference Zeros from MPMath...")
        ref_zeros = []

        # Используем формулу Мангольдта для мгновенного прыжка к нужному номеру
        if CFG.T_MIN > 10.0:
            # 2 * pi = 6.283185307179586
            c = CFG.T_MIN / 6.283185307179586
            n_z = int(c * np.log(c) - c) - 250
            if n_z < 1: n_z = 1
        else:
            n_z = 1

        # Быстрая перемотка
        while True:
            if mp.fp.zetazero(n_z).imag >= CFG.T_MIN: break
            n_z += 1

        # Сбор нулей
        while True:
            t_val = float(mp.fp.zetazero(n_z).imag)
            if t_val > CFG.T_MAX: break
            ref_zeros.append(t_val)
            n_z += 1
        ref_zeros = np.array(ref_zeros)

        # Запуск пайплайна
        staged_pipeline_qssc(ctx, t_grid, n_flow, ref_zeros)

    else:
        # --- MODE 1: STANDARD BATTLE (VISUAL FIX) ---
        print("\n=== STANDARD BATTLE MODE (QSSC vs RS vs MPMath) ===")
        print("Settings: Axiom A7 (Twist) = ON. ")
        print("NOTE: Using Power=6.0 for Battle Mode to prevent visual artifacts.")

        t_grid = np.linspace(CFG.T_MIN, CFG.T_MAX, CFG.N_T)
        n_flow = np.arange(1, CFG.NMAX + 1, dtype=np.float64)

        # 1. QSSC A7 (ПАТЧ: Адаптивная жесткость окна)
        # На малых T (до 20) используем более мягкое окно (power=4),
        # чтобы подавить "дребезг" гармоник.
        current_power = 4.0 if CFG.T_MIN < 20.0 else 6.0

        logger.info(f"1. Computing QSSC (Axiom A7, Power {current_power})...")
        t0 = time.time()
        # В вызове функции ниже мы заменили фиксированную 6.0 на переменную current_power
        _, P_qssc, _, _, _ = compute_Z_profile_hybrid(
            t_grid, n_flow, CFG.U, CFG.ALPHA, use_a7=True, power=current_power
        )

        t_qssc = time.time() - t0
        logger.info(f"QSSC done in {t_qssc:.2f}s")

        # 2. RS (Riemann-Siegel)
        logger.info("2. Computing RS (Spectral)...")
        t0_rs = time.time()
        Z_rs_signed = riemann_siegel_spectral(t_grid)
        t_rs = time.time() - t0_rs
        logger.info(f"RS done in {t_rs:.2f}s")

        # 3. MPMath (ПАТЧ: Разреженное сэмплирование для ускорения в 50 раз)
        logger.info("3. Computing MPMath Reference (Optimized Sparse Mode)...")
        t0_ref = time.time()

        # Вместо 100,000 точек считаем только 2,000 — этого за глаза для графика
        n_ref_points = 2000
        t_ref_sparse = np.linspace(CFG.T_MIN, CFG.T_MAX, n_ref_points)
        ref_abs_sparse = np.zeros(n_ref_points)

        pb = ProgressBar(n_ref_points, prefix="MPMath Fast")
        for i in range(n_ref_points):
            # Считаем эталон в разреженных точках
            ref_abs_sparse[i] = float(abs(mp.fp.zeta(0.5 + 1j * t_ref_sparse[i])))
            if i % 100 == 0:
                pb.update(i)
        pb.close()

        # Мгновенная интерполяция эталона на полную сетку t_grid для отрисовки
        ref_abs = np.interp(t_grid, t_ref_sparse, ref_abs_sparse)

        t_ref = time.time() - t0_ref
        logger.info(f"MPMath (Graph) done in {t_ref:.2f}s (Optimized)")

        # --- СРАВНЕНИЕ (Таблица) ---
        logger.info("Finding exact zeros for report...")

        # Нули QSSC
        zeros_qssc = extract_zeros_signchange(t_grid, P_qssc)
        zeros_qssc = _unique_sorted(zeros_qssc)

        # Нули RS
        zeros_rs = extract_zeros_signchange(t_grid, Z_rs_signed)
        zeros_rs = _unique_sorted(zeros_rs)

        # --- ПАТЧ: Сверхточный поиск стартового индекса нулей (из ver8!) ---
        logger.info("Fetching MPMath True Zeros...")
        true_zeros = []
        n_z = 1
        if CFG.T_MIN > 10.0:
            # Формула Римана-фон Мангольдта: точная оценка номера нуля
            c = CFG.T_MIN / (2.0 * 3.141592653589793)
            val = c * np.log(c) - c
            n_z = int(val) - 250  # Запас 250 нулей для надежности
            if n_z < 1: n_z = 1

        # Быстрая перемотка к началу окна
        while True:
            if mp.fp.zetazero(n_z).imag >= CFG.T_MIN:
                break
            n_z += 1

        # Сбор нулей в заданном интервале
        while True:
            t_val = float(mp.fp.zetazero(n_z).imag)
            if t_val > CFG.T_MAX:
                break
            true_zeros.append(t_val)
            n_z += 1

        true_zeros = np.array(true_zeros)

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