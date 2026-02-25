# -*- coding: utf-8 -*-
"""
QSSC primes -> Xi_g(t) -> Xi_2(tau) -> Xi_3(tau) + spectral diagnostics
Windows multiprocessing + run folder logging & plots.

HARD CONSTRAINT:
- Core QSSC formula for Xi_g(t) must remain unchanged.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# IMPORTANT for Windows multiprocessing + BLAS oversubscription:
# If you enable multiprocessing, each worker must not spawn extra BLAS threads.
# Setting these early is the safest practice.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import matplotlib
matplotlib.use("Agg")  # always save plots to files; no GUI blocking
import matplotlib.pyplot as plt

import logging

# multiprocessing imports (Windows-safe)
import multiprocessing as mp



# =========================
#     PROGRESS BAR
# =========================
class ProgressBar:
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
        print("", file=sys.stdout)


# =========================
#         LOGGING
# =========================
def create_run_dir(base: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("qssc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # file handler
    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger


def save_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv_peaks(path: Path, peaks: List[Tuple[float, float]]) -> None:
    lines = ["rank,lambda,amplitude,period(2pi/lambda)\n"]
    for i, (lam, amp) in enumerate(peaks, 1):
        period = (2.0 * math.pi / lam) if lam > 0 else float("inf")
        lines.append(f"{i},{lam:.12g},{amp:.12g},{period:.12g}\n")
    path.write_text("".join(lines), encoding="utf-8")


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# =========================
#       PRIMES (LIB)
# =========================
def _estimate_nth_prime_upper(n: int) -> int:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n <= 6:
        small = [2, 3, 5, 7, 11, 13]
        return small[n - 1]
    nn = float(n)
    return int(nn * (math.log(nn) + math.log(math.log(nn)))) + 10


def _primes_numpy_sieve(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([2], dtype=np.int64)

    limit = _estimate_nth_prime_upper(n)
    while True:
        m = (limit - 1) // 2  # odds <= limit excluding 2
        sieve = np.ones(m, dtype=np.bool_)
        sqrt_lim = int(math.isqrt(limit))
        i_max = (sqrt_lim - 3) // 2

        for i in range(i_max + 1):
            if sieve[i]:
                p = 2 * i + 3
                start = (p * p - 3) // 2
                sieve[start::p] = False

        odds = (2 * np.flatnonzero(sieve) + 3).astype(np.int64)
        primes = np.concatenate([np.array([2], dtype=np.int64), odds], axis=0)
        if primes.size >= n:
            return primes[:n]

        limit = int(limit * 1.25) + 1000


def get_first_n_primes(n: int, method: str = "auto") -> np.ndarray:
    """
    Returns first n primes as float64 (exact integer values up to 2^53).
    method: auto|primesieve|sympy|numpy
    """
    method = (method or "auto").lower()

    if method in ("auto", "primesieve"):
        try:
            import primesieve  # type: ignore
            arr = np.array(primesieve.primes(n), dtype=np.int64)
            if arr.size != n:
                raise RuntimeError("primesieve returned wrong count")
            return arr.astype(np.float64)
        except Exception:
            if method == "primesieve":
                raise

    if method in ("auto", "sympy"):
        try:
            from sympy import primerange  # type: ignore
            limit = _estimate_nth_prime_upper(n)
            while True:
                arr = np.fromiter(primerange(2, limit + 1), dtype=np.int64)
                if arr.size >= n:
                    return arr[:n].astype(np.float64)
                limit = int(limit * 1.25) + 1000
        except Exception:
            if method == "sympy":
                raise

    if method in ("auto", "numpy"):
        return _primes_numpy_sieve(n).astype(np.float64)

    raise ValueError("Unknown method. Use auto|primesieve|sympy|numpy.")


# =========================
#      SOFT CUTOFF g(lam)
# =========================
def soft_cutoff_heat(lam: np.ndarray, alpha: float) -> tuple[np.ndarray, int]:
    exp_arg = -alpha * lam * lam
    exp_min = -745.0
    clipped = int(np.sum(exp_arg < exp_min))
    exp_arg = np.maximum(exp_arg, exp_min)
    g = np.exp(exp_arg)
    return g, clipped


# =========================
#   CORE QSSC (SERIAL)
# =========================
def compute_Xi_primary_qssc_serial(
    t: np.ndarray,
    lam: np.ndarray,
    u: float,
    alpha: float,
    lam_block: int,
    t_block: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CORE QSSC FORMULA (unchanged):
      A = u^2 + lam^2
      phi = 0.5 * log(A)
      amp = 2 * g(lam) * A^(-1/4) = 2*g*exp(-0.25*logA)

      Re Xi(t) = Σ amp_k cos(t*phi_k)
      Im Xi(t) = -Σ amp_k sin(t*phi_k)
    """
    if lam_block < 1 or t_block < 1:
        raise ValueError("lam_block and t_block must be >= 1")

    g, n_clipped = soft_cutoff_heat(lam, alpha=alpha)
    A = (u * u) + (lam * lam)
    logA = np.log(A)
    phi = 0.5 * logA
    amp = 2.0 * g * np.exp(-0.25 * logA)

    xr = np.zeros_like(t, dtype=np.float64)
    xi = np.zeros_like(t, dtype=np.float64)

    n_modes = lam.size
    n_chunks = int(math.ceil(n_modes / max(1, lam_block)))
    pb = ProgressBar(total=n_chunks, prefix="[compute] Xi_g(t) serial")

    chunk_idx = 0
    for i0 in range(0, n_modes, lam_block):
        i1 = min(i0 + lam_block, n_modes)
        phi_c = phi[i0:i1]
        amp_c = amp[i0:i1]

        for j0 in range(0, t.size, t_block):
            j1 = min(j0 + t_block, t.size)
            t_c = t[j0:j1]
            phase = np.outer(t_c, phi_c)
            xr[j0:j1] += np.dot(np.cos(phase), amp_c)
            xi[j0:j1] -= np.dot(np.sin(phase), amp_c)

        chunk_idx += 1
        pb.update(chunk_idx)

    pb.close()

    Xi = xr + 1j * xi
    P = np.abs(Xi)

    if n_clipped > 0:
        logger.info("[diag] cutoff clipped count = %d / %d", n_clipped, lam.size)

    return Xi, P


# =========================
#   CORE QSSC (PARALLEL)
# =========================
# Shared arrays in worker (NOT SharedMemory; per-worker local copies via initargs)
_W_T: Optional[np.ndarray] = None
_W_PHI: Optional[np.ndarray] = None
_W_AMP: Optional[np.ndarray] = None
_W_T_BLOCK: int = 1024


def _init_worker(t: np.ndarray, phi: np.ndarray, amp: np.ndarray, t_block: int) -> None:
    global _W_T, _W_PHI, _W_AMP, _W_T_BLOCK

    # Ensure contiguous float64 in each worker
    _W_T = np.asarray(t, dtype=np.float64)
    _W_PHI = np.asarray(phi, dtype=np.float64)
    _W_AMP = np.asarray(amp, dtype=np.float64)
    _W_T_BLOCK = int(t_block)


def _worker_sum_lambda_chunk(args: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    i0, i1 = args
    t = _W_T
    phi = _W_PHI
    amp = _W_AMP
    assert t is not None and phi is not None and amp is not None

    phi_c = phi[i0:i1]
    amp_c = amp[i0:i1]


    xr = np.zeros_like(t, dtype=np.float64)
    xi = np.zeros_like(t, dtype=np.float64)

    for j0 in range(0, t.size, _W_T_BLOCK):
        j1 = min(j0 + _W_T_BLOCK, t.size)
        t_c = t[j0:j1]

        phase = np.outer(t_c, phi_c)
        xr[j0:j1] += np.dot(np.cos(phase), amp_c)
        xi[j0:j1] -= np.dot(np.sin(phase), amp_c)

    return xr, xi


def compute_Xi_primary_qssc_parallel(
    t: np.ndarray,
    lam: np.ndarray,
    u: float,
    alpha: float,
    lam_block: int,
    t_block: int,
    n_workers: int,
    logger: logging.Logger,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parallel wrapper around the same core formula.
    Splits lambda modes into chunks and sums in multiple processes (Windows-safe, no SharedMemory).
    """
    g, n_clipped = soft_cutoff_heat(lam, alpha=alpha)
    A = (u * u) + (lam * lam)
    logA = np.log(A)
    phi = 0.5 * logA
    amp = 2.0 * g * np.exp(-0.25 * logA)

    if n_clipped > 0:
        logger.info("[diag] cutoff clipped count = %d / %d", n_clipped, lam.size)

    t = np.asarray(t, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    amp = np.asarray(amp, dtype=np.float64)

    n_modes = lam.size
    chunks = [(i0, min(i0 + lam_block, n_modes)) for i0 in range(0, n_modes, lam_block)]

    if n_workers <= 0:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    chunk_dispatch = max(1, len(chunks) // (n_workers * 4) or 1)

    logger.info(
        "[parallel] workers=%d, lambda_chunks=%d, lam_block=%d, t_block=%d, imap_chunksize=%d",
        n_workers, len(chunks), lam_block, t_block, chunk_dispatch,
    )

    xr = np.zeros_like(t, dtype=np.float64)
    xi = np.zeros_like(t, dtype=np.float64)

    pb = ProgressBar(total=len(chunks), prefix="[compute] Xi_g(t) parallel")

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(t, phi, amp, t_block),
    ) as pool:
        logger.info("[parallel] pool spawned, waiting for first chunk result...")
        t_first = time.time()

        done = 0
        for xr_part, xi_part in pool.imap_unordered(
            _worker_sum_lambda_chunk,
            chunks,
            chunksize=chunk_dispatch,
        ):
            if done == 0:
                logger.info("[parallel] first chunk returned after %.2fs", time.time() - t_first)

            xr += xr_part
            xi += xi_part
            done += 1
            pb.update(done)

    pb.close()

    Xi = xr + 1j * xi
    P = np.abs(Xi)
    return Xi, P


# =========================
#   AUTOCORRELATION (FFT)
# =========================
def autocorr_fft_real(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return np.zeros_like(x)
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, n=nfft)
    r = np.fft.irfft(X * np.conj(X), n=nfft)[:n]
    return r


def build_autocorr_signal(
    grid: np.ndarray,
    signal: np.ndarray,
    *,
    use_signal: str = "real",
    demean: bool = True,
    unbiased: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    if grid.size != signal.size:
        raise ValueError("grid and signal must have same length")
    if grid.size < 4:
        return np.array([0.0]), np.array([0.0])

    dt = float(grid[1] - grid[0])
    twoT = float((grid.size - 1) * dt)

    if use_signal == "real":
        x = np.real(signal).astype(np.float64)
    elif use_signal == "abs":
        x = np.abs(signal).astype(np.float64)
    else:
        raise ValueError("use_signal must be 'real' or 'abs'")

    if demean:
        x = x - float(np.mean(x))

    r = autocorr_fft_real(x)
    y = (dt * r) / max(twoT, 1e-300)

    if unbiased:
        n = grid.size
        denom = np.arange(n, 0, -1, dtype=np.float64)
        y = y * (n / denom)

    tau = np.arange(0, grid.size, dtype=np.float64) * dt
    return tau, y


# =========================
#   SPECTRUM OF A τ-SIGNAL
# =========================
def spectrum_from_even_extension(tau: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if tau.size != y.size:
        raise ValueError("tau and y must have same length")
    if tau.size < 4:
        return np.array([0.0]), np.array([0.0])

    dt = float(tau[1] - tau[0])
    y_pos = np.asarray(y, dtype=np.float64)
    y_even = np.concatenate([y_pos[1:][::-1], y_pos], axis=0)

    n = y_even.size
    twoT = float((n - 1) * dt)

    Y = np.fft.rfft(y_even)
    freqs = np.fft.rfftfreq(n, d=dt)
    lambdas = 2.0 * math.pi * freqs

    scale = dt / max(twoT, 1e-300)
    m = scale * Y
    return lambdas, np.abs(m)


def top_peaks(lambdas: np.ndarray, amp: np.ndarray, k: int = 12, skip_dc: bool = True) -> List[Tuple[float, float]]:
    if lambdas.size == 0:
        return []
    start = 1 if skip_dc and lambdas.size > 1 else 0
    idx = np.argsort(amp[start:])[-k:] + start
    idx = idx[np.argsort(amp[idx])[::-1]]
    return [(float(lambdas[i]), float(amp[i])) for i in idx]


def cluster_peaks_by_period(peaks: List[Tuple[float, float]], rel_tol: float = 0.05) -> List[dict]:
    """
    Convert peaks (lambda, amp) -> periods 2pi/lambda and cluster nearby periods.
    Helps decide if we found a stable periodicity.
    """
    items = []
    for lam, a in peaks:
        if lam <= 0:
            continue
        period = 2.0 * math.pi / lam
        items.append((period, lam, a))
    items.sort(key=lambda x: x[0])

    clusters: List[List[Tuple[float, float, float]]] = []
    for it in items:
        if not clusters:
            clusters.append([it])
            continue
        p = it[0]
        p_ref = clusters[-1][-1][0]
        if abs(p - p_ref) / max(p_ref, 1e-12) <= rel_tol:
            clusters[-1].append(it)
        else:
            clusters.append([it])

    out = []
    for c in clusters:
        periods = [x[0] for x in c]
        amps = [x[2] for x in c]
        out.append({
            "period_mean": float(np.mean(periods)),
            "period_std": float(np.std(periods)),
            "count": int(len(c)),
            "amp_sum": float(np.sum(amps)),
            "members": [{"period": p, "lambda": lam, "amp": a} for (p, lam, a) in c],
        })
    out.sort(key=lambda d: d["amp_sum"], reverse=True)
    return out


# =========================
#  TWIN-PRIME DIAGNOSTICS
# =========================
def compute_twin_indicator(primes_int: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    primes_int: int64 array of primes
    returns:
      gaps: p_{k+1}-p_k
      twin_indicator: 1 if gap==2 else 0 (length n-1)
    """
    gaps = np.diff(primes_int)
    twin = (gaps == 2).astype(np.float64)
    return gaps.astype(np.int64), twin


def periodogram_indicator(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple FFT periodogram for an indicator sequence along prime index.
    Returns angular freq omega in [0, pi] and amplitude.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size < 8:
        return np.array([0.0]), np.array([0.0])

    x = x - float(np.mean(x))
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(x.size, d=1.0)  # cycles per index
    omega = 2.0 * math.pi * freqs
    amp = np.abs(X) / max(1.0, x.size)
    return omega, amp


def match_periods(primary_clusters: List[dict], omega: np.ndarray, amp: np.ndarray) -> dict:
    """
    Attempts to see if any strong periodicity in Xi-layer corresponds
    to periodicity in twin-indicator along prime index.
    """
    result = {
        "twin_periodogram_top": [],
        "matches": [],
    }

    top_tw = top_peaks(omega, amp, k=8, skip_dc=True)
    # note: omega is already angular freq; period in indices = 2pi/omega
    for w, a in top_tw:
        period = (2.0 * math.pi / w) if w > 0 else float("inf")
        result["twin_periodogram_top"].append({"omega": w, "amp": a, "period_idx": period})

    # heuristic matching: compare periods (dimension mismatch!)
    # We do not force equality; we only report if both have a dominant, stable cluster
    # and twin-indicator has a strong period (in index domain).
    if primary_clusters and result["twin_periodogram_top"]:
        best_layer = primary_clusters[0]
        best_twin = result["twin_periodogram_top"][0]
        result["matches"].append({
            "note": "Layer periods are in tau-domain; twin periodogram is along prime index. "
                    "This is not a strict identity, only a qualitative co-presence of periodicity.",
            "layer_best_period_tau": best_layer["period_mean"],
            "layer_cluster_count": best_layer["count"],
            "twin_best_period_index": best_twin["period_idx"],
        })

    return result


# =========================
#          CONFIG
# =========================
@dataclass(frozen=True)
class Config:
    # Input primes: directly from a library (no files)
    N_PRIMES: int = 50_000
    PRIME_METHOD: str = "auto"  # auto|primesieve|sympy|numpy

    # Core QSSC params (formula unchanged)
    ALPHA: float = 1e-12
    U: float = 1e-3

    # Primary grid
    T_MIN: float = 1200.0
    T_MAX: float = 1260.0
    N_T: int = 12_000

    # Performance / parallelism
    PARALLEL: bool = True
    N_WORKERS: int = 8  # 0=auto
    LAM_BLOCK: int = 2048
    T_BLOCK: int = 256

    # Secondary/tertiary extraction
    USE_SIGNAL: str = "real"   # "real" or "abs"
    DEMEAN: bool = True
    UNBIASED: bool = True

    # Optional truncation of tau to reduce noise & compute
    TAU_MAX_FRAC: float = 0.35  # keep first 35% of tau lags

    # Layers
    DO_SECONDARY: bool = True
    DO_TERTIARY: bool = True

    # Output
    RUNS_DIR: str = "runs"
    SAVE_PLOTS: bool = True


# =========================
#           MAIN
# =========================
def main() -> None:
    cfg = Config()

    run_dir = create_run_dir(cfg.RUNS_DIR)
    logger = setup_logger(run_dir)

    logger.info("=== QSSC: primes -> Xi_g(t) -> Xi_2(tau) -> Xi_3(tau) + diagnostics ===")
    logger.info("[cfg] N_PRIMES=%d method=%s", cfg.N_PRIMES, cfg.PRIME_METHOD)
    logger.info("[cfg] core: ALPHA=%.3e U=%.3e", cfg.ALPHA, cfg.U)
    logger.info("[cfg] t-grid: T=[%.6f, %.6f] N_T=%d", cfg.T_MIN, cfg.T_MAX, cfg.N_T)
    logger.info("[cfg] parallel=%s workers=%d lam_block=%d t_block=%d",
                cfg.PARALLEL, cfg.N_WORKERS, cfg.LAM_BLOCK, cfg.T_BLOCK)
    logger.info("[cfg] ACF: use=%s demean=%s unbiased=%s tau_frac=%.2f",
                cfg.USE_SIGNAL, cfg.DEMEAN, cfg.UNBIASED, cfg.TAU_MAX_FRAC)
    logger.info("[cfg] layers: secondary=%s tertiary=%s",
                cfg.DO_SECONDARY, cfg.DO_TERTIARY)

    # Save config snapshot
    save_json(run_dir / "config.json", asdict(cfg))

    # 0) primes
    t0 = time.time()
    primes = get_first_n_primes(cfg.N_PRIMES, method=cfg.PRIME_METHOD)
    primes_int = primes.astype(np.int64)
    logger.info("[primes] loaded=%d p_min=%d p_max=%d (%.2fs)",
                primes.size, int(primes_int[0]), int(primes_int[-1]), time.time() - t0)

    # twin diagnostics on primes
    gaps, twin = compute_twin_indicator(primes_int)
    twin_count = int(np.sum(twin))
    twin_rate = float(twin_count / max(1, twin.size))
    logger.info("[twins] gap==2 count=%d over %d gaps => rate=%.6f",
                twin_count, twin.size, twin_rate)

    # 1) primary correlator Xi_g(t)
    t_grid = np.linspace(cfg.T_MIN, cfg.T_MAX, cfg.N_T, dtype=np.float64)

    t1 = time.time()
    if cfg.PARALLEL:
        Xi_g, P_g = compute_Xi_primary_qssc_parallel(
            t=t_grid,
            lam=primes,  # lam is primes as float64, exact integers
            u=cfg.U,
            alpha=cfg.ALPHA,
            lam_block=cfg.LAM_BLOCK,
            t_block=cfg.T_BLOCK,
            n_workers=cfg.N_WORKERS,
            logger=logger,
        )
    else:
        Xi_g, P_g = compute_Xi_primary_qssc_serial(
            t=t_grid,
            lam=primes,
            u=cfg.U,
            alpha=cfg.ALPHA,
            lam_block=cfg.LAM_BLOCK,
            t_block=cfg.T_BLOCK,
            logger=logger,
        )
    logger.info("[Xi_g] computed in %.2fs", time.time() - t1)

    # 2) secondary wave
    tau2 = None
    Xi2 = None
    lam2 = None
    spec2 = None
    peaks2: List[Tuple[float, float]] = []
    clusters2: List[dict] = []

    if cfg.DO_SECONDARY:
        t2 = time.time()
        logger.info("[layer-2] computing Xi_2(tau) = ACF(Xi_g) ...")
        tau2, Xi2 = build_autocorr_signal(
            grid=t_grid,
            signal=Xi_g,
            use_signal=cfg.USE_SIGNAL,
            demean=cfg.DEMEAN,
            unbiased=cfg.UNBIASED,
        )

        # truncate tau to reduce noise / cost
        k2 = int(max(8, len(tau2) * float(cfg.TAU_MAX_FRAC)))
        tau2 = tau2[:k2]
        Xi2 = Xi2[:k2]

        lam2, spec2 = spectrum_from_even_extension(tau2, Xi2)
        peaks2 = top_peaks(lam2, spec2, k=12, skip_dc=True)
        clusters2 = cluster_peaks_by_period(peaks2, rel_tol=0.05)

        logger.info("[layer-2] done in %.2fs; top peaks saved", time.time() - t2)
        save_csv_peaks(run_dir / "peaks_layer2.csv", peaks2)

        if clusters2:
            best = clusters2[0]
            logger.info("[layer-2] best period cluster: mean=%.6g std=%.3g count=%d amp_sum=%.3g",
                        best["period_mean"], best["period_std"], best["count"], best["amp_sum"])

    # 3) tertiary wave
    tau3 = None
    Xi3 = None
    lam3 = None
    spec3 = None
    peaks3: List[Tuple[float, float]] = []
    clusters3: List[dict] = []

    if cfg.DO_TERTIARY and Xi2 is not None and tau2 is not None:
        t3 = time.time()
        logger.info("[layer-3] computing Xi_3(tau) = ACF(Xi_2) ...")
        # Xi2 is already real; treat it as "real signal"
        tau3, Xi3 = build_autocorr_signal(
            grid=tau2,
            signal=Xi2,
            use_signal="real",
            demean=cfg.DEMEAN,
            unbiased=cfg.UNBIASED,
        )

        k3 = int(max(8, len(tau3) * float(cfg.TAU_MAX_FRAC)))
        tau3 = tau3[:k3]
        Xi3 = Xi3[:k3]

        lam3, spec3 = spectrum_from_even_extension(tau3, Xi3)
        peaks3 = top_peaks(lam3, spec3, k=12, skip_dc=True)
        clusters3 = cluster_peaks_by_period(peaks3, rel_tol=0.05)

        logger.info("[layer-3] done in %.2fs; top peaks saved", time.time() - t3)
        save_csv_peaks(run_dir / "peaks_layer3.csv", peaks3)

        if clusters3:
            best = clusters3[0]
            logger.info("[layer-3] best period cluster: mean=%.6g std=%.3g count=%d amp_sum=%.3g",
                        best["period_mean"], best["period_std"], best["count"], best["amp_sum"])

    # 4) twin indicator spectrum (index-domain)
    omega_twin, amp_twin = periodogram_indicator(twin)
    top_tw = top_peaks(omega_twin, amp_twin, k=10, skip_dc=True)

    # 5) qualitative match / summary
    match2 = match_periods(clusters2, omega_twin, amp_twin) if clusters2 else {"twin_periodogram_top": [], "matches": []}
    match3 = match_periods(clusters3, omega_twin, amp_twin) if clusters3 else {"twin_periodogram_top": [], "matches": []}

    # 6) Save metrics.json (single structured report)
    metrics = {
        "run_dir": str(run_dir),
        "config": asdict(cfg),
        "primes": {
            "count": int(primes.size),
            "p_min": int(primes_int[0]),
            "p_max": int(primes_int[-1]),
        },
        "twins": {
            "gaps_count": int(gaps.size),
            "twin_count": twin_count,
            "twin_rate": twin_rate,
            "twin_periodogram_top": match2.get("twin_periodogram_top", []),
        },
        "layer2": {
            "computed": bool(cfg.DO_SECONDARY),
            "peaks": [{"lambda": L, "amp": A, "period": (2.0 * math.pi / L if L > 0 else None)} for L, A in peaks2],
            "clusters": clusters2,
            "qualitative_match_with_twins": match2.get("matches", []),
        },
        "layer3": {
            "computed": bool(cfg.DO_TERTIARY),
            "peaks": [{"lambda": L, "amp": A, "period": (2.0 * math.pi / L if L > 0 else None)} for L, A in peaks3],
            "clusters": clusters3,
            "qualitative_match_with_twins": match3.get("matches", []),
        },
        "interpretation": {
            "note": (
                "Layer-2/3 periods are extracted in tau-domain from autocorrelation of the QSSC signal. "
                "Twin-periodogram is in index-domain over prime gaps==2. "
                "We report co-presence of periodic structures; it is not a strict identity mapping."
            )
        }
    }
    save_json(run_dir / "metrics.json", metrics)

    # 7) Plots saved to run folder
    if cfg.SAVE_PLOTS:
        # Primary |Xi_g(t)|
        fig1, ax1 = plt.subplots(figsize=(12, 4.5))
        ax1.plot(t_grid, np.abs(Xi_g), linewidth=1.0)
        ax1.set_title("Primary |Xi_g(t)| (QSSC core unchanged)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("|Xi_g(t)|")
        ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig1.tight_layout()
        save_fig(fig1, run_dir / "plot_primary_abs_Xi_g.png")

        # Secondary wave and spectrum
        if Xi2 is not None and tau2 is not None and lam2 is not None and spec2 is not None:
            fig2, ax2 = plt.subplots(figsize=(12, 4.5))
            ax2.plot(tau2, Xi2, linewidth=1.0)
            ax2.set_title("Secondary wave Xi_2(tau) = ACF(Xi_g)")
            ax2.set_xlabel("tau")
            ax2.set_ylabel("Xi_2(tau)")
            ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            fig2.tight_layout()
            save_fig(fig2, run_dir / "plot_layer2_Xi2_tau.png")

            fig3, ax3 = plt.subplots(figsize=(12, 4.5))
            ax3.plot(lam2, spec2, linewidth=1.0)
            ax3.set_title("Layer-2 spectrum |m_2(lambda)|")
            ax3.set_xlabel("lambda")
            ax3.set_ylabel("|m_2(lambda)|")
            ax3.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            fig3.tight_layout()
            save_fig(fig3, run_dir / "plot_layer2_spectrum.png")

        # Tertiary wave and spectrum
        if Xi3 is not None and tau3 is not None and lam3 is not None and spec3 is not None:
            fig4, ax4 = plt.subplots(figsize=(12, 4.5))
            ax4.plot(tau3, Xi3, linewidth=1.0)
            ax4.set_title("Tertiary wave Xi_3(tau) = ACF(Xi_2)")
            ax4.set_xlabel("tau")
            ax4.set_ylabel("Xi_3(tau)")
            ax4.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            fig4.tight_layout()
            save_fig(fig4, run_dir / "plot_layer3_Xi3_tau.png")

            fig5, ax5 = plt.subplots(figsize=(12, 4.5))
            ax5.plot(lam3, spec3, linewidth=1.0)
            ax5.set_title("Layer-3 spectrum |m_3(lambda)|")
            ax5.set_xlabel("lambda")
            ax5.set_ylabel("|m_3(lambda)|")
            ax5.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            fig5.tight_layout()
            save_fig(fig5, run_dir / "plot_layer3_spectrum.png")

        # Prime gaps histogram
        fig6, ax6 = plt.subplots(figsize=(12, 4.5))
        # show gaps up to a cutoff for readability
        g_show = gaps[gaps <= 200]
        ax6.hist(g_show, bins=100)
        ax6.set_title("Prime gaps histogram (<=200)")
        ax6.set_xlabel("gap = p_{k+1}-p_k")
        ax6.set_ylabel("count")
        ax6.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig6.tight_layout()
        save_fig(fig6, run_dir / "plot_prime_gaps_hist.png")

        # Twin indicator periodogram
        fig7, ax7 = plt.subplots(figsize=(12, 4.5))
        ax7.plot(omega_twin, amp_twin, linewidth=1.0)
        ax7.set_title("Twin-indicator periodogram (index-domain)")
        ax7.set_xlabel("omega (rad/index)")
        ax7.set_ylabel("amplitude")
        ax7.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
        fig7.tight_layout()
        save_fig(fig7, run_dir / "plot_twin_indicator_periodogram.png")

    # 8) final console summary
    if clusters2:
        best2 = clusters2[0]
        logger.info("[summary] layer-2 dominant period ~ %.6g (tau-units), cluster_count=%d",
                    best2["period_mean"], best2["count"])
    if clusters3:
        best3 = clusters3[0]
        logger.info("[summary] layer-3 dominant period ~ %.6g (tau-units), cluster_count=%d",
                    best3["period_mean"], best3["count"])

    if top_tw:
        w, a = top_tw[0]
        pidx = (2.0 * math.pi / w) if w > 0 else float("inf")
        logger.info("[summary] twin-indicator dominant period ~ %.6g (prime-index units)", pidx)

    logger.info("Run artifacts saved in: %s", str(run_dir))
    logger.info("Done.")


if __name__ == "__main__":
    mp.freeze_support()
    # Ensure Windows uses spawn
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
