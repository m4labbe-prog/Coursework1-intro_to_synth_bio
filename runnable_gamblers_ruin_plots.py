"""Gambler's ruin model and the four requested plots.

Requirements:
    python -m pip install numpy matplotlib

Run:
    python /path/to/runnable_gamblers_ruin_plots.py

The script creates four PNGs in ``plots_generated`` beside this file:
  1. Single bacterium, log10 hitting probability vs death-to-birth ratio.
  2. Single bacterium, linear hitting probability vs death-to-birth ratio.
  3. Curtis et al. Figure 1c-style replication for four q/p ratios.
  4. Appendix Figure A3-style extinction time with 25 purple noise curves.
  5. Equal-birth/death extinction time with finite carrying capacity and
     25 purple noise curves.

Mathematical model
------------------
The population is a nearest-neighbour random walk on states 0,...,N. At each
population-changing event, birth occurs with probability p, death with q, and
p+q=1. States 0 and N are absorbing for the hitting-probability calculation.
For a starting population i, h_i=P(reach N before 0) satisfies
    h_i = p*h_(i+1) + q*h_(i-1), h_0=0, h_N=1.
Writing r=q/p gives h_i=(1-r**i)/(1-r**N), with h_i=i/N if r=1.

For expected elapsed time in the finite-boundary model, the script uses a mean
waiting time Delta_t/i in state i and solves
    E_i = Delta_t/i + p*E_(i+1) + q*E_(i-1), E_0=E_N=0.
The Appendix A3 plot instead uses the paper's recurrence for extinction under
q/p>1, documented in ``extinction_time_curve`` below.
"""
from pathlib import Path
import math
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUTPUT_DIR = Path(__file__).resolve().parent / "plots_generated"

# ----------------------------- Model settings -----------------------------
MODEL_INITIAL_ABUNDANCE = 1
MODEL_TARGET_ABUNDANCE = 10_000
MODEL_DELTA_T_DAYS = 0.1

# Single-bacterium plots: q/p range and target used in the earlier requests.
RATIO_PLOT_MIN = 0.90
RATIO_PLOT_MAX = 1.05
RATIO_PLOT_POINTS = 1_500
LINEAR_PROBABILITY_YMAX = 0.08

# Figure 1c replication: four ratios, initial abundance i=1, log10(N) up to 10.
FIGURE_1C_I = 1
FIGURE_1C_Q_OVER_P = (1.000025, 1.000050, 1.000075, 1.000100)
FIGURE_1C_MAX_LOG10_N = 10.0
FIGURE_1C_Y_FLOOR = -450_000.0
FIGURE_1C_GRID_POINTS = 3_000
FIGURE_1C_COLORS = ("#1835a5", "#5350a4", "#a64f96", "#d52d81")

# Appendix Figure A3: values from its caption and the authors' linked code.
A3_BASE_Q = 0.5002
A3_NOISE_MAGNITUDE = 2e-4       # uniform absolute perturbation around base q
A3_DELTA_T_DAYS = 1.0 / 24.0    # c=1/24 day in the paper's MATLAB source
A3_MAX_ABUNDANCE = 1_000
A3_NOISE_TRAJECTORIES = 25
A3_RANDOM_SEED = 30_219         # fixed so the 25 illustrative curves reproduce
A3_COLORS = {"baseline": "#111111", "noise": "#7b2cbf"}

# Critical equal-birth/death scenario. A finite carrying capacity is required
# for a finite expected extinction time when p=q=0.5.
CRITICAL_Q = 0.5
CRITICAL_CAPACITY = 10_000
CRITICAL_MAX_ABUNDANCE = 1_000
CRITICAL_NOISE_MAGNITUDE = 2e-4
CRITICAL_NOISE_TRAJECTORIES = 25
SMALL_POPULATION_MAX_ABUNDANCE = 100
A3_SMALL_POPULATION_BASE_Q = 0.5002
A3_SMALL_POPULATION_RANDOM_SEED = 30_221
CRITICAL_RANDOM_SEED = 30_220


def probabilities_from_ratio(q_over_p: float) -> tuple[float, float]:
    """Return probabilities (p, q) from r=q/p, subject to p+q=1."""
    if q_over_p <= 0:
        raise ValueError("q/p must be positive")
    p = 1.0 / (1.0 + q_over_p)
    return p, 1.0 - p


def _log_expm1_positive(z):
    """Stable log(expm1(z)) for positive scalar or array values."""
    values = np.asarray(z, dtype=float)
    result = np.empty_like(values)
    large = values > 50.0
    result[large] = values[large] + np.log1p(-np.exp(-values[large]))
    result[~large] = np.log(np.expm1(values[~large]))
    return float(result) if result.ndim == 0 else result


def log10_hitting_probability(i: int, N, q_over_p: float):
    """Return log10 P(hit N before 0), using a stable form of gambler's ruin.

    N can be a scalar or NumPy array. The stable expm1 form avoids cancellation
    for q/p close to 1 and avoids exponentiating very large q/p powers.
    """
    if q_over_p <= 0:
        raise ValueError("q/p must be positive")
    n = np.asarray(N, dtype=float)
    if i < 1 or np.any(n < i):
        raise ValueError("Require 1 <= i <= N")
    if q_over_p == 1.0:
        result = np.log10(i / n)
    else:
        log_r = math.log(q_over_p)
        if q_over_p < 1.0:
            log_num = np.log(-np.expm1(i * log_r))
            log_den = np.log(-np.expm1(n * log_r))
        else:
            log_num = _log_expm1_positive(i * log_r)
            log_den = _log_expm1_positive(n * log_r)
        result = (log_num - log_den) / math.log(10.0)
    return float(result) if np.ndim(result) == 0 else result


def hitting_probability(i: int, N: int, q_over_p: float) -> float:
    """Probability of reaching target N before extinction at 0."""
    return 10.0 ** log10_hitting_probability(i, N, q_over_p)


def expected_absorption_time(i: int, N: int, q_over_p: float,
                             delta_t: float = MODEL_DELTA_T_DAYS) -> float:
    """Expected time to hit 0 or N, with mean wait delta_t/state.

    Solves the finite tridiagonal recurrence in O(N) time and memory. Returned
    time uses the same units as delta_t (days by default).
    """
    if N < 1 or i < 0 or i > N:
        raise ValueError("Require N >= 1 and 0 <= i <= N")
    if i == 0 or i == N:
        return 0.0
    p, q = probabilities_from_ratio(q_over_p)
    m = N - 1  # unknown transient states E_1,...,E_(N-1)
    upper = -p
    lower = -q
    diag = np.ones(m, dtype=float)
    rhs = delta_t / np.arange(1, N, dtype=float)

    # Thomas algorithm for the tridiagonal linear system.
    cprime = np.zeros(m, dtype=float)
    dprime = np.zeros(m, dtype=float)
    cprime[0] = upper / diag[0] if m > 1 else 0.0
    dprime[0] = rhs[0] / diag[0]
    for row in range(1, m):
        denom = diag[row] - lower * cprime[row - 1]
        cprime[row] = upper / denom if row < m - 1 else 0.0
        dprime[row] = (rhs[row] - lower * dprime[row - 1]) / denom
    values = np.zeros(m, dtype=float)
    values[-1] = dprime[-1]
    for row in range(m - 2, -1, -1):
        values[row] = dprime[row] - cprime[row] * values[row + 1]
    return float(values[i - 1])


def plot_single_bacterium_vs_ratio() -> None:
    """Save log-y and linear-y probability plots for i=1 and N=10^4."""
    ratios = np.linspace(RATIO_PLOT_MIN, RATIO_PLOT_MAX, RATIO_PLOT_POINTS)
    logp = np.array([
        log10_hitting_probability(MODEL_INITIAL_ABUNDANCE,
                                 MODEL_TARGET_ABUNDANCE, r)
        for r in ratios
    ])

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(ratios, logp, color="#1769aa", linewidth=2)
    ax.set_xlabel("Death-to-birth ratio, q/p")
    ax.set_ylabel(r"$\log_{10}(h_1)$")
    ax.set_title(r"One bacterium reaching $N=10^4$ before extinction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "single_bacterium_log10_probability_vs_q_over_p.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)

    probability = np.power(10.0, logp)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(ratios, probability, color="#1769aa", linewidth=2)
    ax.set_ylim(0.0, LINEAR_PROBABILITY_YMAX)
    ax.set_xlabel("Death-to-birth ratio, q/p")
    ax.set_ylabel(r"Hitting probability, $h_1$")
    ax.set_title(r"One bacterium reaching $N=10^4$ before extinction")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "single_bacterium_probability_vs_q_over_p.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_figure1c_replication() -> None:
    """Create four q/p curves on Figure 1c-style log-log axes."""
    xgrid = np.linspace(0.0, FIGURE_1C_MAX_LOG10_N, FIGURE_1C_GRID_POINTS)
    targets = np.maximum(FIGURE_1C_I, np.rint(10.0 ** xgrid)).astype(np.int64)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for ratio, color in zip(FIGURE_1C_Q_OVER_P, FIGURE_1C_COLORS):
        y_all = log10_hitting_probability(FIGURE_1C_I, targets, ratio)
        visible = y_all >= FIGURE_1C_Y_FLOOR
        x, y = xgrid[visible], y_all[visible]
        if len(x) and len(x) < len(xgrid):
            j = len(x)
            xa, xb = xgrid[j - 1], xgrid[j]
            ya, yb = y_all[j - 1], y_all[j]
            x = np.append(x, xa + (FIGURE_1C_Y_FLOOR - ya) *
                          (xb - xa) / (yb - ya))
            y = np.append(y, FIGURE_1C_Y_FLOOR)
        ax.plot(x, y, color=color, linewidth=1.8, marker="o",
                markersize=2.0, markevery=45, label=f"{ratio:.6f}")

    ax.set_xlim(0.0, FIGURE_1C_MAX_LOG10_N)
    ax.set_ylim(FIGURE_1C_Y_FLOOR, 0.0)
    ax.set_xlabel(r"$\log_{10}(N)$ (target population size)")
    ax.set_ylabel(r"$\log_{10}(P_i)$")
    ax.set_title("Single immigrant reaching target abundance")
    ax.grid(True, alpha=0.3)
    ax.legend(title="q/p", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "figure_1c_replication_four_ratios.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def extinction_time_curve(q: float) -> np.ndarray:
    """Appendix 4 recurrence: expected time to extinction from n=1..1000.

    For alpha=q/p>1:
        tau_1 = -(Delta_t/p)*ln(1 - 1/alpha)
        tau_(i+1) = alpha*tau_i - Delta_t/(p*i)
        T_n = sum_(j=1)^n tau_j.
    """
    p = 1.0 - q
    if not (0.0 < p < q < 1.0):
        raise ValueError("This recurrence requires 0 < p < q < 1")
    alpha = q / p
    delta = np.zeros(A3_MAX_ABUNDANCE, dtype=float)
    delta[0] = -(A3_DELTA_T_DAYS / p) * math.log1p(-1.0 / alpha)
    for i in range(1, A3_MAX_ABUNDANCE):
        delta[i] = alpha * delta[i - 1] - A3_DELTA_T_DAYS / (p * i)
    return np.cumsum(delta)


def plot_appendix_a3() -> None:
    """Create the A3-style extinction plot with 25 purple trajectories."""
    rng = random.Random(A3_RANDOM_SEED)
    n = np.arange(1, A3_MAX_ABUNDANCE + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for _ in range(A3_NOISE_TRAJECTORIES):
        # One uniform draw per trajectory, held fixed across all abundance n.
        q = A3_BASE_Q + rng.uniform(-A3_NOISE_MAGNITUDE,
                                    A3_NOISE_MAGNITUDE)
        q = max(q, math.nextafter(0.5, 1.0))  # avoid exact critical q=p=0.5
        ax.plot(n, np.log10(extinction_time_curve(q)),
                color=A3_COLORS["noise"], linewidth=1.0, alpha=0.38)

    baseline = extinction_time_curve(A3_BASE_Q)
    ax.plot(n, np.log10(baseline), color=A3_COLORS["baseline"], linewidth=2.0)
    legend_handles = [
        Line2D([0], [0], color=A3_COLORS["baseline"], lw=2,
               label="Base q = 0.5002"),
        Line2D([0], [0], color=A3_COLORS["noise"], lw=1.5, alpha=0.7,
               label="25 uniform-noise trajectories (±2e-4)"),
    ]
    ax.set_xlim(0, A3_MAX_ABUNDANCE)
    ax.set_ylim(-0.5, 2.5)
    ax.set_xlabel("n")
    ax.set_ylabel(r"$\log_{10}(\mathrm{time\ (days)})$")
    ax.set_title("Time to extinction from state n")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "appendix_A3_extinction_25_purple_trajectories.png",
                dpi=220, bbox_inches="tight")
    plt.close(fig)


def finite_capacity_extinction_curve(q: float, capacity: int,
                                     max_abundance: int,
                                     delta_t: float = A3_DELTA_T_DAYS) -> np.ndarray:
    """Expected extinction time under a reflecting upper capacity.

    Extinction at 0 is absorbing. At the carrying capacity K, births are
    blocked, so the population steps down to K-1 after the mean wait
    delta_t/K. For transient states i<K, solve
        T_i = delta_t/i + p*T_(i+1) + q*T_(i-1).
    Writing d_i=T_i-T_(i-1), the backward recurrence is
        d_i = (p/q)*d_(i+1) + delta_t/(q*i), with d_K=delta_t/K.
    This also gives the appendix's critical p=q formula when q=0.5.
    """
    if not (0.0 < q < 1.0):
        raise ValueError("q must be between 0 and 1")
    if capacity < 2 or not (1 <= max_abundance <= capacity):
        raise ValueError("Require 1 <= max_abundance <= capacity and capacity >= 2")
    p = 1.0 - q
    increments = np.empty(capacity + 1, dtype=float)
    increments[capacity] = delta_t / capacity
    for i in range(capacity - 1, 0, -1):
        increments[i] = (p / q) * increments[i + 1] + delta_t / (q * i)
    return np.cumsum(increments[1:max_abundance + 1])


def plot_equal_birth_death_extinction() -> None:
    """Plot critical p=q extinction times plus 25 fixed-q noise trajectories."""
    rng = random.Random(CRITICAL_RANDOM_SEED)
    n = np.arange(1, CRITICAL_MAX_ABUNDANCE + 1)
    fig, ax = plt.subplots(figsize=(8.5, 5.2))

    for _ in range(CRITICAL_NOISE_TRAJECTORIES):
        q = CRITICAL_Q + rng.uniform(-CRITICAL_NOISE_MAGNITUDE,
                                     CRITICAL_NOISE_MAGNITUDE)
        times = finite_capacity_extinction_curve(
            q, CRITICAL_CAPACITY, CRITICAL_MAX_ABUNDANCE)
        ax.plot(n, np.log10(times), color=A3_COLORS["noise"],
                linewidth=1.0, alpha=0.38)

    baseline = finite_capacity_extinction_curve(
        CRITICAL_Q, CRITICAL_CAPACITY, CRITICAL_MAX_ABUNDANCE)
    ax.plot(n, np.log10(baseline), color=A3_COLORS["baseline"], linewidth=2.0)
    legend_handles = [
        Line2D([0], [0], color=A3_COLORS["baseline"], lw=2,
               label=r"Equal births and deaths ($p=q=0.5$)"),
        Line2D([0], [0], color=A3_COLORS["noise"], lw=1.5, alpha=0.7,
               label=r"25 uniform-$q$ trajectories ($\pm 2\times10^{-4}$)"),
    ]
    ax.set_xlim(0, CRITICAL_MAX_ABUNDANCE)
    ax.set_xlabel("Starting abundance, n")
    ax.set_ylabel(r"$\log_{10}(\mathrm{expected\ time\ to\ extinction\ (days)})$")
    ax.set_title(
        f"Equal birth and death: extinction time (capacity K={CRITICAL_CAPACITY:,})")
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "equal_birth_death_extinction_25_trajectories.png",
                dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "equal_birth_death_extinction_25_trajectories.svg",
                bbox_inches="tight")
    plt.close(fig)


def plot_small_population_comparison() -> None:
    """Compare equal births/deaths and q=0.5002 for starting n=1..100."""
    rng = random.Random(A3_SMALL_POPULATION_RANDOM_SEED)
    n = np.arange(1, SMALL_POPULATION_MAX_ABUNDANCE + 1)
    scenarios = [
        ("Equal births and deaths", CRITICAL_Q),
        ("Earlier A3 baseline", A3_SMALL_POPULATION_BASE_Q),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)

    for ax, (title, base_q) in zip(axes, scenarios):
        for _ in range(CRITICAL_NOISE_TRAJECTORIES):
            q = base_q + rng.uniform(-CRITICAL_NOISE_MAGNITUDE,
                                     CRITICAL_NOISE_MAGNITUDE)
            times = finite_capacity_extinction_curve(
                q, CRITICAL_CAPACITY, SMALL_POPULATION_MAX_ABUNDANCE)
            ax.plot(n, np.log10(times), color=A3_COLORS["noise"],
                    linewidth=1.0, alpha=0.38)

        baseline = finite_capacity_extinction_curve(
            base_q, CRITICAL_CAPACITY, SMALL_POPULATION_MAX_ABUNDANCE)
        ax.plot(n, np.log10(baseline), color=A3_COLORS["baseline"],
                linewidth=2.0, label=f"Baseline q = {base_q:.4f}")
        ax.set_title(title)
        ax.set_xlim(1, SMALL_POPULATION_MAX_ABUNDANCE)
        ax.set_xlabel("Starting abundance, n")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

    axes[0].set_ylabel(r"$\log_{10}(\mathrm{expected\ time\ to\ extinction\ (days)})$")
    fig.suptitle(
        f"Small-population extinction times (n=1–{SMALL_POPULATION_MAX_ABUNDANCE}; "
        f"capacity K={CRITICAL_CAPACITY:,})")
    fig.text(0.5, 0.015,
             r"Purple lines: 25 trajectories per panel; q uniformly perturbed "
             r"by $\pm2\times10^{-4}$, fixed along each curve.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUTPUT_DIR / "small_population_extinction_comparison.png",
                dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "small_population_extinction_comparison.svg",
                bbox_inches="tight")
    plt.close(fig)


def plot_all_extinction_base_curves() -> None:
    """Overlay the four requested extinction-time baseline curves."""
    colors = {
        "A3": "#222222",
        "equal_normal": "#1769aa",
        "equal_small": "#d62728",
        "A3_small": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(9.0, 5.6))

    n_normal = np.arange(1, A3_MAX_ABUNDANCE + 1)
    a3_base = extinction_time_curve(A3_BASE_Q)
    equal_base = finite_capacity_extinction_curve(
        CRITICAL_Q, CRITICAL_CAPACITY, A3_MAX_ABUNDANCE)
    ax.plot(n_normal, np.log10(a3_base), color=colors["A3"], linewidth=2.0,
            label=fr"A3 baseline ($q={A3_BASE_Q:.4f}$; $n=1$–{A3_MAX_ABUNDANCE})")
    ax.plot(n_normal, np.log10(equal_base), color=colors["equal_normal"],
            linewidth=2.0,
            label=fr"Equal births/deaths, normal range ($n=1$–{A3_MAX_ABUNDANCE})")

    n_small = np.arange(1, SMALL_POPULATION_MAX_ABUNDANCE + 1)
    equal_small = finite_capacity_extinction_curve(
        CRITICAL_Q, CRITICAL_CAPACITY, SMALL_POPULATION_MAX_ABUNDANCE)
    a3_small = finite_capacity_extinction_curve(
        A3_SMALL_POPULATION_BASE_Q, CRITICAL_CAPACITY,
        SMALL_POPULATION_MAX_ABUNDANCE)
    ax.plot(n_small, np.log10(equal_small), color=colors["equal_small"],
            linewidth=2.0, linestyle="--",
            label=fr"Equal births/deaths, small range ($n=1$–{SMALL_POPULATION_MAX_ABUNDANCE})")
    ax.plot(n_small, np.log10(a3_small), color=colors["A3_small"],
            linewidth=2.0, linestyle="--",
            label=fr"A3 baseline, small range ($q={A3_SMALL_POPULATION_BASE_Q:.4f}$; "
                  fr"$n=1$–{SMALL_POPULATION_MAX_ABUNDANCE})")

    ax.set_xlim(1, A3_MAX_ABUNDANCE)
    ax.set_xlabel("Starting abundance, n")
    ax.set_ylabel(r"$\log_{10}(\mathrm{expected\ time\ to\ extinction\ (days)})$")
    ax.set_title("Extinction-time baseline curves")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "extinction_time_all_base_curves.png",
                dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "extinction_time_all_base_curves.svg",
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_single_bacterium_vs_ratio()
    plot_figure1c_replication()
    plot_appendix_a3()
    plot_equal_birth_death_extinction()
    plot_small_population_comparison()
    plot_all_extinction_base_curves()
    print(f"Plots saved in: {OUTPUT_DIR}")
    print(f"Model example: p,q for q/p=1.0001 are "
          f"{probabilities_from_ratio(1.0001)}")
    print(f"Model example: P(hit 10,000 before 0) from i=1, q/p=1.0001 is "
          f"{hitting_probability(1, 10_000, 1.0001):.6g}")
    print(f"Model example: expected absorption time from i=1 to 0 or 10,000 "
          f"at q/p=1.0001 is "
          f"{expected_absorption_time(1, 10_000, 1.0001):.6g} days "
          f"(Delta t/state, Delta t={MODEL_DELTA_T_DAYS} days)")
    print(f"A3 baseline log10(T_1000 in days) = "
          f"{math.log10(extinction_time_curve(A3_BASE_Q)[-1]):.4f}")
    print(f"Equal-birth/death baseline log10(T_1000 in days), K="
          f"{CRITICAL_CAPACITY:,} = "
          f"{math.log10(finite_capacity_extinction_curve(CRITICAL_Q, CRITICAL_CAPACITY, CRITICAL_MAX_ABUNDANCE)[-1]):.4f}")


if __name__ == "__main__":
    main()
