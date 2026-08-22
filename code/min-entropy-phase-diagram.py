"""Phase diagram for the self-consistency condition of minimum-entropy learning.

The condition  p_c = p0_c exp(p_c / beta) / Z  with a uniform reference over K
semantic clusters admits, within the symmetric ansatz (one dominant cluster of
mass a, the remaining K-1 sharing 1-a equally), the scalar form

    a = 1 / (1 + (K-1) exp([(1-a)/(K-1) - a] / beta)).

Roots of this equation are traced against beta to give the branch structure.
Produces figures/min-entropy-phase.{svg,png} and prints the numerical tables
quoted in the chapter on minimum-entropy learning.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def residual(a, beta, K):
    """Zero when a is a self-consistent dominant-cluster mass."""
    b = (1 - a) / (K - 1)
    return 1.0 / (1.0 + (K - 1) * np.exp((b - a) / beta)) - a


def fixed_point_full(p_init, beta, K, tol=1e-13, max_iter=500_000):
    """Iterate the full K-dimensional self-consistency map to convergence."""
    p0 = np.ones(K) / K
    p = np.asarray(p_init, dtype=float)
    for _ in range(max_iter):
        w = p0 * np.exp(p / beta)
        p_new = w / w.sum()
        if np.max(np.abs(p_new - p)) < tol:
            return p_new
        p = p_new
    return p


def roots(beta, K, eps=1e-7, grid=20_001):
    """All self-consistent a strictly above 1/K, found by scan and bisection."""
    xs = np.linspace(1 / K + eps, 1 - eps, grid)
    fs = residual(xs, beta, K)
    out = []
    for i in np.nonzero(np.sign(fs[:-1]) * np.sign(fs[1:]) < 0)[0]:
        lo, hi = xs[i], xs[i + 1]
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if np.sign(residual(mid, beta, K)) == np.sign(residual(lo, beta, K)):
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return sorted(out)


def spinodal(K, hi=4.0, steps=200):
    """Largest beta (in units of beta_c = 1/K) at which a collapsed root exists."""
    lo = 1.0
    if not roots(1.0005 / K, K):
        return 1.0
    while roots(hi / K, K) and hi < 64:
        hi *= 2
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if roots(mid / K, K):
            lo = mid
        else:
            hi = mid
    return lo


NAVY, RUST, GREY = "#1f3b57", "#b4451f", "#8a8a8a"
XMIN, XMAX = 0.35, 2.2

fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))

for ax, K in zip(axes, (2, 3)):
    x = np.linspace(XMIN, XMAX, 900)

    # Balanced branch: stable above beta_c, unstable below.
    ax.plot([1.0, XMAX], [1 / K] * 2, color=NAVY, lw=2.4,
            solid_capstyle="round", zorder=4)
    ax.plot([XMIN, 1.0], [1 / K] * 2, color=NAVY, lw=1.7, ls=(0, (3, 2.6)),
            zorder=4)

    # Collapsed branch (upper root) and, where present, the unstable root
    # separating it from the balanced state.
    upper = np.full_like(x, np.nan)
    lower = np.full_like(x, np.nan)
    for i, xi in enumerate(x):
        r = roots(xi / K, K)
        if r:
            upper[i] = r[-1]
            if len(r) > 1:
                lower[i] = r[0]
    ax.plot(x, upper, color=RUST, lw=2.4, solid_capstyle="round", zorder=4)
    ax.plot(x, lower, color=RUST, lw=1.7, ls=(0, (3, 2.6)), zorder=4)

    sp = spinodal(K)
    if sp > 1.001:
        ax.axvspan(1.0, sp, color=RUST, alpha=0.10, lw=0, zorder=0)
        ax.plot([sp, sp], [0, 1.04], color=RUST, lw=0.9, ls=":", zorder=1)
        ax.annotate("both states stable:\nfate depends on history", xy=(0.5 * (1.0 + sp), 0.20),
                    xytext=(1.42, 0.13), fontsize=8.5, color=RUST,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=RUST, lw=0.8,
                                    shrinkA=2, shrinkB=2))
        # Point at the unstable branch: the one-way tipping line between states.
        mid_x = 0.5 * (1.0 + sp)
        r_mid = roots(mid_x / K, K)
        if len(r_mid) > 1:
            ax.annotate("tipping line:\ncrossing it is one-way",
                        xy=(mid_x, r_mid[0]), xytext=(1.42, 0.55),
                        fontsize=8.5, color=RUST, ha="left", va="center",
                        arrowprops=dict(arrowstyle="-", color=RUST, lw=0.8,
                                        shrinkA=2, shrinkB=2))
    else:
        # K = 2: there is no unstable collapsed branch to draw -- the collapsed
        # state grows continuously out of the balanced one, so the transition
        # is reversible and no tipping line exists.
        ax.annotate("branches meet: commitment grows\ngradually, and is reversible\n(no tipping line exists at $K=2$)",
                    xy=(1.0, 0.52), xytext=(1.18, 0.72),
                    fontsize=8.5, color=RUST, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", color=RUST, lw=0.8,
                                    shrinkA=2, shrinkB=2))

    ax.axvline(1.0, color=GREY, lw=0.9, ls="--", zorder=1)
    ax.text(0.985, 0.02, r"$\beta_c$", fontsize=9.5, color=GREY,
            ha="right", va="bottom")

    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(0, 1.04)
    ax.set_xlabel(r"KL coefficient  $\beta\,/\,\beta_c$,   where $\beta_c = 1/K$",
                  fontsize=9.5)
    ax.set_ylabel(r"dominant cluster mass  $\max_c p_c$", fontsize=9.5)
    ax.set_title(
        rf"$K = {K}$:  " + ("continuous" if K == 2 else "discontinuous"),
        fontsize=11, pad=9,
    )
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

axes[0].legend(
    handles=[
        Line2D([], [], color=NAVY, lw=2.4, label="balanced, stable"),
        Line2D([], [], color=NAVY, lw=1.7, ls=(0, (3, 2.6)), label="balanced, unstable"),
        Line2D([], [], color=RUST, lw=2.4, label="collapsed, stable"),
        Line2D([], [], color=RUST, lw=1.7, ls=(0, (3, 2.6)),
               label="collapsed, unstable ($K \\geq 3$ only)"),
    ],
    fontsize=8.2, frameon=False, loc="lower left", bbox_to_anchor=(0.01, 0.03),
    handlelength=2.2, labelspacing=0.5,
)

fig.tight_layout()
fig.savefig("figures/min-entropy-phase.svg", bbox_inches="tight")
fig.savefig("figures/min-entropy-phase.png", dpi=200, bbox_inches="tight")


# ---- numbers quoted in the text -------------------------------------------

print("\nGibbs tilt, K = 3, uniform reference, r = (1.0, 0.5, 0.0)")
p0, r = np.ones(3) / 3, np.array([1.0, 0.5, 0.0])
for beta in (np.inf, 2.0, 1.0, 0.25, 0.05):
    w = p0 * np.exp(r / beta if np.isfinite(beta) else np.zeros(3))
    p = w / w.sum()
    print(f"  beta={beta:>6}: p=({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  "
          f"E[r]={p @ r:.3f}")

print("\nTrajectories from p = (0.36, 0.32, 0.32), K = 3, uniform reference")
for beta in (0.50, 0.25):
    p = np.array([0.36, 0.32, 0.32])
    print(f"  beta = {beta}  ({beta * 3:.2f} beta_c)")
    for t in range(9):
        if t % 2 == 0:
            H = -(p * np.log(p)).sum()
            print(f"    step {t}: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})  "
                  f"gap {p.max() - p.min():.4f}  H {H:.4f}")
        w = np.exp(p / beta) / 3
        p = w / w.sum()

print("\nBistability: same beta, two starting points, K = 3")
# The balanced start carries a 1e-3 perturbation: an exactly uniform start is
# a fixed point of the symmetric map at every beta and so tests nothing.
p_balanced = np.array([1 / 3 + 1e-3, 1 / 3 - 5e-4, 1 / 3 - 5e-4])
for ratio in (1.200, 1.080, 1.035, 1.020, 1.000):
    beta = ratio / 3
    a = fixed_point_full(p_balanced, beta, 3)
    b = fixed_point_full(np.array([0.90, 0.05, 0.05]), beta, 3)
    print(f"  beta={beta:.4f} ({ratio:.3f} beta_c): from balanced {a.max():.4f}, "
          f"from concentrated {b.max():.4f}")

print("\nPer-step amplification of an imbalance, K = 3  (beta_c = 1/3)")
for ratio in (2.0, 1.5, 1.1, 1.0, 0.9, 0.5):
    beta = ratio / 3
    print(f"  beta/beta_c={ratio:>4}: anchored x{1/(3*beta):.3f}   "
          f"online x{1 + 1/(3*beta):.3f}")

print("\nBistable window and jump size")
for K in (2, 3, 5, 8):
    sp = spinodal(K)
    r_at = roots(0.995 / K, K)
    jump = r_at[-1] if r_at else np.nan
    r_sp = roots((sp - 1e-6) / K, K)
    print(f"  K={K}: window [1.000, {sp:.3f}] beta_c   "
          f"balanced mass {1/K:.3f}   "
          f"collapsed mass at 0.995 beta_c {jump:.3f}   "
          f"at spinodal {r_sp[-1] if r_sp else float('nan'):.3f}")
