"""Schematic of predictive coding for the chapter on the free-energy principle.

Three panels, all drawn (no data is fitted):
  (a) the two-level circuit of Example 1 (Bogacz 2017): a prior v_p, a
      state node phi, two prediction-error nodes, and the input u;
      predictions travel down, precision-weighted errors travel up, and
      the state node follows the gradient of the free energy;
  (b) the same motif stacked into a hierarchy, each level predicting the
      one below and receiving only the residual from it;
  (c) the engineering idea behind the name (Elias 1955; Srinivasan et
      al. 1982): a signal, its prediction, and the residual that is all
      the channel has to carry.

It also prints the posterior mode of Example 1 for several ratios of
the prior and sensory precisions, which are the numbers quoted in the
text on precision as gain.

Deterministic; needs only numpy and matplotlib. Run
`python3 code/fep-predictive-coding.py` from the book root; it writes
figures/fep-predictive-coding.png.
"""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figures", "fep-predictive-coding.png")
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
})
NAVY, RUST, GREEN, GOLD, GREY = (
    "#1f3b63", "#b5532a", "#3c7d4a", "#c99a2e", "#777777")


def box(ax, xy, w, h, text, fc="#f3f3f3", ec=GREY, fs=9.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def node(ax, xy, text, color, r=0.17, fs=9.5):
    ax.add_patch(Circle(xy, r, fc="white", ec=color, lw=1.4))
    ax.text(xy[0], xy[1], text, ha="center", va="center", fontsize=fs,
            color=color)


def arrow(ax, p, q, color, label=None, rad=0.0, lw=1.3, lpos=0.5,
          loff=(0, 0), fs=8, ls="-", ha="center"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=4, shrinkB=4))
    if label:
        x = p[0] + lpos * (q[0] - p[0]) + loff[0]
        y = p[1] + lpos * (q[1] - p[1]) + loff[1]
        ax.text(x, y, label, color=color, fontsize=fs, ha=ha, va="center")


def panel_circuit(ax):
    """Two columns: state units on the left, prediction-error units on the
    right. Predictions run from a state to the error unit one level
    below; the error unit compares the prediction with what is there
    and sends the precision-weighted difference back up."""
    ax.set_xlim(-1.75, 2.15)
    ax.set_ylim(-0.95, 3.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) the circuit of Example 1")
    xs, xe = -0.7, 0.8                        # state column, error column
    y2, y1, y0 = 3.0, 1.5, 0.0                # level 2 (prior), 1, 0 (input)
    box(ax, (xs, y2), 1.3, 0.44, r"prior  $v_p,\ \Sigma_p$")
    node(ax, (xs, y1), r"$\phi$", NAVY, r=0.22)
    box(ax, (xs, y0), 1.3, 0.44, r"input  $u,\ \Sigma_u$")
    node(ax, (xe, y1), r"$\varepsilon_p$", RUST, r=0.2)
    node(ax, (xe, y0), r"$\varepsilon_u$", RUST, r=0.2)
    # predictions descend to the error unit of the level below (navy)
    arrow(ax, (xs + 0.35, y2 - 0.22), (xe - 0.1, y1 + 0.2), NAVY)
    ax.text(0.36, 2.21, "prediction $v_p$", color=NAVY, fontsize=8,
            ha="center", va="bottom", rotation=-45, rotation_mode="anchor")
    arrow(ax, (xs + 0.1, y1 - 0.2), (xe - 0.22, y0 + 0.12), NAVY)
    ax.text(-0.23, 0.77, "prediction $\\phi^2$", color=NAVY, fontsize=8,
            ha="center", va="top", rotation=-45, rotation_mode="anchor")
    # the state of each level enters its own error unit (grey)
    arrow(ax, (xs + 0.22, y1 + 0.07), (xe - 0.2, y1 + 0.07), GREY,
          label=r"$\phi$", loff=(0, 0.16), fs=8)
    arrow(ax, (xs + 0.65, y0 + 0.07), (xe - 0.2, y0 + 0.07), GREY,
          label=r"$u$", loff=(0, 0.16), fs=8)
    # precision-weighted errors return to the state unit (rust)
    arrow(ax, (xe - 0.2, y1 - 0.07), (xs + 0.22, y1 - 0.07), RUST, lw=1.5,
          label=r"$-\,\varepsilon_p$", loff=(0, -0.17), fs=8)
    arrow(ax, (xe + 0.05, y0 + 0.22), (xs + 0.3, y1 - 0.1), RUST, lw=1.5)
    ax.text(0.52, 0.67, "$+\\,2\\phi\\,\\varepsilon_u$", color=RUST,
            fontsize=8, ha="center", va="bottom", rotation=-45,
            rotation_mode="anchor")
    ax.text(xe + 0.3, y1 + 0.02, "error\n$\\times$ gain $1/\\Sigma_p$",
            color=RUST, fontsize=8, ha="left", va="center")
    ax.text(xe + 0.3, y0 + 0.02, "error\n$\\times$ gain $1/\\Sigma_u$",
            color=RUST, fontsize=8, ha="left", va="center")
    ax.text(0.2, -0.7,
            r"$\dot\phi = 2\phi\,\varepsilon_u - \varepsilon_p = -\,\partial F/\partial\phi$",
            ha="center", va="center", fontsize=9, color=NAVY)


def panel_hierarchy(ax):
    ax.set_xlim(-1.9, 2.0)
    ax.set_ylim(-0.95, 3.65)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b) the motif stacked into a hierarchy")
    levels = [(0.0, "sensory input"), (1.1, "level 1"), (2.2, "level 2"),
              (3.3, r"level 3 $\cdots$")]
    for y, name in levels:
        fc = "#e9eef5" if y > 0 else "#f3f3f3"
        box(ax, (0, y), 1.5, 0.5, name, fc=fc)
    for (y0, _), (y1, _) in zip(levels[:-1], levels[1:]):
        # prediction down, on the left
        arrow(ax, (-0.3, y1 - 0.25), (-0.3, y0 + 0.25), NAVY)
        # residual up, on the right, with a gain mark
        arrow(ax, (0.3, y0 + 0.25), (0.3, y1 - 0.25), RUST)
        ym = 0.5 * (y0 + y1)
        ax.add_patch(Circle((0.3, ym), 0.085, fc="white", ec=RUST, lw=1.1))
        ax.text(0.3, ym, r"$\Pi$", ha="center", va="center", fontsize=7,
                color=RUST)
    ax.text(-0.85, 1.65, "predictions\ndescend", color=NAVY, fontsize=8.5,
            ha="right", va="center")
    ax.text(0.85, 1.65, "only the residual\nascends, scaled\nby precision $\\Pi$",
            color=RUST, fontsize=8.5, ha="left", va="center")
    ax.text(0.05, -0.7,
            "each level explains the one below;\nwhat is explained is silenced",
            ha="center", va="center", fontsize=8.5, color=GREY)


def panel_residual(ax):
    t = np.linspace(0, 10, 400)
    signal = 1.0 + 0.6 * np.sin(0.9 * t) + 0.25 * np.sin(2.3 * t)
    signal = signal + 0.9 * (t > 6.2) * np.exp(-(t - 6.2) / 1.6)
    # a smooth prediction from the recent past: the signal delayed and
    # smoothed, which tracks the regular part and misses the step
    pred = 1.0 + 0.6 * np.sin(0.9 * (t - 0.25)) + 0.25 * np.sin(2.3 * (t - 0.25))
    resid = signal - pred
    ax.plot(t, signal, color=NAVY, lw=1.6, label="signal")
    ax.plot(t, pred, color=NAVY, lw=1.1, ls="--", label="prediction")
    ax.plot(t, resid, color=RUST, lw=1.6, label="residual (what is sent)")
    ax.axhline(0, color=GREY, lw=0.6)
    ax.axvline(6.2, color=GREY, lw=0.6, ls=":")
    ax.text(6.3, 2.35, "surprise", color=GREY, fontsize=8, ha="left")
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("time")
    ax.set_ylabel("amplitude")
    ax.set_title("(c) the name: transmit only the residual")
    ax.legend(loc="upper left", frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def precision_modes():
    """Posterior mode of Example 1 (prior N(3, Sigma_p), likelihood
    N(u; v^2, Sigma_u), u = 2) for several ratios of the two precisions.
    Only the ratio matters for the mode, which is the point made in the
    text: a precision is the gain of an error unit."""
    v = np.linspace(0.0, 4.0, 4_000_001)
    print("posterior mode of Example 1 versus the precisions")
    for sp, su in [(1.0, 1.0), (1.0, 0.25), (0.25, 1.0), (1.0, 4.0)]:
        f = (v - 3.0) ** 2 / (2 * sp) + (2.0 - v * v) ** 2 / (2 * su)
        m = v[np.argmin(f)]
        print(f"  Sigma_p = {sp:<5} Sigma_u = {su:<5} mode = {m:.4f}")
    print(f"  sqrt(u) = {np.sqrt(2.0):.4f}")


def main():
    precision_modes()
    fig = plt.figure(figsize=(10.4, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.25], wspace=0.25)
    panel_circuit(fig.add_subplot(gs[0]))
    panel_hierarchy(fig.add_subplot(gs[1]))
    panel_residual(fig.add_subplot(gs[2]))
    fig.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.14)
    fig.savefig(OUT, dpi=220)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
