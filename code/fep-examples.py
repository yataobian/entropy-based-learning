"""Numerical examples for the report on the free-energy principle (FEP).

Example 1  Perception as free-energy minimization, with the exercise of
           Bogacz (2017, Sec. 2): exact posterior and surprise, the
           Gaussian variational free energy, the Laplace approximation,
           the predictive-coding network, and the temperature dial.
Example 2  A T-maze with a cue, two arms and a dark room: expected free
           energy and its two decompositions, the precision dial, and
           one-step versus two-step (sophisticated) evaluation.
Example 3  Learning as free-energy minimization: EM for a Gaussian
           mixture at a label temperature T. After every M-step the
           average free energy equals the entropy of the maximum-entropy
           world model minus T times the entropy of the inferred causes;
           T = 1 is EM, T -> 0 is classification EM.
Example 4  Learned preferences: the preference prior is updated from the
           agent's own predicted outcomes, so that policy selection
           becomes a self-referential tilt; the precision then decides
           between diverse preferences and collapse.

Everything is deterministic and needs only numpy and matplotlib. Run
`python3 fep_examples.py` in this folder; it writes
fep_examples_results.json and the figures fig-fep-*.png.
"""

import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcfg")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9.5,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
})
NAVY, RUST, GREEN, GOLD, GREY = (
    "#1f3b63", "#b5532a", "#3c7d4a", "#c99a2e", "#777777")


def entropy(p, axis=None):
    """Shannon entropy in nats, with 0 log 0 = 0."""
    p = np.asarray(p, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(p > 0, -p * np.log(p), 0.0)
    return t.sum(axis=axis)


# ---------------------------------------------------------------------
# Example 1: perception (Bogacz 2017, Sec. 2, Exercises 1-3)
# prior p(v) = N(3, 1), likelihood p(u | v) = N(u; v^2, 1), u = 2
# ---------------------------------------------------------------------
U_OBS, V_PRIOR, S_PRIOR, S_LIK = 2.0, 3.0, 1.0, 1.0
V = np.linspace(-4.0, 8.0, 240_001)
DV = V[1] - V[0]


def energy_prior(v):
    return (0.5 * np.log(2 * np.pi * S_PRIOR)
            + (v - V_PRIOR) ** 2 / (2 * S_PRIOR))


def energy_lik(v):
    return (0.5 * np.log(2 * np.pi * S_LIK)
            + (U_OBS - v**2) ** 2 / (2 * S_LIK))


def energy(v):
    """E(v) = -log p(u, v)."""
    return energy_prior(v) + energy_lik(v)


def d_energy(v):
    return (v - V_PRIOR) / S_PRIOR - 2 * v * (U_OBS - v**2) / S_LIK


def d2_energy(v):
    return 1 / S_PRIOR + (6 * v**2 - 2 * U_OBS) / S_LIK


def gauss_free_energy(m, s):
    """Exact F(q) = E_q[E] - H[q] for q = N(m, s^2), in closed form."""
    ev2 = m**2 + s**2
    ev4 = m**4 + 6 * m**2 * s**2 + 3 * s**4
    e_pri = (0.5 * np.log(2 * np.pi * S_PRIOR)
             + ((m - V_PRIOR) ** 2 + s**2) / (2 * S_PRIOR))
    e_lik = (0.5 * np.log(2 * np.pi * S_LIK)
             + (U_OBS**2 - 2 * U_OBS * ev2 + ev4) / (2 * S_LIK))
    return e_pri + e_lik - 0.5 * np.log(2 * np.pi * np.e * s**2)


def grad_gauss_free_energy(m, s):
    dm = ((m - V_PRIOR) / S_PRIOR
          + (4 * m**3 + 12 * m * s**2 - 4 * U_OBS * m) / (2 * S_LIK))
    ds = (s / S_PRIOR
          + (12 * s**3 + 12 * m**2 * s - 4 * U_OBS * s) / (2 * S_LIK)
          - 1 / s)
    return dm, ds


def gauss_logpdf(x, m, s):
    return -0.5 * np.log(2 * np.pi * s**2) - (x - m) ** 2 / (2 * s**2)


def tempered(log_h, e, T):
    """Mean and mode of q_T proportional to h exp(-E / T) on the grid."""
    a = log_h - e / T
    w = np.exp(a - a.max())
    w /= w.sum() * DV
    return float(np.sum(V * w) * DV), float(V[np.argmax(a)])


def example_perception():
    e = energy(V)
    e0 = e.min()
    evidence = np.exp(-e0) * np.sum(np.exp(-(e - e0))) * DV
    surprise = float(-np.log(evidence))
    log_post = -e + surprise
    post = np.exp(log_post)
    mean = float(np.sum(V * post) * DV)
    sd = float(np.sqrt(np.sum((V - mean) ** 2 * post) * DV))
    mass_neg = float(np.sum(post[V < 0]) * DV)

    v_map = float(V[np.argmin(e)])
    for _ in range(50):
        v_map -= d_energy(v_map) / d2_energy(v_map)

    # gradient descent on the Gaussian free energy, starting at the prior
    lr, m, rho, trace = 0.01, V_PRIOR, 0.5 * np.log(S_PRIOR), []
    for _ in range(3000):
        s = np.exp(rho)
        trace.append(float(gauss_free_energy(m, s)))
        dm, ds = grad_gauss_free_energy(m, s)
        m, rho = m - lr * dm, rho - lr * ds * s
    m_star, s_star = float(m), float(np.exp(rho))
    f_star = float(gauss_free_energy(m_star, s_star))

    # brute-force check that gradient descent found the global minimum
    ms = np.linspace(-3.0, 4.0, 1401)
    ss = np.exp(np.linspace(np.log(0.01), np.log(2.0), 801))
    grid = gauss_free_energy(ms[:, None], ss[None, :])
    i, j = np.unravel_index(np.argmin(grid), grid.shape)

    log_q = gauss_logpdf(V, m_star, s_star)
    kl = float(np.sum(np.exp(log_q) * (log_q - log_post)) * DV)

    s_lap = float(1 / np.sqrt(d2_energy(v_map)))
    lap_surprise = float(energy(v_map)
                         + 0.5 * np.log(d2_energy(v_map) / (2 * np.pi)))
    f_lap = float(gauss_free_energy(v_map, s_lap))

    # Exercise 2: gradient flow on phi; Exercise 3: the network of
    # Bogacz Eqs. (12)-(14) with prediction-error nodes. Euler dt = 0.01.
    dt, phi_direct, flow = 0.01, V_PRIOR, []
    phi, eps_p, eps_u, net = V_PRIOR, 0.0, 0.0, []
    for k in range(2001):
        flow.append((k * dt, phi_direct))
        net.append((k * dt, phi, eps_p, eps_u))
        phi_direct -= dt * d_energy(phi_direct)
        d_phi = eps_u * 2 * phi - eps_p
        d_ep = phi - V_PRIOR - S_PRIOR * eps_p
        d_eu = U_OBS - phi**2 - S_LIK * eps_u
        phi, eps_p, eps_u = (phi + dt * d_phi, eps_p + dt * d_ep,
                             eps_u + dt * d_eu)
    flow, net = np.array(flow), np.array(net)

    # temperature dial: (a) h = prior, E = -log p(u | v);
    # (b) h = flat, E = -log p(u, v). Both give the posterior at T = 1.
    dial = []
    for T in (4.0, 1.0, 0.5, 0.1, 0.02):
        mean_a, mode_a = tempered(-energy_prior(V), energy_lik(V), T)
        row = {"T": T, "prior_base_mean": mean_a,
               "prior_base_mode": mode_a}
        if T <= 1.0:
            mean_b, mode_b = tempered(0.0 * V, e, T)
            row.update(flat_base_mean=mean_b, flat_base_mode=mode_b)
        dial.append(row)

    res = {
        "setting": {"u": U_OBS, "v_p": V_PRIOR, "Sigma_p": S_PRIOR,
                    "Sigma_u": S_LIK, "g": "v^2"},
        "surprise": surprise,
        "posterior": {"mean": mean, "sd": sd, "mode": v_map,
                      "mass_below_zero": mass_neg},
        "gaussian_vfe": {"m": m_star, "s": s_star, "F": f_star,
                         "F_minus_surprise": f_star - surprise,
                         "KL_q_post": kl,
                         "F_at_prior": trace[0],
                         "grid_min": {"m": float(ms[i]),
                                      "s": float(ss[j]),
                                      "F": float(grid[i, j])}},
        "laplace": {"mode": v_map, "s": s_lap,
                    "surprise_estimate": lap_surprise,
                    "F_of_laplace_gaussian": f_lap},
        "predictive_coding": {
            "exercise2_phi_T5": float(flow[500, 1]),
            "network_phi_T5": float(net[500, 1]),
            "network_phi_T20": float(net[-1, 1]),
            "network_eps_p_T20": float(net[-1, 2]),
            "network_eps_u_T20": float(net[-1, 3]),
            "eps_p_at_map": float((v_map - V_PRIOR) / S_PRIOR),
            "eps_u_at_map": float((U_OBS - v_map**2) / S_LIK)},
        "temperature_dial": dial,
    }
    curves = dict(post=post, trace=np.array(trace), net=net[:801],
                  flow=flow[:801], m=m_star, s=s_star, s_lap=s_lap,
                  v_map=v_map, surprise=surprise, kl=kl)
    return res, curves


def figure_perception(c):
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.1))
    sel = (V > 0.0) & (V < 4.6)
    v = V[sel]
    lik = np.exp(-energy_lik(V))
    lik /= lik[V > 0].sum() * DV
    ax[0].plot(v, np.exp(-energy_prior(v)), color=GREY, ls="--",
               label="prior $p(v)$")
    ax[0].plot(v, lik[sel], color=GREY, ls=":",
               label="likelihood (normalized)")
    ax[0].plot(v, c["post"][sel], color="black", lw=1.8,
               label="exact posterior")
    ax[0].plot(v, np.exp(gauss_logpdf(v, c["m"], c["s"])), color=NAVY,
               label="Gaussian $q^\\star$ (min $F$)")
    ax[0].plot(v, np.exp(gauss_logpdf(v, c["v_map"], c["s_lap"])),
               color=RUST, ls="-.", label="Laplace")
    ax[0].axvline(c["v_map"], color=RUST, lw=0.6)
    ax[0].set_xlabel("size $v$")
    ax[0].set_title("(a) beliefs after observing $u=2$")
    ax[0].legend(frameon=False, loc="upper right")

    gap = c["trace"] - c["surprise"]
    ax[1].plot(np.arange(1, gap.size + 1), gap, color=NAVY,
               label="$F(q) + \\log p(u)$ along gradient descent")
    ax[1].axhline(c["kl"], color=RUST, ls="--",
                  label="$\\mathrm{KL}(q^\\star \\Vert p(v\\mid u))"
                        f" = {c['kl']:.4f}$")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("iteration")
    ax[1].set_ylabel("nats above the surprise")
    ax[1].set_title("(b) $F$ stays above the surprise")
    ax[1].legend(frameon=False)

    net, flow = c["net"], c["flow"]
    ax[2].plot(flow[:, 0], flow[:, 1], color=GREY, ls="--",
               label="$\\phi$, gradient flow")
    ax[2].plot(net[:, 0], net[:, 1], color=NAVY, label="$\\phi$, network")
    ax[2].plot(net[:, 0], net[:, 2], color=RUST, label="$\\varepsilon_p$")
    ax[2].plot(net[:, 0], net[:, 3], color=GREEN,
               label="$\\varepsilon_u$")
    ax[2].axhline(c["v_map"], color=GREY, lw=0.6, ls=":")
    ax[2].set_xlabel("time")
    ax[2].set_title("(c) predictive coding finds the mode")
    ax[2].legend(frameon=False, loc="right")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig-fep-perception.png"), dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Example 2: a T-maze with a cue, two arms and a dark room
# hidden context c in {L, R}; arms and the dark room are absorbing
# ---------------------------------------------------------------------
OUTCOMES = ("centre", "cue L", "cue R", "reward", "loss", "dark")
ACTIONS = ("centre", "cue", "left", "right", "dark")
ABSORBING = ("left", "right", "dark")
ARM_VALIDITY = 0.98
C_PREF = np.array([0.0, 0.0, 0.0, 3.0, -3.0, 0.0])
LOG_PREF = C_PREF - np.log(np.exp(C_PREF).sum())


def likelihood(action, cue_v, arm_v=ARM_VALIDITY):
    """A[o, c] = p(o | c, action) for the contexts c = (L, R)."""
    A = np.zeros((len(OUTCOMES), 2))
    if action == "centre":
        A[0] = 1.0
    elif action == "cue":
        A[1], A[2] = (cue_v, 1 - cue_v), (1 - cue_v, cue_v)
    elif action == "left":
        A[3], A[4] = (arm_v, 1 - arm_v), (1 - arm_v, arm_v)
    elif action == "right":
        A[3], A[4] = (1 - arm_v, arm_v), (arm_v, 1 - arm_v)
    else:
        A[5] = 1.0
    return A


def efe_terms(A, qc, log_pref=LOG_PREF):
    """G = risk + ambiguity = cross-entropy - information gain."""
    qo = A @ qc
    h_o = float(entropy(qo))
    ambiguity = float(qc @ entropy(A, axis=0))
    cross = float(-(qo @ log_pref))
    risk = cross - h_o
    return {"G": risk + ambiguity, "risk": risk, "ambiguity": ambiguity,
            "cross_entropy": cross, "outcome_entropy": h_o,
            "info_gain": h_o - ambiguity, "pragmatic": -cross}


def efe(action, qc, cue_v):
    return efe_terms(likelihood(action, cue_v), qc)["G"]


def two_step(a1, qc, cue_v):
    """Sophisticated two-step value in the high-precision limit.

    Returns G(a1) and E_o[min_a2 G(a2 | o)].
    """
    A1 = likelihood(a1, cue_v)
    g1 = efe(a1, qc, cue_v)
    nxt = (a1,) if a1 in ABSORBING else ACTIONS
    qo = A1 @ qc
    g2 = 0.0
    for o, p_o in enumerate(qo):
        if p_o < 1e-14:
            continue
        post = A1[o] * qc / p_o
        g2 += p_o * min(efe(a2, post, cue_v) for a2 in nxt)
    return g1, g2


def softmax(x):
    z = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def example_tmaze():
    qc = np.array([0.5, 0.5])
    out = {"setting": {"arm_validity": ARM_VALIDITY,
                       "C": C_PREF.tolist(),
                       "p_tilde": np.exp(LOG_PREF).tolist(),
                       "prior_context": qc.tolist()}}
    for cue_v in (1.0, 0.9):
        key = f"cue_validity_{cue_v}"
        terms = {a: efe_terms(likelihood(a, cue_v), qc) for a in ACTIONS}
        G = np.array([terms[a]["G"] for a in ACTIONS])
        gam = (0.5, 1.0, 2.0, 4.0, 16.0)
        pol = {str(g): dict(zip(ACTIONS, softmax(-g * G).tolist()))
               for g in gam}
        soph = {}
        for a1 in ("centre", "cue", "left", "dark"):
            g1, g2 = two_step(a1, qc, cue_v)
            soph[a1] = {"G1": g1, "E_G2": g2, "total": g1 + g2}
        classical = {}
        for a1 in ACTIONS:
            for a2 in ((a1,) if a1 in ABSORBING else ACTIONS):
                classical[f"{a1}->{a2}"] = (efe(a1, qc, cue_v)
                                            + efe(a2, qc, cue_v))
        best_cls = min(classical, key=classical.get)
        out[key] = {
            "terms": terms,
            "argmin_G": ACTIONS[int(np.argmin(G))],
            "argmin_outcome_entropy": [
                a for a in ACTIONS
                if terms[a]["outcome_entropy"] < 1e-12],
            "max_info_gain": max(ACTIONS,
                                 key=lambda a: terms[a]["info_gain"]),
            "policy_by_gamma": pol,
            "two_step_sophisticated": soph,
            "two_step_classical_best": [best_cls, classical[best_cls]],
            "two_step_classical": classical,
        }
    return out


def figure_tmaze(res):
    t1 = res["cue_validity_1.0"]["terms"]
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.2))
    x = np.arange(len(ACTIONS))
    ce = [t1[a]["cross_entropy"] for a in ACTIONS]
    ig = [-t1[a]["info_gain"] for a in ACTIONS]
    G = [t1[a]["G"] for a in ACTIONS]
    ax[0].bar(x - 0.27, ce, 0.27, color=GOLD,
              label="cross-entropy to $\\tilde p$")
    ax[0].bar(x, ig, 0.27, color=GREEN, label="$-$information gain")
    ax[0].bar(x + 0.27, G, 0.27, color=NAVY, label="$G$")
    ax[0].axhline(0, color="black", lw=0.6)
    ax[0].set_xticks(x, ACTIONS)
    ax[0].set_ylim(-1.1, 4.6)
    ax[0].set_ylabel("nats")
    ax[0].set_title("(a) one-step terms, cue validity 1")
    ax[0].legend(frameon=False, loc="upper center", ncol=3,
                 handlelength=1.2, columnspacing=0.8)

    gam = np.logspace(-1, 1.5, 300)
    Gv = np.array(G)
    pol = softmax(-gam[:, None] * Gv[None, :])
    shown = (("cue", NAVY, "-", "cue"),
             ("left", RUST, "-", "left (= right)"),
             ("centre", GREY, "--", "centre (= dark)"))
    for a, col, ls, lab in shown:
        k = ACTIONS.index(a)
        ax[1].plot(gam, pol[:, k], color=col, ls=ls, label=lab)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("precision $\\gamma = 1/T$")
    ax[1].set_ylabel("$q(a) \\propto e^{-\\gamma G(a)}$")
    ax[1].set_title("(b) the precision dial")
    ax[1].legend(frameon=False, loc="upper left")

    first = ("centre", "cue", "left", "dark")
    for k, cue_v in enumerate((1.0, 0.9)):
        r = res[f"cue_validity_{cue_v}"]
        one = [r["terms"][a]["G"] for a in first]
        two = [r["two_step_sophisticated"][a]["total"] for a in first]
        xx = np.arange(len(first)) + (k - 0.5) * 0.36
        ax[2].bar(xx, two, 0.34, color=(NAVY, RUST)[k], alpha=0.85,
                  label=f"two-step, cue validity {cue_v}")
        ax[2].plot(xx, one, "o", color="white", mec="black", ms=4.5,
                   label="one-step $G$" if k == 0 else None)
    ax[2].set_xticks(np.arange(len(first)), first)
    ax[2].set_ylim(0, 9.2)
    ax[2].set_xlabel("first move")
    ax[2].set_ylabel("nats")
    ax[2].set_title("(c) myopic versus sophisticated")
    ax[2].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig-fep-tmaze.png"), dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Example 4: learned preferences as a self-referential tilt
# K rooms, room 0 is the dark room, every room yields its own outcome
# without ambiguity, so G(a) = -log p~(o_a) and q(a) ~ p~(o_a)^gamma.
# The learned part p of the preference prior tracks q (Dirichlet counts
# with forgetting, at steady state); p~ = (1 - w) p + w u, where u is a
# fixed uniform reference component of weight w.
# ---------------------------------------------------------------------
def pref_policy(p, gamma, w):
    k = p.shape[-1]
    pt = (1 - w) * p + w / k
    q = pt ** np.asarray(gamma)[..., None]
    return q / q.sum(axis=-1, keepdims=True)


def pref_run(gammas, w, start, lam=0.2, steps=20_000):
    p = np.repeat(start[None, :], len(gammas), axis=0)
    for _ in range(steps):
        p = (1 - lam) * p + lam * pref_policy(p, gammas, w)
    return pref_policy(p, gammas, w)


def example_preferences():
    out = {"lambda_sweep": 0.2, "steps_sweep": 20_000}
    gammas = np.round(np.arange(0.5, 4.0001, 0.005), 4)
    sweeps = {}
    for K in (2, 3, 4):
        bal = np.full(K, 1.0 / K)
        bal[0] *= 1.001
        bal /= bal.sum()
        con = np.full(K, 0.03 / (K - 1))
        con[0] = 0.97
        for w in (0.0, 0.2, 0.5):
            qb = pref_run(gammas, w, bal).max(axis=1)
            qc = pref_run(gammas, w, con).max(axis=1)
            gc = 1.0 / (1.0 - w)
            both = gammas[(qc - qb) > 1e-3]
            above = gammas > gc + 0.02
            jump = float(qb[above][0] - 1.0 / K)
            sweeps[(K, w)] = (qb, qc)
            out[f"K{K}_w{w}"] = {
                "gamma_c_theory": gc,
                "balanced_breaks_at": float(
                    gammas[np.argmax(qb > 1.0 / K + 1e-3)]),
                "bistable_window": ([float(both.min()),
                                     float(both.max())]
                                    if both.size else None),
                "jump_just_above_gamma_c": jump,
            }

    # time courses at w = 0, K = 4, dark room with a 5 % head start
    K, lam, T = 4, 0.05, 400
    start = np.full(K, 1.0)
    start[0] = 1.05
    start /= start.sum()
    tc = {}
    for g in (0.8, 1.0, 1.25, 2.0):
        p, qs = start.copy(), []
        for _ in range(T):
            q = pref_policy(p, g, 0.0)
            qs.append(q[0])
            p = (1 - lam) * p + lam * q
        tc[g] = np.array(qs)
        out[f"timecourse_gamma{g}"] = {"q_dark_t0": float(qs[0]),
                                       "q_dark_final": float(qs[-1])}
    return out, gammas, sweeps, tc


def figure_preferences(gammas, sweeps, tc):
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.3))
    cols = {0.8: GREEN, 1.0: GREY, 1.25: GOLD, 2.0: RUST}
    for g, qs in tc.items():
        ax[0].plot(np.arange(qs.size), qs, color=cols[g],
                   label=f"$\\gamma = {g}$")
    ax[0].axhline(0.25, color="black", lw=0.5, ls=":")
    ax[0].set_xlabel("step $t$")
    ax[0].set_ylabel("$q_t(\\mathrm{dark})$")
    ax[0].set_title("(a) $K=4$ rooms, dark room 5% ahead, $w=0$")
    ax[0].legend(frameon=False, loc="center right")

    for w, col in ((0.0, NAVY), (0.2, GOLD), (0.5, RUST)):
        qb, qc = sweeps[(4, w)]
        ax[1].plot(gammas, qb, color=col, lw=1.6,
                   label=f"$w={w}$, balanced start")
        ax[1].plot(gammas, qc, color=col, lw=1.0, ls="--",
                   label=f"$w={w}$, committed start")
        ax[1].axvline(1 / (1 - w), color=col, lw=0.6, ls=":")
        diff = (qc - qb) > 1e-3
        if diff.any():
            ax[1].axvspan(gammas[diff].min(), gammas[diff].max(),
                          color=col, alpha=0.12)
    ax[1].set_xlabel("precision $\\gamma = 1/T$")
    ax[1].set_ylabel("resting $\\max_a q(a)$")
    ax[1].set_title("(b) resting states, $K=4$")
    ax[1].legend(frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig-fep-preferences.png"), dpi=220)
    plt.close(fig)


# ---------------------------------------------------------------------
# Example 3: learning, a Gaussian mixture fitted at a label temperature
# p(o, s) = pi_s N(o; mu_s, sd_s^2) is an exponential family in (o, s)
# with the statistics (e_s, e_s o, e_s o^2) and a uniform base measure,
# so the M-step is moment matching to the completed data p^(o) q(s | o).
# ---------------------------------------------------------------------
TRUE_GMM = (np.array([0.4, 0.6]), np.array([-1.0, 1.0]),
            np.array([0.7, 1.0]))
INIT_GMM = (np.array([0.5, 0.5]), np.array([-0.3, 0.2]),
            np.array([1.5, 1.5]))


def gmm_sample(n, seed):
    rng = np.random.default_rng(seed)
    pi, mu, sd = TRUE_GMM
    s = rng.choice(2, size=n, p=pi)
    return mu[s] + sd[s] * rng.standard_normal(n)


def gmm_log_joint(o, theta):
    """log p(o, s) for all s, shape (n, 2)."""
    pi, mu, sd = theta
    return (np.log(pi) - 0.5 * np.log(2 * np.pi * sd**2)
            - (o[:, None] - mu) ** 2 / (2 * sd**2))


def logsumexp(a, axis=-1):
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
            ).squeeze(axis)


def gmm_estep(o, theta, T):
    lj = gmm_log_joint(o, theta)
    if T == 0:
        q = np.zeros_like(lj)
        q[np.arange(o.size), lj.argmax(axis=1)] = 1.0
        return q
    return np.exp(lj / T - logsumexp(lj / T)[:, None])


def gmm_mstep(o, q):
    nk = q.sum(axis=0)
    mu = (q * o[:, None]).sum(axis=0) / nk
    var = (q * (o[:, None] - mu) ** 2).sum(axis=0) / nk
    return nk / o.size, mu, np.sqrt(var)


def gmm_joint_entropy(theta):
    """Differential entropy of p(o, s), relative to Lebesgue x counting."""
    pi, _, sd = theta
    return float(entropy(pi)
                 + pi @ (0.5 * np.log(2 * np.pi * np.e * sd**2)))


def gmm_free_energy(o, q, theta, T):
    """Average of E_q[-log p(o, s)] - T H(q) over the data."""
    lj = gmm_log_joint(o, theta)
    return float(np.mean(-(q * lj).sum(axis=1))
                 - T * np.mean(entropy(q, axis=1)))


def gmm_nll(o, theta):
    return float(-np.mean(logsumexp(gmm_log_joint(o, theta))))


def gmm_fit(o, T, iters=400):
    theta, trace, gap = INIT_GMM, [], 0.0
    for _ in range(iters):
        q = gmm_estep(o, theta, T)
        theta = gmm_mstep(o, q)
        f = gmm_free_energy(o, q, theta, T)
        h_world = gmm_joint_entropy(theta)
        h_cause = float(np.mean(entropy(q, axis=1)))
        gap = max(gap, abs(f - (h_world - T * h_cause)))
        trace.append((f, h_world, h_cause))
    return theta, np.array(trace), gap


def example_learning():
    o, o_test = gmm_sample(2000, 7), gmm_sample(200_000, 8)
    out = {"true": [x.tolist() for x in TRUE_GMM],
           "init": [x.tolist() for x in INIT_GMM],
           "n_train": int(o.size), "n_test": int(o_test.size),
           "nll_test_true": gmm_nll(o_test, TRUE_GMM)}
    traces, dial = {}, []
    for T in np.round(np.linspace(0.0, 1.0, 21), 2):
        theta, trace, gap = gmm_fit(o, float(T))
        f = trace[:, 0]
        row = {"T": float(T), "pi": theta[0].tolist(),
               "mu": theta[1].tolist(), "sd": theta[2].tolist(),
               "free_energy": float(f[-1]),
               "H_world": float(trace[-1, 1]),
               "H_cause": float(trace[-1, 2]),
               "nll_train": gmm_nll(o, theta),
               "nll_test": gmm_nll(o_test, theta),
               "max_identity_gap": gap,
               "max_increase": float(np.max(np.diff(f), initial=0.0))}
        dial.append(row)
        traces[float(T)] = (trace, theta)
    out["dial"] = dial
    em = next(r for r in dial if r["T"] == 1.0)
    out["em_free_energy_minus_nll"] = em["free_energy"] - em["nll_train"]
    return out, o, traces


def figure_learning(res, o, traces):
    fig, ax = plt.subplots(1, 3, figsize=(10.4, 3.3))
    trace = traces[1.0][0][:150]
    it = np.arange(1, trace.shape[0] + 1)
    ax[0].plot(it, trace[:, 1], color=NAVY,
               label="$H(p_\\theta(O,S))$, world model")
    ax[0].plot(it, trace[:, 2], color=GREEN,
               label="$\\bar H(q)$, inferred causes")
    ax[0].plot(it, trace[:, 1] - trace[:, 2], color="black",
               label="difference")
    ax[0].plot(it[::6], trace[::6, 0], "o", color=RUST, ms=3.0,
               label="average $F$, computed directly")
    ax[0].set_ylim(0.0, 2.75)
    ax[0].set_xlabel("EM iteration ($T = 1$)")
    ax[0].set_ylabel("nats")
    ax[0].set_title("(a) $F$ = model entropy $-$ cause entropy")
    ax[0].legend(frameon=False, loc="center right",
                 bbox_to_anchor=(1.0, 0.36))

    d = res["dial"]
    T = np.array([r["T"] for r in d])
    ax[1].plot(T, [r["H_world"] for r in d], color=NAVY,
               label="$H(p_\\theta(O,S))$")
    ax[1].plot(T, [r["H_cause"] for r in d], color=GREEN,
               label="$\\bar H(q)$")
    ax[1].set_ylim(0.0, 2.4)
    ax[1].set_xlabel("label temperature $T$")
    ax[1].set_ylabel("nats")
    ax[1].set_title("(b) the dial: $T=1$ is EM")
    excess = 1e3 * (np.array([r["nll_test"] for r in d])
                    - res["nll_test_true"])
    tw = ax[1].twinx()
    tw.plot(T, excess, color=RUST, ls="--",
            label="held-out excess $-\\log p_\\theta(o)$, right")
    tw.set_ylim(0.0, 24.0)
    tw.set_ylabel("millinats above the truth", color=RUST)
    tw.tick_params(axis="y", colors=RUST)
    h1, l1 = ax[1].get_legend_handles_labels()
    h2, l2 = tw.get_legend_handles_labels()
    ax[1].legend(h1 + h2, l1 + l2, frameon=False, loc="center left")

    x = np.linspace(-4.0, 4.5, 600)
    dens = np.exp(logsumexp(gmm_log_joint(x, TRUE_GMM)))
    ax[2].hist(o, bins=60, density=True, color="#dddddd")
    ax[2].plot(x, dens, color="black", lw=1.6, label="truth")
    for t, col, ls in ((1.0, NAVY, "-"), (0.5, GOLD, "--"),
                       (0.0, RUST, "-.")):
        th = traces[t][1]
        ax[2].plot(x, np.exp(logsumexp(gmm_log_joint(x, th))),
                   color=col, ls=ls, label=f"fit at $T = {t:g}$")
    ax[2].set_ylim(0.0, 0.41)
    ax[2].set_xlabel("observation $o$")
    ax[2].set_title("(c) fitted densities")
    ax[2].legend(frameon=False, loc="upper center", ncol=2,
                 handlelength=1.6, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "fig-fep-learning.png"), dpi=220)
    plt.close(fig)


def main():
    perc, curves = example_perception()
    figure_perception(curves)
    tmaze = example_tmaze()
    figure_tmaze(tmaze)
    learn, o, traces = example_learning()
    figure_learning(learn, o, traces)
    prefs, gammas, sweeps, tc = example_preferences()
    figure_preferences(gammas, sweeps, tc)
    res = {"example1_perception": perc, "example2_tmaze": tmaze,
           "example3_learning": learn, "example4_preferences": prefs}
    with open(os.path.join(HERE, "fep_examples_results.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=1)[:200])


if __name__ == "__main__":
    main()
