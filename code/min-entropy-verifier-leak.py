"""Can a learned clustering metric leak supervision past the information ceiling?

Extends the toy world of code/min-entropy-information-ceiling.py with a
verifier belief V, correlated with the truth a*, and a merging rule whose
behaviour may or may not depend on that belief.  Everything is computed by
exact enumeration -- no Monte Carlo.

World:
  a*     ~ Uniform{0, 1}                          the true answer
  theta0 = a*      with probability 0.8           what pre-training learned
  V      = a*      with probability q_V           the verifier's belief
             (q_V = 0.5: uninformed, regex-like;  q_V = 0.9: can often solve q)
  base policy: each of G = 5 samples equals theta0 with probability 0.7
  clustering: samples equal to V always merge into one cluster; samples != V
              each join the main non-V cluster with probability m, else become
              singletons  (m = 1: merging is blind to correctness)
  theta_T = value of the largest cluster (ties broken by fair coin)

Verifies:
  1. I(theta_T ; a* | theta0, V) = 0 in every configuration -- conditioning on
     the frozen metric always screens off the truth (the corrected theorem);
  2. leakage requires knowledge AND a behavioural channel: with q_V = 0.5
     (no knowledge) or m = 1 (no channel), I(theta_T ; a* | theta0) = 0 and the
     one-term ceiling I(theta_T ; a*) <= I(theta0 ; a*) holds;
  3. with both present (q_V = 0.9, m = 0.3) the run adds 0.114 bits beyond
     pre-training and the one-term ceiling is genuinely exceeded, while the
     two-term ceiling I(theta0 ; a*) + I(V ; a* | theta0) still holds.
"""

from collections import defaultdict
from math import comb, log2

G, P_PRETRAIN, P_SAMPLE = 5, 0.8, 0.7


def binom(n, k, p):
    return comb(n, k) * p**k * (1 - p) ** (n - k)


def joint(q_v, m):
    """Exact joint distribution p(a*, theta0, V, theta_T)."""
    p = {}
    for a in (0, 1):
        for t0 in (0, 1):
            pt0 = P_PRETRAIN if t0 == a else 1 - P_PRETRAIN
            for v in (0, 1):
                pv = q_v if v == a else 1 - q_v
                ptT = {0: 0.0, 1: 0.0}
                p1 = P_SAMPLE if t0 == 1 else 1 - P_SAMPLE
                for n1 in range(G + 1):
                    pn1 = binom(G, n1, p1)
                    n_v = n1 if v == 1 else G - n1
                    n_other = G - n_v
                    for b in range(n_other + 1):  # size of the main non-V cluster
                        pb = binom(n_other, b, m)
                        largest_other = b if b >= 1 else (1 if n_other >= 1 else 0)
                        if n_v > largest_other:
                            w = {v: 1.0, 1 - v: 0.0}
                        elif n_v < largest_other:
                            w = {v: 0.0, 1 - v: 1.0}
                        else:
                            w = {v: 0.5, 1 - v: 0.5}
                        for tT in (0, 1):
                            ptT[tT] += pn1 * pb * w[tT]
                for tT in (0, 1):
                    p[(a, t0, v, tT)] = 0.5 * pt0 * pv * ptT[tT]
    assert abs(sum(p.values()) - 1) < 1e-12
    return p


def mutual_information(p, xs, ys, zs):
    """I(X ; Y | Z) in bits; xs, ys, zs index the key tuple (a, t0, V, tT)."""

    def marginal(idx):
        d = defaultdict(float)
        for key, v in p.items():
            d[tuple(key[i] for i in idx)] += v
        return d

    pxyz, pxz, pyz, pz = (marginal(i) for i in (xs + ys + zs, xs + zs, ys + zs, zs))
    out = 0.0
    for key, v in p.items():
        if v <= 0:
            continue
        x = tuple(key[i] for i in xs)
        y = tuple(key[i] for i in ys)
        z = tuple(key[i] for i in zs)
        out += v * log2(pxyz[x + y + z] * (pz[z] if zs else 1) / (pxz[x + z] * pyz[y + z]))
    return out


A, T0, V, TT = [0], [1], [2], [3]

print(f"{'configuration':<44} {'I(tT;a*|t0,V)':>13} {'I(tT;a*|t0)':>11} "
      f"{'I(tT;a*)':>9} {'1-term':>7} {'2-term':>7}")
for q_v, m, label in [
    (0.5, 0.3, "uninformed metric (regex-like)"),
    (0.9, 1.0, "informed, correctness-blind verdicts"),
    (0.9, 0.7, "informed, mild behavioural channel"),
    (0.9, 0.3, "informed, verdicts follow the belief"),
]:
    p = joint(q_v, m)
    i_cond_v = mutual_information(p, TT, A, T0 + V)   # corrected theorem: must be 0
    i_cond = mutual_information(p, TT, A, T0)         # one-term theorem's LHS
    i_t = mutual_information(p, TT, A, [])
    i_0 = mutual_information(p, T0, A, [])
    i_v = mutual_information(p, V, A, T0)             # the metric term
    one = "holds" if i_t <= i_0 + 1e-12 else "BROKEN"
    two = "holds" if i_t <= i_0 + i_v + 1e-12 else "BROKEN"
    print(f"{label:<44} {i_cond_v:>13.1e} {i_cond:>11.4f} {i_t:>9.4f} {one:>7} {two:>7}")

p = joint(0.9, 0.3)
print(f"\npre-training budget I(theta0 ; a*)          = "
      f"{mutual_information(p, T0, A, []):.4f} bits")
print(f"metric term         I(V ; a* | theta0)      = "
      f"{mutual_information(p, V, A, T0):.4f} bits")
print("leakage needs knowledge (q_V > 0.5) AND a channel (m < 1); either alone leaks nothing.")
