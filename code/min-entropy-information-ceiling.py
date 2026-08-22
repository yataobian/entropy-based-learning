"""Checks for the information-ceiling section of the minimum-entropy chapter.

A world small enough to enumerate exactly:

    a*      ~ Uniform{0, 1}                    the true answer
    theta0  = a* with probability 0.8          what pre-training learned
    base policy: each sampled answer equals theta0 with probability 0.7
    training: draw G = 5 i.i.d. answers (randomness R, independent of a*)
              and sharpen towards their majority, theta_T = maj(R)

Verifies, by exact enumeration of the joint distribution (no Monte Carlo):

  1.  I(theta_T ; a* | theta0) = 0 exactly       -- training adds nothing;
  2.  I(theta_T ; a*) <= I(theta0 ; a*)          -- the ceiling inequality,
      with strict loss here, since majority voting is a noisy compression;
  3.  a verifier-guided update (reward touches a*) breaks the bound;
  4.  the pass@k signature: sharpening wins at k = 1 and loses at large k.
"""

from math import comb, log2

import numpy as np

G = 5


def majority_prob(p1):
    """P(majority of G i.i.d. Bernoulli(p1) samples equals 1)."""
    return sum(comb(G, k) * p1**k * (1 - p1) ** (G - k) for k in range(G // 2 + 1, G + 1))


def joint(supervised=False):
    """Exact joint distribution p(a*, theta0, theta_T)."""
    p = {}
    for a in (0, 1):
        for t0 in (0, 1):
            pt0 = 0.8 if t0 == a else 0.2
            for tT in (0, 1):
                if supervised:
                    ptT = 0.9 if tT == a else 0.1  # verifier pulls towards a*
                else:
                    pm1 = majority_prob(0.7 if t0 == 1 else 0.3)
                    ptT = pm1 if tT == 1 else 1 - pm1
                p[(a, t0, tT)] = 0.5 * pt0 * ptT
    return p


def conditional_mi(p):
    """I(theta_T ; a* | theta0) in bits."""
    out = 0.0
    for t0 in (0, 1):
        pt0 = sum(v for (a, s, t), v in p.items() if s == t0)
        for a in (0, 1):
            for tT in (0, 1):
                pj = p[(a, t0, tT)]
                pa = sum(p[(a, t0, t)] for t in (0, 1))
                pt = sum(p[(x, t0, tT)] for x in (0, 1))
                if pj > 0:
                    out += pj * log2(pj * pt0 / (pa * pt))
    return out


def marginal_mi(p):
    """I(theta_T ; a*) in bits."""
    out = 0.0
    for a in (0, 1):
        for tT in (0, 1):
            pj = sum(p[(a, t0, tT)] for t0 in (0, 1))
            pt = sum(p[(x, t0, tT)] for x in (0, 1) for t0 in (0, 1))
            if pj > 0:
                out += pj * log2(pj / (0.5 * pt))
    return out


budget = 1 + 0.8 * log2(0.8) + 0.2 * log2(0.2)  # I(theta0 ; a*)
pu, ps = joint(False), joint(True)

print("unsupervised (theta_T = f(theta0, R), R independent of a*):")
print(f"  I(theta_T ; a* | theta0) = {conditional_mi(pu):.2e} bits   (exactly zero)")
print(f"  I(theta_T ; a*)          = {marginal_mi(pu):.4f} bits   (all inherited)")
print(f"  I(theta0  ; a*)          = {budget:.4f} bits   (the ceiling)")
print(f"  ceiling inequality holds : {marginal_mi(pu) <= budget + 1e-12}")
print("verifier-guided update (reward touches a*):")
print(f"  I(theta_T ; a* | theta0) = {conditional_mi(ps):.4f} bits   (> 0: chain broken)")

# --- the pass@k signature on a two-question toy benchmark -------------------
# Q1: the base majority is right  (correct cluster mass 0.40 of 0.40/0.35/0.25)
# Q2: the base majority is wrong  (correct cluster mass 0.20 of 0.50/0.30/0.20)
# Sharpening moves each question to its majority; the rare correct cluster on
# Q2 receives the group's lowest reward and a negative advantage, and is
# actively suppressed.
questions = {"Q1": {"base": 0.40, "trained": 0.95},
             "Q2": {"base": 0.20, "trained": 0.005}}
print("\npass@k, averaged over the two questions:")
print(f"  {'k':>4}  {'base':>7}  {'sharpened':>9}")
for k in (1, 2, 4, 8, 16, 64, 256):
    b = np.mean([1 - (1 - v["base"]) ** k for v in questions.values()])
    t = np.mean([1 - (1 - v["trained"]) ** k for v in questions.values()])
    print(f"  {k:>4}  {b:7.3f}  {t:9.3f}")
print("  sharpening wins at k = 1 and loses at large k: which answer comes out")
print("  first has changed; which answers are reachable has not.")
