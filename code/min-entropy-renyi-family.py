"""Checks for the "one family, indexed by an order" passage of the
minimum-entropy chapter.

Verifies, in order:
  1.  the power-reward family: E[r_alpha] is monotone in H_alpha, with
      alpha -> 1 giving Shannon and alpha = 2 giving EMPO's reward up to a
      constant that the group baseline removes;
  2.  the endpoints of the entropy band contribute an identically zero update;
  3.  Shannon has no unbiased estimator while collision probability does;
  4.  near uniformity, (log K - H_alpha) / alpha is the same for every order;
  5.  Shannon and collision entropy can rank two groups in opposite orders;
  6.  a block-diagonal similarity kernel reproduces both exactly.
"""

import numpy as np
from numpy.linalg import eigvalsh

rng = np.random.default_rng(3)


def H(p, a):
    p = np.asarray(p, float)
    p = p[p > 0]
    if abs(a - 1) < 1e-12:
        return -(p * np.log(p)).sum()
    return np.log((p**a).sum()) / (1 - a)


# 1 -- the power-reward family
p = rng.dirichlet(np.ones(6) * 0.7)
print("1.  reward r_a(c) = (p_c^{a-1} - 1)/(a - 1)")
for a in (1.0, 1.5, 2.0, 3.0):
    r = np.log(p) if a == 1.0 else (p ** (a - 1) - 1) / (a - 1)
    target = -H(p, 1) if a == 1.0 else ((p**a).sum() - 1) / (a - 1)
    print(f"      a={a:<4} E[r]={(p * r).sum():+.8f}  closed form={target:+.8f}  "
          f"H_a={H(p, a):.6f}  match={np.isclose((p * r).sum(), target)}")
print("      a=2 reward == EMPO reward minus 1 :", np.allclose((p**1 - 1) / 1, p - 1))
print("      E[p_c] == sum p^2 == exp(-H_2)    :",
      np.isclose((p * p).sum(), np.exp(-H(p, 2))))

# 2 -- both endpoints of the band give a zero update
print("\n2.  GRPO advantage at the endpoints of the band (G = 16)")
G = 16
for name, sizes in (("all agree,   H = 0", [16]),
                    ("none agree,  H = log G", [1] * 16),
                    ("interior,    (8,5,3)", [8, 5, 3])):
    q = np.array(sizes) / G
    r = np.concatenate([[s / G] * s for s in sizes])
    A = (r - r.mean()) / (r.std() + 1e-6)
    print(f"      {name:<24} H={H(q, 1):.4f}  max|advantage|={np.abs(A).max():.3e}")

# 3 -- estimability from a finite group
print("\n3.  finite-sample estimability at G = 8, five clusters")
G, T = 8, 400_000
p5 = np.array([0.45, 0.25, 0.15, 0.10, 0.05])
cnt = rng.multinomial(G, p5, size=T)
ph = cnt / G
plug_coll = (ph**2).sum(1)
unbiased = (cnt * (cnt - 1)).sum(1) / (G * (G - 1))
with np.errstate(divide="ignore", invalid="ignore"):
    lg = np.where(ph > 0, np.log(np.where(ph > 0, ph, 1)), 0.0)
plug_H1 = -(ph * lg).sum(1)
true_coll, true_H1 = (p5**2).sum(), H(p5, 1)
print(f"      collision, plug-in : {plug_coll.mean():.6f}  bias {plug_coll.mean()-true_coll:+.6f}")
print(f"      collision, U-stat  : {unbiased.mean():.6f}  bias {unbiased.mean()-true_coll:+.6f}"
      f"   <- unbiased (degree-2 polynomial)")
print(f"      Shannon,   plug-in : {plug_H1.mean():.6f}  bias {plug_H1.mean()-true_H1:+.6f}"
      f"   = {abs(plug_H1.mean()-true_H1)/true_H1*100:.0f}% of H_1 = {true_H1:.4f}")
print("      Shannon has no unbiased estimator at any G (Paninski 2003, Prop. 8)")

# 4 -- near uniformity all orders coincide up to the factor alpha
print("\n4.  (log K - H_a)/a near the uniform distribution, K = 8")
K = 8
for scale in (0.30, 0.10, 0.03, 0.01):
    eps = rng.normal(size=K)
    eps -= eps.mean()
    eps *= scale
    q = (1 + eps) / K
    q /= q.sum()
    cols = "  ".join(f"a={a:.0f}: {(np.log(K)-H(q,a))/a:.7f}" for a in (1.0, 2.0, 3.0))
    print(f"      spread={scale:<5} {cols}   [chi2/2K = {((q-1/K)**2).sum()*K/2:.7f}]")

# 5 -- the two orders can rank two groups oppositely
print("\n5.  an ordering reversal")
A_ = np.array([0.5] + [0.5 / 9] * 9)
B_ = np.array([0.25] * 4)
print(f"      A = (0.5, 0.056 x9) : H_1={H(A_,1):.4f}  H_2={H(A_,2):.4f}")
print(f"      B = uniform over 4  : H_1={H(B_,1):.4f}  H_2={H(B_,2):.4f}")
print(f"      Shannon ranks A more diverse : {H(A_,1) > H(B_,1)};"
      f"  collision ranks A less diverse : {H(A_,2) < H(B_,2)}")

# 6 -- the kernel route reproduces both exactly
print("\n6.  block-diagonal similarity kernel")
sizes = [5, 3, 2, 1, 1]
n = sum(sizes)
pc = np.array(sizes) / n
Km = np.zeros((n, n))
i = 0
for s in sizes:
    Km[i:i + s, i:i + s] = 1.0
    i += s
rho = Km / np.trace(Km)
ev = np.sort(eigvalsh(rho))[::-1]
ev = ev[ev > 1e-12]
print("      eigenvalues of rho == cluster masses :",
      np.allclose(np.sort(ev)[::-1], np.sort(pc)[::-1]))
print(f"      von Neumann S(rho) = {-(ev*np.log(ev)).sum():.6f}   H_1(p) = {H(pc,1):.6f}")
print(f"      -log Tr(rho^2)     = {-np.log(np.trace(rho@rho)):.6f}   H_2(p) = {H(pc,2):.6f}")
print("      Tr(rho^2) == ||rho||_F^2 (no eigendecomposition) :",
      np.isclose(np.trace(rho @ rho), (rho**2).sum()))
