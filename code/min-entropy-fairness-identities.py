"""Checks for the claims in the "one mechanism, six notations" passage of the
minimum-entropy chapter.

Verifies, in order:
  1.  -D_KL(pbar || Unif) = H(pbar) - log K.
  2.  distribution alignment == logit debiasing == one Sinkhorn row-normalisation.
  3.  the analytic gradient of H(pbar) w.r.t. the logits.
  4.  DINO centring subtracts the log *geometric* mean, debiasing the log
      *arithmetic* mean, and the two differ by a class-independent constant.
  5.  one debiasing step does not make the marginal uniform; iterating does.
"""

import numpy as np

rng = np.random.default_rng(0)


def softmax(z, axis=-1):
    e = np.exp(z - z.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


N, K = 7, 4
Z = rng.normal(size=(N, K)) * 1.3          # logits, rows = examples
P = softmax(Z)                             # p_c(x_n)
pbar = P.mean(axis=0)                      # class marginal

# 1 -- the two loss-side notations differ by a constant
H = -(pbar * np.log(pbar)).sum()
KL = (pbar * np.log(pbar * K)).sum()
print("1.  -KL(pbar||Unif) == H(pbar) - log K :", np.isclose(-KL, H - np.log(K)))

# 2 -- three notations, one offset gamma_c = log pbar_c
Q0 = P.T / N                               # K x N; columns sum to 1/N
assert np.allclose(Q0.sum(axis=0), 1 / N)
assert np.allclose(Q0.sum(axis=1), pbar)   # row sums are exactly the marginal
Q1 = Q0 * ((1 / K) / Q0.sum(axis=1))[:, None]        # one Sinkhorn row-scaling
assert np.allclose(Q1.sum(axis=1), 1 / K)

sinkhorn_step = softmax(np.log(Q1.T))                # renormalise per example
debias = softmax(Z - np.log(pbar))                   # logit -= log pbar
align = P / pbar
align /= align.sum(axis=1, keepdims=True)            # q~ prop. to q / pbar
print("2.  one Sinkhorn row-step == logit debiasing :",
      np.allclose(sinkhorn_step, debias))
print("    logit debiasing == distribution alignment:",
      np.allclose(debias, align))

# 3 -- gradient of the marginal-entropy bonus, against finite differences
eps = 1e-6
numeric = np.zeros_like(Z)
for n in range(N):
    for j in range(K):
        Zp, Zm = Z.copy(), Z.copy()
        Zp[n, j] += eps
        Zm[n, j] -= eps
        hp, hm = softmax(Zp).mean(0), softmax(Zm).mean(0)
        numeric[n, j] = (-(hp * np.log(hp)).sum() + (hm * np.log(hm)).sum()) / (2 * eps)

logpbar = np.log(pbar)
analytic = -(P / N) * (logpbar[None, :] - (P * logpbar).sum(1, keepdims=True))
print("3.  analytic gradient == finite differences  :",
      np.allclose(numeric, analytic, atol=1e-7))

# 4 -- centring is the geometric mean, debiasing the arithmetic mean
log_geo = np.log(P).mean(0)                # = mean logit - mean log-partition
gap = Z.mean(0) - log_geo
print("4.  mean logit - log geo mean is constant in c:",
      np.allclose(gap, gap[0]))
print("    log AM >= log GM (Jensen)                 :",
      np.all(logpbar >= log_geo - 1e-12))

# 5 -- one debiasing step is not enough; Sinkhorn iterates for a reason.
# Independently seeded so these figures do not depend on the draws above.
print("5.  TV(marginal, uniform) under repeated debiasing")
for n_, k_, s_ in [(64, 4, 1.5), (256, 10, 2.0), (512, 20, 2.5)]:
    Zi = np.random.default_rng(1).normal(size=(n_, k_)) * s_
    Qi = softmax(Zi)
    tv = lambda p: 0.5 * np.abs(p - 1 / k_).sum()
    trace = [tv(Qi.mean(0))]
    for _ in range(30):
        Qi = softmax(np.log(Qi) - np.log(Qi.mean(0)))   # row scale, then softmax
        trace.append(tv(Qi.mean(0)))
    shown = " -> ".join(f"{v:.4f}" for v in trace[:4])
    print(f"    N={n_:>4} K={k_:>3}: {shown}"
          f"   [12 passes: {trace[12]:.1e}] [30: {trace[30]:.1e}]"
          f"   mean ratio {np.mean(np.array(trace[:4]) / np.array(trace[1:5])):.2f}x")
