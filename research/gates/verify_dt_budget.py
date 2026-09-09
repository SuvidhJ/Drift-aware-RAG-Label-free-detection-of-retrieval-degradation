import numpy as np, ot, sys
sys.path.insert(0, r"E:\drift-aware-rag-research")
rng = np.random.default_rng(7); d=1024
def sphere(n, mu, spread):
    X = mu + spread/np.sqrt(d)*rng.standard_normal((n,d))
    return (X/np.linalg.norm(X,axis=1,keepdims=True)).astype(np.float32)
mu = rng.standard_normal(d); mu/=np.linalg.norm(mu)

print("### A. repo 'CI' collapses with n_projections; real uncertainty does not ###")
X,Y = sphere(1000,mu,0.5), sphere(1000,mu,0.5)
r=np.random.default_rng(1)
for P in (100, 500, 2000):
    boot=[ot.sliced_wasserstein_distance(X,Y,n_projections=P,seed=int(r.integers(0,2**31))) for _ in range(60)]
    lo,hi=np.percentile(boot,[2.5,97.5])
    redraw=[ot.sliced_wasserstein_distance(sphere(1000,mu,0.5),sphere(1000,mu,0.5),
             n_projections=P,seed=int(r.integers(0,2**31))) for _ in range(25)]
    rlo,rhi=np.percentile(redraw,[2.5,97.5])
    print(f"  P={P:5d}  repo CI width={hi-lo:.3e}   true sampling width={rhi-rlo:.3e}   understated {(rhi-rlo)/(hi-lo):5.1f}x")

print("\n### B. D(t) budget: which term actually moves? (tau=0.15, a=b=1, g=0.5) ###")
for gap in (0.0, 0.3, 0.6, 1.0):
    muC = mu.copy()
    muQ = mu + gap*rng.standard_normal(d); muQ/=np.linalg.norm(muQ)
    C1,C2 = sphere(3000,muC,0.5), sphere(3000,muC,0.5)
    Q1,Q2 = sphere(1500,muQ,0.6), sphere(1500,muQ,0.6)
    w1q = ot.sliced_wasserstein_distance(Q1,Q2,n_projections=500,seed=3)
    w1c = ot.sliced_wasserstein_distance(C1,C2,n_projections=500,seed=4)
    cg  = float(np.linalg.norm(Q1.mean(0)-C1.mean(0)))
    D   = w1q + w1c + 0.5*cg
    print(f"  Q-C separation={gap:.1f}: W1_Q={w1q:.5f} W1_C={w1c:.5f} ||mu_Q-mu_C||={cg:.4f}"
          f" -> D={D:.4f} [{'ALARM' if D>0.15 else 'silent'}] centroid={0.5*cg/D:5.1%} of D")
print("  --> drift terms are <2% of D. D(t) alarms iff ||mu_Q-mu_C||>0.30: a property of the")
print("      two TEXT GENRES (questions vs product descriptions), fixed at t=0, not of drift.")

print("\n### C. estimate_lipschitz: the denominator is ~1/document_length ###")
from embeddings.encoder import _char_edit_distance_norm
r2=np.random.default_rng(0)
w=lambda k: " ".join("".join(r2.choice(list("abcdefghij"),size=6)) for _ in range(k))
for k in (5, 20, 80, 300):
    t=w(k); ws=t.split(); ws[0]="zzzzz"; p=" ".join(ws)
    e=_char_edit_distance_norm(t,p)
    print(f"  {k:4d} words ({len(t):5d} chars): norm.edit={e:.5f} -> ratio multiplier = {1/e:8.1f}")
print("  --> L_hat = max_i ||df_i|| / editdist_i is won by the LONGEST doc; it measures")
print("      document length. Plus: max-of-500 + bootstrap CI is inconsistent for extrema.")
