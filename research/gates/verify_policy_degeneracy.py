"""Verification 1: is the shipped policy engine a constant function?"""
import numpy as np, itertools

COST = {"a1_no_op":1.0, "a2_rerank":4.0, "a3_reindex":22.0, "a4_finetune":110.0, "a5_escalate":1e9}

def recovery(a, d, al, be):
    return {"a1_no_op":0.0, "a2_rerank":0.4*al*d, "a3_reindex":0.8*be*d,
            "a4_finetune":0.9*(al+be)*d, "a5_escalate":1.0*d}[a]

def policy(d, al, be, lam):
    loss = d*(al+be)                       # engine.py:30
    objs = {a: COST[a] + lam*(loss - recovery(a,d,al,be)) for a in COST}
    return min(objs, key=objs.get)

# alpha+beta == 1 always (decomposition.py:74-76) -> sweep alpha, D over the
# *achievable* range. research/results/original_dt_audit.json: max D ever = 0.1394
print("=== Sweep over the ACHIEVABLE operating range (D<=0.14), lam=1.0 default ===")
acts=set()
for d in np.linspace(0, 0.14, 60):
    for al in np.linspace(0,1,51):
        acts.add(policy(d, al, 1-al, 1.0))
print("  distinct actions ever selected:", acts)

print("\n=== How large must D or lam be before ANYTHING but no-op is chosen? ===")
# rerank beats no-op iff 4 + lam*(d - 0.4*al*d) < 1 + lam*d  ->  3 < lam*0.4*al*d
for al in (0.5, 1.0):
    d_needed = 3/(0.4*al*1.0)
    lam_needed = 3/(0.4*al*0.14)
    print(f"  alpha={al}: at lam=1 need D > {d_needed:.1f}   |   at D=0.14 (max observed) need lam > {lam_needed:.0f}")

print("\n=== Even generous: sweep D up to 2.0 (unreachable: W1 on unit sphere) ===")
rows=[]
for lam in (1,10,100,1000):
    a=set(policy(d,0.5,0.5,lam) for d in np.linspace(0,2,200))
    rows.append((lam,sorted(a)))
for lam,a in rows: print(f"  lam={lam:5d} -> {a}")

print("\n=== Scale-invariance of the regime classifier (decomposition.py) ===")
def regime(w1q, w1c, d, tau=0.15, ta=0.4, tb=0.4):
    if d < tau: return "stable"
    tot=w1q+w1c+1e-10; a,b=w1q/tot, w1c/tot
    if a>=ta and b>=tb: return "joint_drift"
    if a>=ta: return "query_drift"
    if b>=tb: return "corpus_drift"
    return "stable"
for scale in (1.0, 0.1, 0.01):
    print(f"  w1q={0.9*scale:<8.4g} w1c={0.9*scale:<8.4g} (d forced above tau) -> {regime(0.9*scale,0.9*scale,1.0)}")
print("  -> identical regime at every magnitude: alpha/beta are ratios, magnitude-blind.")

print("\n=== Is the 4th quadrant reachable? alpha+beta==1 exactly ===")
seen={}
for al in np.linspace(0,1,10001):
    r = regime(al, 1-al, 1.0)
    seen.setdefault(r, [al,al])
    seen[r][1]=al
for r,(lo,hi) in seen.items(): print(f"  {r:14s} alpha in [{lo:.3f}, {hi:.3f}]")
print("  -> the '2x2 plane' is a 1-D partition of alpha; final `return STABLE` is dead code.")
