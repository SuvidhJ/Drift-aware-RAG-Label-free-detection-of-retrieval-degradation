"""
E2 - THRESHOLD TRANSFER UNDER QUERY-VOLUME SHIFT.  The paper gate.

The repo's own bake-off says a plain sliced-W1 beats the calibrated detector on
AUROC (0.9995 vs 0.9668).  The defence is that raw distances are volume-
confounded, so their THRESHOLD does not transfer, and AUROC - which is
threshold-free - cannot see that.  Nobody has tested it.  This does.

Protocol
  1. calibrate each detector's threshold to a 5% false-alarm rate on stable
     windows at query volume A  (calibration split only)
  2. freeze the threshold
  3. deploy on HELD-OUT stable windows at volume A  -> in-domain FAR (sanity)
  4. deploy on stable windows at volume B           -> TRANSFER FAR
  5. deploy on drifted windows at volume B          -> TRANSFER POWER

A detector whose score depends on sample size will either go silent (FAR and
power collapse to 0) or alarm on everything, depending on the direction of the
volume shift.  Both are operational failures a threshold-free metric hides.
"""
import numpy as np, time, pandas as pd

D = 256
def nrm(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)

def sample(mu, n, rng, spread=0.55):
    return nrm(mu + (spread/np.sqrt(D))*rng.standard_normal((n, D)))

def _w1(px,py,M=200):
    g=(np.arange(M)+0.5)/M
    return np.abs(np.quantile(px,g,axis=0)-np.quantile(py,g,axis=0)).mean(axis=0)

def sliced(A,B,rng,n_proj=64):
    th=nrm(rng.standard_normal((n_proj,D))); w=_w1(A@th.T,B@th.T)
    return float(w.mean()), float(w.max())

def sqd(A,B):
    aa=np.einsum('ij,ij->i',A,A)[:,None]; bb=np.einsum('ij,ij->i',B,B)[None,:]
    return np.maximum(aa+bb-2*A@B.T,0.0)

def mmd2(A,B,sigma):
    g=1/(2*sigma*sigma)
    Kxx=np.exp(-g*sqd(A,A)); np.fill_diagonal(Kxx,0)
    Kyy=np.exp(-g*sqd(B,B)); np.fill_diagonal(Kyy,0)
    n,m=len(A),len(B)
    return Kxx.sum()/(n*(n-1))+Kyy.sum()/(m*(m-1))-2*np.exp(-g*sqd(A,B)).mean()

def perm_z(A,B,rng,Bp=40,n_proj=48,cap=900):
    """calibrated: standardise max-slice against a permutation null at THESE sizes"""
    if len(A)>cap: A=A[rng.choice(len(A),cap,replace=False)]
    if len(B)>cap: B=B[rng.choice(len(B),cap,replace=False)]
    pool=np.vstack([A,B]); na=len(A); N=len(pool)
    obs=sliced(A,B,rng,n_proj)[1]
    null=np.empty(Bp)
    for b in range(Bp):
        p=rng.permutation(N)
        null[b]=sliced(pool[p[:na]],pool[p[na:]],rng,n_proj)[1]
    return (obs-null.mean())/(null.std()+1e-12)

def window(mu_ref, mu_cur, n_ref, n_cur, rng):
    R=sample(mu_ref,n_ref,rng); C=sample(mu_cur,n_cur,rng)
    sm,sx=sliced(C,R,rng)
    cen=float(np.linalg.norm(C.mean(0)-R.mean(0)))
    sig=float(np.sqrt(np.median(sqd(R[:400],R[:400]))/2+1e-12))
    return dict(w1_mean=sm, w1_max=sx, centroid=cen, mmd=mmd2(C,R,sig),
                cal_z=perm_z(C,R,rng))

DET=["w1_mean","w1_max","centroid","mmd","cal_z"]
VOL={"A (low, n=400)":400, "B (high, n=4000)":4000}
N_REF=2000; NW=70; DRIFT=0.11

def build(vol_n, drifted, rng, nw=NW):
    rows=[]
    for _ in range(nw):
        mu=nrm(rng.standard_normal(D))
        mu_cur = nrm(mu + DRIFT*nrm(rng.standard_normal(D))) if drifted else mu
        rows.append(window(mu, mu_cur, N_REF, vol_n, rng))
    return pd.DataFrame(rows)

t0=time.time(); rng=np.random.default_rng(0)
nullA=build(VOL["A (low, n=400)"],  False, rng); print(f"  nullA  {time.time()-t0:5.1f}s",flush=True)
nullA2=build(VOL["A (low, n=400)"], False, rng); print(f"  nullA' {time.time()-t0:5.1f}s",flush=True)
nullB=build(VOL["B (high, n=4000)"],False, rng); print(f"  nullB  {time.time()-t0:5.1f}s",flush=True)
drB  =build(VOL["B (high, n=4000)"],True,  rng); print(f"  driftB {time.time()-t0:5.1f}s",flush=True)
drA  =build(VOL["A (low, n=400)"],  True,  rng); print(f"  driftA {time.time()-t0:5.1f}s",flush=True)

print("\nMean score on STABLE windows - does the score itself depend on volume?")
print(f"{'detector':<12}{'vol A (n=400)':>16}{'vol B (n=4000)':>16}{'ratio B/A':>12}")
for d in DET:
    a,b=nullA[d].mean(), nullB[d].mean()
    print(f"{d:<12}{a:>16.5f}{b:>16.5f}{b/a if a else float('nan'):>12.2f}")

print("\n" + "="*74)
print("CALIBRATE AT VOLUME A (5% FAR), FREEZE THRESHOLD, DEPLOY ELSEWHERE")
print("="*74)
print(f"{'detector':<12}{'FAR @A held-out':>18}{'FAR @B transfer':>18}{'POWER @B':>12}{'POWER @A':>12}")
res=[]
for d in DET:
    thr=np.quantile(nullA[d].values,0.95)
    farA=float((nullA2[d].values>thr).mean())
    farB=float((nullB[d].values>thr).mean())
    powB=float((drB[d].values>thr).mean())
    powA=float((drA[d].values>thr).mean())
    res.append(dict(detector=d,thr=thr,far_A=farA,far_B=farB,pow_B=powB,pow_A=powA))
    print(f"{d:<12}{farA:>18.3f}{farB:>18.3f}{powB:>12.3f}{powA:>12.3f}")

print("\n" + "="*74)
print("REVERSE DIRECTION - CALIBRATE AT VOLUME B, DEPLOY AT VOLUME A")
print("="*74)
print(f"{'detector':<12}{'FAR @B held-out':>18}{'FAR @A transfer':>18}{'POWER @A':>12}")
for d in DET:
    thr=np.quantile(nullB[d].values,0.95)
    farB=float((nullB[d].values>thr).mean())
    farA=float((nullA[d].values>thr).mean())
    powA=float((drA[d].values>thr).mean())
    print(f"{d:<12}{farB:>18.3f}{farA:>18.3f}{powA:>12.3f}")
pd.DataFrame(res).to_csv("e2_transfer.csv",index=False)
