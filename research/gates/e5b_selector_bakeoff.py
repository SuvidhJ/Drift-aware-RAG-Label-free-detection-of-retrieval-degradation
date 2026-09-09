"""E5 (final) - selective re-index across three staleness regimes, with honest baselines."""
import numpy as np, time, pandas as pd

D, N, NCL, NQ, TOPK = 256, 6000, 20, 2000, 10
SPREAD, QNOISE, PROBE = 0.70, 0.45, 0.02
def nrm(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)

def build(rng):
    means = nrm(rng.standard_normal((NCL,D))); w = 1.0/np.arange(1,NCL+1); w/=w.sum()
    cnt = rng.multinomial(N,w); E,CL=[],[]
    for k,c in enumerate(cnt):
        if c==0: continue
        E.append(nrm(means[k]+(SPREAD/np.sqrt(D))*rng.standard_normal((c,D)))); CL.append(np.full(c,k))
    return np.vstack(E), np.concatenate(CL), means

def stale_world(C0, cl, means, rng, hot_k=5, rewrite_frac=0.60, migrate_frac=0.0):
    sizes=np.bincount(cl,minlength=NCL); hot=rng.choice(np.argsort(sizes)[NCL//4:],size=hot_k,replace=False)
    rewritten = np.isin(cl,hot) & (rng.random(N)<rewrite_frac)
    C1 = C0.copy(); idx=np.where(rewritten)[0]; mig=rng.random(len(idx))<migrate_frac
    for j,i in enumerate(idx):
        base = nrm(rng.standard_normal(D)) if mig[j] else means[cl[i]]
        C1[i]=nrm(base+(SPREAD/np.sqrt(D))*rng.standard_normal(D))
    return C1, rewritten

def topk_idx(Q,E,k,block=500):
    out=np.empty((Q.shape[0],k),dtype=int)
    for s in range(0,Q.shape[0],block):
        sims=Q[s:s+block]@E.T
        p=np.argpartition(-sims,k,axis=1)[:,:k]
        out[s:s+block]=p
    return out
def recall(Q,E,gold,k=TOPK): return (topk_idx(Q,E,k)==gold[:,None]).any(axis=1).mean()

def _w1(px,py,M=200):
    g=(np.arange(M)+0.5)/M
    return np.abs(np.quantile(px,g,axis=0)-np.quantile(py,g,axis=0)).mean(axis=0)
def theta_star(A,B,rng,n_proj=192):
    th=nrm(rng.standard_normal((n_proj,D))); return th[int(np.argmax(_w1(A@th.T,B@th.T)))]

def scores(sel,C0,C1,probe,Q,rng):
    if sel=="random":  return rng.random(N)
    if sel=="oracle":  return np.linalg.norm(C1-C0,axis=1)
    if sel=="retrieval_freq":                       # refresh what gets served most
        t=topk_idx(Q,C0,TOPK); return np.bincount(t.ravel(),minlength=N).astype(float)
    if sel=="low_conf_prox":                        # refresh near queries retrieval struggles with
        conf=np.empty(NQ)
        for s in range(0,NQ,500): conf[s:s+500]=(Q[s:s+500]@C0.T).max(axis=1)
        weak=Q[conf<=np.quantile(conf,0.25)]
        return (C0@weak.T).max(axis=1)
    if sel=="theta_star":
        th=theta_star(C1[probe],C0[probe],rng)
        return np.abs(C0@th-np.median(C0[probe]@th))
    if sel=="probe_knn":
        mv=np.linalg.norm(C1[probe]-C0[probe],axis=1)
        S=C0@C0[probe].T; m=min(15,len(probe))
        nb=np.argpartition(-S,m-1,axis=1)[:,:m]
        w=np.maximum(np.take_along_axis(S,nb,axis=1),0.0)**8; w/=w.sum(axis=1,keepdims=True)+1e-12
        return (w*mv[nb]).sum(axis=1)

SEL=["random","retrieval_freq","low_conf_prox","theta_star","probe_knn","oracle"]
BUD=[0.05,0.10,0.20,0.35]
REG=[("in-place",0.0),("mixed",0.5),("migrating",1.0)]

rows=[];t0=time.time()
for rname,mig in REG:
    for seed in range(3):
        rng=np.random.default_rng(seed); C0,cl,means=build(rng)
        r=np.random.default_rng(100+seed)
        C1,rew=stale_world(C0,cl,means,r,migrate_frac=mig)
        pick=r.integers(0,N,size=NQ); Q=nrm(C1[pick]+(QNOISE/np.sqrt(D))*r.standard_normal((NQ,D)))
        probe=r.choice(N,size=int(PROBE*N),replace=False)
        rs,rf=recall(Q,C0,pick),recall(Q,C1,pick)
        rows.append(dict(regime=rname,seed=seed,selector="_stale",budget=0.0,recall=rs))
        rows.append(dict(regime=rname,seed=seed,selector="_full",budget=1.0,recall=rf))
        sc={s:scores(s,C0,C1,probe,Q,r) for s in SEL}
        for s in SEL:
            order=np.argsort(-sc[s]); oh=PROBE if s in("theta_star","probe_knn") else 0.0
            for b in BUD:
                k=int(max(0.0,b-oh)*N); E=C0.copy()
                if oh: E[probe]=C1[probe]
                if k>0: E[order[:k]]=C1[order[:k]]
                rows.append(dict(regime=rname,seed=seed,selector=s,budget=b,recall=recall(Q,E,pick)))
    print(f"  {rname:10s} done {time.time()-t0:5.1f}s",flush=True)

df=pd.DataFrame(rows); df.to_csv("e5_final.csv",index=False)
for rname,_ in REG:
    d=df[df.regime==rname]
    base=d[d.selector=="_stale"].recall.mean(); full=d[d.selector=="_full"].recall.mean()
    piv=d[d.selector.isin(SEL)].pivot_table(index="budget",columns="selector",values="recall",aggfunc="mean")[SEL]
    print(f"\n===== {rname.upper()} REWRITE =====   stale {base:.4f} -> full {full:.4f}   gap {full-base:.4f}")
    print("share of the full-reindex gain recovered, by embedding budget:")
    print(((piv-base)/(full-base)).round(3).to_string())
