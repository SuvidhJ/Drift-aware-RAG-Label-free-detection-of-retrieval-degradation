"""
E5 (v4) - ITERATIVE selective re-index.

Key idea under test: every document you re-embed becomes a NEW LABEL for the
selector.  You learn ||C1[i]-C0[i]|| for free the moment you refresh document i.
So instead of one probe -> one batch, run R rounds and refit between them.

That is an active-learning loop over the index, and it costs nothing extra:
the labels are a by-product of work you were already paying for.
"""
import numpy as np, time, pandas as pd

D,N,NCL,NQ,TOPK = 256,6000,20,2000,10
SPREAD,QNOISE = 0.70,0.45
def nrm(x): return x/(np.linalg.norm(x,axis=-1,keepdims=True)+1e-12)

def build(rng):
    means=nrm(rng.standard_normal((NCL,D))); w=1.0/np.arange(1,NCL+1); w/=w.sum()
    cnt=rng.multinomial(N,w); E,CL=[],[]
    for k,c in enumerate(cnt):
        if c==0: continue
        E.append(nrm(means[k]+(SPREAD/np.sqrt(D))*rng.standard_normal((c,D)))); CL.append(np.full(c,k))
    return np.vstack(E),np.concatenate(CL),means

def stale_world(C0,cl,means,rng,hot_k=5,rewrite_frac=0.60,migrate_frac=0.0):
    sizes=np.bincount(cl,minlength=NCL); hot=rng.choice(np.argsort(sizes)[NCL//4:],size=hot_k,replace=False)
    rew=np.isin(cl,hot)&(rng.random(N)<rewrite_frac); C1=C0.copy()
    idx=np.where(rew)[0]; mig=rng.random(len(idx))<migrate_frac
    for j,i in enumerate(idx):
        base=nrm(rng.standard_normal(D)) if mig[j] else means[cl[i]]
        C1[i]=nrm(base+(SPREAD/np.sqrt(D))*rng.standard_normal(D))
    return C1,rew

def recall(Q,E,gold,k=TOPK,block=500):
    h=0
    for s in range(0,Q.shape[0],block):
        p=np.argpartition(-(Q[s:s+block]@E.T),k,axis=1)[:,:k]
        h+=(p==gold[s:s+block,None]).any(axis=1).sum()
    return h/Q.shape[0]

def knn_predict(C0, lab_idx, lab_mv, target_idx, m=15, sharp=8):
    """Predict movement for target docs from labelled ones, in STALE space."""
    S = C0[target_idx] @ C0[lab_idx].T
    m = min(m, len(lab_idx))
    nb = np.argpartition(-S, m-1, axis=1)[:, :m]
    w = np.maximum(np.take_along_axis(S, nb, axis=1), 0.0)**sharp
    w /= w.sum(axis=1, keepdims=True)+1e-12
    return (w*lab_mv[nb]).sum(axis=1)

def oneshot(C0,C1,rng,budget,probe_frac):
    probe=rng.choice(N,size=int(probe_frac*N),replace=False)
    k=int(max(0.0,budget-probe_frac)*N)
    if k<=0: return probe
    rest=np.setdiff1d(np.arange(N),probe)
    mv=np.linalg.norm(C1[probe]-C0[probe],axis=1)
    s=knn_predict(C0,probe,mv,rest)
    return np.concatenate([probe,rest[np.argsort(-s)[:k]]])

def iterative(C0,C1,rng,budget,probe_frac,rounds):
    probe=rng.choice(N,size=int(probe_frac*N),replace=False)
    done=list(probe); done_set=set(done.copy() if isinstance(done,list) else done)
    per=int(max(0.0,budget-probe_frac)*N/rounds)
    for _ in range(rounds):
        if per<=0: break
        lab=np.array(done); mv=np.linalg.norm(C1[lab]-C0[lab],axis=1)
        rest=np.array([i for i in range(N) if i not in done_set])
        if len(rest)==0: break
        s=knn_predict(C0,lab,mv,rest)
        pick=rest[np.argsort(-s)[:per]]
        done.extend(pick.tolist()); done_set.update(pick.tolist())
    return np.array(done)

BUD=[0.10,0.20,0.35]
REG=[("in-place",0.0),("migrating",1.0)]
rows=[];t0=time.time()
for rname,mig in REG:
    for seed in range(3):
        rg=np.random.default_rng(seed); C0,cl,means=build(rg)
        r=np.random.default_rng(100+seed); C1,_=stale_world(C0,cl,means,r,migrate_frac=mig)
        pick=r.integers(0,N,size=NQ); Q=nrm(C1[pick]+(QNOISE/np.sqrt(D))*r.standard_normal((NQ,D)))
        base=recall(Q,C0,pick); full=recall(Q,C1,pick)
        rows.append(dict(regime=rname,seed=seed,method="_stale",budget=0,recall=base))
        rows.append(dict(regime=rname,seed=seed,method="_full",budget=1,recall=full))
        for b in BUD:
            cfgs={
              "oneshot p=2%":  lambda rr,bb: oneshot(C0,C1,rr,bb,0.02),
              "oneshot p=5%":  lambda rr,bb: oneshot(C0,C1,rr,bb,0.05),
              "iter 3 rounds": lambda rr,bb: iterative(C0,C1,rr,bb,0.01,3),
              "iter 6 rounds": lambda rr,bb: iterative(C0,C1,rr,bb,0.01,6),
              "oracle":        lambda rr,bb: np.argsort(-np.linalg.norm(C1-C0,axis=1))[:int(bb*N)],
              "random":        lambda rr,bb: rr.choice(N,size=int(bb*N),replace=False),
            }
            for name,fn in cfgs.items():
                sel=fn(np.random.default_rng(900+seed),b)
                E=C0.copy(); E[sel]=C1[sel]
                rows.append(dict(regime=rname,seed=seed,method=name,budget=b,
                                 recall=recall(Q,E,pick),spent=len(sel)/N))
    print(f"  {rname:10s} {time.time()-t0:5.1f}s",flush=True)

df=pd.DataFrame(rows); df.to_csv("e5_iter.csv",index=False)
ORDER=["random","oneshot p=2%","oneshot p=5%","iter 3 rounds","iter 6 rounds","oracle"]
for rname,_ in REG:
    d=df[df.regime==rname]
    base=d[d.method=="_stale"].recall.mean(); full=d[d.method=="_full"].recall.mean()
    piv=d[d.method.isin(ORDER)].pivot_table(index="budget",columns="method",values="recall",aggfunc="mean")[ORDER]
    print(f"\n===== {rname.upper()} =====  stale {base:.4f} -> full {full:.4f}  gap {full-base:.4f}")
    print("SHARE OF FULL-REINDEX GAIN RECOVERED"); print(((piv-base)/(full-base)).round(3).to_string())
    print("ABSOLUTE recall@10");                   print(piv.round(4).to_string())
