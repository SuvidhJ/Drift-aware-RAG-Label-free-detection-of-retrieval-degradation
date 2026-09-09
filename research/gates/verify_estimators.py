"""Verification 2: the W1 'confidence interval', the MMD allocation, the null decay."""
import numpy as np, ot, time

rng = np.random.default_rng(0)
d = 1024                       # bge-m3 dim
def sphere(n, mu=None, spread=0.5):
    mu = mu if mu is not None else np.zeros(d)
    X = mu + spread/np.sqrt(d)*rng.standard_normal((n,d))
    return (X/np.linalg.norm(X,axis=1,keepdims=True)).astype(np.float32)

mu = rng.standard_normal(d); mu/=np.linalg.norm(mu)
X, Y = sphere(1200, mu), sphere(1200, mu)      # SAME distribution

print("### A. drift/wasserstein.py 'bootstrap CI' -- what does it resample? ###")
# repo: re-runs sliced_wasserstein_distance(X, Y) with a new PROJECTION seed; X,Y fixed
r = np.random.default_rng(42)
point = ot.sliced_wasserstein_distance(X,Y,n_projections=500,seed=int(r.integers(0,2**31)))
repo_boot = np.array([ot.sliced_wasserstein_distance(X,Y,n_projections=500,seed=int(r.integers(0,2**31)))
                      for _ in range(200)])
repo_lo, repo_hi = np.percentile(repo_boot,[2.5,97.5])
# honest: resample the DATA rows
true_boot=[]
for _ in range(200):
    xi = r.integers(0,len(X),len(X)); yi = r.integers(0,len(Y),len(Y))
    true_boot.append(ot.sliced_wasserstein_distance(X[xi],Y[yi],n_projections=500,seed=int(r.integers(0,2**31))))
true_boot=np.array(true_boot); true_lo,true_hi=np.percentile(true_boot,[2.5,97.5])
print(f"  point estimate           : {point:.6f}")
print(f"  repo 'CI' (seed-only)    : [{repo_lo:.6f}, {repo_hi:.6f}]   width={repo_hi-repo_lo:.6f}")
print(f"  honest CI (data bootstrap): [{true_lo:.6f}, {true_hi:.6f}]   width={true_hi-true_lo:.6f}")
print(f"  --> repo CI understates uncertainty by {(true_hi-true_lo)/(repo_hi-repo_lo):.1f}x")

print("\n### B. Null level: W1 between two i.i.d. samples is NOT zero, and depends on n ###")
for n in (300, 600, 1200, 2400, 4800):
    vals=[ot.sliced_wasserstein_distance(sphere(n,mu),sphere(n,mu),n_projections=500,seed=int(r.integers(0,2**31)))
          for _ in range(8)]
    print(f"  n={n:5d}  W1(null) = {np.mean(vals):.6f}  (sd {np.std(vals):.6f})")
print("  --> a FIXED threshold tau on a raw W1 is a query-VOLUME threshold, not a drift threshold.")

print("\n### C. drift/mmd.py memory: (n,m,d) intermediate ###")
for n in (500, 2000):
    gb = n*n*d*4/1e9
    print(f"  _rbf_kernel with n=m={n}, d={d}: allocates {n}x{n}x{d} float32 = {gb:.2f} GB")
print("  (default subsample=2000 -> 16.4 GB for ONE of three kernel matrices)")
# demonstrate the identity gives the same answer at ~n*m memory
def sqd_naive(A,B): return np.sum((A[:,None,:]-B[None,:,:])**2,axis=-1)
def sqd_gram(A,B):
    aa=np.einsum('ij,ij->i',A,A)[:,None]; bb=np.einsum('ij,ij->i',B,B)[None,:]
    return np.maximum(aa+bb-2*A@B.T,0.0)
A,B = X[:400].astype(np.float64), Y[:400].astype(np.float64)
t0=time.time(); Dn=sqd_naive(A,B); t1=time.time(); Dg=sqd_gram(A,B); t2=time.time()
print(f"  identity check: max|naive-gram| = {np.abs(Dn-Dg).max():.2e}   speed {(t1-t0)/(t2-t1):.0f}x faster")
print(f"  memory ratio at n=2000: {2000*2000*d*4/1e9:.1f} GB -> {2000*2000*8/1e6:.0f} MB  ({d/2:.0f}x less)")
