import numpy as np, ot
rng=np.random.default_rng(7); d=1024
def sphere(n,mu,s):
    X=mu+s/np.sqrt(d)*rng.standard_normal((n,d)); return (X/np.linalg.norm(X,axis=1,keepdims=True)).astype(np.float32)
mu=rng.standard_normal(d); mu/=np.linalg.norm(mu)
print("### Coverage of the shipped 95% 'bootstrap CI' when the TRUE W1 is 0 ###")
r=np.random.default_rng(1); cov=0; TR=15
for _ in range(TR):
    X,Y=sphere(800,mu,0.5),sphere(800,mu,0.5)
    b=[ot.sliced_wasserstein_distance(X,Y,n_projections=500,seed=int(r.integers(0,2**31))) for _ in range(40)]
    lo,hi=np.percentile(b,[2.5,97.5]); cov += (lo<=0.0<=hi)
print(f"  X,Y drawn from the SAME law  => true W1 = 0")
print(f"  shipped CI contained 0 in {cov}/{TR} trials -> COVERAGE {cov/TR:.0%}  (nominal 95%)")
print("\n### estimate_lipschitz: denominator ~ 1/doc_length ###")
def edit(a,b):
    lb=len(b); prev=list(range(lb+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[lb]/max(len(a),lb)
r2=np.random.default_rng(0)
w=lambda k:" ".join("".join(r2.choice(list("abcdefghij"),size=6)) for _ in range(k))
for k in (5,20,80,300):
    t=w(k); ws=t.split(); ws[0]="zzzzz"; p=" ".join(ws); e=edit(t,p)
    print(f"  {k:4d} words ({len(t):5d} chars)  norm.edit = {e:.5f}  -> L_hat multiplier {1/e:8.1f}x")
print("  --> 60x length range => 60x denominator range. L_hat = max_i ||df_i||/edit_i is won by")
print("      the longest doc: it measures LENGTH, not encoder smoothness.")
print("      (Repo truncates corpus at 2048 chars, questions at 512 -> built-in 4x systematic gap.)")
