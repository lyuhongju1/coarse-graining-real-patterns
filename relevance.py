import numpy as np, json
rng=np.random.default_rng(1); import sys
N,T,J,b=4096,float(sys.argv[1]),1.0,16; nb=N//b
s=rng.choice([-1,1],size=N)
def sweep(s):
    for parity in (0,1):
        idx=np.arange(parity,N,2); nbr=s[idx-1]+s[(idx+1)%N]
        dE=2*J*s[idx]*nbr; flip=rng.random(idx.size)<1/(1+np.exp(dE/T)); s[idx[flip]]*=-1
    return s
cfg=[]
for t in range(4300):
    s=sweep(s)
    if t>=300: cfg.append(s.copy())
X=np.array(cfg,dtype=float); Tn=X.shape[0]
A=X.reshape(Tn,nb,b).mean(2)
W=(X!=np.roll(X,-1,axis=1)).astype(float).reshape(Tn,nb,b).mean(2)
def nbrf(C,t,deg=2):
    F=np.stack([np.roll(C[t],-d) for d in (-1,0,1)],1); return np.hstack([F**p for p in range(1,deg+1)])
def pooled(featfn,target,ridge=1e-4):
    h=Tn//2; F=np.stack([featfn(t) for t in range(target.shape[0])]); F=np.concatenate([F,np.ones(F.shape[:2]+(1,))],2)
    Ftr=F[:h].reshape(-1,F.shape[2]); ytr=target[:h].reshape(-1)
    Wt=np.linalg.solve(Ftr.T@Ftr+ridge*np.eye(Ftr.shape[1]),Ftr.T@ytr)
    Fte=F[h:].reshape(-1,F.shape[2]); yte=target[h:].reshape(-1); p=Fte@Wt
    return 1-((yte-p)**2).sum()/((yte-yte.mean())**2).sum()
# use a longer horizon too (4 sweeps) to make walls' own dynamics visible
res={}
for lag in (1,4):
    r={}
    r["A_next_from_A"]=pooled(lambda t:nbrf(A,t),A[lag:])
    r["A_next_from_W"]=pooled(lambda t:nbrf(W,t),A[lag:])
    r["W_next_from_W"]=pooled(lambda t:nbrf(W,t),W[lag:])
    r["W_next_from_A"]=pooled(lambda t:nbrf(A,t),W[lag:])
    r["A_next_from_both"]=pooled(lambda t:np.hstack([nbrf(A,t),nbrf(W,t)]),A[lag:])
    r["W_next_from_both"]=pooled(lambda t:np.hstack([nbrf(A,t),nbrf(W,t)]),W[lag:])
    res[f"lag{lag}"]={k:round(float(v),3) for k,v in r.items()}
print(json.dumps(res,indent=1)); json.dump(res,open(f"relevance_T{sys.argv[1]}.json","w"),indent=1)
# variance shares: how much of the microstate's next-step predictable variance is magnetization-like vs wall-like? skip
