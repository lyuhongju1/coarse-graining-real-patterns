import numpy as np, json, sys
def run(seed,N=4096,T=1.6,b=16,sweeps=4000,burn=300):
    rng=np.random.default_rng(seed); J=1.0; nb=N//b
    s=rng.choice([-1,1],size=N)
    def sweep(s):
        for parity in (0,1):
            idx=np.arange(parity,N,2); nbr=s[idx-1]+s[(idx+1)%N]
            dE=2*J*s[idx]*nbr; flip=rng.random(idx.size)<1/(1+np.exp(dE/T)); s[idx[flip]]*=-1
        return s
    cfg=[]
    for t in range(sweeps+burn):
        s=sweep(s)
        if t>=burn: cfg.append(s.copy())
    X=np.array(cfg,dtype=float); spins=X.reshape(X.shape[0],nb,b)
    def bmag(off): return np.roll(X,-off,axis=1).reshape(X.shape[0],nb,b).mean(2)
    def bwall(off):
        Y=np.roll(X,-off,axis=1); W=(Y!=np.roll(Y,-1,axis=1)).astype(float); return W.reshape(X.shape[0],nb,b).mean(2)
    def nbrf(C,t,deg=2):
        F=np.stack([np.roll(C[t],-d) for d in (-1,0,1)],1)
        return np.hstack([F**p for p in range(1,deg+1)])
    def microf(t,off=0,w=8):
        Y=np.roll(X[t],-off); idx=(np.arange(nb)[:,None]*b+np.arange(-w,b+w)[None,:])%N; return Y[idx]
    def pooled(featfn,target,ridge=1e-4):
        Tn=target.shape[0]; h=Tn//2
        F=np.stack([featfn(t) for t in range(Tn)]); F=np.concatenate([F,np.ones(F.shape[:2]+(1,))],2)
        Ftr=F[:h].reshape(-1,F.shape[2]); ytr=target[:h].reshape(-1)
        W=np.linalg.solve(Ftr.T@Ftr+ridge*np.eye(Ftr.shape[1]),Ftr.T@ytr)
        Fte=F[h:].reshape(-1,F.shape[2]); yte=target[h:].reshape(-1); p=Fte@W
        return 1-((yte-p)**2).sum()/((yte-yte.mean())**2).sum()
    def perpos(featfn):
        Tn=X.shape[0]; h=Tn//2; ss=tt=0
        F=np.stack([featfn(t) for t in range(Tn)]); F=np.concatenate([F,np.ones(F.shape[:2]+(1,))],2)
        for p in range(b):
            y=spins[:,:,p]; Ftr=F[:h].reshape(-1,F.shape[2]); ytr=y[:h].reshape(-1)
            W=np.linalg.solve(Ftr.T@Ftr+1e-4*np.eye(Ftr.shape[1]),Ftr.T@ytr)
            Fte=F[h:].reshape(-1,F.shape[2]); yte=y[h:].reshape(-1); pr=Fte@W
            ss+=((yte-pr)**2).sum(); tt+=((yte-yte.mean())**2).sum()
        return 1-ss/tt
    A=bmag(0); B=bmag(b//2); Cw=bwall(0)
    out=dict(seed=seed,T=T,b=b,
        corr1=float(np.mean(X*np.roll(X,-1,axis=1))),
        derivB_lin=pooled(lambda t:nbrf(A,t,1),B), derivB_quad=pooled(lambda t:nbrf(A,t,2),B),
        derivB1_quad=pooled(lambda t:nbrf(A,t,2),bmag(1)),
        derivW_lin=pooled(lambda t:nbrf(A,t,1),Cw), derivW_quad=pooled(lambda t:nbrf(A,t,2),Cw),
        selfA_lin=pooled(lambda t:nbrf(A,t,1),A[1:]), selfA_quad=pooled(lambda t:nbrf(A,t,2),A[1:]), selfA_cubic=pooled(lambda t:nbrf(A,t,3),A[1:]),
        selfB_quad=pooled(lambda t:nbrf(B,t,2),B[1:]),
        micA=pooled(lambda t:microf(t,0),A[1:]), micB=pooled(lambda t:microf(t,b//2),B[1:]),
        selfW_quad=pooled(lambda t:nbrf(Cw,t,2),Cw[1:]), selfW_cubic=pooled(lambda t:nbrf(Cw,t,3),Cw[1:]),
        reconA=perpos(lambda t:nbrf(A,t,1)), reconAB=perpos(lambda t:np.hstack([nbrf(A,t,1),nbrf(B,t,1)])))
    return out
res=[]
for seed in (1,2,3): res.append(run(seed)); print(json.dumps(res[-1])); sys.stdout.flush()
res.append(run(1,b=8)); print(json.dumps(res[-1])); sys.stdout.flush()
res.append(run(1,b=32)); print(json.dumps(res[-1])); sys.stdout.flush()
res.append(run(1,T=1.2)); print(json.dumps(res[-1])); sys.stdout.flush()
res.append(run(1,T=2.2)); print(json.dumps(res[-1])); sys.stdout.flush()
json.dump(res,open("robust.json","w"),indent=1)
