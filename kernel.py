import numpy as np, json
rng=np.random.default_rng(1); N,T,J,b=4096,1.6,1.0,16; nb=N//b
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
X=np.array(cfg,dtype=float); spins=X.reshape(X.shape[0],nb,b)
def kern_field(w):  # w: kernel weights centered; sample at block centers (offset 0 blocks -> centers at j*b + b/2)
    w=np.asarray(w,float); w/=w.sum(); L=len(w); half=L//2
    centers=np.arange(nb)*b + b//2
    idx=(centers[:,None]+np.arange(-half,L-half)[None,:])%N
    return X[:,idx]@w
box=np.ones(b)                                    # = A
tri=np.concatenate([np.arange(1,b//2+1),np.arange(b//2,0,-1)]).astype(float)  # triangular width 16
x=np.arange(-12,13); gau=np.exp(-x**2/(2*4.0**2))  # gaussian sigma 4, support 25
A=kern_field(box); Tr=kern_field(tri); G=kern_field(gau)
def nbrf(C,t,deg=2):
    F=np.stack([np.roll(C[t],-d) for d in (-1,0,1)],1); return np.hstack([F**p for p in range(1,deg+1)])
def microf(t,w=8):
    idx=(np.arange(nb)[:,None]*b+np.arange(-w,b+w)[None,:])%N; return X[t][idx]
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
out={}
for name,C in (("triangular",Tr),("gaussian s=4",G)):
    out[name]=dict(deriv_from_A=pooled(lambda t:nbrf(A,t),C),self=pooled(lambda t:nbrf(C,t),C[1:]),micro=pooled(lambda t:microf(t),C[1:]))
    print(name, "deriv from A: %.3f"%pooled(lambda t:nbrf(A,t),C), " A from it: %.3f"%pooled(lambda t:nbrf(C,t),A),
          " self: %.4f (A %.4f)"%(pooled(lambda t:nbrf(C,t),C[1:]),pooled(lambda t:nbrf(A,t),A[1:])),
          " micro: %.3f (A %.3f)"%(pooled(lambda t:microf(t),C[1:]),pooled(lambda t:microf(t),A[1:])),
          " recon A %.3f -> A+it %.3f"%(perpos(lambda t:nbrf(A,t,1)),perpos(lambda t:np.hstack([nbrf(A,t,1),nbrf(C,t,1)]))))
out["A"]=dict(self=pooled(lambda t:nbrf(A,t),A[1:]),micro=pooled(lambda t:microf(t),A[1:]))
json.dump({a:{b:float(c) for b,c in v.items()} for a,v in out.items()},open("kernel.json","w"),indent=1)
