import numpy as np, json
rng=np.random.default_rng(1)
N,T,J=4096,1.6,1.0; sweeps,burn=4000,300; b=16; nb=N//b
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
def nbrf(C,t,sq=True):
    F=np.stack([np.roll(C[t],-d) for d in (-1,0,1)],1); return np.hstack([F,F**2]) if sq else F
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
A=bmag(0); Cw=bwall(0)
offs=list(range(0,9)); deriv=[]; gain=[]; auton=[]
base=perpos(lambda t: nbrf(A,t,False))
for k in offs:
    Bk=bmag(k)
    deriv.append(pooled(lambda t: nbrf(A,t),Bk))
    gain.append(perpos(lambda t: np.hstack([nbrf(A,t,False),nbrf(Bk,t,False)]))-base)
    auton.append(pooled(lambda t: nbrf(Bk,t),Bk[1:]))
autA_mic=pooled(lambda t: microf(t,0),A[1:]); autB8_mic=pooled(lambda t: microf(t,8),bmag(8)[1:])
autCw=pooled(lambda t: nbrf(Cw,t),Cw[1:]); autCw_mic=pooled(lambda t: microf(t,0),Cw[1:])
gainCw=perpos(lambda t: np.hstack([nbrf(A,t,False),nbrf(Cw,t,False)]))-base
derivCw=pooled(lambda t: nbrf(A,t),Cw)
res=dict(offsets=offs,derivability=deriv,gain=gain,autonomy=auton,autA_micro=autA_mic,autB8_micro=autB8_mic,
         autCw=autCw,autCw_micro=autCw_mic,gainCw=gainCw,derivCw=derivCw,base_recon=base,
         corr=[float(np.mean(X*np.roll(X,-r,axis=1))) for r in (1,2,4,8)])
json.dump(res,open("results.json","w"),indent=1); print(json.dumps(res,indent=1))
