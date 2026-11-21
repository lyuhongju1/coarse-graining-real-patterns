# Kawasaki (spin-exchange) dynamics: self-prediction and autonomy of A, B8 and the kernel fields.
# Usage: python kawasaki.py LAG SEED   (LAG in stored frames; frames are 4 sweeps apart)
import numpy as np, json, sys
LAG=int(sys.argv[1]) if len(sys.argv)>1 else 32
SEED=int(sys.argv[2]) if len(sys.argv)>2 else 1
rng=np.random.default_rng(SEED); N,T,J,b=4096,1.6,1.0,16; nb=N//b
s=rng.choice([-1,1],size=N)
def glauber(s):
    for parity in (0,1):
        idx=np.arange(parity,N,2); nbr=s[idx-1]+s[(idx+1)%N]
        dE=2*J*s[idx]*nbr; flip=rng.random(idx.size)<1/(1+np.exp(dE/T)); s[idx[flip]]*=-1
    return s
def kawasaki(s):
    # exchange neighbouring antiparallel pairs (i, i+1); bonds are visited on two sublattices
    for parity in (0,1):
        i=np.arange(parity,N,2); j=(i+1)%N; anti=s[i]!=s[j]
        left=s[(i-1)%N]; right=s[(j+1)%N]
        # energy change of the swap: only the bonds (i-1,i) and (j,j+1) change
        dE=J*((s[i]*left + s[j]*right) - (s[j]*left + s[i]*right))
        p=1/(1+np.exp(dE/T)); do=anti&(rng.random(i.size)<p)
        si=s[i].copy(); s[i[do]]=s[j[do]]; s[j[do]]=si[do]
    return s
for t in range(300): s=glauber(s)          # equilibrate at temperature T before conserving magnetization
cfg=[]
for t in range(16300):
    s=kawasaki(s)
    if t>=300 and t%4==0: cfg.append(s.copy())
X=np.array(cfg,dtype=float); Tn=X.shape[0]
corr=[round(float(np.mean(X*np.roll(X,-r,axis=1))),3) for r in (1,2,4,8)]
print("magnetization range:",np.ptp(X.sum(1)),"corr:",corr)
def kern(w):
    w=np.asarray(w,float); w/=w.sum(); L=len(w); half=L//2
    centers=np.arange(nb)*b+b//2; idx=(centers[:,None]+np.arange(-half,L-half)[None,:])%N
    return X[:,idx]@w
A=kern(np.ones(16)); Tr=kern(np.concatenate([np.arange(1,9),np.arange(8,0,-1)]).astype(float)); x=np.arange(-12,13); G=kern(np.exp(-x**2/32.0))
B8=np.roll(X,-8,axis=1).reshape(Tn,nb,b).mean(2)
def nbrf(C,t,deg=2):
    F=np.stack([np.roll(C[t],-d) for d in (-1,0,1)],1); return np.hstack([F**p for p in range(1,deg+1)])
def microf(t,off=0,w=8):
    # spins in each block (starting at site off) and w sites either side
    idx=(off+np.arange(nb)[:,None]*b+np.arange(-w,b+w)[None,:])%N; return X[t][idx]
def pooled(featfn,target,ridge=1e-4):
    h=target.shape[0]//2; F=np.stack([featfn(t) for t in range(target.shape[0])]); F=np.concatenate([F,np.ones(F.shape[:2]+(1,))],2)
    Ftr=F[:h].reshape(-1,F.shape[2]); ytr=target[:h].reshape(-1)
    Wt=np.linalg.solve(Ftr.T@Ftr+ridge*np.eye(Ftr.shape[1]),Ftr.T@ytr)
    Fte=F[h:].reshape(-1,F.shape[2]); yte=target[h:].reshape(-1); p=Fte@Wt
    return 1-((yte-p)**2).sum()/((yte-yte.mean())**2).sum()
MF=np.stack([microf(t) for t in range(Tn)]); MF8=np.stack([microf(t,8) for t in range(Tn)])   # window centred on each field's own blocks
res={"corr":corr}
for name,C in (("A",A),("B8",B8),("Tri",Tr),("Gau",G)):
    sv=pooled(lambda t:nbrf(C,t),C[LAG:]); M=MF8 if name=="B8" else MF; mv=pooled(lambda t:M[t],C[LAG:],ridge=1.0)
    res[name]={"self":round(float(sv),4),"micro":round(float(mv),4),"autonomy":round(float(sv/mv),3)}
res["deriv_Tri_from_A"]=round(float(pooled(lambda t:nbrf(A,t),Tr)),3); res["deriv_B8_from_A"]=round(float(pooled(lambda t:nbrf(A,t),B8)),3)
res["deriv_Gau_from_A"]=round(float(pooled(lambda t:nbrf(A,t),G)),3)
print(json.dumps(res,indent=1)); json.dump(res,open(f"kawasaki_lag{LAG}_seed{SEED}.json","w"),indent=1)
