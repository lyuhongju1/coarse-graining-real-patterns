# Joint information gain per spin, H(B|A)/N, for the shifted fields B_k and the kernel fields.
# Given the whole field A, B_k is fixed by the sums over the first k sites of each block, so H(B_k|A) is
# estimated from those partial sums with a truncated chain rule (an upper bound, tight for short correlations).
import numpy as np, json, sys
N,T,J,b=4096,1.6,1.0,16; nb=N//b
def H(c): p=c/c.sum(); p=p[p>0]; return float(-(p*np.log2(p)).sum())
def ce(y,*conds):
    key=np.zeros(y.shape,dtype=np.int64)
    for c in conds: key=key*(int(c.max())+1)+c
    yk=key*(int(y.max())+1)+y
    _,cj=np.unique(yk,return_counts=True); _,ck=np.unique(key,return_counts=True); return H(cj)-H(ck)
def simulate(mode,seed):
    rng=np.random.default_rng(seed); s=rng.choice([-1,1],size=N)
    def glauber(s):
        for parity in (0,1):
            idx=np.arange(parity,N,2); nbr=s[idx-1]+s[(idx+1)%N]
            dE=2*J*s[idx]*nbr; flip=rng.random(idx.size)<1/(1+np.exp(dE/T)); s[idx[flip]]*=-1
        return s
    def kawasaki(s):
        for parity in (0,1):
            i=np.arange(parity,N,2); j=(i+1)%N; anti=s[i]!=s[j]
            left=s[(i-1)%N]; right=s[(j+1)%N]
            dE=J*((s[i]*left + s[j]*right) - (s[j]*left + s[i]*right))
            p=1/(1+np.exp(dE/T)); do=anti&(rng.random(i.size)<p)
            si=s[i].copy(); s[i[do]]=s[j[do]]; s[j[do]]=si[do]
        return s
    sweep=glauber if mode=="glauber" else kawasaki; nsw=4300 if mode=="glauber" else 16300; cfg=[]
    if mode=="kawasaki":
        for t in range(300): s=glauber(s)      # equilibrate before conserving magnetization
    for t in range(nsw):
        s=sweep(s)
        if t>=300 and (mode=="glauber" or t%4==0): cfg.append(s.copy())
    return np.array(cfg)
def analyse(X):
    Tn=X.shape[0]; S=((X+1)//2).reshape(Tn,nb,b).astype(np.int64)
    A=S.sum(2); Al=np.roll(A,1,1); Ar=np.roll(A,-1,1)
    def kern(w):
        w=np.asarray(w,float); w/=w.sum(); L=len(w); half=L//2
        centers=np.arange(nb)*b+b//2; idx=(centers[:,None]+np.arange(-half,L-half)[None,:])%N
        C=X[:,idx]@w; return np.clip(np.round((C+1)/2*16),0,16).astype(np.int64)
    out={"shift":[0.0]}
    B8=np.roll(X,-8,axis=1).reshape(Tn,nb,b).sum(2)+16; out["H_B8"]=H(np.unique(B8,return_counts=True)[1]); out["H_B8_given_A_Aplus"]=ce(B8.ravel(),A.ravel(),Ar.ravel())
    for k in range(1,9):
        P=S[:,:,:k].sum(2)                       # given A, B_k is fixed by these partial sums
        out["shift"].append(ce(P.ravel(),A.ravel(),np.roll(P,1,1).ravel(),Al.ravel())/b)
    for name,w in (("tri",np.concatenate([np.arange(1,9),np.arange(8,0,-1)]).astype(float)),("gau",np.exp(-np.arange(-12,13)**2/32.0))):
        C=kern(w); Cl=np.roll(C,1,1)
        b1=ce(C.ravel(),A.ravel(),Al.ravel(),Ar.ravel()); b2=ce(C.ravel(),A.ravel(),Al.ravel(),Cl.ravel())
        out["H_"+name+"_given_own_block"]=ce(C.ravel(),A.ravel())
        out[name]=min(b1,b2)/b; out[name+"_bounds"]=[b1,b2]; out["H_"+name]=H(np.unique(C,return_counts=True)[1])
        out["Hfield_"+name]=ce(C.ravel(),Cl.ravel(),np.roll(C,2,1).ravel())/b
    out["Hfield_A"]=ce(A.ravel(),Al.ravel(),np.roll(A,2,1).ravel())/b
    sp=((X+1)//2).astype(np.int64); out["h_spin"]=ce(sp.ravel(),np.roll(sp,1,1).ravel())
    return out
res={}
for mode,seeds in (("glauber",(1,2)),("kawasaki",(1,2,3))):
    res[mode]=[analyse(simulate(mode,sd)) for sd in seeds]
    for r in res[mode]:
        print(mode,"shift:",[round(v,3) for v in r["shift"]],"tri %.3f"%r["tri"],[round(x,3) for x in r["tri_bounds"]],"gau %.3f"%r["gau"],[round(x,3) for x in r["gau_bounds"]],
              "| h %.3f HA %.3f Htri %.3f -> H(D|A) %.3f H(D|tri) %.3f"%(r["h_spin"],r["Hfield_A"],r["Hfield_tri"],r["h_spin"]-r["Hfield_A"],r["h_spin"]-r["Hfield_tri"]))
json.dump(res,open("joint.json","w"),indent=1)
