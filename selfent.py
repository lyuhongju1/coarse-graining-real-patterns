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
X=np.array(cfg); Tn=X.shape[0]
def bsum(off): return np.roll(X,-off,axis=1).reshape(Tn,nb,b).sum(2)+16
def kern(w):
    w=np.asarray(w,float); w/=w.sum(); L=len(w); half=L//2
    centers=np.arange(nb)*b+b//2; idx=(centers[:,None]+np.arange(-half,L-half)[None,:])%N
    C=X[:,idx]@w; return np.clip(np.round((C+1)/2*16),0,16).astype(int)
def H(c): p=c/c.sum(); p=p[p>0]; return -(p*np.log2(p)).sum()
def ce(y,*conds):
    key=np.zeros(y.shape,dtype=np.int64)
    for c in conds: key=key*(int(c.max())+1)+c
    yk=key*(int(y.max())+1)+y
    _,cj=np.unique(yk,return_counts=True); _,ck=np.unique(key,return_counts=True); return H(cj)-H(ck)
tri=np.concatenate([np.arange(1,9),np.arange(8,0,-1)]).astype(float); x=np.arange(-12,13); gau=np.exp(-x**2/32.0)
out={}
for name,C in (("A",bsum(0)),("B8",bsum(8)),("Tri",kern(tri)),("Gau",kern(gau))):
    y=C[1:]; c0=C[:-1]; cl=np.roll(c0,1,axis=1); cr=np.roll(c0,-1,axis=1)
    out[name]={"H":H(np.unique(C,return_counts=True)[1]),"H_next_given_self3":ce(y.ravel(),c0.ravel(),cl.ravel(),cr.ravel())}
    out[name]["info_ratio"]=1-out[name]["H_next_given_self3"]/out[name]["H"]
print(json.dumps(out,indent=1)); json.dump(out,open("selfent.json","w"),indent=1)
