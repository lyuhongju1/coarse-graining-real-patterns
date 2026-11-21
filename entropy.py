import numpy as np, json, sys
def H(counts):
    p=counts/counts.sum(); p=p[p>0]; return -(p*np.log2(p)).sum()
def cond_entropy(y, *conds):
    # y, conds: integer arrays same shape; returns H(y|conds) in bits
    key=np.zeros(y.shape,dtype=np.int64)
    for c in conds: key=key*(c.max()+1)+c
    yk=key*(y.max()+1)+y
    _,cj=np.unique(yk,return_counts=True); _,ck=np.unique(key,return_counts=True)
    return H(cj)-H(ck)
def run(seed):
    rng=np.random.default_rng(seed); N,T,J,b=4096,1.6,1.0,16; nb=N//b
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
    def bsum(off): return np.roll(X,-off,axis=1).reshape(Tn,nb,b).sum(2)   # integer -16..16
    def kern(w):
        w=np.asarray(w,float); w/=w.sum(); L=len(w); half=L//2
        centers=np.arange(nb)*b+b//2; idx=(centers[:,None]+np.arange(-half,L-half)[None,:])%N
        return X[:,idx]@w
    A=bsum(0); B8=bsum(8); B1=bsum(1)
    x=np.arange(-12,13); G=kern(np.exp(-x**2/(2*16.0)))
    tri=np.concatenate([np.arange(1,9),np.arange(8,0,-1)]).astype(float); Tr=kern(tri)
    Wd=(X!=np.roll(X,-1,axis=1)).astype(int).reshape(Tn,nb,b).sum(2)
    # quantize continuous kernels to 17 levels
    def q(C): return np.clip(np.round((C+1)/2*16),0,16).astype(int)
    Gq=q(G); Trq=q(Tr)
    # per-spin conditional entropy: spin i in block j, given A_j (and given A_j plus the other pattern's block covering i)
    spins=((X+1)//2).reshape(Tn,nb,b).astype(int)
    pos=np.broadcast_to(np.arange(b)[None,None,:],spins.shape)
    Aj=np.broadcast_to((A+16)[:,:,None],spins.shape)
    # B8 block covering site i=j*b+p: offset-8 block index = floor((i-8)/b) mod nb
    i=(np.arange(nb)[:,None]*b+np.arange(b)[None,:])
    def cover(off,C):
        jj=((i-off)//b)%nb; return (C[:,jj]+16)
    out={}
    base=cond_entropy(spins.ravel(), Aj.ravel(), pos.ravel())
    out["H(s|A)"]=base
    for name,C,off in (("B8",B8,8),("B1",B1,1)):
        out[f"H(s|A,{name})"]=cond_entropy(spins.ravel(), Aj.ravel(), pos.ravel(), cover(off,C).ravel())
    out["H(s|A,G)"]=cond_entropy(spins.ravel(), Aj.ravel(), pos.ravel(), np.broadcast_to(Gq[:,:,None],spins.shape).ravel())
    out["H(s|A,Tr)"]=cond_entropy(spins.ravel(), Aj.ravel(), pos.ravel(), np.broadcast_to(Trq[:,:,None],spins.shape).ravel())
    out["H(s|A,W)"]=cond_entropy(spins.ravel(), Aj.ravel(), pos.ravel(), np.broadcast_to(Wd[:,:,None],spins.shape).ravel())
    # pattern-level: H(B8_j | A_j, A_{j+1}) etc. (distinctness in bits)
    Ar=np.roll(A,-1,axis=1)
    out["H(B8|A,A+)"]=cond_entropy((B8+16).ravel(),(A+16).ravel(),(Ar+16).ravel()); out["H(B8)"]=H(np.unique(B8,return_counts=True)[1])
    out["H(G|A,A+,A-)"]=cond_entropy(Gq.ravel(),(A+16).ravel(),(Ar+16).ravel(),(np.roll(A,1,axis=1)+16).ravel()); out["H(G)"]=H(np.unique(Gq,return_counts=True)[1])
    # autonomy via entropy: H(A_j' | A_j, A_j±1) vs H(A_j' | A_j,A_j±1, B8 covering, ...) skip; keep R2 for autonomy
    return out
res=[run(s) for s in (1,2)]
for r in res: print({k:round(v,4) for k,v in r.items()})
json.dump(res,open("entropy.json","w"),indent=1)
