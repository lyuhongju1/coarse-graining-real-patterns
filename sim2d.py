import numpy as np, json, sys
rng=np.random.default_rng(1); L=128; T=2.5; J=1.0; b=8; nb=L//b
s=rng.choice([-1,1],size=(L,L))
mask=(np.indices((L,L)).sum(0)%2)
def sweep(s):
    for parity in (0,1):
        nbr=np.roll(s,1,0)+np.roll(s,-1,0)+np.roll(s,1,1)+np.roll(s,-1,1)
        dE=2*J*s*nbr; p=1/(1+np.exp(dE/T)); flip=(rng.random((L,L))<p)&(mask==parity); s[flip]*=-1
    return s
cfg=[]
for t in range(2300):
    s=sweep(s)
    if t>=300 and t%2==0: cfg.append(s.copy())
X=np.array(cfg,dtype=float); Tn=X.shape[0]
print("frames",Tn,"corr r=1,2,4,8:",[round(float(np.mean(X*np.roll(X,-r,axis=1))),3) for r in (1,2,4,8)]); sys.stdout.flush()
def box(off):
    Y=np.roll(np.roll(X,-off,1),-off,2); return Y.reshape(Tn,nb,b,nb,b).mean((2,4))
def kern(w2):  # w2: b x b kernel on same blocks (offset 0)
    w2=w2/w2.sum(); return np.einsum('tiajb,ab->tij',X.reshape(Tn,nb,b,nb,b),w2)
tri1=np.concatenate([np.arange(1,b//2+1),np.arange(b//2,0,-1)]).astype(float); pyr=np.outer(tri1,tri1)
A=box(0); B=box(b//2); P=kern(pyr)
def nbrf(C,t):
    F=np.stack([np.roll(np.roll(C[t],dx,0),dy,1) for dx in (-1,0,1) for dy in (-1,0,1)],-1).reshape(nb*nb,9)
    return np.hstack([F,F**2])
def microf(t,w=4):
    Y=X[t]; idx_i=(np.arange(nb)[:,None]*b+np.arange(-w,b+w)[None,:])%L
    out=np.empty((nb,nb,(b+2*w)**2))
    for i in range(nb):
        rows=Y[idx_i[i]]
        for j in range(nb):
            out[i,j]=rows[:,idx_i[j]].ravel()
    return out.reshape(nb*nb,-1)
def pooled(featfn,target,ridge=1e-3):
    T2=target.shape[0]; h=T2//2; F=np.stack([featfn(t) for t in range(T2)]); F=np.concatenate([F,np.ones(F.shape[:2]+(1,))],2)
    tg=target.reshape(T2,-1)
    Ftr=F[:h].reshape(-1,F.shape[2]); ytr=tg[:h].ravel()
    W=np.linalg.solve(Ftr.T@Ftr+ridge*np.eye(Ftr.shape[1]),Ftr.T@ytr)
    Fte=F[h:].reshape(-1,F.shape[2]); yte=tg[h:].ravel(); p=Fte@W
    return 1-((yte-p)**2).sum()/((yte-yte.mean())**2).sum()
res={}
MF=np.stack([microf(t) for t in range(Tn)])
microf_c=lambda t: MF[t]
res["deriv_B_from_A"]=pooled(lambda t:nbrf(A,t),B); res["deriv_P_from_A"]=pooled(lambda t:nbrf(A,t),P)
for name,C in (("A",A),("B",B),("P",P)):
    res[f"self_{name}"]=pooled(lambda t:nbrf(C,t),C[1:]); res[f"micro_{name}"]=pooled(lambda t:microf_c(t),C[1:],ridge=1.0)
# entropy gain per spin: H(s | A-block, pos) vs with B / P
def H(c): p=c/c.sum(); p=p[p>0]; return -(p*np.log2(p)).sum()
def cond_entropy(y,*conds):
    key=np.zeros(y.shape,dtype=np.int64)
    for c in conds: key=key*(int(c.max())+1)+c
    yk=key*(int(y.max())+1)+y
    _,cj=np.unique(yk,return_counts=True); _,ck=np.unique(key,return_counts=True); return H(cj)-H(ck)
Xi=((X+1)//2).astype(int); sp=Xi.reshape(Tn,nb,b,nb,b)
pos=np.broadcast_to((np.arange(b)[:,None]*b+np.arange(b)[None,:])[None,None,:,None,:],sp.shape)
# map to shape (Tn,nb,b,nb,b): position index pos = a*b+c
posf=np.broadcast_to((np.arange(b)[:,None]*b+np.arange(b)[None,:]).reshape(1,1,b,1,b)*0+ (np.arange(b)[:,None]*b+np.arange(b)[None,:]).reshape(1,1,b,1,b), sp.shape)
def q(C): return np.clip(np.round((C+1)/2*(b*b)),0,b*b).astype(int)
Aq=q(A); Bq=q(B); Pq=q(P)
Ab=np.broadcast_to(Aq[:,:,None,:,None],sp.shape)
# B block covering site (i,j): index floor((i-4)/8)
ii=(np.arange(nb)[:,None]*b+np.arange(b)[None,:]); jj=((ii-b//2)//b)%nb
Bcov=Bq[:, jj[:,:,None,None], jj[None,None,:,:]]  # shape Tn,nb,b,nb,b
Pb=np.broadcast_to(Pq[:,:,None,:,None],sp.shape)
base=cond_entropy(sp.ravel(),Ab.ravel(),posf.ravel())
res["H_s_given_A"]=base; res["H_s_given_A_B"]=cond_entropy(sp.ravel(),Ab.ravel(),posf.ravel(),Bcov.ravel()); res["H_s_given_A_P"]=cond_entropy(sp.ravel(),Ab.ravel(),posf.ravel(),Pb.ravel())
res={k:float(v) for k,v in res.items()}; print(json.dumps(res,indent=1)); json.dump(res,open("sim2d.json","w"),indent=1)
