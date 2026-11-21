import numpy as np, sys
T=1.6
def run(n,bb,sh):
    idx=np.arange(2**n,dtype=np.int64)
    st=(((idx[:,None]>>np.arange(n))&1)*2-1).astype(np.int8)
    E=-(st*np.roll(st,-1,1)).sum(1,dtype=np.int32); p=np.exp(-E/T); p/=p.sum()
    def Hw(cols):
        key=np.zeros(len(p),dtype=np.int64)
        for c in cols: key=key*40+(c.astype(np.int64)-int(c.min()))
        u,inv=np.unique(key,return_inverse=True); q=np.bincount(inv,weights=p); q=q[q>0]; return float(-(q*np.log2(q)).sum())
    nbk=n//bb
    a=[st[:,k*bb:(k+1)*bb].sum(1,dtype=np.int16) for k in range(nbk)]
    r=np.roll(st,-sh,1); bs=[r[:,k*bb:(k+1)*bb].sum(1,dtype=np.int16) for k in range(nbk)]
    joint=Hw(a+bs)-Hw(a)                       # H(B|A) = H(D|A)-H(D|A,B)
    single=0.0
    for k in range(n):
        aj=a[k//bb]; bj=bs[((k-sh)//bb)%nbk]; sk=st[:,k].astype(np.int16)
        single+=(Hw([sk,aj])-Hw([aj]))-(Hw([sk,aj,bj])-Hw([aj,bj]))
    print("ring n=%d, block %d, shift %d (%d blocks): joint gain %.4f bits/spin, single-site sum %.4f bits/spin, ratio %.2f"%(n,bb,sh,nbk,joint/n,single/n,joint/single))
run(16,8,4); run(16,4,2); run(20,4,2); run(24,4,2); run(24,8,4)
