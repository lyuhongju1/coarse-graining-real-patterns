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
X=np.array(cfg,dtype=float); Tn=X.shape[0]
F=np.fft.rfft(X,axis=1)              # modes k=0..N/2
F0=F[:-1]; F1=F[1:]
# per-mode one-step predictability: |corr(m_k(t), m_k(t+1))| (complex correlation magnitude)
num=np.abs((np.conj(F0)*F1).mean(0)); den=np.sqrt((np.abs(F0)**2).mean(0)*(np.abs(F1)**2).mean(0))
rho=num/den
ks=np.arange(len(rho))
print("rho at k=0,1,2,4,8,16,32,64,128,256,512,1024,2048:",[round(float(rho[k]),3) for k in (0,1,2,4,8,16,32,64,128,256,512,1024,2048)])
# monotonicity check: fraction of adjacent pairs where rho decreases
print("monotone decreasing fraction:", float(np.mean(np.diff(rho)<=0)))
# strictly: are there ties beyond +/-k? rfft already folds -k. Check how distinct neighbouring rho are
print("min gap between consecutive rho (k<300):", float(np.min(np.abs(np.diff(rho[:300])))))
# band-limited optimum at 'rate' matching one value per block: m=nb/2 complex modes ~ nb real numbers -> keep |k|<=nb/2=128
m=nb//2
def band_fraction(kernel_taps, centers_stride=b):
    # variance fraction of the kernel-filtered field lying in |k|<=m (compute on full-resolution filtered field)
    w=np.asarray(kernel_taps,float); w/=w.sum(); L=len(w); half=L//2
    Wk=np.fft.rfft(np.roll(np.pad(w,(0,N-L)),-half))   # kernel transfer function
    P=np.abs(F)**2 * np.abs(Wk)**2                      # filtered power spectrum per mode (per frame)
    P=P.mean(0); P[0]=0
    return float(P[:m+1].sum()/P.sum())
x=np.arange(-12,13); gau=np.exp(-x**2/32.0); tri=np.concatenate([np.arange(1,9),np.arange(8,0,-1)]).astype(float); box=np.ones(16)
out={"rho_k":[float(r) for r in rho[:600]],"band_fraction":{"box":band_fraction(box),"triangular":band_fraction(tri),"gaussian":band_fraction(gau)}}
print(out["band_fraction"])
# predictive information captured: sum over kept modes of -log(1-rho^2) proxy (Gaussian MI), optimum vs kernels' effective
def pred_info_weights(kernel_taps):
    w=np.asarray(kernel_taps,float); w/=w.sum(); L=len(w); half=L//2
    Wk=np.abs(np.fft.rfft(np.roll(np.pad(w,(0,N-L)),-half)))**2
    return Wk
gmi=-np.log(1-rho**2+1e-12)
opt=float(gmi[:m+1].sum())
res={}
for name,kt in (("box",box),("triangular",tri),("gaussian",gau)):
    Wk=pred_info_weights(kt); Wk=Wk/Wk[:m+1].mean()  # normalise passband
    res[name]=float((gmi*np.minimum(Wk,1)).sum()/opt)
print("predictive-information proxy relative to band-limited optimum:",res)
out["pred_info_rel"]=res; out["opt_modes"]=m
json.dump(out,open("ib.json","w"))
