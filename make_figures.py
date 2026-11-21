# Draws the three figures of the paper from the stored results.
import json, numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ink="#1f3a5f"; warm="#b5542a"

# ---- Figure 1: schematic of two block partitions ----
plt.rcParams.update({"font.size":9.5,"font.family":"serif"})
rng=np.random.default_rng(7); n=32; b=8
s=np.ones(n,int)
for i in range(1,n): s[i]=s[i-1] if rng.random()<0.72 else -s[i-1]
fig,ax=plt.subplots(figsize=(6.4,2.2))
ax.set_xlim(-0.6,n+0.6); ax.set_ylim(-1.75,1.55); ax.axis("off")
for i,v in enumerate(s):
    ax.add_patch(FancyBboxPatch((i+0.08,-0.42),0.84,0.84,boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=ink if v>0 else "0.86",edgecolor="none"))
for i in range(0,n+1,4): ax.text(i,-0.6,str(i),ha="center",va="top",fontsize=7,color="0.45")
def brackets(off,y,color,label,sign):
    for j in range(off,n,b):
        if j+b>n: break
        m=s[j:j+b].mean()
        ax.plot([j+0.1,j+0.1,j+b-0.1,j+b-0.1],[y-0.14*sign,y,y,y-0.14*sign],color=color,lw=1.4,solid_capstyle="round")
        ax.text(j+b/2,y+0.12*sign,f"{m:+.2f}",ha="center",va="bottom" if sign>0 else "top",fontsize=8,color=color)
    ax.text(-0.5,y+0.1*sign,label,ha="right",va="center",fontsize=9,color=color)
brackets(0, 0.72, ink, "$A$ (offset 0)", +1)
brackets(4,-1.05, warm, "$B_4$ (offset 4)", -1)
fig.tight_layout(pad=0.3); fig.savefig("figs/fig_schematic.pdf"); plt.close(fig)

plt.rcParams.update({"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"0.35","axes.labelcolor":"0.15",
    "xtick.color":"0.35","ytick.color":"0.35","axes.grid":True,"grid.color":"0.88","grid.linewidth":0.6,"axes.axisbelow":True})

# ---- Figure 2: shifted fields against the offset ----
r=json.load(open("results.json")); offs=r["offsets"]; d=r["derivability"]
g=json.load(open("joint.json"))["glauber"][0]["shift"]
fig,(a1,a2)=plt.subplots(1,2,figsize=(6.9,2.7))
for ax,y,yl,lo,hi in ((a1,d,"$R^2$ predicting $B_k$ from $A$",0.5,1.03),(a2,g,"$H(B_k \\mid A)$ (bits per spin)",-0.01,0.17)):
    ax.plot(offs,y,"-",color=ink,lw=2,zorder=2); ax.plot(offs,y,"o",color=ink,ms=6,mfc="white",mew=2,zorder=3)
    ax.set_xticks(offs); ax.set_xlabel("offset $k$ (sites)"); ax.set_ylabel(yl); ax.set_ylim(lo,hi); ax.tick_params(length=3)
a2.axhline(0,color="0.35",lw=0.8)
a1.annotate(f"{d[8]:.2f}",(8,d[8]),xytext=(-4,-14),textcoords="offset points",fontsize=8,color="0.2",ha="right")
fig.tight_layout(pad=0.4,w_pad=4.0); fig.savefig("figs/fig_offsets.pdf"); plt.close(fig)

# ---- Figure 3: one-step predictability of Fourier modes ----
ib=json.load(open("ib.json")); rho=ib["rho_k"]; m=ib["opt_modes"]
fig,ax=plt.subplots(figsize=(4.6,2.8))
ax.plot(range(len(rho)),rho,"-",color=ink,lw=1.6)
ax.axvline(m,color="0.35",lw=0.9,ls="--"); ax.text(m+6,0.905,"rate of a block-16 field",fontsize=8,color="0.3",va="center")
ax.set_xlim(0,600); ax.set_ylim(0.3,0.95); ax.set_xlabel("wavenumber index $q$"); ax.set_ylabel("one-step predictability $\\rho_q$"); ax.tick_params(length=3)
fig.tight_layout(pad=0.4); fig.savefig("figs/fig_modes.pdf"); plt.close(fig)
