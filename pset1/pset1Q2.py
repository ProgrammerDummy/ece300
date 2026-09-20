import numpy as np
import matplotlib.pyplot as plt

W = 1
A = 1


def xt(t, A=1, W=1):
    return A*np.sinc(2*W*t) + (A/2)*np.sinc(2*W*t+1) + (A/2)*np.sinc(2*W*t-1)

def Xf(f, A=1, W=1):
    gate = (np.abs(f) <= W) #II(f/2W) is 0 if abs(f) not within [-W, W] band
    return (A/(2*W))*(1+np.cos(np.pi*f/W))*gate

for f0 in [0, W/2, -W/2, W, 1.5*W]:
    print(f"  X({f0:+.2f}) = {Xf(np.array([f0]))[0]:.4f}")
    
f = np.linspace(-2*W, 2*W, 1000)
T = 1/(2*W)
t = np.linspace(-4*T, 4*T, 1000)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7))


ks = np.arange(-4, 5)                  
t_k = ks * T                          
x_pred = np.where(ks == 0, A,
         np.where(np.abs(ks) == 1, A/2, 0.0))


ax1.plot(f, Xf(f), label="X(f)")
ax1.scatter([0, W/2, -W/2, W, -W],
            [A/W, A/(2*W), A/(2*W), 0, 0],
            color="red", zorder=5, label="predicted (part a)")
ax1.axhline(0, color="gray", lw=0.5)
ax1.set_xlabel("f")
ax1.set_ylabel("X(f)")
ax1.set_title(f"Raised-cosine spectrum (A={A}, W={W})")
ax1.grid(alpha=0.3)
ax1.legend()

ax2.plot(t, xt(t), label="x(t)")
ax2.scatter(t_k, x_pred, color="red", zorder=5, label="predicted x(kT), parts d,e")
ax2.axhline(0, color="gray", lw=0.5)
ax2.set_xlabel("t")
ax2.set_ylabel("x(t)")
ax2.set_title(f"Inverse transform, T = 1/(2W) = {T}")
ax2.grid(alpha=0.3)
ax2.legend()

print(f"\n  T = {T}")
print(f"  {'k':>3s} {'t = kT':>8s} {'x(kT)':>12s} {'predicted':>11s}")
for k in range(-4, 5):
    val = xt(k*T)
    pred = A if k == 0 else (A/2 if abs(k) == 1 else 0.0)
    print(f"  {k:+3d} {k*T:8.2f} {val:12.6f} {pred:11.4f}")
    
fig.tight_layout()
fig.savefig("2f.png", dpi=150)