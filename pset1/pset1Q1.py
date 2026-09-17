#matthew jeong pset1 q1
#run the code to see results

import numpy as np

def computeDmin(points):
    d = np.abs(points[:, None] - points[None, :])
    np.fill_diagonal(d, np.inf)
    return d.min()

def psk(M, dmin=1.0):
    A = dmin / (2*np.sin(np.pi/M))
    angles = np.arange(M)
    angles = angles*2*np.pi/M

    return A*np.exp(1j*angles)



def qam_rect(n, dmin=1.0):
    coords = np.arange(n)
    coords = coords - np.mean(coords)
    coords = coords*dmin
    X, Y = np.meshgrid(coords, coords)
    return (X + 1j*Y).flatten()

def qam32(dmin=1.0):
    grid = qam_rect(6, dmin)
    mx = np.abs(grid.real).max()
    is_corner = (np.abs(grid.real) == mx) & (np.abs(grid.imag) == mx)
    return grid[~is_corner]

qpsk = psk(4)
psk8 = psk(8)

q16 = qam_rect(4)

q32 = qam32()


for d in [1.0, 2.0, 3.7]:
    assert np.isclose(computeDmin(psk(4, d)), d)
    assert np.isclose(computeDmin(psk(8, d)), d)
    assert np.isclose(computeDmin(qam_rect(4, d)), d)
    assert np.isclose(computeDmin(qam32(d)), d)
    assert qam32(d).size == 32
    
#part 1A finished


#part 1B starts here

def computeEs(points):
    return np.mean(np.square(np.abs(points)))
    

def computeEta(points, N=2):
    return np.log2(points.size)/N

def computeEb(points):
    return computeEs(points)/np.log2(points.size)

constellations = {
    "QPSK":   qpsk,
    "8-PSK":  psk8,
    "16-QAM": q16,
    "32-QAM": q32,
}


assert np.isclose(computeEb(psk8), 0.5690, atol=1e-4), "8-PSK doesn't match the slide"

names = list(constellations.keys())
ebs   = [computeEb(p) for p in constellations.values()]
etas  = [computeEta(p) for p in constellations.values()]

print(f"{'':8s} {'M':>3s} {'E_s':>9s} {'E_b':>9s} {'eta':>6s}")
for name, pts in constellations.items():
    print(f"{name:8s} {pts.size:3d} {computeEs(pts):9.4f} "
          f"{computeEb(pts):9.4f} {computeEta(pts):6.2f}")
    
    
def dB(ratio):
    return 10*np.log10(ratio)

def section(title):
    print(f"\n{title}")
    print("-"*60)


print("all constellations scaled so that d_min = 1")


section("(a) constellations constructed")
for name, pts in constellations.items():
    print(f"  {name:8s} {pts.size:3d} points,  d_min = {computeDmin(pts):.4f}")
    print(f"    {np.round(pts, 4)}")

section("(b),(c) energy per bit and spectral efficiency")
print(f"  {'':8s} {'M':>3s} {'E_s':>9s} {'E_b':>9s} {'eta':>7s}")
for name, pts in constellations.items():
    print(f"  {name:8s} {pts.size:3d} {computeEs(pts):9.4f} "
          f"{computeEb(pts):9.4f} {computeEta(pts):7.2f}")

section("(d),(e) rankings")
i = np.argmin(ebs); j = np.argmax(etas)
print(f"  most power efficient:       {names[i]:8s} (E_b = {ebs[i]:.4f})")
print(f"  most spectrally efficient:  {names[j]:8s} (eta = {etas[j]:.2f})")

section("(f) cost of each step up in spectral efficiency")
for i in range(len(names)-1):
    print(f"  {names[i]:7s} -> {names[i+1]:8s} "
          f"d_eta = {etas[i+1]-etas[i]:+.2f}    "
          f"cost = {dB(ebs[i+1]/ebs[i]):+6.2f} dB")

print("\n  control case to help answer f:")
p16 = psk(16)
print(f"    16-PSK   eta = {computeEta(p16):.2f}   E_b = {computeEb(p16):.4f}")
print(f"    16-QAM   eta = {computeEta(q16):.2f}   E_b = {computeEb(q16):.4f}")
print(f"    16-PSK costs {dB(computeEb(p16)/computeEb(q16)):+.2f} dB more at identical eta")


print("all internal checks passed")

