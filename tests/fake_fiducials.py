"""Generate a synthetic fiducial set that behaves like real instrument output.

Maps pixel points through a known similarity transform, then adds a little
click/stage noise so the fit lands between min-fiducial-residual-um and
max-fiducial-residual-um instead of tripping either guard.

    python fake_fiducials.py [n] [seed]
"""
import sys, math
import numpy as np

n    = int(sys.argv[1]) if len(sys.argv) > 1 else 3
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7

W, H          = 10002, 7551   # Slide_Image.tif
UM_PER_PX     = 4.85
ROT_DEG       = 0.42
OFFSET_UM     = np.array([2150.0, 11400.0])
NOISE_UM      = 1.5           # >0.05 floor, <<25 limit

rng = np.random.default_rng(seed)
th  = math.radians(ROT_DEG)
R   = np.array([[math.cos(th), -math.sin(th)],
                [math.sin(th),  math.cos(th)]])

# Spread the points wide and off-axis so min-fiducial-spread passes.
m = 0.12
base = np.array([[m, m], [1 - m, m * 1.4], [m * 1.6, 1 - m],
                 [1 - m, 1 - m], [0.5, 0.28], [0.28, 0.62],
                 [0.72, 0.55], [0.5, 0.85], [0.16, 0.42], [0.84, 0.2],
                 [0.38, 0.14], [0.62, 0.9], [0.9, 0.7], [0.1, 0.8],
                 [0.45, 0.5]])[:n]
src = base * np.array([W, H])
src += rng.uniform(-40, 40, src.shape)          # not on a perfect grid

dst = (UM_PER_PX * src @ R.T) + OFFSET_UM
dst += rng.normal(0, NOISE_UM, dst.shape)       # the bit that matters

print("fiducials:")
for (xp, yp), (xu, yu) in zip(src, dst):
    print(f"  - x_px: {xp:.2f}")
    print(f"    y_px: {yp:.2f}")
    print(f"    x_um: {xu:.2f}")
    print(f"    y_um: {yu:.2f}")
