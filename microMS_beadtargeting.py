#!/usr/bin/env python3
"""
microMS_beadtargeting.py
========================

Image-guided MALDI-MSI targeting of SPPS resin beads on ITO slides,
for a Bruker timsTOF fleX.

Every tunable parameter lives in the CONFIG dict near the top of
this file.

The workflow ordering, the point-based similarity registration, the
nearest-neighbour distance filter and the fiducial click-training
interaction all follow microMS:

    Comi TJ, Neumann EK, Do TD, Sweedler JV. microMS: A Python
    Platform for Image-Guided Mass Spectrometry Profiling. J. Am. Soc.
    Mass Spectrom. 2017, 28(9), 1919-1928.
    DOI 10.1007/s13361-017-1704-1

Algorithms and interactions here are reimplemented independently. Only
file-format constants (the .xeo header/footer) are reproduced so that
files interoperate; they are marked FORMAT SPEC. microMS carries an
Illinois copyright with no explicit licence, so nothing else is copied.

Usage
-----
    python microMS_beadtargeting.py doctor    # environment check
    python microMS_beadtargeting.py convert   # image -> TIFF
    python microMS_beadtargeting.py pick      # click fiducials -> saved here
    python microMS_beadtargeting.py select    # bead manual selection
    python microMS_beadtargeting.py check     # registration quality only
    python microMS_beadtargeting.py review    # show planned shots, no export
    python microMS_beadtargeting.py run       # detect, filter, shoot, export
    python microMS_beadtargeting.py selftest  # synthetic end-to-end test

Add -v (or --verbose) to any command for step-by-step tracing with
timings. Start with 'doctor' if something will not run.
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(__file__).resolve()

XEO_MAX_POSITIONS = 400          # autoXecute per-file cap

VERSION = "1.1"


# =====================================================================
# TRACING
#
# Everything is flushed immediately. Windows cmd buffers stdout when
# it is not a tty, and a crash mid-run then loses exactly the lines
# that would have told you where it stopped.
# =====================================================================

VERBOSE = False
_T0 = None


def say(*a) -> None:
    """Always printed."""
    print(*a, flush=True)


def log(*a) -> None:
    """Printed only with -v. Prefixed with elapsed seconds."""
    if not VERBOSE:
        return
    import time
    global _T0
    if _T0 is None:
        _T0 = time.time()
    print(f"  [{time.time() - _T0:7.2f}s]", *a, flush=True)


def banner(cmd: str) -> None:
    import platform
    say(f"microMS_beadtargeting {VERSION}   command: {cmd}")
    if VERBOSE:
        log(f"python      {platform.python_version()} on {platform.system()}")
        log(f"script dir  {HERE}")
        log(f"config      {CONFIG_PATH}"
            f"{'' if CONFIG_PATH.exists() else '   MISSING'}")
        for mod in ("numpy", "scipy", "cv2", "matplotlib"):
            try:
                m = __import__(mod)
                log(f"{mod:11s} {getattr(m, '__version__', '?')}")
            except ImportError:
                log(f"{mod:11s} NOT INSTALLED")


# =====================================================================
# CONFIG
# =====================================================================

# Filled in by:  python microMS_beadtargeting.py pick
# x_px/y_px = pixel in the scan.  x_um/y_um = stage reading in microns.
#
# Do not reuse these across sessions once the slide has been remounted:
# repositioning shows up as a systematic error at every target.
FIDUCIALS = [
    {"x_px": 263.05, "y_px": 252.85, "x_um": 18601.50, "y_um": -20310.80},
    {"x_px": 9655.71, "y_px": 268.03, "x_um": 86083.10, "y_um": -20161.00},
    {"x_px": 249.13, "y_px": 7362.01, "x_um": 18646.70, "y_um": -69830.80},
    {"x_px": 9628.05, "y_px": 7371.59, "x_um": 86124.70, "y_um": -69700.20},
]


# Every tunable lives in CONFIG. Edit it directly; nothing else in the
# file needs changing to alter geometry, filtering or output.
#
# Distances are MICRONS unless the key says otherwise.
#
# Values that must be measured on the instrument are None. Nothing
# guesses them -- the writer refuses instead, because a plausible wrong
# constant produces a file that loads cleanly and fires in the wrong
# place.

CONFIG = {

    # ---- input -------------------------------------------------------
    "input": {
        # Scan of the MATRIX-COATED slide. TIFF, not JPEG. Fiducials
        # must be visible in it.
        "scan": "slide01.tif",
        # Optional bead list (x_px,y_px,diameter_px) instead of
        # detecting. None to detect.
        "beads": None,
    },

    # ---- bead geometry -----------------------------------------------
    "bead-diameter": 90,
    "bead-diameter-tolerance": 0.35,

    # "edge" placement scales shot distance with the MEASURED radius,
    # so a mis-measured bead is placed wrongly and silently.
    #
    # MUST be tighter than bead-diameter-tolerance, or nothing can be
    # both accepted and suspect and the check is unreachable. A bead at
    # the edge of the accepted band -- 122 um measured on a 90 um bead
    # -- still gets its shots 16 um too far out.
    "suspect-diameter-tolerance": 0.20,

    # Isolation filter, centre to centre, run against EVERY detected
    # object including debris. A bead beside a speck of dust is not
    # isolated, and shape-filtering first would delete the dust and let
    # the bead falsely pass.
    "min-bead-separation": 150,

    # Target localization error: the distance between where a shot was
    # requested and where the laser actually fired.
    #
    # Comi et al. 2017 give two rules that follow from it:
    #
    #   probe radius    >= target localization error
    #   distance filter >  target localization error + probe radius
    #
    # They measured 38.3 +/- 3.9 um on a 2017 ultrafleXtreme with a
    # ~100 um laser footprint. That is NOT this instrument: the fleX
    # has smartbeam 3D targeting to 10 um, and microGRID to 5 um, so
    # its real error is likely far smaller.
    #
    # None means the check is skipped. Measure yours with the burn-mark
    # test -- fire a target list on a sacrificial slide, rescan, and
    # measure the offset from the intended positions -- then put that
    # number here and the two rules start being checked.
    "target-localization-error-um": None,

    # The paper recommends at least 12 fiducials; error falls as
    # 1/sqrt(n), and the fiducial training set was the ONLY factor that
    # significantly affected accuracy in their ANOVA.
    "recommended-fiducials": 12,

    # ---- shot pattern ------------------------------------------------
    # Counter-clockwise from +x (image right).
    "laser-shot-angles": [0, 90, 180, 270],

    "shot-placement": {
        # "edge"   each bead's own measured radius + edge-offset
        # "center" fixed laser-distance from every bead centre
        #
        # Measured diameter is threshold-dependent and not yet reliable
        # on these slides, so "center" ignores it entirely.
        "distance-reference": "edge",
        "edge-offset": 15,
        "laser-distance": 60,
        # "edge" only: clamp so one bad measurement cannot fling shots
        # across the slide.
        "min-radius": 25,
        "max-radius": 70,

        # Shot count per bead.
        #
        # False   use laser-shot-angles exactly as listed.
        # True    microMS's circularPackPoints: the count follows the
        #         bead radius, so a large bead gets more shots while
        #         shot-to-shot spacing stays above spot-spacing. A
        #         small bead keeps min-spots.
        #
        # max-spots == min-spots reproduces a fixed count.
        "dynamic-spots": True,
        "min-spots": 4,
        "max-spots": 12,
        "spot-spacing": 60,
        "rotation-offset-deg": 0.0,
    },

    # Laser footprint. Affects ONLY the crater-overlap check; it moves
    # no shot. UNVERIFIED -- confirm with the instrument operator.
    "focal-spot-um": 10,
    "beam-scan": True,
    "beam-scan-um": 20,

    # Drop a shot whose crater would overlap the bead itself or a
    # neighbour. Genuine overlap only, never a cosmetic margin.
    "enforce-bead-clearance": True,

    # ---- registration -------------------------------------------------
    # Similarity, not affine: an affine fit through exactly 3 fiducials
    # is exactly determined and reports a meaningless zero residual.
    # What the fiducial x_um / y_um values are.
    #
    # "stage"  raw motor microns, as read off the instrument. This is
    #          what the measured MTP calibration below expects, and it
    #          is the timsTOF fleX path.
    #
    # "plate"  plate units of 10 um. Only for a workflow that enters
    #          fiducials as displayed plate positions.
    "fiducial-units": "stage",

    "allow-reflection": True,
    "max-fiducial-residual-um": 25,

    # ---- detection ----------------------------------------------------
    "detection": {
        # Beads are usually darker than matrix, and detection
        # thresholds for bright objects, so invert.
        "invert": True,

        # "flatfield" subtracts a large-kernel median then takes
        #             connected components; handles the illumination
        #             and matrix drift of a real slide.
        # "blob"      OpenCV SimpleBlobDetector; assumes a flat
        #             background.
        "method": "flatfield",

        # Restrict detection to one slide, in PIXELS, e.g.
        #   {"x0": 800, "y0": 3450, "x1": 6900, "y1": 5500}
        # None searches the whole image, and dark mounting hardware
        # then inverts to bright and floods the object list.
        # MUST exclude the slide edges and adapter clamps: inverted,
        # they become huge bright blobs and swamp the object list.
        "roi": None,

        "min-circularity": 0.70,
        # blob method only; flatfield uses min-solidity below.
        "min-convexity": 0.80,

        # Clump screening. Touching beads merge into ONE component, so
        # a clump arrives as a single object with nothing near it and
        # passes isolation cleanly. A clump is MARKED, not deleted, so
        # nearby singles still fail isolation against it.
        "screen-clumps": True,
        "max-aspect-ratio": 1.8,
        "min-solidity": 0.90,
        "clump-core-fraction": 0.62,

        # flatfield only
        "background-kernel-px": 101,
        "threshold": 6,
        # blob only
        "threshold-step": 10,

        # Loose pre-filter in pixels, before the micron size filter.
        # Generous on purpose: the isolation filter needs to see debris.
        "min-diameter-px": 4,
        "max-diameter-px": 30,
    },

    # ---- output --------------------------------------------------------
    "output": {
        "prefix": "targets",
        "serpentine-order": True,
        "write-xeo": True,
        "overlay": True,
        "overlay-show": False,
        # 'review' opens a window showing the planned shots; set False
        # to save the picture only (headless / remote sessions).
        "review-show": True,
        "zoom": True,
        "zoom-window-px": 700,
        "zoom-scale": 3,

        # The .xeo holds coordinates; the .run is the ordered list of
        # position NAMES autoXecute executes, resolving them through
        # geometry -> <stem>.xeo. Both are written together and their
        # names must match exactly.
        # flexImaging target list in stage coordinates. Needs no MTP
        # calibration -- the fiducial registration is enough, and the
        # fiducial rows let flexImaging register the slide itself.
        "write-flex-txt": True,
        "flex-region": "01",
        "flex-negate-y": True,

        "write-run": True,
        # microMS's own convention. Its loadXEO parses this back into
        # pixel positions, so a file written here can be reopened in
        # microMS. Change it only if something downstream needs a
        # different scheme.
        # None picks the convention matching fiducial-units:
        # R<region>X<x>Y<y> in plate units, as the instrument
        # names its own positions.
        "position-name": None,
        "flip-y": False,
        "region": 0,
        "chip": 0,

        # Copied into the .run header; everything else defaults to the
        # reference file. "type" is FastImaging there, which rastered
        # tissue -- we fire discrete positions, so confirm it.
        "run": {
            "acqMethod": "D:\\Methods\\your_method.m",
            "directory": "D:\\Data\\beads",
            "sampleName": "beadrun",
            "type": "FastImaging",
        },
    },

    # Manual overrides from the selection window, matched by pixel
    # position because detection indices shift when parameters change.
    "manual-selection": {"match-radius-px": 12},

    # ---- MTP calibration -------------------------------------------------
    # UnitCoord in a .xeo is a SIGNED plate fraction, about +/-0.73 in
    # X and +/-0.55 in Y, measured from the plate centre. The header's
    # alpha/beta give the scale: 51.75 mm per unit, both axes.
    #
    # The fractions of the named MTP grid positions are fixed plate
    # geometry and are built in (MTP_MAP_X / MTP_MAP_Y). What must be
    # measured on YOUR instrument is the STAGE COORDINATE of two or
    # more of those named positions -- the same thing microMS keeps in
    # <mapper>Coords.txt. Its shipped ultrafleXtreme file reads:
    #
    #     C20  -23215  -13605
    #     C5   -90705  -13715
    #     G20  -23190  -31610
    #     G5   -90680  -31715
    #
    # Either point this at such a file:
    #
    # ---- MTP calibration ------------------------------------------
    # UnitCoord in a .xeo is a SIGNED plate fraction: about +/-0.729 in
    # X and +/-0.551 in Y, measured from the plate centre. The header's
    # alpha/beta give the scale, 51.75 mm per unit on both axes.
    #
    # The fractions of the named MTP grid positions are fixed plate
    # geometry and are built in (MTP_MAP_X / MTP_MAP_Y). Only the STAGE
    # COORDINATES of those positions are instrument specific.
    #
    # MEASURED on the timsTOF fleX: the four corners of the two-slide
    # array. Converted to UnitCoord about their own centre they land
    # within 0.0012 of C5/C20/N5/N20 -- about 60 um on a 75 mm plate --
    # so these corners ARE the MTP grid corners, not an approximation:
    #
    #   top left     -0.652415 +0.477100    C5  -0.652174 +0.478261
    #   top right    +0.651577 +0.479994    C20 +0.652174 +0.478261
    #   bottom left  -0.651542 -0.479809    N5  -0.652174 -0.478261
    #   bottom right +0.652381 -0.477285    N20 +0.652174 -0.478261
    #
    # A microMS <mapper>Coords.txt path works too:
    #     "mtp_calibration": "flexCoords.txt",
    "mtp_calibration": [
        {"name": "C5",  "x_um": 18601.5, "y_um": -20310.8},
        {"name": "C20", "x_um": 86083.1, "y_um": -20161.0},
        {"name": "N5",  "x_um": 18646.7, "y_um": -69830.8},
        {"name": "N20", "x_um": 86124.7, "y_um": -69700.2},
    ],
}


def load_config(path=None) -> dict:
    """Return a validated working copy of CONFIG."""
    import copy
    cfg = copy.deepcopy(CONFIG)
    cfg["fiducials"] = [dict(f) for f in FIDUCIALS]

    if not cfg["laser-shot-angles"]:
        sys.exit("laser-shot-angles is empty; nothing to fire.")

    sus = float(cfg.get("suspect-diameter-tolerance", 0.2))
    acc = float(cfg.get("bead-diameter-tolerance", 0.35))
    if sus >= acc:
        sys.exit(f"suspect-diameter-tolerance ({sus}) must be smaller than "
                 f"bead-diameter-tolerance ({acc}); otherwise no accepted "
                 f"bead can ever be flagged and the check is unreachable.")

    fu = cfg.get("fiducial-units", "stage")
    if fu not in ("stage", "plate"):
        sys.exit(f"fiducial-units must be 'stage' or 'plate', got {fu!r}")

    ref = cfg["shot-placement"].get("distance-reference")
    if ref not in ("edge", "center"):
        sys.exit(f"shot-placement['distance-reference'] must be "
                 f"'edge' or 'center', got {ref!r}")

    log(f"config OK: {len(cfg['fiducials'])} fiducials, "
        f"{len(cfg['mtp_calibration'])} MTP positions, reference={ref}")
    return cfg


def save_fiducials(fids: list[dict], path: Path = None) -> None:
    """
    Rewrite the FIDUCIALS block in this file.

    Fiducials are the one setting produced by clicking rather than
    typing, so `pick` writes them back into the source. Everything
    around the block is left untouched.
    """
    path = path or Path(__file__).resolve()
    lines = path.read_text().splitlines()

    start = next((i for i, ln in enumerate(lines)
                  if ln.startswith("FIDUCIALS = [")), None)
    if start is None:
        sys.exit(f"Could not find the FIDUCIALS block in {path.name}")
    end = next((i for i, ln in enumerate(lines)
                if i > start and ln.startswith("]")), None)
    if end is None:
        sys.exit(f"FIDUCIALS block in {path.name} is not closed")

    block = ["FIDUCIALS = ["]
    for f in fids:
        block.append('    {"x_px": %.2f, "y_px": %.2f, '
                     '"x_um": %.2f, "y_um": %.2f},'
                     % (f["x_px"], f["y_px"], f["x_um"], f["y_um"]))
    block.append("]")

    path.write_text("\n".join(lines[:start] + block + lines[end + 1:]) + "\n")


def footprint_um(cfg: dict) -> float:
    """Diameter of the ablated crater."""
    spot = float(cfg.get("focal-spot-um", 10))
    if cfg.get("beam-scan", False):
        spot += float(cfg.get("beam-scan-um", 0))
    return spot


# =====================================================================
# REGISTRATION
# similarity: uniform scale + rotation [+ reflection] + translation
# =====================================================================

class Transform:
    """Maps scan pixels to stage microns."""

    def __init__(self, scale: float, R: np.ndarray, t: np.ndarray):
        self.scale, self.R, self.t = float(scale), R, t

    def px_to_um(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, float))
        return (self.scale * p @ self.R.T) + self.t

    def um_to_px(self, pts) -> np.ndarray:
        p = np.atleast_2d(np.asarray(pts, float))
        return (p - self.t) @ self.R / self.scale

    @property
    def um_per_px(self) -> float:
        return self.scale

    @property
    def rotation_deg(self) -> float:
        return math.degrees(math.atan2(self.R[1, 0], self.R[0, 0]))

    @property
    def reflected(self) -> bool:
        return bool(np.linalg.det(self.R) < 0)


def fit_similarity(src: np.ndarray, dst: np.ndarray,
                   allow_reflection: bool = True) -> Transform:
    """
    Umeyama least-squares similarity fit. 4 DOF, 5 with reflection.

    Deliberately NOT affine: an affine fit through exactly 3 points is
    exactly determined and reports a zero residual that tells you
    nothing about registration quality.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError("Need at least 3 matched point pairs.")

    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d

    C = D.T @ S / len(src)
    U, sv, Vt = np.linalg.svd(C)

    F = np.eye(2)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        F[1, 1] = -1.0

    R = U @ F @ Vt
    var_s = (S ** 2).sum() / len(src)
    scale = float(np.trace(np.diag(sv) @ F) / var_s) if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    return Transform(scale, R, t)


def residuals(T: Transform, src, dst) -> np.ndarray:
    pred = T.px_to_um(src)
    return np.linalg.norm(pred - np.asarray(dst, float), axis=1)


def loo_residuals(src, dst, allow_reflection=True) -> np.ndarray | None:
    """
    Leave-one-out cross validation. Needs >=4 points, since dropping
    one from 3 leaves too few to fit. This is the honest error
    estimate; the in-sample residual always flatters the fit.
    """
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    n = len(src)
    if n < 4:
        return None
    out = np.zeros(n)
    for i in range(n):
        keep = np.arange(n) != i
        Ti = fit_similarity(src[keep], dst[keep], allow_reflection)
        out[i] = np.linalg.norm(Ti.px_to_um(src[i])[0] - dst[i])
    return out


def to_microns(T: Transform, cfg: dict) -> Transform:
    """
    Rescale a fitted transform so it outputs MICRONS.

    With fiducial-units "plate" the fit lands in plate units of 10 um.
    Everything downstream -- min-bead-separation, bead diameter, shot
    distance, crater overlap -- is in microns, so the scaling happens
    once here and the export converts back. Filtering in plate units
    made every bead fail isolation.
    """
    if cfg.get("fiducial-units", "stage") != "plate":
        return T
    k = PLATE_UNIT_UM
    return Transform(T.scale * k, T.R, T.t * k)


def transform_from_config(cfg: dict) -> Transform:
    fids = cfg["fiducials"]
    if len(fids) < 3:
        sys.exit("At least 3 fiducials are required. "
                 "Run:  python microMS_beadtargeting.py pick")
    src = np.array([[f["x_px"], f["y_px"]] for f in fids], float)
    dst = np.array([[f["x_um"], f["y_um"]] for f in fids], float)
    return fit_similarity(src, dst, cfg.get("allow-reflection", True))


def check_fiducial_geometry(src: np.ndarray, tol: float = 0.05) -> list[str]:
    """
    Catch fiducial layouts that fit perfectly and register badly.

    A least-squares fit reports the residual of the points it was
    given, so a degenerate arrangement produces RMS 0 and looks
    ideal. Duplicated marks collapse the fit to scale 1.0; collinear
    marks leave the direction perpendicular to the line
    unconstrained, so any error there is invisible and uncorrected.

    The guide's advice is to surround the target area with fiducials.
    """
    warn = []
    n = len(src)

    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(src[i] - src[j]) < 1.0:
                warn.append(f"fiducials {i} and {j} are at the same pixel "
                            f"-- the fit collapses and reports RMS 0")

    if n >= 3:
        centred = src - src.mean(0)
        sv = np.linalg.svd(centred, compute_uv=False)
        ratio = sv[1] / sv[0] if sv[0] > 0 else 0.0
        if ratio < tol:
            warn.append(
                f"fiducials are nearly collinear (spread ratio "
                f"{ratio:.4f}). The residual is meaningless "
                f"perpendicular to that line; spread them across the "
                f"slide instead")
    return warn


def report_registration(cfg: dict) -> Transform:
    fids = cfg["fiducials"]
    src = np.array([[f["x_px"], f["y_px"]] for f in fids], float)
    dst = np.array([[f["x_um"], f["y_um"]] for f in fids], float)

    T = transform_from_config(cfg)
    res = residuals(T, src, dst)
    limit = float(cfg.get("max-fiducial-residual-um", 25))

    print(f"\nFiducials      : {len(fids)}")
    print(f"Scale          : {T.um_per_px:.4f} um/px")
    print(f"Rotation       : {T.rotation_deg:+.3f} deg")
    print(f"Reflected      : {T.reflected}")
    print(f"RMS residual   : {np.sqrt((res ** 2).mean()):.2f} um")
    print(f"Max residual   : {res.max():.2f} um")

    print("\n  #      x_px      y_px         x_um         y_um   resid_um")
    for i, (f, r) in enumerate(zip(fids, res)):
        flag = "  <-- OVER LIMIT" if r > limit else ""
        print(f" {i:2d} {f['x_px']:9.1f} {f['y_px']:9.1f} "
              f"{f['x_um']:12.1f} {f['y_um']:12.1f} {r:10.2f}{flag}")

    rec = int(cfg.get("recommended-fiducials", 12))
    if len(fids) < rec:
        factor = (rec / len(fids)) ** 0.5
        print(f"\nNOTE: {len(fids)} fiducials. Comi et al. 2017 recommend "
              f"at least {rec};\n  localization error falls as 1/sqrt(n), so "
              f"this is roughly {factor:.1f}x worse\n  than {rec} would be. "
              f"The fiducial set was the only factor that\n  significantly "
              f"affected accuracy in their ANOVA.")

    loo = loo_residuals(src, dst, cfg.get("allow-reflection", True))
    if loo is None:
        print("\nLeave-one-out  : needs >=4 fiducials (have "
              f"{len(fids)}). In-sample residual only -- treat it as a "
              "lower bound on true error.")
    else:
        print(f"\nLeave-one-out  : RMS {np.sqrt((loo ** 2).mean()):.2f} um, "
              f"max {loo.max():.2f} um")

    for w in check_fiducial_geometry(src):
        print(f"\nWARNING: {w}")

    if res.max() > limit:
        print(f"\nWARNING: a fiducial exceeds max-fiducial-residual-um "
              f"({limit} um). Check for a mistyped stage coordinate or a "
              f"misclicked mark before trusting any targets.")
    return T


# =====================================================================
# DETECTION
# =====================================================================

@dataclass
class Bead:
    x_px: float
    y_px: float
    diameter_px: float
    x_um: float = 0.0
    y_um: float = 0.0
    diameter_um: float = 0.0
    nn_um: float = float("inf")
    clumped: bool = False
    accepted: bool = False
    reject_reason: str = ""
    reject_category: str = ""
    manual: str = ""
    shots: list = field(default_factory=list)


def _roi_slice(img, cfg):
    """Return (cropped image, x_offset, y_offset)."""
    roi = cfg["detection"].get("roi")
    if not roi:
        return img, 0, 0
    x0, y0 = int(roi["x0"]), int(roi["y0"])
    x1 = int(roi["x1"]) if roi.get("x1") else img.shape[1]
    y1 = int(roi["y1"]) if roi.get("y1") else img.shape[0]
    return img[y0:y1, x0:x1], x0, y0


def _detect_blobdetector(g, d):
    """SimpleBlobDetector. Closest to microMS's global-threshold sweep."""
    import cv2
    p = cv2.SimpleBlobDetector_Params()
    p.thresholdStep = float(d.get("threshold-step", 10))
    p.minThreshold, p.maxThreshold = 10, 245

    # OpenCV defaults filterByColor=True with blobColor=0, i.e. DARK
    # blobs. After inverting, beads are bright, so the default silently
    # rejects every bead and returns only dark artefacts. Turn it off.
    p.filterByColor = False

    p.filterByArea = True
    p.minArea = math.pi * (float(d.get("min-diameter-px", 3)) / 2) ** 2
    p.maxArea = math.pi * (float(d.get("max-diameter-px", 60)) / 2) ** 2
    p.filterByCircularity = True
    p.minCircularity = float(d.get("min-circularity", 0.70))
    p.filterByConvexity = True
    p.minConvexity = float(d.get("min-convexity", 0.80))
    p.filterByInertia = False

    kps = cv2.SimpleBlobDetector_create(p).detect(g)
    # SimpleBlobDetector gives no component mask, so clumps cannot be
    # screened directly here -- only approximated by minConvexity.
    return [(k.pt[0], k.pt[1], k.size, False) for k in kps]


def _count_cores(mask, d) -> int:
    """
    Count bead cores inside one connected component.

    Touching beads merge into a single component. Distance-transform
    peaks survive that merge: two touching circles give two maxima
    separated by a saddle at the contact point, so counting peaks
    counts beads regardless of how the outline looks.
    """
    import cv2
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    peak = float(dist.max())
    if peak <= 0:
        return 1

    # A core must be a locally maximal region at least this deep. The
    # saddle between two touching equal circles sits near 0 depth, so
    # a moderate fraction of the peak separates them cleanly.
    frac = float(d.get("clump-core-fraction", 0.62))
    _, cores = cv2.threshold(dist, frac * peak, 255, cv2.THRESH_BINARY)
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        cores.astype(np.uint8), 8)

    # Ignore specks thrown off by a ragged edge.
    min_core = max(2, int(0.03 * mask.sum() / 255))
    return sum(1 for i in range(1, n)
               if stats[i, cv2.CC_STAT_AREA] >= min_core)


def _detect_flatfield(g, d):
    """
    Flat-field subtraction then connected components.

    A global threshold sweep assumes the background is roughly uniform.
    On a matrix-coated slide it is not: illumination and matrix
    thickness both drift across the field, and bead-to-background
    contrast can be under 10 grey levels. Subtracting a large-kernel
    median removes the drift and leaves the beads.

    Returns (x, y, diameter_px, is_clump). Clumps are RETURNED, not
    discarded, so the isolation filter still counts them as
    neighbouring objects for the singles around them.
    """
    import cv2
    k = int(d.get("background-kernel-px", 101))
    if k % 2 == 0:
        k += 1
    sub = cv2.subtract(g, cv2.medianBlur(g, k))

    thr = int(d.get("threshold", 6))
    _, bw = cv2.threshold(sub, thr, 255, cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    dmin = float(d.get("min-diameter-px", 3))
    dmax = float(d.get("max-diameter-px", 60))
    a_min, a_max = math.pi * (dmin / 2) ** 2, math.pi * (dmax / 2) ** 2
    min_circ = float(d.get("min-circularity", 0.70))
    max_ar = float(d.get("max-aspect-ratio", 1.8))
    min_sol = float(d.get("min-solidity", 0.90))
    screen = d.get("screen-clumps", True)

    n, lab, stats, cent = cv2.connectedComponentsWithStats(bw, 8)
    out = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if not (a_min <= a <= a_max):
            continue

        # Crop to this component's bounding box before building the
        # mask. Working on the full ROI per component made detection
        # O(n * image) and cost minutes on an 8000 px scan.
        bx, by = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw_, bh_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        pad = 2
        y0 = max(by - pad, 0)
        x0 = max(bx - pad, 0)
        sub_lab = lab[y0:by + bh_ + pad, x0:bx + bw_ + pad]
        mask = (sub_lab == i).astype(np.uint8) * 255
        cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                 cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        c = cs[0]
        per = cv2.arcLength(c, True)
        if per <= 0:
            continue

        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        aspect = max(w, h) / max(min(w, h), 1)
        circ = 4 * math.pi * a / per ** 2
        hull_a = cv2.contourArea(cv2.convexHull(c))
        solidity = a / hull_a if hull_a > 0 else 0.0

        clump = False
        if screen:
            # Any one of these is enough. A touching pair shows up as
            # an elongated outline, a notched hull, or two cores --
            # which of the three fires depends on how deep the overlap
            # is, so all three are checked.
            if aspect > max_ar or solidity < min_sol or circ < min_circ:
                clump = True
            elif _count_cores(mask, d) > 1:
                clump = True
        else:
            if aspect > max_ar or circ < min_circ:
                continue

        out.append((cent[i][0], cent[i][1],
                    2 * math.sqrt(a / math.pi), clump))
    return out


def detect_blobs(scan_path: Path, cfg: dict) -> list[Bead]:
    """
    Loose detection. Intentionally permissive: the isolation filter
    downstream must be able to see debris, so nothing is rejected here
    beyond the coarse settings in the config.
    """
    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required for detection. "
                 "pip install opencv-python")

    if not scan_path.exists():
        sys.exit(f"Scan not found: {scan_path}")
    log(f"reading {scan_path.name} "
        f"({scan_path.stat().st_size / 1e6:.1f} MB)")
    img = cv2.imread(str(scan_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        sys.exit(f"Could not read image: {scan_path}\n"
                 f"Input must be TIFF, not JPEG. Convert with:\n"
                 f"  python microMS_beadtargeting.py convert <file>")
    log(f"image {img.shape[1]}x{img.shape[0]} {img.dtype}, "
        f"grey range {img.min()}-{img.max()}")

    d = cfg["detection"]
    sub, ox, oy = _roi_slice(img, cfg)
    if (ox, oy) != (0, 0) or sub.shape != img.shape:
        log(f"ROI {sub.shape[1]}x{sub.shape[0]} at offset ({ox}, {oy})")
    else:
        log("no ROI set, searching the whole image")

    g = cv2.bitwise_not(sub) if d.get("invert", True) else sub

    method = d.get("method", "flatfield")
    log(f"invert={d.get('invert', True)}  method={method}")
    if method == "flatfield":
        log(f"flatfield: kernel {d.get('background-kernel-px', 101)} px, "
            f"threshold {d.get('threshold', 6)} grey levels")
        found = _detect_flatfield(g, d)
        if VERBOSE:
            log(f"{sum(1 for f in found if f[3])} of {len(found)} "
                f"components flagged as clumps")
    elif method == "blob":
        found = _detect_blobdetector(g, d)
    else:
        sys.exit(f"detection.method must be 'flatfield' or 'blob', "
                 f"got {method!r}")

    log(f"detection returned {len(found)} objects")
    if not found:
        say("\nNothing detected. Things to check, in order:\n"
            "  - detection.invert  (beads darker than matrix -> true)\n"
            "  - detection.roi     (is it over the right slide?)\n"
            "  - detection.threshold (lower it; 6 is typical)\n"
            "  - min/max-diameter-px against your um/px scale")
    return [Bead(x + ox, y + oy, size, clumped=cl)
            for x, y, size, cl in found]


def load_beads_csv(path: Path) -> list[Bead]:
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append(Bead(float(row["x_px"]), float(row["y_px"]),
                            float(row["diameter_px"]),
                            clumped=str(row.get("clumped", "")).strip().lower()
                            in ("1", "true", "yes")))
    return out


# =====================================================================
# FILTERS  (order follows microMS and matters -- see below)
# =====================================================================

def to_stage(beads: list[Bead], T: Transform) -> None:
    if not beads:
        return
    px = np.array([[b.x_px, b.y_px] for b in beads], float)
    um = T.px_to_um(px)
    for b, (x, y) in zip(beads, um):
        b.x_um, b.y_um = float(x), float(y)
        b.diameter_um = b.diameter_px * T.um_per_px


def isolation_filter(beads: list[Bead], min_sep_um: float) -> None:
    """
    Nearest-neighbour distance filter, run against EVERY detected
    object including debris that will later fail the shape filter.

    Ordering is the point: shape-filtering first would delete the dust
    beside a bead, and the bead would then falsely pass isolation.
    """
    if len(beads) < 2:
        for b in beads:
            b.nn_um = float("inf")
        return

    # Same rule as microMS blobList.distanceFilter: a blob fails if
    # ANY neighbour is closer than the cutoff, and both members of a
    # too-close pair fail. Recording the nearest-neighbour distance
    # and comparing it to the cutoff is equivalent, and keeps the
    # number for the CSV and the histogram.
    pts = np.array([[b.x_um, b.y_um] for b in beads], float)
    dist, _ = cKDTree(pts).query(pts, k=2)
    for b, dd in zip(beads, dist[:, 1]):
        b.nn_um = float(dd)


def shape_filter(beads: list[Bead], cfg: dict) -> None:
    """Size window, applied after isolation. Sets accept/reject."""
    nominal = float(cfg["bead-diameter"])
    tol = float(cfg.get("bead-diameter-tolerance", 0.35))
    lo, hi = nominal * (1 - tol), nominal * (1 + tol)
    min_sep = float(cfg["min-bead-separation"])

    for b in beads:
        if b.clumped:
            # A clump is one connected component containing more than
            # one bead. Its centroid is not a bead centre and its
            # measured diameter is meaningless, so no shot pattern
            # placed on it can be trusted. It stays in the object list
            # so nearby singles still fail isolation against it.
            b.accepted, b.reject_reason = False, "clumped"
            b.reject_category = "clumped"
        elif b.nn_um < min_sep:
            b.accepted, b.reject_reason = False, (
                f"not isolated ({b.nn_um:.0f} < {min_sep:.0f} um)")
            b.reject_category = "not isolated"
        elif not (lo <= b.diameter_um <= hi):
            b.accepted, b.reject_reason = False, (
                f"size {b.diameter_um:.0f} um outside "
                f"{lo:.0f}-{hi:.0f} um")
            b.reject_category = ("undersize" if b.diameter_um < lo
                                 else "oversize")
        else:
            b.accepted, b.reject_reason = True, ""
            b.reject_category = ""


# =====================================================================
# SHOT PLACEMENT
# =====================================================================

def suspect_radius(bead: "Bead", cfg: dict) -> bool:
    """
    True when this bead's measured diameter is far enough from nominal
    that scaling its shot distance is not trustworthy.

    In "edge" mode shot distance is derived from the measured radius,
    so the clearance check compares that radius against itself and can
    never fail. A wrongly measured bead therefore produces a clean run
    with shots placed at the wrong distance -- 165 um measured on a
    90 um bead puts them 40 um out into plain matrix, off the halo,
    with nothing to show for it.

    "center" mode has the opposite behaviour: the same bead fails the
    crater-overlap check loudly. This flag restores that signal
    without giving up the scaling.
    """
    tol = float(cfg.get("suspect-diameter-tolerance", 0.4))
    nominal = float(cfg["bead-diameter"])
    return abs(bead.diameter_um - nominal) > tol * nominal


@dataclass
class Shot:
    bead_id: int
    angle_deg: float
    x_um: float
    y_um: float
    x_px: float = 0.0
    y_px: float = 0.0
    dropped: bool = False
    drop_reason: str = ""


def circular_pack(radius_um: float, cfg: dict) -> list[float]:
    """
    Angles for one bead, following microMS's
    blobList.circularPackPoints exactly.

        maxR = maxSpots * spacing / 2pi - offset
        minR = minSpots * spacing / 2pi - offset

        radius > maxR   ->  maxSpots
        radius < minR   ->  minSpots      (spacing ignored)
        otherwise       ->  floor(2pi * (radius + offset) / spacing)

    Targets are then equally spaced around the circumference. The
    effect is that a large bead gets more shots and a small one keeps
    the minimum, while shot-to-shot spacing never drops below
    `spacing`.

    Setting max-spots == min-spots reproduces a fixed count.
    """
    sp = cfg["shot-placement"]
    spacing = float(sp.get("spot-spacing", 60))
    offset = float(sp.get("edge-offset", 15))
    min_spots = int(sp.get("min-spots", 4))
    max_spots = max(int(sp.get("max-spots", 4)), min_spots)

    max_r = max_spots * spacing / (2 * math.pi) - offset
    min_r = min_spots * spacing / (2 * math.pi) - offset

    if radius_um > max_r:
        n = max_spots
    elif radius_um < min_r:
        n = min_spots
    else:
        n = int(math.floor(2 * math.pi * (radius_um + offset) / spacing))
        n = max(n, min_spots)

    rot = float(sp.get("rotation-offset-deg", 0.0))
    return [rot + 360.0 * k / n for k in range(n)]


def shot_radius(bead: Bead, cfg: dict) -> float:
    """Distance from bead centre to shot centre."""
    sp = cfg["shot-placement"]
    if sp["distance-reference"] == "center":
        return float(sp["laser-distance"])

    r = bead.diameter_um / 2.0
    r = min(max(r, float(sp.get("min-radius", 0))),
            float(sp.get("max-radius", 1e9)))
    return r + float(sp["edge-offset"])


def place_shots(beads: list[Bead], cfg: dict,
                T: "Transform | None" = None) -> list[Shot]:
    fixed_angles = [float(a) for a in cfg["laser-shot-angles"]]
    dynamic = cfg["shot-placement"].get("dynamic-spots", False)
    T_um_per_px = T.um_per_px if T is not None else 0.0
    crater = footprint_um(cfg)
    enforce = cfg.get("enforce-bead-clearance", True)

    all_pts = np.array([[b.x_um, b.y_um] for b in beads], float) \
        if beads else np.zeros((0, 2))
    tree = cKDTree(all_pts) if len(all_pts) else None

    shots: list[Shot] = []
    for i, b in enumerate(beads):
        if not b.accepted:
            continue
        R = shot_radius(b, cfg)
        # microMS sizes the ring from the bead; the fixed list is the
        # simpler alternative.
        angles = (circular_pack(b.diameter_um / 2.0, cfg) if dynamic
                  else fixed_angles)
        ring: list[Shot] = []
        for a in angles:
            rad = math.radians(a)
            # Pixel position too: microMS's PositionName convention
            # encodes it, which is how its loadXEO reads the file back.
            R_px = R / T_um_per_px if T_um_per_px else 0.0
            s = Shot(i, a,
                     b.x_um + R * math.cos(rad),
                     b.y_um + R * math.sin(rad),
                     b.x_px + R_px * math.cos(rad),
                     b.y_px + R_px * math.sin(rad))

            # No software travel-limit check. The stage enforces its
            # own limits in hardware; a guessed coordinate window here
            # silently discarded entire target lists.
            if enforce and (R - crater / 2.0) < (b.diameter_um / 2.0):
                s.dropped, s.drop_reason = True, "crater overlaps own bead"

            elif enforce and tree is not None:
                # nearest OTHER object, whether or not it was accepted
                # cKDTree pads missing neighbours with distance inf and
                # index == len(points), which is out of range. With a
                # single detected object k=2 always returns one such
                # pad, so this must be checked before indexing.
                for d, j in zip(*tree.query([s.x_um, s.y_um], k=2)):
                    if j == i or j >= len(beads) or not math.isfinite(d):
                        continue
                    edge = beads[j].diameter_um / 2.0
                    if d - crater / 2.0 < edge:
                        s.dropped = True
                        s.drop_reason = "crater overlaps neighbour"
                    break
            ring.append(s)

        # Genuine crater-crater overlap between shots on the same bead.
        # Nothing is trimmed for a cosmetic margin -- only real overlap.
        live = [s for s in ring if not s.dropped]
        for m in range(len(live)):
            for n in range(m + 1, len(live)):
                d = math.hypot(live[m].x_um - live[n].x_um,
                               live[m].y_um - live[n].y_um)
                if d < crater:
                    live[n].dropped = True
                    live[n].drop_reason = "crater overlaps adjacent shot"

        shots.extend(ring)
        b.shots = ring
    return shots


def serpentine(beads: list[Bead], shots: list[Shot],
               band_um: float = 1000.0) -> list[Shot]:
    """Boustrophedon ordering to shorten stage travel."""
    live = [s for s in shots if not s.dropped]
    if not live:
        return live
    order = {}
    for i, b in enumerate(beads):
        band = int(b.y_um // band_um)
        order[i] = (band, b.x_um if band % 2 == 0 else -b.x_um)
    return sorted(live, key=lambda s: (*order[s.bead_id], s.angle_deg))


# =====================================================================
# EXPORT -- CSV
# =====================================================================

def write_csv(path: Path, beads: list[Bead], shots: list[Shot]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "position", "bead_id", "angle_deg",
                    "x_um", "y_um", "bead_x_um", "bead_y_um",
                    "bead_diameter_um", "nn_um", "clumped"])
        for k, s in enumerate(shots):
            b = beads[s.bead_id]
            w.writerow([k, f"B{s.bead_id:04d}_A{int(s.angle_deg):03d}",
                        s.bead_id, f"{s.angle_deg:.1f}",
                        f"{s.x_um:.2f}", f"{s.y_um:.2f}",
                        f"{b.x_um:.2f}", f"{b.y_um:.2f}",
                        f"{b.diameter_um:.2f}", f"{b.nn_um:.2f}",
                        int(b.clumped)])


# =====================================================================
# EXPORT -- .xeo
#
# FORMAT SPEC. The header and footer below are the Bruker autoXecute
# geometry wrapper. They are reproduced because interoperability
# requires the exact strings; the 13-line header and 12-line footer
# are what make microMS's own lines[13:-12] position slice work.
#
# PlateTypeName is CONFIRMED. A real autoXecute run file from the
# target instrument (Imaging_Run.run, AutoExecute 7.6.6.0) carries
# baseGeometry="MTP Slide Adapter II" -- exactly the string below.
#
# The surrounding XML wrapper is still UNVERIFIED; no genuine .xeo has
# been seen. Diff one against this writer's output before acquiring.
# =====================================================================

# FORMAT SPEC, reproduced verbatim from microMS's brukerMapper.py so
# that files interoperate. This is the one place microMS source is
# copied; see ATTRIBUTION.md.
#
# The header is 12 lines, plus the <PlateSpots> line written per file,
# giving the 13 that microMS's own loadXEO skips. The footer is 12.

XEO_HEADER = [
    '<!-- $Revision: 1.5 $-->',
    '<PlateType>',
    '\t<GlobalParameters PlateTypeName="MTP Slide Adapter II" ProbeType="MTP"',
    '\t                  RowsNumber="100" ChipNumber="1" ChipsInRow="1"',
    '\t                  X_ChipOffsetSize="0" Y_ChipOffsetSize="0"',
    '\t                  HasDirectLabels="false" HasColRowLabels="true"',
    '\t                  HasNearNeighbourCalibrants="false"',
    '\t                  ProbeDiameterX="103.5" SampleDiameter="2"',
    '\t                  SamplePixelRadius="5" ZoomFactor="1"',
    '\t                  FirstCalibrant="TPX1" SecondCalibrant="TPX2" '
    'ThirdCalibrant="TPX3"',
    '\t                  />',
    '\t<MappingParameters mox="56.239998" moy="42.635009" sinphi="0.000000" '
    'cosphi="1.000000" alpha="51.750000" beta="51.750000" tansigma="0.000000"/>',
]

XEO_FOOTER = [
    '\t</PlateSpots>',
    '    \t<AutoTeachSpots>',
    '    \t\t<PlateSpot PositionIndex="0" PositionName="TPX1" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="0.550725"/>',
    '    \t\t<PlateSpot PositionIndex="1" PositionName="TPX2" '
    'UnitCoord_X="0.729469" UnitCoord_Y="0.550725"/>',
    '    \t\t<PlateSpot PositionIndex="2" PositionName="TPX3" '
    'UnitCoord_X="0.729469" UnitCoord_Y="0.057971"/>',
    '    \t\t<PlateSpot PositionIndex="3" PositionName="TPX4" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="0.057971"/>',
    '    \t\t<PlateSpot PositionIndex="4" PositionName="TPY1" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="-0.057971"/>',
    '    \t\t<PlateSpot PositionIndex="5" PositionName="TPY2" '
    'UnitCoord_X="0.729469" UnitCoord_Y="-0.057971"/>',
    '    \t\t<PlateSpot PositionIndex="6" PositionName="TPY3" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="-0.550725"/>',
    '    \t\t<PlateSpot PositionIndex="7" PositionName="TPY4" '
    'UnitCoord_X="0.729469" UnitCoord_Y="-0.550725"/>',
    '    \t</AutoTeachSpots>',
    '    </PlateType>',
]

# Named MTP grid positions and their UnitCoord fractions, from
# brukerMapper.py. FORMAT SPEC -- plate geometry, not a measurement.
# UnitCoord is a SIGNED fraction about the plate centre; the header's
# alpha and beta give the scale, 51.75 mm per unit on both axes.
MTP_MAP_Y = {'C': 0.478261, 'D': 0.391304, 'E': 0.304348, 'F': 0.217391,
             'G': 0.130435, 'J': -0.130435, 'K': -0.217391, 'L': -0.304348,
             'M': -0.391304, 'N': -0.478261}

MTP_MAP_X = {'5': -0.652174, '6': -0.565217, '7': -0.478261, '8': -0.391304,
             '9': -0.304348, '10': -0.217391, '11': -0.130435,
             '12': -0.043478, '13': 0.043478, '14': 0.130435,
             '15': 0.217391, '16': 0.304348, '17': 0.391304, '18': 0.478261,
             '19': 0.565217, '20': 0.652174}

MTP_UNIT_MM = 51.750     # header alpha / beta


# =====================================================================
# PLATE COORDINATES  (MTP Slide Adapter II)
#
# Derived entirely from constants already in this file plus the
# reference run file. Nothing here is measured or assumed.
#
#   brukerMapper's header declares alpha = beta = 51.750 mm per
#   UnitCoord, and its AutoTeachSpots put the teach points at
#   UnitCoord_X = +/-0.729469 and UnitCoord_Y = +/-0.550725.
#
#   2 x 0.729469 x 51.750 mm = 75.500 mm
#   2 x 0.550725 x 51.750 mm = 57.000 mm
#
# Exact round numbers: the adapter is 75.5 x 57.0 mm, i.e. 7550 x 5700
# units of 10 um. Every position in Dr Neumann's run file falls inside
# that box (X 1868-7162 of 7550, Y 1308-4801 of 5700), and the gap
# between its two slide bands converts to 26.23 mm -- the slide pitch
# of a two-slide adapter.
#
# So a position name R<region>X<x>Y<y> carries plate position directly,
# and UnitCoord follows from constants alone. No stage calibration and
# no MTP coordinate file are needed for this path.
# =====================================================================

PLATE_UNIT_UM = 10.0
TEACH_X = 0.729469          # brukerMapper AutoTeachSpots
TEACH_Y = 0.550725
PLATE_X_UNITS = 2 * TEACH_X * MTP_UNIT_MM * 1000 / PLATE_UNIT_UM   # 7550
PLATE_Y_UNITS = 2 * TEACH_Y * MTP_UNIT_MM * 1000 / PLATE_UNIT_UM   # 5700


def plate_to_unitcoord(x_units: float, y_units: float,
                       flip_y: bool = False) -> tuple[float, float]:
    """Plate units (as in a position name) -> UnitCoord for a .xeo."""
    ux = x_units / PLATE_X_UNITS * 2 * TEACH_X - TEACH_X
    uy = y_units / PLATE_Y_UNITS * 2 * TEACH_Y - TEACH_Y
    return ux, (-uy if flip_y else uy)


def unitcoord_to_plate(ux: float, uy: float,
                       flip_y: bool = False) -> tuple[float, float]:
    if flip_y:
        uy = -uy
    return ((ux + TEACH_X) / (2 * TEACH_X) * PLATE_X_UNITS,
            (uy + TEACH_Y) / (2 * TEACH_Y) * PLATE_Y_UNITS)


def mtp_name_to_unit(name: str) -> tuple[float, float] | None:
    """'C20' -> (UnitCoord_X, UnitCoord_Y). None if unparseable."""
    row, col = name[0].upper(), name[1:]
    if row in MTP_MAP_Y and col in MTP_MAP_X:
        return MTP_MAP_X[col], MTP_MAP_Y[row]
    return None


def fit_mtp(cfg: dict) -> Transform | None:
    """
    Stage microns -> UnitCoord, the signed plate fraction a .xeo uses.

    UnitCoord is NOT microns and NOT a 0-1 fraction. It runs about
    -0.73 to +0.73 in X and -0.55 to +0.55 in Y, measured from the
    plate centre, and the header's alpha/beta give the scale as
    51.75 mm per unit on both axes.

    The fractions of the named MTP grid positions are fixed plate
    geometry and are built in. What has to be measured on the
    instrument is the STAGE COORDINATE of two or more of those named
    positions -- exactly the file microMS keeps as
    <mapper>Coords.txt, e.g.

        C20  -23215  -13605
        C5   -90705  -13715
        G20  -23190  -31610
        G5   -90680  -31715

    Config entries mirror that:

        {"name": "C20", "x_um": -23215, "y_um": -13605}
    """
    cal = cfg.get("mtp_calibration") or []

    # A microMS <mapper>Coords.txt can be used directly. Same file
    # brukerMapper.loadStagePoints reads: name<TAB>x<TAB>y.
    if isinstance(cal, str):
        path = Path(cal)
        if not path.is_absolute():
            path = HERE / path
        if not path.exists():
            sys.exit(f"mtp_calibration file not found: {path}")
        rows = []
        for line in path.read_text().splitlines():
            t = line.replace("\t", " ").split()
            if len(t) >= 3:
                rows.append({"name": t[0], "x_um": float(t[1]),
                             "y_um": float(t[2])})
        log(f"read {len(rows)} MTP positions from {path.name}")
        cal = rows

    if len(cal) < 2:
        return None

    src, dst, bad = [], [], []
    for c in cal:
        unit = mtp_name_to_unit(str(c["name"]))
        if unit is None:
            bad.append(c["name"])
            continue
        src.append([float(c["x_um"]), float(c["y_um"])])
        dst.append(list(unit))

    if bad:
        print(f"\nWARNING: unrecognised MTP position name(s): {bad}. "
              f"Rows are C-G and J-N, columns 5-20.")
    if len(src) < 2:
        return None

    src, dst = np.array(src, float), np.array(dst, float)
    if len(src) == 2:
        # Two points fix scale, rotation and translation for a
        # similarity fit exactly. microMS ships four.
        M = fit_similarity(np.vstack([src, src.mean(0)]),
                           np.vstack([dst, dst.mean(0)]),
                           allow_reflection=True)
    else:
        M = fit_similarity(src, dst, allow_reflection=True)

    res = np.linalg.norm(M.px_to_um(src) - dst, axis=1)
    mm_per_unit = MTP_UNIT_MM
    print(f"\nMTP fit        : {len(src)} named positions, "
          f"max residual {res.max():.6f} unit "
          f"({res.max() * mm_per_unit * 1000:.0f} um)")
    print(f"  recovered scale {1 / M.um_per_px / 1000:.3f} mm per unit "
          f"(header declares {mm_per_unit:.3f})")
    if abs(1 / M.um_per_px / 1000 - mm_per_unit) > 0.5:
        print("  WARNING: recovered scale disagrees with the header. Check "
              "the measured\n  stage coordinates and the position names.")
    return M


def write_xeo(prefix: Path, shots: list[Shot], beads: list[Bead],
              cfg: dict, M: "Transform | None" = None) -> list[Path]:
    """
    Write .xeo files using microMS's own brukerMapper.writeXEO.

    Not a reimplementation: flex_mapper.flexMapper subclasses
    brukerMapper, so the header, footer, spot-line format, MTP grid
    fractions and the motor-to-plate-fraction map all come from
    microMS unchanged.

    brukerMapper.writeXEO does not split -- that is solarixMapper --
    so the 400-position cap is applied here.
    """
    cal = cfg.get("mtp_calibration") or []
    if isinstance(cal, str) or len(cal) < 2:
        cal = _calibration_rows(cfg)
    if len(cal) < 2:
        print("\nSKIPPED .xeo: mtp_calibration needs at least two named MTP\n"
              "  positions with their stage coordinates. The CSV is still "
              "written.")
        return []

    import flex_mapper

    coord_file = HERE / "flexCoords.txt"
    flex_mapper.write_coord_file(coord_file, cal)
    mapper = flex_mapper.flexMapper(str(coord_file))
    log(f"brukerMapper loaded, motor2MTP fitted from {len(cal)} positions")

    # Hand it the fiducial training set. Its own PBSR then reproduces
    # the pixel -> motor transform we already fitted and reported.
    for f in cfg["fiducials"]:
        mapper.addPoints((f["x_px"], f["y_px"]), (f["x_um"], f["y_um"]))
    mapper.PBSR()

    written = []
    chunks = [shots[i:i + XEO_MAX_POSITIONS]
              for i in range(0, len(shots), XEO_MAX_POSITIONS)] or [[]]
    for n, chunk in enumerate(chunks, start=1):
        blobs = [flex_mapper.make_blob(s.x_px, s.y_px) for s in chunk]
        out = prefix.with_name(f"{prefix.name}_{n:03d}.xeo")
        mapper.saveInstrumentFile(str(out), blobs)
        written.append(out)
    return written


def _calibration_rows(cfg: dict) -> list[dict]:
    """mtp_calibration as rows, whether inline or a Coords.txt path."""
    cal = cfg.get("mtp_calibration") or []
    if not isinstance(cal, str):
        return list(cal)
    path = Path(cal)
    if not path.is_absolute():
        path = HERE / path
    if not path.exists():
        sys.exit(f"mtp_calibration file not found: {path}")
    rows = []
    for line in path.read_text().splitlines():
        t = line.replace("\t", " ").split()
        if len(t) >= 3:
            rows.append({"name": t[0], "x_um": float(t[1]),
                         "y_um": float(t[2])})
    return rows


def read_xeo(path: Path) -> list[str]:
    """microMS's position slice: 13 header lines, 12 footer lines."""
    return path.read_text().splitlines()[13:-12]



# =====================================================================
# EXPORT -- flexImaging .txt
#
# FORMAT SPEC, from microMS's flexImagingSolarix.saveInstrumentFile.
#
# This path needs NO MTP calibration. The .xeo route has to convert
# stage microns into UnitCoord plate fractions, which requires the
# stage coordinates of named MTP positions. This one writes stage
# coordinates directly and lets flexImaging do its own registration
# from the fiducial rows written at the top of the file.
#
# Everything it needs is already known: the fiducial registration.
#
#   # X-pos Y-pos spot-name region
#   -56960 22660 fiducial0 01
#   -56901 22615 x1234_y5678 01
#
# Y is negated on write, as microMS does -- image y increases
# downward, stage y increases upward.
# =====================================================================

def write_flex_txt(path: Path, shots: list[Shot], cfg: dict,
                   T: Transform) -> Path:
    """Write a flexImaging target list in stage coordinates."""
    rows = ["# X-pos Y-pos spot-name region"]
    region = str(cfg["output"].get("flex-region", "01"))
    negate = -1.0 if cfg["output"].get("flex-negate-y", True) else 1.0

    # Fiducials first, so flexImaging can register the slide itself.
    for i, f in enumerate(cfg["fiducials"]):
        rows.append(f"{f['x_um']:.0f} {negate * f['y_um']:.0f} "
                    f"fiducial{i} {region}")

    for sh in shots:
        rows.append(f"{sh.x_um:.0f} {negate * sh.y_um:.0f} "
                    f"x{sh.x_px:.0f}_y{sh.y_px:.0f} {region}")

    path.write_text("\n".join(rows) + "\n")
    return path


def read_flex_txt(path: Path) -> list[tuple]:
    """(x, y, name, region) per row, header skipped."""
    out = []
    for line in path.read_text().splitlines()[1:]:
        t = line.split()
        if len(t) >= 4:
            out.append((float(t[0]), float(t[1]), t[2], t[3]))
    return out


# =====================================================================
# EXPORT -- .run  (autoXecute)
#
# FORMAT SPEC, derived from a real run file produced by the target
# instrument: Imaging_Run.run, AutoExecute 7.6.6.0, 922611 positions
# across 7 regions.
#
# The .run carries NO coordinates. It is a flat ordered list of
# position NAMES plus acquisition settings; the coordinates for those
# names live in the .xeo identified by the `geometry` attribute. The
# two files are therefore written together, and `geometry` must equal
# the .xeo filename stem or autoXecute cannot resolve a single point.
#
# Naming: R<region:02d>X<x>Y<y>. X and Y are PHYSICAL COORDINATES on
# the adapter, in whole units, not raster indices. Evidence from the
# reference file:
#
#   - Y falls into two bands 2623 units apart. A two-slide adapter
#     puts slide centres ~26 mm apart. At 10 um/unit that is 26.2 mm.
#   - X spans 5294 units across both slides. At 10 um/unit that is
#     52.9 mm, sitting inboard of a 75 mm slide.
#   - Each region is 448 x 295 units. At 10 um/unit that is
#     4.48 x 2.95 mm, the size of a mouse kidney section.
#   - sampleName is kidneyslides34 -- slides 3 and 4 -- and the two Y
#     bands hold 3 and 4 regions respectively.
#
# So one unit is 10 um and BOTH AXES SHARE THAT SCALE. The scale is
# inferred, not documented; confirm it against a real .xeo before an
# acquisition run.
# =====================================================================

RUN_ATTRS_DEFAULT = {
    "AnalysisSpectraType": "Imaging",
    "DataStorage": "Container",
    "appname": "AutoExecute",
    "appversion": "7.6.6.0_036f43428109dad9058d7f326002a644b119244f_1",
    "barcode": "-1",
    "baseGeometry": "MTP Slide Adapter II",
    "binDataPoints": "8000",
    "cleanSourceAfterMeasurement": "Off",
    "doBaselineSub": "false",
    "doSmoothing": "false",
    "ejectTargetAfterMeasurement": "true",
    "executeExternalCalibration": "true",
    "fragmentMass": "0.0000",
    "parentMass": "0.0000",
    "runID": "",
    "stopAfterMsMeasurement": "false",
    "targetID": "",
    "type": "FastImaging",
    "use1to1Preteaching": "true",
    "version": "1.0",
}


def _xml_escape(v: str) -> str:
    return (v.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def position_name(cfg: dict, index: int, shot) -> str:
    """
    Name for one position, shared by the .xeo and the .run.

    The two files are matched by name alone, so whatever this returns
    must be used identically in both. Default follows the convention
    seen in the reference run file.
    """
    out = cfg["output"]

    default = ("R{region:02d}X{i:.0f}Y{j:.0f}"
               if cfg.get("fiducial-units", "stage") == "plate"
               else "x_{px:.0f}y_{py:.0f}")
    pattern = out.get("position-name") or default
    return pattern.format(region=out.get("region", 0),
                          n=index + 1,
                          i=shot.x_um / PLATE_UNIT_UM,   # plate units
                          j=shot.y_um / PLATE_UNIT_UM,
                          px=shot.x_px, py=shot.y_px,
                          bead=shot.bead_id, angle=int(shot.angle_deg))


def write_run(path: Path, names: list[str], cfg: dict) -> Path:
    """Write an autoXecute .run listing positions by name."""
    import datetime

    run = dict(RUN_ATTRS_DEFAULT)
    run.update({k: str(v) for k, v in (cfg["output"].get("run") or {}).items()})
    run["date"] = datetime.datetime.now().astimezone().isoformat(
        timespec="seconds")
    run["geometry"] = path.stem            # must match the .xeo stem

    attrs = " ".join(f'{k}="{_xml_escape(str(v))}"'
                     for k, v in sorted(run.items()))
    chip = cfg["output"].get("chip", 0)

    lines = ['<?xml version="1.0" encoding="UTF-8"?>', f"<table {attrs}>"]
    lines += [f'\t<cont Chip_on_Scout="{chip}" Pos_on_Scout="{n}"/>'
              for n in names]
    lines.append("</table>")

    # The reference file uses CRLF throughout.
    path.write_bytes(("\r\n".join(lines) + "\r\n").encode("utf-8"))
    return path


def read_run(path: Path) -> list[str]:
    """Position names from a .run, in order."""
    import re
    return re.findall(r'Pos_on_Scout="([^"]+)"', path.read_text())


# =====================================================================
# QC OVERLAY
# =====================================================================

def draw_overlay(path: Path, beads: list[Bead], shots: list[Shot],
                 cfg: dict, T: Transform, scan: Path | None,
                 show: bool = False) -> None:
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(14, 9))

    if scan and scan.exists():
        try:
            import cv2
            img = cv2.imread(str(scan), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                ax.imshow(img, cmap="gray", origin="upper")
        except ImportError:
            pass

    crater_px = footprint_um(cfg) / T.um_per_px

    for b in beads:
        if b.accepted:
            colour, lw = "#2ca02c", 1.2
        elif b.clumped:
            colour, lw = "#9467bd", 1.6          # clump
        else:
            colour, lw = "#d62728", 1.2
        ax.add_patch(Circle((b.x_px, b.y_px), b.diameter_px / 2,
                            fill=False, ec=colour, lw=lw))
        for s in b.shots:
            px = T.um_to_px([s.x_um, s.y_um])[0]
            if s.dropped:
                ax.add_patch(Circle(tuple(px), crater_px / 2, fill=False,
                                    ec="#ff7f0e", lw=0.8, ls=":"))
            else:
                ax.add_patch(Circle(tuple(px), crater_px / 2,
                                    fc="#1f77b4", ec="none", alpha=0.65))

    n_acc = sum(b.accepted for b in beads)
    n_cl = sum(b.clumped for b in beads)
    n_live = sum(not s.dropped for s in shots)
    ax.set_title(
        f"green = accepted ({n_acc})   purple = clumped ({n_cl})   "
        f"red = rejected ({len(beads) - n_acc - n_cl})   "
        f"blue = shot ({n_live})   "
        f"orange dotted = dropped ({len(shots) - n_live})")
    ax.set_xlabel("scan pixels")
    ax.set_aspect("equal")

    # Patches do not trigger autoscale, so set the view explicitly.
    if not (scan and scan.exists()) and beads:
        xs = [b.x_px for b in beads]
        ys = [b.y_px for b in beads]
        pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)   # image convention
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)



def draw_zoom(path: Path, beads: list[Bead], cfg: dict, T: Transform,
              scan: Path | None) -> bool:
    """
    High-magnification render of shot placement, centred on the
    densest patch of accepted beads. This is the picture to check
    before committing an acquisition: it shows whether shots actually
    land on clean matrix just off each bead edge.
    """
    try:
        import cv2
    except ImportError:
        return False
    if scan is None or not scan.exists():
        return False
    acc = [b for b in beads if b.accepted]
    if not acc:
        return False

    img = cv2.imread(str(scan), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    z = cfg["output"]
    w = int(z.get("zoom-window-px", 700))
    h = int(w * 0.66)
    xs = np.array([b.x_px for b in acc])
    ys = np.array([b.y_px for b in acc])

    best, bx, by = -1, float(xs.min()), float(ys.min())
    for cx in np.linspace(xs.min(), max(xs.max() - w, xs.min()), 40):
        for cy in np.linspace(ys.min(), max(ys.max() - h, ys.min()), 30):
            n = int(((xs >= cx) & (xs < cx + w) &
                     (ys >= cy) & (ys < cy + h)).sum())
            if n > best:
                best, bx, by = n, float(cx), float(cy)

    x0, y0 = int(max(bx, 0)), int(max(by, 0))
    x1, y1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
    sc = int(z.get("zoom-scale", 3))

    vis = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
                     ((x1 - x0) * sc, (y1 - y0) * sc),
                     interpolation=cv2.INTER_LANCZOS4)
    crater_px = footprint_um(cfg) / T.um_per_px

    for b in beads:
        if not (x0 < b.x_px < x1 and y0 < b.y_px < y1):
            continue
        if b.accepted:
            col = (0, 170, 0)
        elif b.clumped:
            col = (180, 60, 140)          # purple, BGR
        else:
            col = (0, 0, 225)
        cv2.circle(vis, (int((b.x_px - x0) * sc), int((b.y_px - y0) * sc)),
                   max(int(b.diameter_px / 2 * sc), 3), col, 2)
        for sh in b.shots:
            px = T.um_to_px([sh.x_um, sh.y_um])[0]
            pt = (int((px[0] - x0) * sc), int((px[1] - y0) * sc))
            r = max(int(crater_px / 2 * sc), 3)
            if sh.dropped:
                cv2.circle(vis, pt, r, (0, 165, 255), 2)
            else:
                cv2.circle(vis, pt, r, (230, 120, 0), -1)

    cv2.imwrite(str(path), vis)
    return True


# =====================================================================
# FIDUCIAL PICKER
# =====================================================================

def pick_fiducials(cfg: dict) -> None:
    """
    Interactive fiducial picker.

        right-click on the image   set the pending pixel
        type stage x / stage y     into the boxes at the bottom
        Add fiducial               commit the pair
        Remove nearest             delete the fiducial nearest the
                                   last right-click
        Reset                      clear the list
        close the window           write FIDUCIALS into this file

    Coordinate entry is IN THE WINDOW, not the terminal. An earlier
    version prompted with input() from inside the click callback,
    which blocks the GUI event loop and makes the window appear to
    freeze. Never call input() from a matplotlib callback.

    The worst-fitting fiducial is drawn red and live RMS sits in the
    title, so a mistyped stage coordinate shows up immediately.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import TextBox, Button

    backend = plt.get_backend()
    if backend.lower() in ("agg", "pdf", "ps", "svg", "template"):
        sys.exit(f"matplotlib is using the non-interactive '{backend}' "
                 f"backend, so no window can open.\n"
                 f"Install a GUI toolkit (pip install pyqt5) or set "
                 f"MPLBACKEND=TkAgg and try again.")

    scan = Path(cfg["input"]["scan"])
    if not scan.is_absolute():
        scan = HERE / scan
    if not scan.exists():
        sys.exit(f"Scan not found: {scan}")

    try:
        import cv2
        img = cv2.imread(str(scan), cv2.IMREAD_GRAYSCALE)
    except ImportError:
        sys.exit("opencv-python is required to open the scan.")
    if img is None:
        sys.exit(f"Could not read {scan}. TIFF, not JPEG -- convert with:\n"
                 f"  python microMS_beadtargeting.py convert <file>")

    fids: list[dict] = list(cfg.get("fiducials") or [])
    pending = {"px": None}

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_axes([0.05, 0.16, 0.92, 0.79])
    show_pyramid(fig, ax, img, cmap="gray")

    def redraw():
        for art in list(ax.lines) + list(ax.texts):
            art.remove()
        worst = -1
        title = f"{len(fids)} fiducials  (3 needed)"
        if len(fids) >= 3:
            src = np.array([[f["x_px"], f["y_px"]] for f in fids], float)
            dst = np.array([[f["x_um"], f["y_um"]] for f in fids], float)
            T = fit_similarity(src, dst, cfg.get("allow-reflection", True))
            res = residuals(T, src, dst)
            worst = int(np.argmax(res))
            title = (f"{len(fids)} fiducials | RMS "
                     f"{np.sqrt((res ** 2).mean()):.1f} um | worst "
                     f"{res[worst]:.1f} um | {T.um_per_px:.3f} um/px"
                     + (" | REFLECTED" if T.reflected else ""))
        for i, f in enumerate(fids):
            c = "red" if i == worst else "lime"
            ax.plot(f["x_px"], f["y_px"], "+", ms=16, mew=2, color=c)
            ax.text(f["x_px"] + 60, f["y_px"] - 60, str(i), color=c,
                    fontsize=10)
        if pending["px"] is not None:
            x, y = pending["px"]
            ax.plot(x, y, "x", ms=18, mew=3, color="gold")
            ax.text(x + 60, y - 60, "pending", color="gold", fontsize=10)
        ax.set_title(title)
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is not ax or ev.button != 3 or ev.xdata is None:
            return
        pending["px"] = (float(ev.xdata), float(ev.ydata))
        status.set_text(f"pending pixel ({ev.xdata:.0f}, {ev.ydata:.0f})"
                        f"  -- enter stage coords, then Add")
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    bx = TextBox(fig.add_axes([0.125, 0.06, 0.085, 0.045]), "stage x (um) ")
    by = TextBox(fig.add_axes([0.325, 0.06, 0.085, 0.045]), "stage y (um) ")
    status = fig.text(0.05, 0.012, "right-click a fiducial on the image",
                      fontsize=9, color="#555555")
    fig.text(0.55, 0.012, "scroll = zoom at cursor   middle-drag = pan   "
                          "+ / - = zoom   f = fit",
             fontsize=9, color="#555555")

    def add(_ev=None):
        if pending["px"] is None:
            status.set_text("right-click a fiducial on the image first")
            fig.canvas.draw_idle()
            return
        try:
            x_um, y_um = float(bx.text), float(by.text)
        except ValueError:
            status.set_text("stage x and stage y must both be numbers")
            fig.canvas.draw_idle()
            return
        fids.append({"x_px": pending["px"][0], "y_px": pending["px"][1],
                     "x_um": x_um, "y_um": y_um})
        pending["px"] = None
        bx.set_val("")
        by.set_val("")
        status.set_text(f"added fiducial {len(fids) - 1}")
        redraw()

    def remove(_ev=None):
        if not fids:
            return
        if pending["px"] is not None:
            ref = pending["px"]
        else:
            ref = (fids[-1]["x_px"], fids[-1]["y_px"])
        d = [math.hypot(f["x_px"] - ref[0], f["y_px"] - ref[1]) for f in fids]
        i = int(np.argmin(d))
        fids.pop(i)
        status.set_text(f"removed fiducial {i}")
        redraw()

    def reset(_ev=None):
        fids.clear()
        pending["px"] = None
        status.set_text("cleared")
        redraw()

    zin, zout, zfit = attach_zoom(fig, ax)

    keep = [Button(fig.add_axes([0.435, 0.06, 0.105, 0.045]), "Add fiducial"),
            Button(fig.add_axes([0.550, 0.06, 0.115, 0.045]),
                   "Remove nearest"),
            Button(fig.add_axes([0.675, 0.06, 0.060, 0.045]), "Reset"),
            Button(fig.add_axes([0.755, 0.06, 0.058, 0.045]), "Zoom +"),
            Button(fig.add_axes([0.818, 0.06, 0.058, 0.045]), "Zoom -"),
            Button(fig.add_axes([0.881, 0.06, 0.058, 0.045]), "Fit")]
    keep[0].on_clicked(add)
    keep[1].on_clicked(remove)
    keep[2].on_clicked(reset)
    keep[3].on_clicked(zin)
    keep[4].on_clicked(zout)
    keep[5].on_clicked(zfit)

    # Enter in either box commits, so the whole entry is keyboard-only.
    bx.on_submit(lambda _t: by.set_val(by.text) or None)
    by.on_submit(lambda _t: add())

    redraw()
    plt.show()

    save_fiducials(fids)
    print(f"Wrote {len(fids)} fiducials to {CONFIG_PATH.name}")
    if len(fids) >= 3:
        report_registration({**cfg, "fiducials": fids})
    else:
        print("Fewer than 3 fiducials -- registration cannot be fitted yet.")





# =====================================================================
# IMAGE PYRAMID
#
# A whole-slide scan is 8000 x 6039 -- 48 million pixels. matplotlib
# redraws the entire array on every zoom and pan, which is why the
# picker crawls.
#
# The fix is what every slide viewer does: keep the image at several
# resolutions and draw the coarsest one that still has more pixels
# than the screen area it fills. Zoomed out you are looking at a
# 1/8-scale copy; zoomed in you get full detail over a small region.
#
# The displayed array changes, the COORDINATES DO NOT. `extent` is
# always the full-resolution frame, so clicks, fiducials and bead
# positions are unaffected by which level happens to be showing.
#
# microMS decimates plain TIFFs for the same reason.
# =====================================================================

def build_pyramid(img, levels: int = 4, min_side: int = 512):
    """[full, half, quarter, ...] while the result stays usable."""
    import cv2
    out = [img]
    for _ in range(levels - 1):
        h, w = out[-1].shape[:2]
        if min(h, w) // 2 < min_side:
            break
        out.append(cv2.resize(out[-1], (w // 2, h // 2),
                              interpolation=cv2.INTER_AREA))
    return out


def show_pyramid(fig, ax, img, **imshow_kw):
    """
    imshow that redraws only the visible region, at a resolution
    matched to the screen.

    Choosing a pyramid level alone is not enough: matplotlib processes
    the entire array on every draw and simply clips what falls outside
    the axes, so zooming into a corner of an 8000 px scan is no faster
    than viewing all of it. The visible rectangle has to be CROPPED
    out before it is handed over.

    Level and crop together keep the drawn array near the size of the
    axes in screen pixels, whatever the zoom.

    `extent` tracks the crop in FULL-RESOLUTION coordinates, so clicks,
    fiducials and bead positions are unaffected.
    """
    pyr = build_pyramid(img)
    h, w = img.shape[:2]
    full = (-0.5, w - 0.5, h - 0.5, -0.5)

    state = {"key": None, "busy": False}
    im = ax.imshow(pyr[-1], extent=full, **imshow_kw)
    ax.set_xlim(full[0], full[1])
    ax.set_ylim(full[2], full[3])

    def refresh(_evt=None):
        if state["busy"]:
            return
        x0, x1 = sorted(ax.get_xlim())
        y0, y1 = sorted(ax.get_ylim())
        x0 = max(x0, 0.0); y0 = max(y0, 0.0)
        x1 = min(x1, float(w)); y1 = min(y1, float(h))
        if x1 - x0 < 1 or y1 - y0 < 1:
            return

        # coarsest level whose crop still has a pixel per screen pixel
        screen = max(ax.bbox.width, 1.0)
        level = len(pyr) - 1
        for i in range(len(pyr) - 1, 0, -1):
            if (x1 - x0) * (pyr[i].shape[1] / w) >= screen:
                level = i
                break
        else:
            level = 0

        f = pyr[level].shape[1] / w
        cx0 = int(max(x0 * f, 0)); cy0 = int(max(y0 * f, 0))
        cx1 = int(min(round(x1 * f), pyr[level].shape[1]))
        cy1 = int(min(round(y1 * f), pyr[level].shape[0]))
        if cx1 - cx0 < 2 or cy1 - cy0 < 2:
            return

        key = (level, cx0, cy0, cx1, cy1)
        if key == state["key"]:
            return

        state["busy"] = True
        try:
            state["key"] = key
            im.set_data(pyr[level][cy0:cy1, cx0:cx1])
            # back to full-resolution coordinates
            im.set_extent((cx0 / f - 0.5, cx1 / f - 0.5,
                           cy1 / f - 0.5, cy0 / f - 0.5))
            fig.canvas.draw_idle()
        finally:
            state["busy"] = False

    ax.callbacks.connect("xlim_changed", refresh)
    ax.callbacks.connect("ylim_changed", refresh)
    fig.canvas.mpl_connect("draw_event", refresh)

    log(f"image pyramid: {len(pyr)} levels, "
        f"{' '.join(f'{q.shape[1]}x{q.shape[0]}' for q in pyr)}")
    return im


# =====================================================================
# ZOOM / PAN
#
# Shared by the fiducial picker and the manual selection window.
#
# The matplotlib toolbar's own zoom is deliberately not used: it binds
# left-drag, which is already the box selector in 'select', and
# entering toolbar zoom mode silently swallows the clicks that add or
# toggle beads. Scroll and middle-drag stay clear of both buttons.
#
# Pan is computed from PIXEL deltas, not data coordinates. Reading
# ev.xdata while the limits are being changed feeds the new limits
# back into the next delta, and the image accelerates away under the
# cursor.
# =====================================================================

def attach_zoom(fig, ax, step: float = 1.3):
    """Wire scroll zoom, middle-drag pan and keyboard shortcuts.

    Returns (zoom_in, zoom_out, fit) so the same actions can be bound
    to buttons.
    """
    home = {"lim": None}
    pan = {"px": None}

    def remember_home():
        if home["lim"] is None:
            home["lim"] = (ax.get_xlim(), ax.get_ylim())

    def scale_about(xd, yd, f):
        remember_home()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(xd - (xd - x0) * f, xd + (x1 - xd) * f)
        ax.set_ylim(yd - (yd - y0) * f, yd + (y1 - yd) * f)
        fig.canvas.draw_idle()

    def centre():
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        return (x0 + x1) / 2, (y0 + y1) / 2

    def zoom_in(*_a):
        scale_about(*centre(), 1 / step)

    def zoom_out(*_a):
        scale_about(*centre(), step)

    def fit(*_a):
        if home["lim"] is not None:
            ax.set_xlim(*home["lim"][0])
            ax.set_ylim(*home["lim"][1])
            fig.canvas.draw_idle()

    def on_scroll(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        # Zoom about the cursor so the feature under the pointer stays
        # put -- the behaviour every image viewer has.
        scale_about(ev.xdata, ev.ydata,
                    1 / step if ev.button == "up" else step)

    def on_press(ev):
        if ev.button == 2 and ev.inaxes is ax:
            remember_home()
            pan["px"] = (ev.x, ev.y, ax.get_xlim(), ax.get_ylim())

    def on_motion(ev):
        if pan["px"] is None or ev.x is None:
            return
        x0px, y0px, xl, yl = pan["px"]
        sx = (xl[1] - xl[0]) / max(ax.bbox.width, 1)
        sy = (yl[1] - yl[0]) / max(ax.bbox.height, 1)
        dx = -(ev.x - x0px) * sx
        dy = -(ev.y - y0px) * sy
        ax.set_xlim(xl[0] + dx, xl[1] + dx)
        ax.set_ylim(yl[0] + dy, yl[1] + dy)
        fig.canvas.draw_idle()

    def on_release(ev):
        if ev.button == 2:
            pan["px"] = None

    def on_key(ev):
        if ev.key in ("+", "=", "up"):
            zoom_in()
        elif ev.key in ("-", "_", "down"):
            zoom_out()
        elif ev.key in ("f", "home", "0"):
            fit()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("draw_event", lambda _e: remember_home())
    return zoom_in, zoom_out, fit


# =====================================================================
# TIFF CONVERSION
#
# Neither this pipeline nor microMS reads JPEG. microMS additionally
# wants a _c1 channel suffix and thresholds for BRIGHT objects, so a
# file destined for microMS must also be inverted -- beads on matrix
# are darker than their background.
#
# This pipeline inverts internally at detection time, so a file for
# THIS tool should NOT be pre-inverted. Inverting twice puts you back
# where you started with no error message.
# =====================================================================

def convert_image(src: Path, dst: Path | None = None,
                  invert: bool = False, microms: bool = False,
                  downsample: float = 1.0) -> Path:
    try:
        import cv2
    except ImportError:
        sys.exit("opencv-python is required. pip install opencv-python")

    log(f"reading {src}")
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f"Could not read {src}")
    log(f"input {img.shape} {img.dtype}")

    if img.ndim == 3:
        say(f"  {img.shape[2]}-channel input, converting to greyscale")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img.dtype != np.uint8:
        say(f"  {img.dtype} input, scaling to 8-bit")
        img = cv2.normalize(img, None, 0, 255,
                            cv2.NORM_MINMAX).astype(np.uint8)

    if downsample != 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w * downsample), int(h * downsample)),
                         interpolation=cv2.INTER_AREA)
        say(f"  downsampled by {downsample} -> {img.shape[1]}x{img.shape[0]}"
            f"   REMEMBER: um/px changes by the same factor")

    if invert or microms:
        img = cv2.bitwise_not(img)
        say("  inverted (beads now bright)")

    if dst is None:
        stem = src.stem + ("_inverted_c1" if microms else "")
        dst = src.with_name(stem).with_suffix(".tif")
        import os
        if not os.access(dst.parent, os.W_OK):
            dst = HERE / dst.name
            say(f"  {src.parent} is not writable, writing to {HERE}")

    ok = False
    try:
        ok = cv2.imwrite(str(dst), img)
    except Exception as e:
        log(f"imwrite raised {e}")
        ok = False
    if not ok:
        sys.exit(f"Could not write {dst}")

    say(f"  wrote {dst.name}  {img.shape[1]}x{img.shape[0]}  "
        f"{dst.stat().st_size / 1e6:.1f} MB")
    if microms:
        say("  microMS-ready: TIFF, inverted, _c1 channel suffix.\n"
            "  Do NOT point this pipeline at it -- detection inverts "
            "again internally.")
    return dst


def cli_convert(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("-")]
    flags = {a for a in argv if a.startswith("--")}

    if args:
        src = Path(args[0])
        if not src.is_absolute():
            src = HERE / src
    else:
        src = HERE / load_config()["input"]["scan"]
        say("No input given, using CONFIG input scan")

    if not src.exists():
        sys.exit(f"Not found: {src}")

    dst = None
    if len(args) > 1:
        dst = Path(args[1])
        if not dst.is_absolute():
            dst = HERE / dst

    ds = 1.0
    for f in flags:
        if f.startswith("--downsample="):
            ds = float(f.split("=", 1)[1])

    say(f"Converting {src.name}")
    convert_image(src, dst, invert="--invert" in flags,
                  microms="--microms" in flags, downsample=ds)


# =====================================================================
# MANUAL SELECTION
#
# Auto filtering is a starting point, not a verdict. At low contrast
# detection both merges real singles into false clumps and lets ragged
# pairs through, so the operator gets the final say.
#
# Overrides are stored by PIXEL POSITION, not index, because detection
# indices shift the moment any detection parameter changes. On reload
# each override is matched to the nearest detected object within
# match-radius-px; anything unmatched is reported, not silently
# dropped.
# =====================================================================

SELECTION_PATH = HERE / "manual_selection.csv"


def load_manual(path: Path = SELECTION_PATH) -> list[tuple]:
    if not path.exists():
        return []
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append((float(row["x_px"]), float(row["y_px"]),
                        row["decision"].strip()))
    return out


def save_manual(entries: list[tuple], path: Path = SELECTION_PATH) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_px", "y_px", "decision"])
        for x, y, d in entries:
            w.writerow([f"{x:.2f}", f"{y:.2f}", d])


def apply_manual(beads: list[Bead], cfg: dict,
                 path: Path = SELECTION_PATH) -> None:
    entries = load_manual(path)
    if not entries or not beads:
        log("no manual overrides to apply")
        return
    tol = float(cfg.get("manual-selection", {}).get("match-radius-px", 12))
    tree = cKDTree(np.array([[b.x_px, b.y_px] for b in beads], float))

    hit = miss = 0
    for x, y, decision in entries:
        d, i = tree.query([x, y])
        if d > tol:
            miss += 1
            continue
        hit += 1
        b = beads[i]
        b.accepted = (decision == "accept")
        b.manual = decision
        b.reject_reason = "" if b.accepted else "manually rejected"
        b.reject_category = "" if b.accepted else "manual"
    say(f"  manual overrides applied            : {hit}"
        + (f"  ({miss} matched nothing within {tol:.0f} px)" if miss else ""))


def bead_manual_selection(cfg: dict) -> None:
    """
    Bead manual selection window.

    green accepted   red rejected   purple clumped   blue overridden

      right-click a bead    toggle it
      drag a box            select a region
      Accept box            accept every bead inside
      Reject box            reject every bead inside
      Clear box             drop the region selection
      Reset                 discard all manual overrides
      close the window      write manual_selection.csv
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import (RectangleSelector, Button,
                                    CheckButtons)
    from matplotlib.patches import Circle

    backend = plt.get_backend()
    if backend.lower() in ("agg", "pdf", "ps", "svg", "template"):
        sys.exit(f"matplotlib is using the non-interactive '{backend}' "
                 f"backend, so no window can open.\n"
                 f"Install a GUI toolkit (pip install pyqt5) or set "
                 f"MPLBACKEND=TkAgg and try again.")

    T = to_microns(transform_from_config(cfg), cfg)
    beads, scan = build_beads(cfg, T)
    auto = [b.accepted for b in beads]

    img = None
    if scan:
        import cv2
        img = cv2.imread(str(scan), cv2.IMREAD_GRAYSCALE)

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_axes([0.04, 0.12, 0.93, 0.83])
    if img is not None:
        show_pyramid(fig, ax, img, cmap="gray")
    ax.set_aspect("equal")

    box = {"rect": None}

    # Which categories are drawn. Hiding a category only affects the
    # display -- a hidden bead keeps its accept/reject state, still
    # counts as an isolation neighbour, and is still exported.
    show = {"accepted": True, "clumped": True, "rejected": True}

    def category(b):
        if b.accepted:
            return "accepted"
        if b.clumped:
            return "clumped"
        return "rejected"

    def colour(b):
        if b.accepted:
            return "#2ca02c"
        if b.clumped:
            return "#9467bd"
        if b.reject_category == "manual":
            return "#1f77b4"
        return "#d62728"

    patches = []
    for b in beads:
        c = Circle((b.x_px, b.y_px), max(b.diameter_px / 2, 4),
                   fill=False, ec=colour(b), lw=1.3)
        ax.add_patch(c)
        patches.append(c)

    def refresh():
        for b, c in zip(beads, patches):
            c.set_edgecolor(colour(b))
            c.set_linewidth(2.0 if b.manual else 1.3)
            c.set_visible(show[category(b)])
        hidden = [k for k, v in show.items() if not v]
        note = f"   |   HIDDEN: {', '.join(hidden)}" if hidden else ""
        ax.set_title(f"Bead manual selection   |   accepted "
                     f"{sum(b.accepted for b in beads)} / {len(beads)}   |   "
                     f"manual overrides {sum(1 for b in beads if b.manual)}"
                     f"{note}")
        fig.canvas.draw_idle()

    def in_box():
        # Only VISIBLE beads. Acting on a hidden category would toggle
        # things the operator cannot see and did not mean to touch.
        if not box["rect"]:
            return []
        x0, x1, y0, y1 = box["rect"]
        return [i for i, b in enumerate(beads)
                if x0 <= b.x_px <= x1 and y0 <= b.y_px <= y1
                and show[category(b)]]

    def on_box(eclick, erelease):
        x0, x1 = sorted((eclick.xdata, erelease.xdata))
        y0, y1 = sorted((eclick.ydata, erelease.ydata))
        box["rect"] = (x0, x1, y0, y1)
        status.set_text(f"box selects {len(in_box())} beads")
        fig.canvas.draw_idle()

    rs = RectangleSelector(ax, on_box, useblit=True, button=[1],
                           minspanx=8, minspany=8, interactive=True,
                           props=dict(facecolor="#1f77b4", alpha=0.15,
                                      edgecolor="#1f77b4", lw=1.5))

    def set_many(decision):
        idx = in_box()
        if not idx:
            status.set_text("draw a box first")
            fig.canvas.draw_idle()
            return
        for i in idx:
            b = beads[i]
            b.accepted = (decision == "accept")
            b.manual = decision
            b.reject_category = "" if b.accepted else "manual"
            b.reject_reason = "" if b.accepted else "manually rejected"
        status.set_text(f"{decision}ed {len(idx)} beads in box")
        say(f"{decision}ed {len(idx)} beads in box")
        refresh()

    def on_click(ev):
        if ev.inaxes is not ax or ev.button != 3 or ev.xdata is None:
            return
        # Hidden beads are not clickable, for the same reason.
        d = [math.hypot(b.x_px - ev.xdata, b.y_px - ev.ydata)
             if show[category(b)] else float("inf") for b in beads]
        i = int(np.argmin(d))
        if not math.isfinite(d[i]) or d[i] > 40:
            return
        b = beads[i]
        b.accepted = not b.accepted
        b.manual = "accept" if b.accepted else "reject"
        b.reject_category = "" if b.accepted else "manual"
        b.reject_reason = "" if b.accepted else "manually rejected"
        status.set_text(f"bead {i} -> {b.manual}")
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    status = fig.text(0.04, 0.005, "right-click a bead to toggle   |   "
                                   "left-drag to draw a box   |   "
                                   "close to save",
                      fontsize=9, color="#555555")
    fig.text(0.60, 0.005, "scroll = zoom at cursor   middle-drag = pan   "
                          "+ / - = zoom   f = fit",
             fontsize=9, color="#555555")

    def clear_box():
        box["rect"] = None
        rs.set_visible(False)
        status.set_text("box cleared")
        fig.canvas.draw_idle()

    def reset_all():
        for b, a in zip(beads, auto):
            b.accepted, b.manual = a, ""
        status.set_text("manual overrides reset")
        refresh()

    zin, zout, zfit = attach_zoom(fig, ax)

    keep = []
    for x, w, lbl, fn in ((0.04, 0.12, "Accept box",
                           lambda: set_many("accept")),
                          (0.17, 0.12, "Reject box",
                           lambda: set_many("reject")),
                          (0.30, 0.11, "Clear box", clear_box),
                          (0.42, 0.08, "Reset", reset_all),
                          (0.53, 0.07, "Zoom +", zin),
                          (0.61, 0.07, "Zoom -", zout),
                          (0.69, 0.07, "Fit", zfit)):
        btn = Button(fig.add_axes([x, 0.045, w, 0.045]), lbl)
        btn.on_clicked(lambda _ev, f=fn: f())
        keep.append(btn)

    # Show/hide by category. On a crowded slide the red and purple
    # circles bury the green ones -- 549 of 1003 objects were clumps
    # on the reference scan -- so hiding them is the only way to see
    # what will actually be acquired.
    cax = fig.add_axes([0.78, 0.03, 0.20, 0.085])
    cax.set_frame_on(False)
    checks = CheckButtons(
        cax,
        ["accepted (green)", "clumped (purple)", "rejected (red)"],
        [show["accepted"], show["clumped"], show["rejected"]])
    for lbl_obj, col in zip(checks.labels,
                            ("#2ca02c", "#9467bd", "#d62728")):
        lbl_obj.set_color(col)
        lbl_obj.set_fontsize(9)

    def toggle(label):
        key = {"accepted (green)": "accepted",
               "clumped (purple)": "clumped",
               "rejected (red)": "rejected"}[label]
        show[key] = not show[key]
        refresh()

    checks.on_clicked(toggle)
    keep.append(checks)

    refresh()
    plt.show()

    entries = [(b.x_px, b.y_px, b.manual) for b in beads if b.manual]
    save_manual(entries)
    say(f"\nWrote {SELECTION_PATH.name} with {len(entries)} overrides")
    say(f"Accepted after manual selection: "
        f"{sum(b.accepted for b in beads)}")


# =====================================================================
# RUN
# =====================================================================

def build_beads(cfg: dict, T: Transform) -> tuple[list[Bead], Path | None]:
    """
    Detect (or load) objects and run every automatic filter, in the
    order microMS uses. Shared by 'run' and 'select' so the two can
    never diverge.
    """
    src = cfg["input"]
    if src.get("beads"):
        path = Path(src["beads"])
        if not path.is_absolute():
            path = HERE / path
        beads, scan = load_beads_csv(path), None
        say(f"Loaded {len(beads)} objects from {path.name}")
    else:
        scan = Path(src["scan"])
        if not scan.is_absolute():
            scan = HERE / scan
        beads = detect_blobs(scan, cfg)
        say(f"Detected {len(beads)} objects in {scan.name}")

    to_stage(beads, T)
    log("converted pixel centroids to stage microns")
    isolation_filter(beads, float(cfg["min-bead-separation"]))
    log("isolation filter done (run against ALL objects, debris included)")
    shape_filter(beads, cfg)
    log("shape filter done")
    apply_manual(beads, cfg)
    return beads, scan


def run(cfg: dict) -> None:
    log("fitting registration from fiducials")
    T = to_microns(report_registration(cfg), cfg)
    print()
    beads, scan = build_beads(cfg, T)

    if beads:
        dm = np.array([b.diameter_um for b in beads])
        print(f"  measured diameter um: median {np.median(dm):.0f}, "
              f"10th {np.percentile(dm, 10):.0f}, "
              f"90th {np.percentile(dm, 90):.0f}")
        if cfg["shot-placement"]["distance-reference"] == "edge":
            nom = float(cfg["bead-diameter"])
            if abs(np.median(dm) - nom) > 0.2 * nom:
                print(f"  WARNING: median measured diameter is far from "
                      f"bead-diameter ({nom:.0f} um). Threshold choice "
                      f"biases\n  the measurement, and 'edge' placement "
                      f"inherits that bias directly. Check the overlay, "
                      f"or use\n  distance-reference: center.")

    # Comi et al. 2017: probe radius >= target localization error, and
    # distance filter > that error + probe radius.
    tle = float(cfg.get("target-localization-error-um", 0) or 0)
    if tle:
        probe_r = footprint_um(cfg) / 2.0
        sep = float(cfg["min-bead-separation"])
        need = tle + probe_r
        print(f"\n  target localization error : {tle:.1f} um "
              f"(probe radius {probe_r:.1f} um)")
        if probe_r < tle:
            print(f"  NOTE: probe radius is smaller than the localization "
                  f"error. Comi et al.\n  recommend the probe be at least as "
                  f"large, so a mistargeted shot still\n  lands on the bead.")
        if sep <= need:
            print(f"  WARNING: min-bead-separation ({sep:.0f} um) does not "
                  f"exceed error + probe\n  radius ({need:.0f} um). "
                  f"Neighbouring beads may be sampled together.")
        else:
            print(f"  min-bead-separation {sep:.0f} um exceeds "
                  f"{need:.0f} um: OK")

    n_clump = sum(b.clumped for b in beads)
    n_iso = sum(1 for b in beads
                if b.nn_um >= float(cfg["min-bead-separation"]))
    n_acc = sum(b.accepted for b in beads)
    print(f"  screened as clumps                  : {n_clump}")
    print(f"  isolated (>= {cfg['min-bead-separation']} um)       : {n_iso}")
    print(f"  accepted after all filters          : {n_acc}")

    reasons = {}
    for b in beads:
        if not b.accepted:
            key = b.reject_category or "other"
            reasons[key] = reasons.get(key, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {c:5d}  rejected: {r}")

    log(f"placing {len(cfg['laser-shot-angles'])} shots per accepted bead, "
        f"crater {footprint_um(cfg):.0f} um")
    shots = place_shots(beads, cfg, T)
    log(f"{len(shots)} shot positions generated")
    ordered = serpentine(beads, shots) if cfg["output"].get(
        "serpentine-order", True) else [s for s in shots if not s.dropped]

    sp = cfg["shot-placement"]
    ref = sp["distance-reference"]

    if ref == "edge":
        suspect = [b for b in beads if b.accepted and suspect_radius(b, cfg)]
        if suspect:
            tol = float(cfg.get("suspect-diameter-tolerance", 0.4))
            nominal = float(cfg["bead-diameter"])
            print(f"\nWARNING: {len(suspect)} of "
                  f"{sum(b.accepted for b in beads)} accepted beads differ "
                  f"from bead-diameter\n  ({nominal:.0f} um) by more than "
                  f"{tol * 100:.0f}%. 'edge' placement scales shot distance "
                  f"from the\n  measured radius, so those shots are placed "
                  f"on a radius that may be wrong --\n  and the "
                  f"crater-overlap check cannot catch it, because it compares "
                  f"that\n  radius against itself.")
            d = np.array([b.diameter_um for b in suspect])
            print(f"  their diameters: {d.min():.0f} to {d.max():.0f} um "
                  f"(median {np.median(d):.0f})")
            print(f"  Cross-check by running once with "
                  f"distance-reference: center -- shots\n  dropped there for "
                  f"'crater overlaps own bead' are these same beads.")
    detail = (f"measured radius + {sp['edge-offset']} um"
              if ref == "edge" else f"{sp['laser-distance']} um from centre")
    print(f"\nShot placement : {ref} ({detail})")
    print(f"  placed  : {len(shots)}")
    print(f"  dropped : {len(shots) - len(ordered)}")
    for reason in sorted({s.drop_reason for s in shots if s.dropped}):
        n = sum(1 for s in shots if s.dropped and s.drop_reason == reason)
        print(f"      {n:5d}  {reason}")
    print(f"  written : {len(ordered)}")

    prefix = HERE / cfg["output"]["prefix"]
    log(f"writing outputs with prefix {prefix}")
    csv_path = prefix.with_suffix(".csv")
    write_csv(csv_path, beads, ordered)
    print(f"\nWrote {csv_path.name}  ({len(ordered)} positions, stage um)")

    if cfg["output"].get("overlay", True):
        log("rendering overlay")
        png = prefix.with_name(prefix.name + "_overlay.png")
        draw_overlay(png, beads, shots, cfg, T, scan,
                     cfg["output"].get("overlay-show", False))
        print(f"Wrote {png.name}")

    if cfg["output"].get("zoom", True):
        log("rendering zoom")
        zp = prefix.with_name("shot_placement_zoom.png")
        if draw_zoom(zp, beads, cfg, T, scan):
            print(f"Wrote {zp.name}")

    if cfg["output"].get("write-flex-txt", True) and ordered:
        fp = prefix.with_name(prefix.name + "_flexImaging.txt")
        write_flex_txt(fp, ordered, cfg, T)
        print(f"Wrote {fp.name}  ({len(read_flex_txt(fp))} rows: "
              f"{len(cfg['fiducials'])} fiducials + {len(ordered)} targets, "
              f"stage um)")

    if cfg["output"].get("write-xeo", True) and ordered:
        M = fit_mtp(cfg)
        files = write_xeo(prefix, ordered, beads, cfg, M)
        for f in files:
            print(f"Wrote {f.name}  ({len(read_xeo(f))} positions)")

        if files and cfg["output"].get("write-run", True):
            names = [position_name(cfg, i, sh) for i, sh in enumerate(ordered)]
            for idx, f in enumerate(files):
                s0 = idx * XEO_MAX_POSITIONS
                rp = write_run(f.with_suffix(".run"),
                               names[s0:s0 + XEO_MAX_POSITIONS], cfg)
                print(f"Wrote {rp.name}  ({len(read_run(rp))} positions)"
                      f"  geometry={rp.stem}")



# =====================================================================
# REVIEW
# =====================================================================

def review(cfg: dict) -> None:
    """
    Show where the shots would land, before anything is exported.

    Sits between 'select' and 'run': fits the registration, detects
    and filters beads (honouring manual_selection.csv), places shots,
    and renders them on the scan. Saves <prefix>_review.png and, with
    output.review-show true and a display available, opens the same
    picture in a window. Writes NO target files -- 'run' remains the
    only command that exports.
    """
    log("fitting registration from fiducials")
    T = to_microns(transform_from_config(cfg), cfg)

    fids = cfg["fiducials"]
    src = np.array([[f["x_px"], f["y_px"]] for f in fids], float)
    dst = np.array([[f["x_um"], f["y_um"]] for f in fids], float)
    r = residuals(T, src, dst)
    say(f"Registration   : {len(fids)} fiducials, "
        f"RMS residual {np.sqrt((r ** 2).mean()):.2f} um")
    limit = float(cfg.get("max-fiducial-residual-um", 25))
    if r.max() > limit:
        say(f"  WARNING: max fiducial residual {r.max():.2f} um exceeds "
            f"{limit:.0f} um.\n  The drawn shot positions inherit that "
            f"error. Run 'check' and fix the\n  fiducials before "
            f"trusting this picture.")

    beads, scan = build_beads(cfg, T)
    log(f"placing {len(cfg['laser-shot-angles'])} shots per accepted bead")
    shots = place_shots(beads, cfg, T)
    n_live = sum(not s.dropped for s in shots)
    say(f"Planned shots  : {n_live} on "
        f"{sum(b.accepted for b in beads)} accepted beads "
        f"({len(shots) - n_live} dropped)")
    for reason in sorted({s.drop_reason for s in shots if s.dropped}):
        n = sum(1 for s in shots if s.dropped and s.drop_reason == reason)
        say(f"      {n:5d}  {reason}")

    prefix = HERE / cfg["output"]["prefix"]
    png = prefix.with_name(prefix.name + "_review.png")
    show = bool(cfg["output"].get("review-show", True))
    if show:
        import matplotlib
        if matplotlib.get_backend().lower() in (
                "agg", "pdf", "ps", "svg", "template"):
            say("  no display backend: saving the picture only "
                "(see doctor)")
            show = False
    draw_overlay(png, beads, shots, cfg, T, scan, show)
    say(f"Wrote {png.name}")
    say("Nothing exported. Run 'run' to write the target files.")


# =====================================================================
# DOCTOR
# =====================================================================

def doctor() -> None:
    """Environment and configuration check. Run this first when
    something will not start."""
    import platform, os
    ok = True

    say("--- environment ---")
    say(f"python        {platform.python_version()}  ({sys.executable})")
    if sys.version_info < (3, 10):
        say("  FAIL: Python 3.10 or newer is required")
        ok = False
    say(f"platform      {platform.system()} {platform.release()}")
    say(f"script dir    {HERE}")
    say(f"writable      {os.access(HERE, os.W_OK)}")

    say("\n--- packages ---")
    need = {"numpy": "numpy", "scipy": "scipy",
            "cv2": "opencv-python", "matplotlib": "matplotlib"}
    for mod, pkg in need.items():
        try:
            m = __import__(mod)
            say(f"{pkg:16s} {getattr(m, '__version__', '?')}")
        except ImportError:
            say(f"{pkg:16s} NOT INSTALLED   pip install {pkg}")
            ok = False

    say("\n--- matplotlib backend ---")
    try:
        import matplotlib
        b = matplotlib.get_backend()
        say(f"backend       {b}")
        if b.lower() in ("agg", "pdf", "ps", "svg", "template"):
            say("  FAIL for 'pick' and 'select': no window can open.")
            say("  Fix: pip install pyqt5   or   set MPLBACKEND=TkAgg")
            ok = False
    except ImportError:
        pass

    say("\n--- config ---")
    try:
        cfg = load_config()
    except SystemExit as e:
        say(f"CONFIG  INVALID: {e}")
        return
    say("CONFIG  OK")
    sp = cfg["shot-placement"]
    say(f"placement         {sp['distance-reference']}"
        + (f"   ({sp['laser-distance']} um from centre)"
           if sp['distance-reference'] == 'center'
           else f"   (measured radius + {sp['edge-offset']} um)"))
    roi = cfg["detection"].get("roi")
    say(f"detection roi     {roi if roi else 'whole image'}")

    nf = len(cfg.get("fiducials") or [])
    say(f"fiducials         {nf}"
        + ("   need >=3, run 'pick'" if nf < 3 else
           "   (>=4 enables leave-one-out)" if nf < 4 else ""))
    if nf < 3:
        ok = False

    nm = len(cfg.get("mtp_calibration") or [])
    say(f"mtp_calibration   {nm}"
        + ("   .xeo will be SKIPPED until 3 are measured" if nm < 3 else ""))
    say(f"write-xeo         {cfg['output'].get('write-xeo', False)}")

    say("\n--- input ---")
    src = cfg["input"]
    if src.get("beads"):
        bp = HERE / src["beads"]
        say(f"bead list     {bp}   {'OK' if bp.exists() else 'MISSING'}")
        ok = ok and bp.exists()
    else:
        sp = HERE / src["scan"]
        if not sp.exists():
            say(f"scan          {sp}   MISSING")
            ok = False
        else:
            say(f"scan          {sp.name}  "
                f"{sp.stat().st_size / 1e6:.1f} MB")
            if sp.suffix.lower() in (".jpg", ".jpeg"):
                say("  FAIL: JPEG is not read. Convert with:")
                say("    python microMS_beadtargeting.py convert "
                    f"{sp.name}")
                ok = False
            else:
                try:
                    import cv2
                    im = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        say("  FAIL: opencv could not decode this file")
                        ok = False
                    else:
                        say(f"  decoded {im.shape[1]}x{im.shape[0]} "
                            f"{im.dtype}, grey {im.min()}-{im.max()}")
                        roi = cfg["detection"].get("roi")
                        if roi:
                            if (roi["x1"] > im.shape[1]
                                    or roi["y1"] > im.shape[0]):
                                say("  WARNING: detection.roi extends past "
                                    "the image edge")
                            else:
                                say(f"  roi {roi['x0']},{roi['y0']} -> "
                                    f"{roi['x1']},{roi['y1']}  inside image")
                except ImportError:
                    pass

    sel = SELECTION_PATH
    say(f"\nmanual_selection.csv  "
        f"{len(load_manual()) if sel.exists() else 'none'}"
        f"{' overrides' if sel.exists() else ''}")

    say("\n" + ("All checks passed." if ok else
                "One or more checks FAILED -- see above."))


# =====================================================================
# SELF TEST
# =====================================================================

def selftest() -> None:
    print("1. registration round-trip")
    rng = np.random.default_rng(0)
    true = Transform(9.74,
                     np.array([[math.cos(0.05), -math.sin(0.05)],
                               [math.sin(0.05), math.cos(0.05)]]),
                     np.array([1200.0, 3400.0]))
    src = rng.uniform(0, 3000, (5, 2))
    dst = true.px_to_um(src)
    T = fit_similarity(src, dst)
    assert abs(T.um_per_px - 9.74) < 1e-6, T.um_per_px
    assert residuals(T, src, dst).max() < 1e-6
    assert loo_residuals(src, dst).max() < 1e-6
    print("   scale, rotation and LOO recovered exactly")

    print("2. affine-vs-similarity rationale")
    T3 = fit_similarity(src[:3], dst[:3])
    assert residuals(T3, src[:3], dst[:3]).max() < 1e-6
    assert loo_residuals(src[:3], dst[:3]) is None
    print("   3 fiducials give no LOO estimate, as designed")

    print("3. clump screen on synthetic images")
    import cv2
    canvas = np.zeros((200, 600), np.uint8)
    cv2.circle(canvas, (100, 100), 20, 255, -1)                 # single
    cv2.circle(canvas, (280, 100), 20, 255, -1)                 # touching
    cv2.circle(canvas, (312, 100), 20, 255, -1)                 #   pair
    for c in ((470, 90), (500, 108), (472, 126)):               # triple
        cv2.circle(canvas, c, 20, 255, -1)
    dd = {"min-diameter-px": 10, "max-diameter-px": 200,
          "background-kernel-px": 151, "threshold": 60,
          "min-circularity": 0.70, "max-aspect-ratio": 1.8,
          "min-solidity": 0.90, "screen-clumps": True}
    found = _detect_flatfield(canvas, dd)
    found.sort(key=lambda f: f[0])
    flags = [f[3] for f in found]
    assert flags == [False, True, True], (len(found), flags)
    print("   single kept, touching pair and triple both flagged")

    dd["screen-clumps"] = False
    assert len(_detect_flatfield(canvas, dd)) == 1, "unscreened: shape only"
    print("   with screening off, only the single survives shape filters")

    print("4. synthetic slide")
    beads = ([Bead(500 + 10 * i, 500, 8.2) for i in range(6)] +
             [Bead(1000 + 200 * i, 1200, 8.2) for i in range(10)] +
             [Bead(1005, 1210, 2.0)])
    cfg = load_config()
    cfg["fiducials"] = [
        {"x_px": 0, "y_px": 0, "x_um": 0, "y_um": 0},
        {"x_px": 2000, "y_px": 0, "x_um": 19480, "y_um": 0},
        {"x_px": 0, "y_px": 1500, "x_um": 0, "y_um": 14610},
    ]
    T = transform_from_config(cfg)
    to_stage(beads, T)
    isolation_filter(beads, cfg["min-bead-separation"])
    shape_filter(beads, cfg)
    acc = [b for b in beads if b.accepted]
    assert not any(b.accepted for b in beads[:6]), "packed row must fail"
    assert not beads[16].accepted, "debris must fail"
    assert not beads[6].accepted, "bead beside debris must fail isolation"
    print(f"   {len(acc)}/{len(beads)} accepted; packed row, debris and "
          f"the bead beside debris all correctly rejected")

    print("5. edge vs centre placement")
    cfg["shot-placement"]["distance-reference"] = "edge"
    r_edge = shot_radius(acc[0], cfg)
    cfg["shot-placement"]["distance-reference"] = "center"
    r_ctr = shot_radius(acc[0], cfg)
    print(f"   edge: {r_edge:.1f} um   centre: {r_ctr:.1f} um   "
          f"(bead measured {acc[0].diameter_um:.1f} um)")
    cfg["shot-placement"]["distance-reference"] = "edge"

    print("6. shots")
    shots = place_shots(beads, cfg)
    live = serpentine(beads, shots)
    expected = sum(len(circular_pack(b.diameter_um / 2, cfg)) for b in acc)
    assert len(shots) == expected, (len(shots), expected)
    print(f"   {len(shots)} placed, {len(live)} survive validation")

    print("7. .xeo split and round-trip")
    # microMS's shipped ultrafleXtreme calibration
    cfg["mtp_calibration"] = [
        {"name": "C20", "x_um": -23215, "y_um": -13605},
        {"name": "C5", "x_um": -90705, "y_um": -13715},
        {"name": "G20", "x_um": -23190, "y_um": -31610},
        {"name": "G5", "x_um": -90680, "y_um": -31715},
    ]
    M = fit_mtp(cfg)
    assert M is not None
    fake = [Shot(0, 0, -50000.0 + i, -20000.0, float(i), 0.0)
            for i in range(950)]
    tmp = HERE / "_selftest"
    files = write_xeo(tmp, fake, beads, cfg, M)
    counts = [len(read_xeo(f)) for f in files]
    assert counts == [400, 400, 150], counts
    print(f"   950 positions -> {len(files)} files {counts} via lines[13:-12]")
    for f in files:
        f.unlink()

    print("8. empty mtp_calibration blocks .xeo")
    cfg["mtp_calibration"] = []
    assert fit_mtp(cfg) is None
    print("   returns None, CSV still written")

    print("\nall checks passed")


# =====================================================================

def main() -> None:
    global VERBOSE
    VERBOSE = bool({"-v", "--verbose"} & set(sys.argv))

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd and not cmd.startswith("-"):
        banner(cmd)

    if cmd == "doctor":
        doctor()
    elif cmd == "convert":
        cli_convert(sys.argv[2:])
    elif cmd == "pick":
        pick_fiducials(load_config())
    elif cmd == "select":
        bead_manual_selection(load_config())
    elif cmd == "check":
        report_registration(load_config())
    elif cmd == "review":
        review(load_config())
    elif cmd == "run":
        run(load_config())
    elif cmd == "selftest":
        selftest()
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
