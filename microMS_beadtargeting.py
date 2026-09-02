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
files interoperate; they are marked FORMAT SPEC. microMS is MIT
licensed ((c) 2016 troycomi) and is vendored unmodified under microms/;
see ATTRIBUTION.md.

Usage
-----
    python microMS_beadtargeting.py gui       # windows: parameters -> beads -> fiducials
    python microMS_beadtargeting.py select    # bead selection
    python microMS_beadtargeting.py pick      # click fiducials -> saved here
    python microMS_beadtargeting.py review    # planned shots + check.txt, no export
    python microMS_beadtargeting.py run       # detect, filter, shoot, export

Add -v (or --verbose) to any command for step-by-step tracing with
timings.
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
# PROGRESS
#
# One console line per command that fills as the stages complete, so
# a 30-second detection on a 48-megapixel scan does not look like a
# hang:
#
#     run    [#########-------------]  41%  4/9 detecting beads     8.2s
#
# A background thread keeps the current notch pulsing while a blocking
# call runs (cv2.imread, the 101-px median filter, savefig), because
# those calls cannot report progress themselves; loops that can report
# it call bar_tick and the notch becomes a real fraction.
#
#     bar_start("run", 9)            at the top of a command
#     bar_step("reading scan")       one call per stage, anywhere below
#     bar_tick(i, n)                 optional, inside a long loop
#     bar_done()                     at the end
#
# The stage calls are no-ops when no bar is running, so build_beads and
# friends behave exactly as before under pytest. When
# stdout is not a terminal (output redirected to a file) the carriage-
# return redraw would be garbage, so each stage prints one plain line
# instead. Other output while a bar line is pending -- say(), print(),
# a traceback -- is moved to a fresh line by the stdout proxy, so
# nothing is ever overwritten.
# =====================================================================

_BAR = None


class _BarStdout:
    """sys.stdout wrapper: starts a new line before anything that is
    not a bar redraw, and serialises the pulse thread's writes. Other
    output marks the bar as interrupted, so it is not redrawn again
    underneath until the next stage -- one bar line per stage at most."""

    def __init__(self, raw):
        import threading
        self.raw, self.pending, self.lock = raw, False, threading.Lock()
        self.interrupted = False

    def write(self, s):
        with self.lock:
            if s.startswith("\r"):
                self.pending = True
            elif s:
                if self.pending:
                    self.raw.write("\n")
                    self.pending = False
                self.interrupted = True
            return self.raw.write(s)

    def flush(self):
        self.raw.flush()

    def __getattr__(self, name):
        return getattr(self.raw, name)


class Bar:
    WIDTH = 22

    def __init__(self, title: str, steps: int):
        import threading
        import time
        self.title = title
        self.steps = max(int(steps), 1)
        self.i = 0                      # stages started so far
        self.label = ""
        self.frac = None                # progress inside the stage, if known
        self.t0 = time.time()
        self.shown = False              # quick commands never draw a bar
        self.DELAY = 0.5
        self.tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.out = sys.stdout
        if self.tty:
            self.out = sys.stdout = _BarStdout(sys.stdout)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pulse, daemon=True)
        self._thread.start()

    def _line(self, k=None) -> str:
        import time
        done = 0.0
        if self.i:
            done = (self.i - 1 + (self.frac or 0.0)) / self.steps
        done = min(max(done, 0.0), 1.0)
        n = int(done * self.WIDTH)
        fill = "#" * n + "-" * (self.WIDTH - n)
        free = self.WIDTH - n
        if k is not None and self.frac is None and free >= 5:
            span = free - 3                     # a 3-wide block bounces
            p = k % (2 * span)                  # in the unfilled part
            if p > span:
                p = 2 * span - p
            j = n + p
            fill = fill[:j] + "###" + fill[j + 3:]
        return (f"\r{self.title:<7.7}[{fill}] {done * 100:3.0f}% "
                f"{self.i:>2}/{self.steps} {self.label:<26.26} "
                f"{time.time() - self.t0:5.1f}s")

    def _draw(self, k=None) -> None:
        self.out.write(self._line(k))
        self.out.flush()
        self.shown = True

    def _pulse(self):
        import time
        k = 0
        while not self._stop.wait(0.12):
            if (self.tty and self.i and time.time() - self.t0 >= self.DELAY
                    and not self.out.interrupted):
                self._draw(k)
            k += 1

    def step(self, label: str) -> None:
        import time
        self.i = min(self.i + 1, self.steps)
        self.label, self.frac = label, None
        if self.tty:
            self.out.interrupted = False
            if time.time() - self.t0 >= self.DELAY:
                self._draw(0)
        else:
            self.out.write(f"  [{self.i}/{self.steps}] {label}  "
                           f"({time.time() - self.t0:.1f}s)\n")
            self.out.flush()

    def tick(self, i: int, n: int) -> None:
        self.frac = min(max(i / max(n, 1), 0.0), 1.0)   # drawn by _pulse

    def done(self) -> None:
        import time
        self._stop.set()
        self._thread.join(timeout=1.0)
        self.i, self.frac, self.label = self.steps, 1.0, "done"
        if self.tty:
            if self.shown:
                self.out.interrupted = False
                self.out.write(self._line())
                self.out.write("\n")
            if sys.stdout is self.out:
                sys.stdout = self.out.raw
        elif time.time() - self.t0 >= self.DELAY:
            self.out.write(f"  done in {time.time() - self.t0:.1f}s\n")
        self.out.flush()


def bar_start(title: str, steps: int) -> Bar:
    global _BAR
    if _BAR is not None:
        _BAR.done()
    _BAR = Bar(title, steps)
    return _BAR


def bar_step(label: str) -> None:
    if _BAR is not None:
        _BAR.step(label)


def bar_tick(i: int, n: int) -> None:
    if _BAR is not None:
        _BAR.tick(i, n)


def bar_done() -> None:
    global _BAR
    if _BAR is not None:
        _BAR.done()
        _BAR = None


def preload_gui_libs() -> None:
    """Start-up bar for the imports that take seconds on Windows. A
    missing package is reported later by the command that needs it."""
    bar_start("start", 2)
    bar_step("importing opencv")
    try:
        import cv2                                           # noqa: F401
    except ImportError:
        pass
    bar_step("importing matplotlib")
    try:
        import matplotlib.figure                             # noqa: F401
    except ImportError:
        pass
    bar_done()


# =====================================================================
# CONFIG
# =====================================================================

# Filled in by:  python microMS_beadtargeting.py pick
# x_px/y_px = pixel in the scan.  x_um/y_um = stage reading in microns.
#
# Do not reuse these across sessions once the slide has been remounted:
# repositioning shows up as a systematic error at every target.
FIDUCIALS = [
    {"x_px": 262.46, "y_px": 253.34, "x_um": 18601.50, "y_um": -20310.80},
    {"x_px": 9702.07, "y_px": 265.34, "x_um": 86083.10, "y_um": -20161.00},
    {"x_px": 248.16, "y_px": 7364.29, "x_um": 18646.70, "y_um": -69830.80},
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
        # Every generated file lands under this folder, created next
        # to the script on demand. Each invocation gets its own
        # timestamped subfolder -- "<timestamp> run" holding the full
        # export, "<timestamp> review" holding the review images -- so
        # nothing ever overwrites an earlier result. flexCoords.txt
        # stays beside the script: it doubles as an mtp_calibration
        # input path.
        "results-dir": "RESULTS",
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
        bar_tick(i, n - 1)
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
    bar_step(f"reading {scan_path.name}")
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
    bar_step(f"detecting beads ({method})")
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
                 show: bool = False, only_accepted: bool = False) -> None:
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
        if only_accepted and not b.accepted:
            continue
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

def _floats_in(text: str) -> list[str]:
    """Numeric tokens in pasted text, in order. Accepts any separator
    (comma, tab, space, newline) and scientific notation, so a stage
    readout copied as '18601.5, -20310.8' yields both numbers."""
    import re
    return re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)


def _clipboard_text(fig) -> str:
    """Best-effort system clipboard read using whichever GUI toolkit
    is already running the window -- no new dependencies."""
    try:
        return fig.canvas.get_tk_widget().clipboard_get()      # TkAgg
    except Exception:
        pass
    try:
        from matplotlib.backends.qt_compat import QtWidgets    # Qt*Agg
        return QtWidgets.QApplication.clipboard().text()
    except Exception:
        pass
    try:
        import tkinter                                         # fallback
        r = tkinter.Tk()
        r.withdraw()
        try:
            return r.clipboard_get()
        finally:
            r.destroy()
    except Exception:
        return ""


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


# =====================================================================
# RUN
# =====================================================================

def timestamp() -> str:
    """File-name-safe local time, e.g. 2026-08-25_143059."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d_%H%M%S")


def results_dir(cfg: dict, subfolder: str | None = None) -> Path:
    """
    HERE/<output.results-dir>, created on demand -- the home of every
    generated file, so outputs never mix with code and scans.
    """
    root = HERE / cfg["output"].get("results-dir", "RESULTS")
    if subfolder:
        root = root / subfolder
    root.mkdir(parents=True, exist_ok=True)
    return root


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
        bar_step(f"loading {path.name}")
        beads, scan = load_beads_csv(path), None
        say(f"Loaded {len(beads)} objects from {path.name}")
        # the scan is still the background of the overlay and the zoom
        if src.get("scan"):
            scan = Path(src["scan"])
            if not scan.is_absolute():
                scan = HERE / scan
            if not scan.exists():
                scan = None
    else:
        scan = Path(src["scan"])
        if not scan.is_absolute():
            scan = HERE / scan
        beads = detect_blobs(scan, cfg)
        say(f"Detected {len(beads)} objects in {scan.name}")

    bar_step("filtering")
    to_stage(beads, T)
    log("converted pixel centroids to stage microns")
    isolation_filter(beads, float(cfg["min-bead-separation"]))
    log("isolation filter done (run against ALL objects, debris included)")
    shape_filter(beads, cfg)
    log("shape filter done")
    apply_manual(beads, cfg)
    return beads, scan


def run(cfg: dict) -> None:
    bar_start("run", 9)
    bar_step("fitting registration")
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

    bar_step("placing shots")
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

    outdir = results_dir(cfg, timestamp() + " run")
    prefix = outdir / cfg["output"]["prefix"]
    print(f"\nOutput folder  : {outdir.relative_to(HERE)}")
    log(f"writing outputs with prefix {prefix}")
    csv_path = prefix.with_suffix(".csv")
    bar_step(f"writing {csv_path.name}")
    write_csv(csv_path, beads, ordered)
    print(f"\nWrote {csv_path.name}  ({len(ordered)} positions, stage um)")

    if cfg["output"].get("overlay", True):
        bar_step("rendering overlay")
        log("rendering overlay")
        png = prefix.with_name(prefix.name + "_overlay.png")
        draw_overlay(png, beads, shots, cfg, T, scan,
                     cfg["output"].get("overlay-show", False))
        print(f"Wrote {png.name}")

    if cfg["output"].get("zoom", True):
        bar_step("rendering zoom")
        log("rendering zoom")
        zp = prefix.with_name("shot_placement_zoom.png")
        if draw_zoom(zp, beads, cfg, T, scan):
            print(f"Wrote {zp.name}")

    bar_step("writing instrument files")
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
    bar_done()



# =====================================================================
# REVIEW
# =====================================================================

def review(cfg: dict) -> None:
    """
    Show where the shots would land, before anything is exported.

    Sits between 'select' and 'run': fits the registration, detects
    and filters beads (honouring manual_selection.csv), places shots,
    and renders them on the scan. Saves both pictures into their own
    "RESULTS/<timestamp> review" folder: review.png (the full overlay)
    and review_zoom.png (a close-up of the densest patch of selected
    beads). With output.review-show true and a display available the
    overlay also opens in a window. Writes NO target files -- 'run'
    remains the only command that exports.
    """
    bar_start("review", 8)
    bar_step("fitting registration")
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
            f"error. Fix the fiducials in 'pick' before\n  trusting "
            f"this picture.")

    beads, scan = build_beads(cfg, T)
    bar_step("placing shots")
    log(f"placing {len(cfg['laser-shot-angles'])} shots per accepted bead")
    shots = place_shots(beads, cfg, T)
    n_live = sum(not s.dropped for s in shots)
    say(f"Planned shots  : {n_live} on "
        f"{sum(b.accepted for b in beads)} accepted beads "
        f"({len(shots) - n_live} dropped)")
    for reason in sorted({s.drop_reason for s in shots if s.dropped}):
        n = sum(1 for s in shots if s.dropped and s.drop_reason == reason)
        say(f"      {n:5d}  {reason}")

    outdir = results_dir(cfg, timestamp() + " review")

    # 'check' folded in: the registration report goes into the same
    # folder as the pictures, so one folder holds everything needed to
    # judge the run.
    import contextlib
    import io
    bar_step("writing check.txt")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_registration(cfg)
    (outdir / "check.txt").write_text(buf.getvalue().lstrip("\n"))
    say(f"Wrote {(outdir / 'check.txt').relative_to(HERE)}")

    png = outdir / "review.png"
    show = bool(cfg["output"].get("review-show", True))
    if show:
        import matplotlib
        if matplotlib.get_backend().lower() in (
                "agg", "pdf", "ps", "svg", "template"):
            say("  no display backend: saving the picture only")
            show = False
    bar_step("rendering review.png"
             + ("  (close the window to continue)" if show else ""))
    # review.png shows only the beads that will be shot; the zoom keeps
    # every category so clumps and rejects can still be checked.
    draw_overlay(png, beads, shots, cfg, T, scan, show, only_accepted=True)
    say(f"Wrote {png.relative_to(HERE)}")

    zp = outdir / "review_zoom.png"
    bar_step("rendering review_zoom.png")
    if draw_zoom(zp, beads, cfg, T, scan):
        say(f"Wrote {zp.relative_to(HERE)}  (close-up of the densest "
            f"patch of selected beads)")
    say("Nothing exported. Run 'run' to write the target files.")
    bar_done()
    return beads, shots, T


# =====================================================================
# WINDOWED FLOW  ('select' and 'pick')
#
# Windows 95/98-styled tkinter shell for the reordered workflow:
#
#     select parameters  ->  bead selection  ->  correlate fiducials
#
# The original matplotlib windows (bead_manual_selection,
# pick_fiducials) were removed on 2026-09-01; git history has them.
#
# How it connects to the pipeline -- nothing below re-implements it:
#   * a box drawn on the scan is analysed with _detect_blobdetector
#     ("default") or _detect_flatfield ("quick", "strict") on that crop,
#     and the objects go through
#     to_stage, isolation_filter and shape_filter exactly as build_beads
#     runs them;
#   * before any fiducial exists the scale comes from the provisional
#     um/px in the Image Settings panel (a synthetic 3-point
#     registration), and pick replaces it with the real one;
#   * "save and review" / "save and run" hand the window's objects to
#     review() and run() through CONFIG's own input.beads (a CSV) and
#     manual_selection.csv, so the console commands see the same beads
#     and the same decisions as the window.
#
# Saves live in SAVES/ next to the script: <timestamp>[_name].json holds
# the parameters, the scan path, the analysed boxes and every object
# with its decision; last_settings.json remembers the parameter values
# between runs.
#
# Parameters marked * in the window are stored with the save but do not
# change detection yet: their meaning still has to be pinned down.
#
# tkinter is standard library, so this adds no dependency. matplotlib's
# TkAgg canvas is embedded in the window so show_pyramid and
# attach_zoom are reused unchanged. Fonts are tk NAMED fonts: every
# widget refers to a name and the family is chosen once in init_fonts
# (Aptos Slab first, then the fallbacks).
# =====================================================================

try:
    import tkinter as tk
    from tkinter import filedialog, simpledialog
except ImportError:                      # headless install; see _gui_root
    tk = None

GUI_BG = "#808080"          # medium dark grey: window background
GUI_BG_BAR = "#8e8e8e"      # the "v Image Settings v" bar
GUI_BOX = "#d4d4d4"         # lighter grey value boxes and dropdowns
GUI_BTN = "#c0c0c0"         # classic Win95 button face
GUI_LIGHT = "#c8c8c8"       # bevel highlight
GUI_DARK = "#404040"        # bevel shadow
GUI_TEAL = "#008080"        # title strip (Win95 desktop teal)
GUI_LAVENDER = "#e6e0f5"    # faint lavender drop-down panel
GUI_NAVY = "#000080"        # Win95 menu highlight
GUI_TXT = "#000000"
GUI_DIM = "#a6a6a6"         # greyed-out text
GUI_WHITE = "#ffffff"

# Labels and buttons: the Windows system face. Value boxes: a monospace
# face so digits stay sharp and line up. First installed family wins.
GUI_FONT_PREFERENCE = ("Segoe UI", "Tahoma", "DejaVu Sans")
GUI_BOX_FONT_PREFERENCE = ("Consolas", "Courier New", "DejaVu Sans Mono")
GUI_FONT_SIZE = 10
F_LABEL = "ui.label"                     # base size
F_BOX = "ui.box"                         # base size, value boxes
F_SMALL = "ui.small"                     # base - 2
F_TITLE = "ui.title"                     # base bold
F_BTN = "ui.button"                      # base
F_BIG = "ui.big"                         # base + 1 bold
F_ROW = "ui.row"                         # base - 1, table boxes
F_TINY_B = "ui.tinybold"                 # 7 bold, the x buttons
F_SHOW = "ui.show"                       # base - 1 bold, show/hide colours
_FONTS: dict = {}                        # keeps the named fonts alive
GUI_SCALE = 1.0                          # pixels per 96-dpi pixel, see _gui_root


def px(n: float) -> int:
    """Pixel size scaled for the screen's DPI (1.0 at 96 dpi)."""
    return int(round(n * GUI_SCALE))

# bead colours, shared by the window and the review pictures
GUI_COL = {"accepted": "#2ca02c", "clumped": "#9467bd",
           "rejected": "#d62728", "manual": "#1f77b4"}


def init_fonts(root) -> str:
    """Create the named fonts. Runs once, right after tk.Tk()."""
    from tkinter import font as tkfont
    installed = set(tkfont.families(root))
    labels, boxes, base = (GUI_FONT_PREFERENCE, GUI_BOX_FONT_PREFERENCE,
                           GUI_FONT_SIZE)
    family = next((f for f in labels if f in installed), labels[-1])
    box_family = next((f for f in boxes if f in installed), boxes[-1])
    for name, size, weight in ((F_LABEL, base, "normal"),
                               (F_BOX, base, "normal"),
                               (F_SMALL, base - 2, "normal"),
                               (F_TITLE, base, "bold"),
                               (F_BTN, base, "normal"),
                               (F_BIG, base + 1, "bold"),
                               (F_ROW, base - 1, "normal"),
                               (F_TINY_B, 7, "bold"),
                               (F_SHOW, base - 1, "bold")):
        fam = box_family if name in (F_BOX, F_ROW) else family
        if name in _FONTS:
            _FONTS[name].configure(family=fam, size=size, weight=weight)
        else:
            # tkinter deletes a named font when its Python object is
            # garbage-collected, hence the module-level dict.
            _FONTS[name] = tkfont.Font(root, name=name, family=fam,
                                       size=size, weight=weight)
    log(f"gui font: {family}; boxes: {box_family}")
    return family


# ---- About ----------------------------------------------------------
# <...> are placeholders still to be filled in.

ABOUT_LINES = [
    ("Grace <surname>", F_TITLE),
    ("<title>, Lam Lab, <Department>, University of California, Davis",
     F_SMALL),
    ("<email>", F_SMALL),
    ("", F_SMALL),
    ("Dr. Elizabeth K. Neumann", F_TITLE),
    ("<title>, <Department>, <Institution>", F_SMALL),
    ("Co-author of microMS", F_SMALL),
    ("", F_SMALL),
    ("Based on microMS", F_TITLE),
    ("Comi TJ, Neumann EK, Do TD, Sweedler JV. microMS: A Python Platform "
     "for Image-Guided Mass Spectrometry Profiling. J. Am. Soc. Mass "
     "Spectrom. 2017, 28(9), 1919-1928.", F_SMALL),
    ("DOI 10.1007/s13361-017-1704-1   (click to open)", F_SMALL),
    ("microMS is MIT licensed, (c) 2016 troycomi; vendored under microms/",
     F_SMALL),
]
PAPER_URL = "https://doi.org/10.1007/s13361-017-1704-1"


def _open_paper() -> None:
    import webbrowser
    webbrowser.open(PAPER_URL)


def about_dialog(parent):
    dlg = tk.Toplevel(parent, bg=GUI_BG)
    dlg.title("About microMS_beadtargeting")
    dlg.resizable(False, False)
    gui_title_strip(dlg, "About microMS_beadtargeting")
    body = tk.Frame(dlg, bg=GUI_BG, padx=16, pady=12)
    body.pack(fill="both", expand=True)
    gui_label(body, f"microMS_beadtargeting {VERSION}", font=F_BIG).pack(
        anchor="w")
    gui_small(body, "image-guided MALDI-MSI targeting of SPPS beads on the "
                    "Bruker timsTOF fleX").pack(anchor="w", pady=(0, 8))
    for text, font in ABOUT_LINES:
        lb = gui_label(body, text, font=font, wraplength=px(520),
                       justify="left")
        lb.pack(anchor="w")
        if text.startswith("DOI"):
            lb.config(fg=GUI_NAVY, cursor="hand2")
            lb.bind("<Button-1>", lambda _e: _open_paper())
    gui_button(body, "OK", dlg.destroy, width=8).pack(anchor="e",
                                                       pady=(12, 0))
    dlg.transient(parent)
    return dlg


def about_menu_items(win) -> list:
    return [("About microMS_beadtargeting...", lambda: about_dialog(win)),
            ("Open the microMS paper (DOI)", _open_paper)]


# ---- widget helpers -------------------------------------------------
# Every control is built from these so the look stays consistent.

def gui_title_strip(parent, text: str):
    """Teal Win95 title strip with the decorative _ [] X squares."""
    strip = tk.Frame(parent, bg=GUI_TEAL, height=px(26))
    strip.pack(fill="x", side="top")
    strip.pack_propagate(False)
    lab = tk.Label(strip, text=text, bg=GUI_TEAL, fg=GUI_WHITE, font=F_TITLE,
                   padx=8)
    lab.pack(side="left")
    for glyph in ("X", "□", "_"):
        tk.Label(strip, text=glyph, bg=GUI_BTN, fg=GUI_TXT, font=F_SMALL,
                 width=2, relief="raised", bd=2).pack(side="right", padx=1,
                                                      pady=3)
    strip.label = lab
    return strip


def gui_menu_bar(win, items: dict):
    """items = {"File": [("Save", cb), None, ("Load...", cb)], ...}
    None inserts a separator."""
    bar = tk.Menu(win, bg=GUI_BTN, fg=GUI_TXT, activebackground=GUI_NAVY,
                  activeforeground=GUI_WHITE, font=F_BTN, relief="flat",
                  bd=0)
    for name, entries in items.items():
        m = tk.Menu(bar, tearoff=0, bg=GUI_BOX, fg=GUI_TXT,
                    activebackground=GUI_NAVY, activeforeground=GUI_WHITE,
                    font=F_BTN, relief="raised", bd=2)
        for e in entries:
            if e is None:
                m.add_separator()
            else:
                m.add_command(label=e[0], command=e[1])
        bar.add_cascade(label=name, menu=m)
    win.config(menu=bar)
    return bar


def gui_label(parent, text, font=F_LABEL, fg=GUI_TXT, bg=GUI_BG, **kw):
    return tk.Label(parent, text=text, font=font, fg=fg, bg=bg, **kw)


def gui_small(parent, text, **kw):
    """8 pt helper text beside a box, e.g. um or +/-."""
    return gui_label(parent, text, font=F_SMALL, **kw)


def gui_box(parent, var, width: int = 6):
    """Light grey sunken value box, 12 pt."""
    return tk.Entry(parent, textvariable=var, width=width, font=F_BOX,
                    bg=GUI_BOX, fg=GUI_TXT, relief="sunken", bd=2,
                    insertbackground=GUI_TXT, highlightthickness=0,
                    disabledbackground=GUI_BG_BAR,
                    disabledforeground=GUI_DIM)


def gui_check(parent, var, text: str = "", font=F_LABEL, bg=GUI_BG, **kw):
    return tk.Checkbutton(parent, text=text, variable=var, font=font,
                          bg=bg, fg=GUI_TXT, activebackground=bg,
                          activeforeground=GUI_TXT, selectcolor=GUI_BOX,
                          disabledforeground=GUI_DIM, highlightthickness=0,
                          **kw)


def gui_toggle(parent, var, text: str, command=None, font=F_BTN):
    """A button that stays pressed: Win95's toolbar toggle."""
    return tk.Checkbutton(parent, text=text, variable=var, command=command,
                          font=font, indicatoron=False, bg=GUI_BTN,
                          fg=GUI_TXT, selectcolor=GUI_LAVENDER,
                          activebackground=GUI_BOX, activeforeground=GUI_TXT,
                          relief="raised", bd=2, highlightthickness=0,
                          padx=6, pady=2)


def gui_dropdown(parent, var, options: list, command=None):
    """Light grey raised dropdown, sized to the longest option so it
    never changes width. Hovered entries highlight navy/white."""
    width = max(len(o) for o in options) + 2
    mb = tk.Menubutton(parent, textvariable=var, width=width, font=F_BOX,
                       bg=GUI_BOX, fg=GUI_TXT, activebackground=GUI_BOX,
                       activeforeground=GUI_TXT, relief="raised", bd=2,
                       anchor="w", indicatoron=False, highlightthickness=0)
    menu = tk.Menu(mb, tearoff=0, bg=GUI_BOX, fg=GUI_TXT,
                   activebackground=GUI_NAVY, activeforeground=GUI_WHITE,
                   font=F_BOX, relief="raised", bd=2)
    for o in options:
        menu.add_radiobutton(label=o, variable=var, value=o,
                             command=command)
    mb.config(menu=menu)
    arrow = tk.Label(mb, text="▼", bg=GUI_BOX, fg=GUI_TXT, font=F_TINY_B)
    arrow.place(relx=1.0, rely=0.5, anchor="e", x=-4)
    # the label sits on top of the button and would swallow the click
    arrow.bind("<Button-1>", lambda _e: menu.tk_popup(
        mb.winfo_rootx(), mb.winfo_rooty() + mb.winfo_height()))
    return mb


def gui_button(parent, text, command=None, font=F_BTN, width=None, **kw):
    return tk.Button(parent, text=text, command=command, font=font,
                     bg=GUI_BTN, fg=GUI_TXT, activebackground=GUI_BOX,
                     activeforeground=GUI_TXT, relief="raised", bd=2,
                     width=width, highlightthickness=0, padx=6, **kw)


def gui_hrule(parent, bg=GUI_BG):
    f = tk.Frame(parent, bg=bg)
    tk.Frame(f, bg=GUI_DARK, height=1).pack(fill="x")
    tk.Frame(f, bg=GUI_LIGHT, height=1).pack(fill="x")
    return f


def gui_ask(parent, title: str, text: str, yes: str = "Yes",
            no: str = "Exit") -> bool:
    """Modal Win95 question box with two named buttons."""
    dlg = tk.Toplevel(parent, bg=GUI_BG)
    dlg.title(title)
    dlg.resizable(False, False)
    gui_title_strip(dlg, title)
    body = tk.Frame(dlg, bg=GUI_BG, padx=16, pady=12)
    body.pack(fill="both", expand=True)
    gui_label(body, text, wraplength=px(420), justify="left").pack(anchor="w")
    answer = {"yes": False}

    def choose(val):
        answer["yes"] = val
        dlg.destroy()

    bf = tk.Frame(body, bg=GUI_BG)
    bf.pack(anchor="e", pady=(12, 0))
    yes_btn = gui_button(bf, yes, lambda: choose(True), width=8)
    yes_btn.pack(side="left", padx=4)
    gui_button(bf, no, lambda: choose(False), width=8).pack(side="left")
    dlg.bind("<Return>", lambda _e: choose(True))
    dlg.bind("<Escape>", lambda _e: choose(False))
    dlg.transient(parent)
    # grab_set on a window that is not on screen yet fails ("window not
    # viewable") and left the box open but dead. Draw it, then keep
    # trying the grab briefly; the box works without a grab anyway.
    dlg.update_idletasks()
    dlg.update()

    def try_grab(attempt=0):
        if not dlg.winfo_exists():
            return
        try:
            dlg.grab_set()
        except tk.TclError:
            if attempt < 20:
                dlg.after(50, try_grab, attempt + 1)

    try_grab()
    dlg.focus_force()
    yes_btn.focus_set()
    parent.wait_window(dlg)
    return answer["yes"]


def gui_image_canvas(parent, img, bg=GUI_BG):
    """matplotlib figure embedded in tk, drawn through show_pyramid so
    the 48-megapixel scan stays responsive. Returns (fig, ax, canvas)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    fig = Figure(figsize=(9, 6), dpi=100, facecolor=bg)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    w = canvas.get_tk_widget()
    w.config(bg=bg, highlightthickness=0, relief="sunken", bd=2)
    w.pack(fill="both", expand=True)
    # keyboard focus follows the mouse so +/-/f and Shift reach the canvas
    w.bind("<Enter>", lambda _e: w.focus_set())
    if img is not None:
        show_pyramid(fig, ax, img, cmap="gray")
        ax.set_aspect("equal")
    return fig, ax, canvas


def gui_attach_drag_pan(fig, ax, widget=None, when=None, button=1):
    """Dragging with `button` moves the view. Same pixel-delta method
    as the middle-drag pan in attach_zoom, so the image follows the
    mouse exactly instead of accelerating away. `when(ev)` can veto a
    drag (the box tool may share the button)."""
    drag = {"px": None}

    def on_press(ev):
        if (ev.button == button and ev.inaxes is ax
                and (when is None or when(ev))):
            drag["px"] = (ev.x, ev.y, ax.get_xlim(), ax.get_ylim())
            if widget is not None:
                widget.config(cursor="fleur")

    def on_motion(ev):
        if drag["px"] is None or ev.x is None:
            return
        x0px, y0px, xl, yl = drag["px"]
        sx = (xl[1] - xl[0]) / max(ax.bbox.width, 1)
        sy = (yl[1] - yl[0]) / max(ax.bbox.height, 1)
        dx = -(ev.x - x0px) * sx
        dy = -(ev.y - y0px) * sy
        ax.set_xlim(xl[0] + dx, xl[1] + dx)
        ax.set_ylim(yl[0] + dy, yl[1] + dy)
        fig.canvas.draw_idle()

    def on_release(ev):
        if ev.button == button and drag["px"] is not None:
            drag["px"] = None
            if widget is not None:
                widget.config(cursor="")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    return drag


class GuiBoxTool:
    """Rubber-band rectangle on the image, drawn with a left-drag
    whenever `mode_var` (the manual accept/reject toggle) is off. The
    box stays until cleared so the same area can be analysed, accepted
    or rejected again."""

    def __init__(self, fig, ax, widget, mode_var, on_change=None):
        from matplotlib.patches import Rectangle
        self.fig, self.ax, self.widget = fig, ax, widget
        self.mode_var, self.on_change = mode_var, on_change
        self.box = None                     # (x0, x1, y0, y1) data coords
        self.start = None
        self.rect = Rectangle((0, 0), 0, 0, fill=True, fc="#1f77b4",
                              alpha=0.15, ec="#1f77b4", lw=1.5,
                              visible=False)
        ax.add_patch(self.rect)
        fig.canvas.mpl_connect("button_press_event", self._press)
        fig.canvas.mpl_connect("motion_notify_event", self._motion)
        fig.canvas.mpl_connect("button_release_event", self._release)

    def active(self, ev) -> bool:
        return not bool(self.mode_var.get())

    def _press(self, ev):
        if ev.button != 1 or ev.inaxes is not self.ax or not self.active(ev):
            return
        self.start = (ev.xdata, ev.ydata)
        self.rect.set_bounds(ev.xdata, ev.ydata, 0, 0)
        self.rect.set_visible(True)
        self.widget.config(cursor="crosshair")

    def _motion(self, ev):
        if self.start is None or ev.xdata is None:
            return
        x0, y0 = self.start
        self.rect.set_bounds(min(x0, ev.xdata), min(y0, ev.ydata),
                             abs(ev.xdata - x0), abs(ev.ydata - y0))
        self.fig.canvas.draw_idle()

    def _release(self, ev):
        if self.start is None or ev.button != 1:
            return
        x0, y0 = self.start
        self.start = None
        self.widget.config(cursor="")
        if ev.xdata is None or abs(ev.xdata - x0) < 8 or abs(ev.ydata - y0) < 8:
            return                         # a click, not a box
        self.box = (min(x0, ev.xdata), max(x0, ev.xdata),
                    min(y0, ev.ydata), max(y0, ev.ydata))
        self.fig.canvas.draw_idle()
        if self.on_change:
            self.on_change(self.box)

    def clear(self):
        self.box = None
        self.rect.set_visible(False)
        self.fig.canvas.draw_idle()
        if self.on_change:
            self.on_change(None)


def gui_load_scan(path: Path):
    """Greyscale scan for the windows, or None with the reason printed."""
    if not path.exists():
        say(f"Scan not found: {path}")
        return None
    try:
        import cv2
    except ImportError:
        say("opencv-python is required to open the scan.")
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        say(f"Could not read {path}. TIFF, not JPEG -- convert with:\n"
            f"  python microMS_beadtargeting.py convert <file>")
    return img


# ---- parameters, defaults and remembered values ----------------------

GUI_DEFAULTS = {
    "scan": "",
    "image_type": "highres scanner",
    "microscope_slide": False,
    "microscope_zoom": "10",
    "um_per_px": "7.0",             # provisional scale before fiducials
    "global_sweep": True,
    "flat_field": False,
    "bead_um": "90",
    "bead_dev_um": "30",
    "isolation_on": True,
    "isolation_um": "150",
    "max_points": "1000",
    "matrix": "yes",
    "method": "default",
}

# stored with the save, no effect on detection yet (marked * in the window)
GUI_NO_EFFECT = ("image_type", "microscope_slide", "microscope_zoom",
                 "matrix")

# strict method: fraction of ALL objects found that is kept as beads,
# best fit first (size closest to nominal, then most isolated)
GUI_STRICT_KEEP = 0.05

SAVES_DIR = HERE / "SAVES"
LAST_SETTINGS = SAVES_DIR / "last_settings.json"


def gui_saves_dir() -> Path:
    SAVES_DIR.mkdir(parents=True, exist_ok=True)
    return SAVES_DIR


def gui_params(v: dict) -> dict:
    return {k: v[k].get() for k in GUI_DEFAULTS}


def gui_settings_save(v: dict) -> None:
    """Remember the current values for the next run."""
    import json
    try:
        gui_saves_dir()
        LAST_SETTINGS.write_text(json.dumps(gui_params(v), indent=1))
    except OSError as e:
        log(f"could not write {LAST_SETTINGS.name}: {e}")


def gui_settings_load() -> dict:
    import json
    try:
        data = json.loads(LAST_SETTINGS.read_text())
        return {k: data[k] for k in GUI_DEFAULTS if k in data}
    except (OSError, ValueError):
        return {}


def gui_make_vars(master, cfg: dict) -> dict:
    """Defaults, then CONFIG's geometry, then whatever was used last."""
    v = {}
    for k, d in GUI_DEFAULTS.items():
        v[k] = (tk.BooleanVar(master, d) if isinstance(d, bool)
                else tk.StringVar(master, d))
    v["scan"].set(str(cfg["input"]["scan"]))
    v["bead_um"].set(f"{float(cfg['bead-diameter']):g}")
    v["isolation_um"].set(f"{float(cfg['min-bead-separation']):g}")
    for k, val in gui_settings_load().items():
        if k == "scan" and not val:
            continue
        v[k].set(val)
    return v


def gui_restore_defaults(v: dict, cfg: dict) -> None:
    for k, d in GUI_DEFAULTS.items():
        v[k].set(d)
    v["scan"].set(str(cfg["input"]["scan"]))


def gui_float(var, fallback: float) -> float:
    try:
        return float(var.get())
    except (TypeError, ValueError):
        return fallback


def gui_apply_params(cfg: dict, v: dict) -> dict:
    """Write the window's values into a CONFIG copy. Only the keys the
    pipeline already has; everything else is recorded, not applied."""
    bead = gui_float(v["bead_um"], float(cfg["bead-diameter"]))
    dev = gui_float(v["bead_dev_um"], 0.35 * bead)
    cfg["bead-diameter"] = bead
    cfg["bead-diameter-tolerance"] = max(dev / bead, 0.01) if bead else 0.35
    cfg["min-bead-separation"] = (
        gui_float(v["isolation_um"], float(cfg["min-bead-separation"]))
        if v["isolation_on"].get() else 0.0)
    cfg["detection"]["method"] = ("blob" if v["method"].get() == "default"
                                  else "flatfield")
    cfg["gui-max-points"] = int(gui_float(v["max_points"], 1000))
    cfg["gui-method"] = v["method"].get()
    return cfg


# ---- state shared by the three windows ---------------------------------

class GuiState:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.v = None
        self.img = None
        self.path = None                 # Path of the loaded scan
        self.beads: list = []            # every object found so far
        self.boxes: list = []            # analysed regions, data coords
        self.fids = [dict(f) for f in cfg.get("fiducials") or []]
        self.save_path = None            # current SAVES/*.json
        self.T = None
        self.on_beads_changed = None     # the bead window hooks this

    def active_fids(self) -> list:
        return [f for f in self.fids if not f.get("hide")]


def gui_provisional_fiducials(v: dict) -> list:
    """Three synthetic fiducials that encode only a scale, so the
    pipeline can filter in microns before the slide has been on the
    stage. Y is reflected like every Bruker mapper."""
    s = gui_float(v["um_per_px"], 7.0)
    return [{"x_px": 0.0, "y_px": 0.0, "x_um": 0.0, "y_um": 0.0},
            {"x_px": 1000.0, "y_px": 0.0, "x_um": 1000.0 * s, "y_um": 0.0},
            {"x_px": 0.0, "y_px": 1000.0, "x_um": 0.0, "y_um": -1000.0 * s}]


def gui_fit_fiducials(state: GuiState) -> tuple:
    """(fiducials to fit with, True if they are the real ones)."""
    real = state.active_fids()
    if len(real) >= 3:
        return real, True
    return gui_provisional_fiducials(state.v), False


def gui_transform(state: GuiState, cfg: dict):
    fids, _ = gui_fit_fiducials(state)
    return to_microns(transform_from_config({**cfg, "fiducials": fids}), cfg)


def gui_detect_region(img, box, cfg: dict, method: str) -> list:
    """Run the pipeline's detector on one crop of the scan. Returns
    Bead objects in full-image pixel coordinates."""
    import cv2
    h, w = img.shape[:2]
    x0, x1 = int(max(box[0], 0)), int(min(box[1], w))
    y0, y1 = int(max(box[2], 0)), int(min(box[3], h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return []
    d = dict(cfg["detection"])
    sub = img[y0:y1, x0:x1]
    g = cv2.bitwise_not(sub) if d.get("invert", True) else sub
    if method == "default":
        found = _detect_blobdetector(g, d)            # global threshold sweep
    elif method == "quick":
        found = _detect_flatfield(g, d)
    else:
        # strict: both detectors, objects combined (an object seen by
        # both counts once); gui_refilter then keeps only the best few
        found = list(_detect_flatfield(g, d))
        extra = _detect_blobdetector(g, d)
        if found and extra:
            tol = float(cfg.get("manual-selection", {}).get("match-radius-px",
                                                            12))
            tree = cKDTree(np.array([[f[0], f[1]] for f in found]))
            dd, _ = tree.query(np.array([[f[0], f[1]] for f in extra]))
            extra = [f for f, d_ in zip(extra, dd) if d_ > tol]
        found.extend(extra)
    return [Bead(x + x0, y + y0, size, clumped=cl) for x, y, size, cl in found]


def gui_merge(state: GuiState, new: list, cfg: dict) -> int:
    """Add objects not already known (boxes may overlap)."""
    tol = float(cfg.get("manual-selection", {}).get("match-radius-px", 12))
    if state.beads and new:
        tree = cKDTree(np.array([[b.x_px, b.y_px] for b in state.beads]))
        d, _ = tree.query(np.array([[b.x_px, b.y_px] for b in new]))
        new = [b for b, dd in zip(new, d) if dd > tol]
    state.beads.extend(new)
    return len(new)


def gui_set_manual(b, decision: str) -> None:
    b.manual = decision
    b.accepted = (decision == "accept")
    b.reject_reason = "" if b.accepted else "manually rejected"
    b.reject_category = "" if b.accepted else "manual"


def gui_refilter(state: GuiState) -> dict:
    """The pipeline's own filters over everything found so far, then
    the manual decisions, the strict ranking and the max-points cap."""
    import copy
    cfg = gui_apply_params(copy.deepcopy(state.cfg), state.v)
    beads = state.beads
    T = gui_transform(state, cfg)
    state.T = T
    to_stage(beads, T)
    isolation_filter(beads, float(cfg["min-bead-separation"]))
    shape_filter(beads, cfg)
    for b in beads:
        if b.manual:
            gui_set_manual(b, b.manual)

    nominal = float(cfg["bead-diameter"])
    auto = [b for b in beads if b.accepted and not b.manual]
    ranked = sorted(auto, key=lambda b: (abs(b.diameter_um - nominal),
                                         -b.nn_um))
    if cfg["gui-method"] == "strict":
        # keep only the best-fitting few: 5 % of everything found
        keep = max(1, int(math.ceil(GUI_STRICT_KEEP * len(beads))))
        for b in ranked[keep:]:
            b.accepted = False
            b.reject_reason = (f"strict: outside the top "
                               f"{GUI_STRICT_KEEP:.0%} by fit")
            b.reject_category = "strict"
        ranked = ranked[:keep]
    cap = cfg["gui-max-points"] - sum(1 for b in beads
                                      if b.accepted and b.manual)
    for b in ranked[max(cap, 0):]:
        b.accepted = False
        b.reject_reason = "over max number of points"
        b.reject_category = "over max points"
    return cfg


def gui_bead_category(b) -> str:
    if b.accepted:
        return "accepted"
    if b.clumped:
        return "clumped"
    return "rejected"


def gui_bead_colour(b) -> str:
    if b.accepted:
        return GUI_COL["accepted"]
    if b.clumped:
        return GUI_COL["clumped"]
    if b.reject_category == "manual":
        return GUI_COL["manual"]
    return GUI_COL["rejected"]


# ---- saves ----------------------------------------------------------------

def gui_beads_to_json(beads: list) -> list:
    return [{"x_px": round(b.x_px, 2), "y_px": round(b.y_px, 2),
             "diameter_px": round(b.diameter_px, 2),
             "clumped": bool(b.clumped), "manual": b.manual}
            for b in beads]


def gui_beads_from_json(rows: list) -> list:
    out = []
    for r in rows:
        b = Bead(float(r["x_px"]), float(r["y_px"]), float(r["diameter_px"]),
                 clumped=bool(r.get("clumped", False)))
        b.manual = str(r.get("manual", "") or "")
        out.append(b)
    return out


def gui_save(state: GuiState, win, ask_name: bool = False):
    """Write SAVES/<timestamp>[_name].json. Returns the path or None."""
    import json
    if state.save_path is None or ask_name:
        name = simpledialog.askstring(
            "Save", "Name to add after the time stamp (optional):",
            parent=win)
        if name is None:
            return None
        name = "".join(c if c.isalnum() or c in "-_ " else "_"
                       for c in name).strip().replace(" ", "_")
        stem = timestamp() + (f"_{name}" if name else "")
        state.save_path = gui_saves_dir() / f"{stem}.json"
    data = {"format": "microMS_beadtargeting select 1",
            "saved": timestamp(),
            "scan": str(state.path) if state.path else state.v["scan"].get(),
            "params": gui_params(state.v),
            "boxes": [list(map(float, b)) for b in state.boxes],
            "beads": gui_beads_to_json(state.beads)}
    state.save_path.write_text(json.dumps(data, indent=1))
    gui_settings_save(state.v)
    say(f"Saved {state.save_path.relative_to(HERE)}  "
        f"({len(state.beads)} objects, "
        f"{sum(b.accepted for b in state.beads)} accepted)")
    return state.save_path


def gui_load(state: GuiState, win) -> bool:
    import json
    p = filedialog.askopenfilename(
        parent=win, title="Load a saved selection",
        initialdir=str(gui_saves_dir()),
        filetypes=[("Selection saves", "*.json"), ("All files", "*.*")])
    if not p:
        return False
    try:
        data = json.loads(Path(p).read_text())
        params = data.get("params", {})
        for k, val in params.items():
            if k in state.v:
                state.v[k].set(val)
        if data.get("scan"):
            state.v["scan"].set(data["scan"])
        state.beads = gui_beads_from_json(data.get("beads", []))
        state.boxes = [tuple(b) for b in data.get("boxes", [])]
        state.save_path = Path(p)
    except (OSError, ValueError, KeyError) as e:
        say(f"Could not load {p}: {e}")
        return False
    say(f"Loaded {Path(p).name}: {len(state.beads)} objects, "
        f"{len(state.boxes)} boxes")
    if state.on_beads_changed:
        state.on_beads_changed()
    return True


def gui_write_run_inputs(state: GuiState, cfg: dict) -> Path:
    """The window's objects and decisions in the form CONFIG already
    understands: input.beads (a CSV) plus manual_selection.csv. Every
    object gets an explicit decision so run/review reproduce the
    window exactly."""
    stem = state.save_path.stem if state.save_path else timestamp()
    csv_path = gui_saves_dir() / f"{stem}_beads.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x_px", "y_px", "diameter_px", "clumped"])
        for b in state.beads:
            w.writerow([f"{b.x_px:.2f}", f"{b.y_px:.2f}",
                        f"{b.diameter_px:.2f}", int(b.clumped)])
    save_manual([(b.x_px, b.y_px, "accept" if b.accepted else "reject")
                 for b in state.beads])
    cfg["input"]["beads"] = str(csv_path)
    return csv_path


def gui_run_cfg(state: GuiState, fids: list) -> dict:
    import copy
    cfg = gui_apply_params(copy.deepcopy(state.cfg), state.v)
    cfg["fiducials"] = [dict(f) for f in fids]
    if state.path is not None:
        cfg["input"]["scan"] = str(state.path)
    cfg["output"]["review-show"] = False
    if state.beads:
        gui_refilter(state)
        gui_write_run_inputs(state, cfg)
    return cfg


def gui_review_into(state: GuiState, layer, status, master=None) -> None:
    """review() on the window's objects; the planned shots are drawn on
    the canvas and the review pictures open in a pop-up window."""
    fids, real = gui_fit_fiducials(state)
    cfg = gui_run_cfg(state, fids)
    try:
        beads, shots, T = review(cfg)
    except SystemExit as e:
        bar_done()
        status(f"review stopped: {e}")
        return
    live = sum(not s.dropped for s in shots)
    layer.show_shots(beads, shots, T, cfg)
    status(f"review: {live} shots on {sum(b.accepted for b in beads)} beads"
           + ("" if real else "  (provisional scale, no fiducials yet)")
           + "  -- pictures and check.txt in RESULTS")
    # the folder review() just wrote: the newest "<timestamp> review"
    folders = sorted((HERE / cfg["output"].get("results-dir", "RESULTS")).glob(
        "* review"), key=lambda p: p.stat().st_mtime)
    if folders and master is not None:
        gui_review_window(master, folders[-1])


def gui_review_window(master, folder: Path):
    """Pop-up with the review pictures: review.png (accepted beads and
    their shots) and review_zoom.png (close-up, every category), one at
    a time, with the usual zoom and drag."""
    import cv2
    pics = [p for p in (folder / "review.png", folder / "review_zoom.png")
            if p.exists()]
    if not pics:
        return None
    win = tk.Toplevel(master, bg=GUI_BG)
    win.title(f"microMS_beadtargeting -- review  {folder.name}")
    win.geometry(f"{px(1100)}x{px(760)}")
    gui_title_strip(win, f"Review   --   {folder.name}")

    bar = tk.Frame(win, bg=GUI_BG, padx=8, pady=6)
    bar.pack(fill="x", side="top")
    which = tk.StringVar(win, pics[0].name)
    for p in pics:
        tk.Radiobutton(bar, text=p.name, variable=which, value=p.name,
                       command=lambda: show(), font=F_BTN, indicatoron=False,
                       bg=GUI_BTN, fg=GUI_TXT, selectcolor=GUI_LAVENDER,
                       activebackground=GUI_BOX, relief="raised", bd=2,
                       highlightthickness=0, padx=8).pack(side="left",
                                                          padx=(0, 6))
    gui_small(bar, "scroll = zoom     left-drag = move     f = fit     "
                   f"folder: {folder}", anchor="w").pack(side="left",
                                                         padx=(12, 0))
    body = tk.Frame(win, bg=GUI_BG)
    body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    holder = {"widget": None}

    def show():
        if holder["widget"] is not None:
            holder["widget"].destroy()
        img = cv2.imread(str(folder / which.get()), cv2.IMREAD_COLOR)
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fig, ax, canvas = gui_image_canvas(body, img)
        holder["widget"] = canvas.get_tk_widget()
        attach_zoom(fig, ax)
        gui_attach_drag_pan(fig, ax, holder["widget"])

    show()
    return win


class GuiBeadLayer:
    """Bead circles (and, after a review, shot circles) on the canvas."""

    def __init__(self, fig, ax, show_vars: dict):
        self.fig, self.ax, self.show = fig, ax, show_vars
        self.patches = []
        self.shot_patches = []
        self.beads = []

    def rebuild(self, beads: list) -> None:
        from matplotlib.patches import Circle
        for p in self.patches + self.shot_patches:
            p.remove()
        self.patches, self.shot_patches = [], []
        self.beads = beads
        for b in beads:
            c = Circle((b.x_px, b.y_px), max(b.diameter_px / 2, 4),
                       fill=False, ec=gui_bead_colour(b), lw=1.3)
            self.ax.add_patch(c)
            self.patches.append(c)
        self.refresh()

    def refresh(self) -> None:
        for b, c in zip(self.beads, self.patches):
            c.set_edgecolor(gui_bead_colour(b))
            c.set_linewidth(2.0 if b.manual else 1.3)
            c.set_visible(bool(self.show[gui_bead_category(b)].get()))
        self.fig.canvas.draw_idle()

    def visible(self, b) -> bool:
        return bool(self.show[gui_bead_category(b)].get())

    def show_shots(self, beads, shots, T, cfg) -> None:
        from matplotlib.patches import Circle
        for p in self.shot_patches:
            p.remove()
        self.shot_patches = []
        r = footprint_um(cfg) / T.um_per_px / 2
        for s in shots:
            px = T.um_to_px([s.x_um, s.y_um])[0]
            if s.dropped:
                c = Circle(tuple(px), r, fill=False, ec="#ff7f0e", lw=0.8,
                           ls=":")
            else:
                c = Circle(tuple(px), r, fc="#1f77b4", ec="none", alpha=0.65)
            self.ax.add_patch(c)
            self.shot_patches.append(c)
        self.fig.canvas.draw_idle()

    def clear_shots(self) -> None:
        for p in self.shot_patches:
            p.remove()
        self.shot_patches = []
# ---- window 1: select parameters -------------------------------------

def gui_select_window(master, state: GuiState, on_continue=None):
    v = state.v
    win = tk.Toplevel(master, bg=GUI_BG)
    win.title("microMS_beadtargeting -- select")
    win.resizable(False, False)

    def do_load():
        if gui_load(state, win):
            status.config(text=f"loaded {state.save_path.name}")

    gui_menu_bar(win, {"File": [("Save", lambda: gui_save(state, win)),
                                ("Save As...",
                                 lambda: gui_save(state, win, True)),
                                ("Load...", do_load),
                                None,
                                ("Restore default values",
                                 lambda: gui_restore_defaults(v, state.cfg))],
                       "About": about_menu_items(win)})
    gui_title_strip(win, "Select parameters")

    body = tk.Frame(win, bg=GUI_BG, padx=14, pady=10)
    body.pack(fill="both", expand=True)
    body.columnconfigure(1, weight=1)
    row = [0]

    def line(text: str = "", widget=None):
        """One parameter line: label in the left column, a frame of
        controls in the right column."""
        r = row[0]
        if widget is not None:
            widget.grid(row=r, column=0, sticky="w", pady=4)
        elif text:
            gui_label(body, text).grid(row=r, column=0, sticky="w", pady=4)
        f = tk.Frame(body, bg=GUI_BG)
        f.grid(row=r, column=1, sticky="w", padx=(18, 0), pady=4)
        row[0] += 1
        return f

    # -- scan image ----------------------------------------------------
    f = line("Scan image")

    def browse():
        p = filedialog.askopenfilename(
            parent=win, title="Select the slide scan",
            filetypes=[("Images", "*.tif *.tiff *.jpg *.jpeg *.png"),
                       ("All files", "*.*")])
        if not p:
            return
        src = Path(p)
        if src.suffix.lower() in (".jpg", ".jpeg", ".png"):
            if not gui_ask(win, "Convert to TIFF",
                           f"{src.name} is not a TIFF. Detection needs "
                           f"one.\nConvert it now? The .tif is written "
                           f"next to the original and used as the slide "
                           f"image."):
                status.config(text="not converted -- choose a .tif to go on")
                return
            bar_start("convert", 1)
            bar_step(f"converting {src.name}")
            try:
                dst = convert_image(src)
            except SystemExit as e:
                bar_done()
                status.config(text=f"conversion failed: {e}")
                return
            bar_done()
            src = dst
        v["scan"].set(str(src))
        status.config(text=f"scan: {src.name}")

    gui_button(f, "Upload image...", browse).pack(side="left")
    gui_box(f, v["scan"], width=34).pack(side="left", padx=(8, 0))
    gui_small(f, ".jpg is offered for .tif conversion").pack(side="left",
                                                              padx=(8, 0))

    # -- matrix + drop-down image settings ------------------------------
    f = line("Matrix *")
    gui_dropdown(f, v["matrix"], ["yes", "no", "beta"]).pack(side="left")
    zoom_check = [None]
    panel_open = [False]

    def set_panel(open_: bool):
        panel_open[0] = open_
        if open_:
            panel.grid()
            bar_btn.config(text="^   Image Settings   ^")
        else:
            panel.grid_remove()
            bar_btn.config(text="v   Image Settings   v")

    def on_type(*_a):
        if v["image_type"].get() == "Microscope":
            zoom_check[0].config(state="normal", fg=GUI_TXT)
            v["microscope_slide"].set(True)
        else:
            zoom_check[0].config(state="disabled")
            v["microscope_slide"].set(False)

    r = row[0]
    bar_btn = tk.Button(body, text="v   Image Settings   v", font=F_SMALL,
                        bg=GUI_BG_BAR, fg=GUI_TXT, activebackground=GUI_BOX,
                        relief="flat", bd=0, highlightthickness=0,
                        command=lambda: set_panel(not panel_open[0]))
    bar_btn.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(6, 0))
    row[0] += 1
    r = row[0]
    panel = tk.Frame(body, bg=GUI_LAVENDER, padx=10, pady=6,
                     relief="sunken", bd=2)
    panel.grid(row=r, column=0, columnspan=2, sticky="ew")
    row[0] += 1
    prow0 = tk.Frame(panel, bg=GUI_LAVENDER)
    prow0.pack(anchor="w", pady=(0, 4))
    gui_label(prow0, "Type of image *", bg=GUI_LAVENDER).pack(side="left")
    gui_dropdown(prow0, v["image_type"],
                 ["Microscope", "highres scanner", "beta"],
                 on_type).pack(side="left", padx=(6, 0))
    prow = tk.Frame(panel, bg=GUI_LAVENDER)
    prow.pack(anchor="w")
    zoom_check[0] = gui_check(prow, v["microscope_slide"],
                              "microscope zoom *", bg=GUI_LAVENDER,
                              state="disabled")
    zoom_check[0].pack(side="left")
    gui_box(prow, v["microscope_zoom"], width=4).pack(side="left",
                                                       padx=(6, 4))
    zoom_hint = gui_small(prow, "(10x)", bg=GUI_LAVENDER)
    zoom_hint.pack(side="left")
    v["microscope_zoom"].trace_add(
        "write", lambda *_a: zoom_hint.config(
            text=f"({v['microscope_zoom'].get() or '?'}x)"))
    prow2 = tk.Frame(panel, bg=GUI_LAVENDER)
    prow2.pack(anchor="w", pady=(4, 0))
    gui_label(prow2, "scale", bg=GUI_LAVENDER).pack(side="left")
    gui_box(prow2, v["um_per_px"], width=6).pack(side="left", padx=(6, 4))
    gui_small(prow2, "um/px  -- used for the isolation filter until "
                     "fiducials are picked", bg=GUI_LAVENDER).pack(
                         side="left")
    set_panel(False)
    on_type()

    # -- location of targets --------------------------------------------
    f = line("Location of targets")
    gui_check(f, v["global_sweep"], "Global threshold sweep",
              font=F_SMALL).pack(side="left")
    gui_label(f, ";", font=F_SMALL).pack(side="left", padx=4)
    gui_check(f, v["flat_field"], "flat field subtraction",
              font=F_SMALL).pack(side="left")

    # -- bead size -------------------------------------------------------
    f = line("Average bead size + deviation")
    gui_box(f, v["bead_um"], 5).pack(side="left")
    gui_small(f, "um").pack(side="left", padx=(3, 10))
    gui_small(f, "+/-").pack(side="left", padx=(0, 3))
    gui_box(f, v["bead_dev_um"], 4).pack(side="left")
    gui_small(f, "um").pack(side="left", padx=(3, 0))

    # -- isolation window ------------------------------------------------
    f = line(widget=gui_check(body, v["isolation_on"], "Isolation window"))
    gui_box(f, v["isolation_um"], 5).pack(side="left")
    gui_small(f, "um").pack(side="left", padx=(3, 0))

    # -- max points ------------------------------------------------------
    f = line("Max number of points")
    gui_box(f, v["max_points"], 6).pack(side="left")

    # -- footnote + continue -----------------------------------------------
    gui_hrule(body).grid(row=row[0], column=0, columnspan=2, sticky="ew",
                         pady=(10, 6))
    row[0] += 1
    gui_small(body, "* stored with the save; no effect on detection yet",
              fg="#3a3a3a").grid(row=row[0], column=0, columnspan=2,
                                 sticky="w")
    row[0] += 1
    fb = tk.Frame(body, bg=GUI_BG)
    fb.grid(row=row[0], column=0, columnspan=2, sticky="ew", pady=(6, 0))
    status = gui_small(fb, "", anchor="w")
    status.pack(side="left", fill="x", expand=True)
    gui_button(fb, "Continue to bead selection  >",
               on_continue or (lambda: None),
               font=F_BIG).pack(side="right")
    win.status = status
    return win


# ---- window 2: bead selection ------------------------------------------

def gui_beads_window(master, state: GuiState, on_continue=None):
    import copy
    v = state.v
    win = tk.Toplevel(master, bg=GUI_BG)
    win.title("microMS_beadtargeting -- select")
    win.geometry(f"{px(1280)}x{px(800)}")
    gui_title_strip(win, "Bead selection   --   "
                         f"{state.path.name if state.path else 'no scan'}")

    # bottom bar, packed first so the image can never squeeze it out
    bottom = tk.Frame(win, bg=GUI_BG, padx=8, pady=6)
    bottom.pack(fill="x", side="bottom")
    status = gui_small(bottom, "left-drag = draw the box     "
                               "right-drag = move the view     "
                               "scroll / + / - / f = zoom / fit", anchor="w")
    status.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))

    def set_status(text: str) -> None:
        status.config(text=text)

    main = tk.Frame(win, bg=GUI_BG)
    main.pack(fill="both", expand=True, padx=8, pady=(8, 0))

    # right-hand panel
    side = tk.Frame(main, bg=GUI_BG, width=px(270))
    side.pack(side="right", fill="y", padx=(8, 0))
    side.pack_propagate(False)

    gui_label(side, "Bead matching", font=F_TITLE).pack(anchor="w")
    # soft-chosen from the parameters window; still changeable here
    if v["global_sweep"].get() and not v["flat_field"].get():
        v["method"].set("default")
    elif v["flat_field"].get():
        v["method"].set("quick")
    if v["method"].get() not in ("default", "quick", "strict"):
        v["method"].set("default")          # a save made under the old names
    gui_dropdown(side, v["method"], ["default", "quick", "strict"]).pack(
        anchor="w", pady=(2, 2), fill="x")
    gui_small(side, "default   global threshold sweep\n"
                    "quick      flat-field subtraction\n"
                    "strict     both combined, keep the best 5 %",
              justify="left", anchor="w").pack(anchor="w", fill="x",
                                               pady=(0, 8))
    gui_button(side, "Analyze box", lambda: analyze(),
               font=F_BIG).pack(fill="x", pady=(4, 0))
    gui_small(side, "left-drag a box on the image, then analyze;\n"
                    "the box stays -- Clear box empties it so the same\n"
                    "area can be analyzed again with another method",
              justify="left", anchor="w").pack(anchor="w", fill="x",
                                               pady=(2, 8))
    for t, fn in (("Accept box", lambda: set_many("accept")),
                  ("Reject box", lambda: set_many("reject")),
                  ("Clear box", lambda: clear_box())):
        gui_button(side, t, fn).pack(fill="x", pady=2)
    gui_hrule(side).pack(fill="x", pady=6)
    manual_mode = tk.BooleanVar(win, False)
    gui_toggle(side, manual_mode, "Manual accept / reject",
               command=lambda: mode_changed()).pack(fill="x")
    gui_small(side, "while on: left-click a bead to flip it",
              anchor="w").pack(anchor="w", fill="x", pady=(2, 6))
    gui_hrule(side).pack(fill="x", pady=6)
    gui_label(side, "Show", font=F_TITLE).pack(anchor="w")
    show_vars = {}          # kept here: an unreferenced tk variable is
    for key, t, col in (("accepted", "accepted (green)", "#1e7d1e"),
                        ("clumped", "clumped (purple)", "#5b3a9e"),
                        ("rejected", "rejected (red)", "#b01c1c")):
        show_vars[key] = tk.BooleanVar(win, True)      # garbage-collected
        cb = gui_check(side, show_vars[key], t, font=F_SHOW,
                       command=lambda: layer.refresh())
        cb.config(fg=col, activeforeground=col)
        cb.pack(anchor="w")
    win.show_vars = show_vars
    counts = gui_label(side, "", font=F_SMALL, justify="left")
    counts.pack(anchor="w", pady=(10, 0))

    # image, bead layer, tools: left-drag = box, right-drag = move,
    # scroll / keys = zoom (attach_zoom); in manual mode left-click flips
    fig, ax, canvas = gui_image_canvas(main, state.img)
    widget = canvas.get_tk_widget()
    layer = GuiBeadLayer(fig, ax, show_vars)
    box = GuiBoxTool(fig, ax, widget, manual_mode,
                     on_change=lambda b: update_counts())
    if state.img is not None:
        attach_zoom(fig, ax)
        gui_attach_drag_pan(fig, ax, widget, button=3)
    widget.config(cursor="crosshair")

    def mode_changed() -> None:
        on = manual_mode.get()
        widget.config(cursor="hand2" if on else "crosshair")
        set_status("manual mode: left-click a bead to flip accept / reject"
                   if on else "left-drag = draw the box     "
                              "right-drag = move the view")

    def in_box() -> list:
        if box.box is None:
            return []
        x0, x1, y0, y1 = box.box
        return [b for b in state.beads
                if x0 <= b.x_px <= x1 and y0 <= b.y_px <= y1
                and layer.visible(b)]

    def update_counts() -> None:
        n_acc = sum(b.accepted for b in state.beads)
        n_man = sum(1 for b in state.beads if b.manual)
        sel = in_box()
        text = (f"accepted {n_acc} / {len(state.beads)}\n"
                f"overrides {n_man}\n"
                f"boxes analysed {len(state.boxes)}")
        if box.box:
            text += f"\nin current box {len(sel)}"

            def stats(label, vals):
                if not vals:
                    return ""
                a = np.array(vals)
                return (f"\n{label} size um: min {a.min():.0f}  "
                        f"med {np.median(a):.0f}  avg {a.mean():.0f}  "
                        f"max {a.max():.0f}")

            text += stats("bead", [b.diameter_um for b in sel
                                   if b.accepted])
            text += stats("contaminant", [b.diameter_um for b in sel
                                          if not b.accepted])
        counts.config(text=text)

    def redraw_all() -> None:
        if state.beads:
            gui_refilter(state)
        layer.rebuild(state.beads)
        update_counts()

    def analyze() -> None:
        if state.img is None:
            set_status("no scan loaded")
            return
        if box.box is None:
            set_status("draw a box first: left-drag on the image")
            return
        method = v["method"].get()
        bar_start("analyze", 3)
        bar_step(f"detecting ({method})")
        cfg = gui_apply_params(copy.deepcopy(state.cfg), v)
        found = gui_detect_region(state.img, box.box, cfg, method)
        added = gui_merge(state, found, cfg)
        state.boxes.append(tuple(box.box))
        bar_step("filtering")
        gui_refilter(state)
        bar_step("drawing")
        layer.rebuild(state.beads)
        bar_done()
        update_counts()
        set_status(f"{method}: {len(found)} objects in the box, {added} new"
                   f"   |   accepted {sum(b.accepted for b in state.beads)}"
                   f" of {len(state.beads)}")

    def set_many(decision: str) -> None:
        sel = in_box()
        if not sel:
            set_status("draw a box around some beads first")
            return
        for b in sel:
            gui_set_manual(b, decision)
        gui_refilter(state)
        layer.refresh()
        update_counts()
        set_status(f"{decision}ed {len(sel)} beads in the box")

    def on_click(ev):
        if (ev.button != 1 or not manual_mode.get() or ev.inaxes is not ax
                or ev.xdata is None):
            return
        if not state.beads:
            return
        d = [math.hypot(b.x_px - ev.xdata, b.y_px - ev.ydata)
             if layer.visible(b) else float("inf") for b in state.beads]
        i = int(np.argmin(d))
        if not math.isfinite(d[i]) or d[i] > 40:
            return
        b = state.beads[i]
        gui_set_manual(b, "reject" if b.accepted else "accept")
        gui_refilter(state)
        layer.refresh()
        update_counts()
        set_status(f"bead {i} -> {b.manual}")

    fig.canvas.mpl_connect("button_press_event", on_click)

    def clear_box() -> None:
        """Remove the objects inside the box; the box itself stays so
        the same area can be analyzed again with another method."""
        if box.box is None:
            set_status("no box drawn")
            return
        x0, x1, y0, y1 = box.box
        inside = [b for b in state.beads
                  if x0 <= b.x_px <= x1 and y0 <= b.y_px <= y1]
        state.beads = [b for b in state.beads if b not in inside]
        state.boxes = [bx for bx in state.boxes if tuple(bx) != tuple(box.box)]
        layer.clear_shots()
        redraw_all()
        set_status(f"removed {len(inside)} objects inside the box -- "
                   f"pick a method and analyze again")

    def reset_all() -> None:
        for b in state.beads:
            b.manual = ""
        redraw_all()
        set_status("manual overrides discarded")

    def clear_beads() -> None:
        state.beads, state.boxes = [], []
        layer.rebuild([])
        box.clear()
        set_status("all objects removed -- draw a box and analyze again")

    def no_selection() -> bool:
        if not any(b.accepted for b in state.beads):
            set_status("no beads selected -- draw a box and analyze first")
            return True
        return False

    def save_and_review() -> None:
        if no_selection():
            return
        if gui_save(state, win) is None:
            set_status("not saved")
            return
        gui_review_into(state, layer, set_status, win)

    def go_on() -> None:
        if no_selection():
            return
        gui_settings_save(v)
        if on_continue:
            on_continue()

    def do_load() -> None:
        if gui_load(state, win):
            set_status(f"loaded {state.save_path.name}")

    gui_menu_bar(win, {"File": [("Save", lambda: gui_save(state, win)),
                                ("Save As...",
                                 lambda: gui_save(state, win, True)),
                                ("Load...", do_load),
                                None,
                                ("Discard manual overrides", reset_all),
                                ("Clear all beads", clear_beads),
                                None,
                                ("Restore default values",
                                 lambda: gui_restore_defaults(v, state.cfg))],
                       "About": about_menu_items(win)})
    gui_button(bottom, "Continue to correlate fiducials  (ready for MSI)  >",
               go_on, font=F_BIG).pack(side="right")
    gui_button(bottom, "Save and review", save_and_review,
               font=F_BIG).pack(side="right", padx=(0, 10))

    state.on_beads_changed = redraw_all
    redraw_all()
    win.protocol("WM_DELETE_WINDOW", lambda: (setattr(state, "on_beads_changed", None), win.destroy()))
    return win


# ---- window 3: correlate fiducials (pick) ----------------------------

def gui_pick_window(master, state: GuiState):
    win = tk.Toplevel(master, bg=GUI_BG)
    win.title("microMS_beadtargeting -- pick")
    win.geometry(f"{px(1280)}x{px(800)}")
    gui_menu_bar(win, {"About": about_menu_items(win)})
    gui_title_strip(win, "Correlate fiducials   --   "
                         f"{state.path.name if state.path else 'no scan'}")

    # bottom bar, packed first so the image can never squeeze it out
    bottom = tk.Frame(win, bg=GUI_BG, padx=8, pady=6)
    bottom.pack(fill="x", side="bottom")
    status = gui_small(bottom, "right-click = set pending pixel     "
                               "Tab = next box     Ctrl+V = paste     "
                               "Enter = add     scroll = zoom     "
                               "left-drag = move", anchor="w")
    status.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))

    def set_status(text: str) -> None:
        status.config(text=text)

    main = tk.Frame(win, bg=GUI_BG)
    main.pack(fill="both", expand=True, padx=8, pady=(8, 0))

    # left-hand fiducial table: header stays put, rows scroll (up to 100)
    side = tk.Frame(main, bg=GUI_BG, width=px(400))
    side.pack(side="left", fill="y", padx=(0, 8))
    side.pack_propagate(False)

    gui_label(side, "Fiducials", font=F_TITLE).pack(anchor="w")
    head = tk.Frame(side, bg=GUI_BG)
    head.pack(anchor="w", fill="x")
    COLW = (12, 9, 9, 4, 2)                 # characters per column (rows)
    HEADW = (15, 13, 13, 5, 2)              # same columns in the 8 pt font
    for c, h in enumerate(("#   pixel", "stage X (um)", "stage Y (um)",
                           "hide", "")):
        gui_small(head, h, width=HEADW[c], anchor="w").grid(
            row=0, column=c, padx=2, sticky="w")
    gui_small(side, "hidden = left out of the fit, not deleted",
              fg="#3a3a3a").pack(anchor="w")

    holder = tk.Frame(side, bg=GUI_BG, relief="sunken", bd=2)
    holder.pack(anchor="w", fill="both", pady=(2, 4))
    cv = tk.Canvas(holder, bg=GUI_BG, highlightthickness=0, height=px(300),
                   width=px(372))
    sb = tk.Scrollbar(holder, orient="vertical", command=cv.yview,
                      bg=GUI_BTN, troughcolor=GUI_BOX,
                      activebackground=GUI_BTN, relief="raised", bd=2,
                      width=14)
    cv.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y")
    cv.pack(side="left", fill="both", expand=True)
    table = tk.Frame(cv, bg=GUI_BG)
    cv.create_window((0, 0), window=table, anchor="nw")
    table.bind("<Configure>",
               lambda _e: cv.configure(scrollregion=cv.bbox("all")))
    def wheel(e):
        # rows are children of the canvas, so bind on the window and
        # act only while the pointer is over the table
        x, y = holder.winfo_pointerxy()
        if (holder.winfo_rootx() <= x <= holder.winfo_rootx() + holder.winfo_width()
                and holder.winfo_rooty() <= y <= holder.winfo_rooty() + holder.winfo_height()):
            cv.yview_scroll(-1 if e.delta > 0 else 1, "units")
    win.bind_all("<MouseWheel>", wheel, add="+")

    rows: list = []
    marks: list = []
    pending = {"px": None, "art": []}

    def edit_row(i, key, var):
        try:
            state.fids[i][key] = float(var.get())
        except ValueError:
            set_status(f"fiducial {i}: '{var.get()}' is not a number")
            return
        refit()

    def add_row(i, f):
        r = len(rows)
        w = {}
        w["name"] = gui_label(table, f"{i:>2}  {f['x_px']:.0f},{f['y_px']:.0f}",
                              font=F_SMALL, width=COLW[0], anchor="w")
        w["name"].grid(row=r, column=0, sticky="w", padx=2)
        w["x"] = tk.StringVar(win, f"{f['x_um']:.1f}")
        w["y"] = tk.StringVar(win, f"{f['y_um']:.1f}")
        for c, key in ((1, "x"), (2, "y")):
            e = tk.Entry(table, textvariable=w[key], width=COLW[c],
                         font=F_ROW, bg=GUI_BOX, fg=GUI_TXT, relief="sunken",
                         bd=1, insertbackground=GUI_TXT,
                         highlightthickness=0)
            e.grid(row=r, column=c, padx=2, pady=1)
            e.bind("<Return>", lambda _e, i=i, k=key + "_um", var=w[key]:
                   edit_row(i, k, var))
        w["hide"] = tk.BooleanVar(win, bool(f.get("hide")))
        gui_check(table, w["hide"], pady=0,
                  command=lambda i=i, var=w["hide"]: toggle_hide(i, var)).grid(
                      row=r, column=3, padx=6)
        tk.Button(table, text="x", font=F_TINY_B, bg=GUI_BTN, fg=GUI_TXT,
                  relief="raised", bd=1, width=2, padx=0, pady=0,
                  highlightthickness=0,
                  command=lambda i=i: delete_fid(i)).grid(row=r, column=4,
                                                          padx=2)
        rows.append(w)

    def rebuild_table():
        for child in table.winfo_children():
            child.destroy()
        rows.clear()
        for i, f in enumerate(state.fids):
            add_row(i, f)

    def toggle_hide(i, var):
        state.fids[i]["hide"] = bool(var.get())
        refit()

    def delete_fid(i):
        state.fids.pop(i)
        rebuild_table()
        refit()
        set_status(f"removed fiducial {i}")

    gui_small(side, "edit a value and press Enter to re-fit").pack(anchor="w")
    fit_label = gui_label(side, "", font=F_SMALL, wraplength=px(380),
                          justify="left", anchor="w")
    fit_label.pack(anchor="w", fill="x", pady=(2, 8))
    gui_hrule(side).pack(fill="x", pady=4)

    gui_label(side, "Add a fiducial", font=F_TITLE).pack(anchor="w")
    gui_small(side, "right-click the mark on the image, then type or "
                    "Ctrl+V the stage reading", wraplength=px(380),
              justify="left", anchor="w").pack(anchor="w", fill="x")
    pend_label = gui_small(side, "pending pixel: (none)")
    pend_label.pack(anchor="w", pady=(2, 4))
    ef = tk.Frame(side, bg=GUI_BG)
    ef.pack(anchor="w")
    sx, sy = tk.StringVar(win), tk.StringVar(win)
    gui_small(ef, "stage x").grid(row=0, column=0, sticky="w")
    ex = gui_box(ef, sx, 10)
    ex.grid(row=0, column=1, padx=4, pady=2)
    gui_small(ef, "um").grid(row=0, column=2)
    gui_small(ef, "stage y").grid(row=1, column=0, sticky="w")
    ey = gui_box(ef, sy, 10)
    ey.grid(row=1, column=1, padx=4, pady=2)
    gui_small(ef, "um").grid(row=1, column=2)

    def paste(_ev=None):
        try:
            nums = _floats_in(win.clipboard_get())
        except tk.TclError:
            nums = []
        if not nums:
            set_status("clipboard has no number in it")
        elif len(nums) == 1:
            (sy if win.focus_get() is ey else sx).set(nums[0])
            set_status(f"pasted {nums[0]}")
        else:
            sx.set(nums[0])
            sy.set(nums[1])
            set_status(f"pasted x={nums[0]}  y={nums[1]}  -- check, then Add")
        return "break"

    ex.bind("<Control-v>", paste)
    ey.bind("<Control-v>", paste)
    ex.bind("<Return>", lambda _e: ey.focus_set())
    ey.bind("<Return>", lambda _e: add())

    bf = tk.Frame(side, bg=GUI_BG)
    bf.pack(anchor="w", pady=(4, 0))
    gui_button(bf, "Add fiducial", lambda: add()).pack(side="left")
    gui_button(bf, "Remove nearest", lambda: remove()).pack(side="left",
                                                            padx=4)
    gui_button(bf, "Reset", lambda: reset()).pack(side="left")
    gui_hrule(side).pack(fill="x", pady=8)
    zf = tk.Frame(side, bg=GUI_BG)
    zf.pack(anchor="w")
    zoom_btns = [gui_button(zf, t, width=6) for t in ("Zoom +", "Zoom -",
                                                      "Fit")]
    for b in zoom_btns:
        b.pack(side="left", padx=1)

    # image with the fiducial marks
    fig, ax, canvas = gui_image_canvas(main, state.img)
    widget = canvas.get_tk_widget()
    layer = GuiBeadLayer(fig, ax, {"accepted": tk.BooleanVar(win, True),
                                   "clumped": tk.BooleanVar(win, True),
                                   "rejected": tk.BooleanVar(win, True)})
    if state.img is not None:
        for b, fn in zip(zoom_btns, attach_zoom(fig, ax)):
            b.config(command=fn)
        gui_attach_drag_pan(fig, ax, widget)

    def refit():
        for a in marks:
            a.remove()
        marks.clear()
        active = state.active_fids()
        worst = None
        text = f"{len(active)} fiducials  (3 needed)"
        if len(active) >= 3:
            src = np.array([[f["x_px"], f["y_px"]] for f in active], float)
            dst = np.array([[f["x_um"], f["y_um"]] for f in active], float)
            T = fit_similarity(src, dst, state.cfg.get("allow-reflection",
                                                       True))
            res = residuals(T, src, dst)
            worst = active[int(np.argmax(res))]
            text = (f"{len(active)} fiducials | RMS "
                    f"{np.sqrt((res ** 2).mean()):.1f} um | worst "
                    f"{res.max():.1f} um | {T.um_per_px:.3f} um/px"
                    + (" | REFLECTED" if T.reflected else ""))
        fit_label.config(text=text)
        for i, f in enumerate(state.fids):
            if f.get("hide"):
                c = "#bbbbbb"
            elif f is worst:
                c = "red"
            else:
                c = "lime"
            marks.append(ax.plot(f["x_px"], f["y_px"], "+", ms=16, mew=2,
                                 color=c)[0])
            marks.append(ax.text(f["x_px"] + 60, f["y_px"] - 60, str(i),
                                 color=c, fontsize=10))
        fig.canvas.draw_idle()

    def set_pending(px):
        for a in pending["art"]:
            a.remove()
        pending["art"] = []
        pending["px"] = px
        if px is None:
            pend_label.config(text="pending pixel: (none)")
        else:
            pend_label.config(text=f"pending pixel: ({px[0]:.0f}, {px[1]:.0f})")
            pending["art"].append(ax.plot(px[0], px[1], "x", ms=18, mew=3,
                                          color="gold")[0])
            pending["art"].append(ax.text(px[0] + 60, px[1] - 60, "pending",
                                          color="gold", fontsize=10))
        fig.canvas.draw_idle()

    def on_click(ev):
        if ev.inaxes is not ax or ev.button != 3 or ev.xdata is None:
            return
        set_pending((float(ev.xdata), float(ev.ydata)))
        set_status(f"pending pixel ({ev.xdata:.0f}, {ev.ydata:.0f})"
                   f"  -- enter stage coords, then Add")
        ex.focus_set()

    fig.canvas.mpl_connect("button_press_event", on_click)

    def add():
        if pending["px"] is None:
            set_status("right-click a fiducial mark on the image first")
            return
        try:
            x_um, y_um = float(sx.get()), float(sy.get())
        except ValueError:
            set_status("stage x and stage y must both be numbers")
            return
        state.fids.append({"x_px": pending["px"][0], "y_px": pending["px"][1],
                           "x_um": x_um, "y_um": y_um})
        set_pending(None)
        sx.set("")
        sy.set("")
        rebuild_table()
        refit()
        set_status(f"added fiducial {len(state.fids) - 1}")

    def remove():
        if not state.fids:
            return
        ref = pending["px"] or (state.fids[-1]["x_px"], state.fids[-1]["y_px"])
        d = [math.hypot(f["x_px"] - ref[0], f["y_px"] - ref[1])
             for f in state.fids]
        delete_fid(int(np.argmin(d)))

    def reset():
        state.fids.clear()
        set_pending(None)
        rebuild_table()
        refit()
        set_status("all fiducials cleared")

    def save_and(what: str):
        active = state.active_fids()
        if len(active) < 3:
            set_status("at least 3 fiducials are needed (hidden ones do "
                       "not count)")
            return
        save_fiducials(active)
        say(f"Wrote {len(active)} fiducials to {CONFIG_PATH.name}")
        if not any(b.accepted for b in state.beads):
            set_status(f"fiducials saved; no beads selected -- go back to "
                       f"bead selection, or use '{what}' from the console "
                       f"for whole-scan detection")
            return
        if what == "review":
            gui_review_into(state, layer, set_status, win)
            return
        cfg = gui_run_cfg(state, active)
        try:
            run(cfg)
        except SystemExit as e:
            bar_done()
            set_status(f"run stopped: {e}")
            return
        set_status("run complete -- target files are in RESULTS "
                   "(see the console for the summary)")

    gui_button(bottom, "Save and run  >", lambda: save_and("run"),
               font=F_BIG).pack(side="right")
    gui_button(bottom, "Save and review", lambda: save_and("review"),
               font=F_BIG).pack(side="right", padx=(0, 10))

    rebuild_table()
    refit()
    return win


# ---- entry points ------------------------------------------------------

def _gui_root(cfg: dict):
    """One hidden Tk root, the named fonts and the shared state."""
    if tk is None:
        sys.exit("tkinter is not installed, so no window can open. On "
                 "Windows it ships with python.org Python; on Linux "
                 "install python3-tk.")
    global GUI_SCALE
    import platform
    if platform.system() == "Windows":
        # Without this Windows bitmap-scales the whole window on a
        # 125 % / 150 % display and every glyph comes out fuzzy.
        try:
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    root = tk.Tk()
    root.withdraw()
    GUI_SCALE = max(root.winfo_fpixels("1i") / 96.0, 0.5)
    init_fonts(root)
    state = GuiState(cfg)
    state.v = gui_make_vars(root, cfg)
    return root, state


def _gui_scan_path(state: GuiState) -> Path:
    p = Path(state.v["scan"].get() or state.cfg["input"]["scan"])
    return p if p.is_absolute() else HERE / p


def _gui_load_scan(state: GuiState, status=None) -> bool:
    p = _gui_scan_path(state)
    if state.path != p or state.img is None:
        bar_start("select", 1)
        bar_step(f"reading {p.name}")
        if status:
            status(f"loading {p.name} ...")
        state.img, state.path = gui_load_scan(p), p
        bar_done()
    if state.img is None:
        if status:
            status(f"could not open {p.name} -- see console")
        return False
    if status:
        status("")
    return True


def gui_select(cfg: dict) -> None:
    """select: parameters -> bead selection -> correlate fiducials."""
    root, state = _gui_root(cfg)

    def open_pick():
        gui_pick_window(root, state)

    def open_beads():
        sel.update_idletasks()
        if _gui_load_scan(state, lambda t: (sel.status.config(text=t),
                                             sel.update_idletasks())):
            gui_settings_save(state.v)
            gui_beads_window(root, state, on_continue=open_pick)

    sel = gui_select_window(root, state, on_continue=open_beads)

    def close():
        gui_settings_save(state.v)
        root.destroy()

    sel.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
    say("select closed")


def gui_pick(cfg: dict) -> None:
    """pick: straight to the correlate-fiducials window on the CONFIG
    scan. With no bead selection loaded, 'save and run' detects over
    the whole scan exactly as the console 'run' does."""
    root, state = _gui_root(cfg)
    if not _gui_load_scan(state):
        sys.exit(f"Scan not found or unreadable: {_gui_scan_path(state)}")
    win = gui_pick_window(root, state)
    win.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    say("pick closed")


def main() -> None:
    global VERBOSE
    VERBOSE = bool({"-v", "--verbose"} & set(sys.argv))

    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd and not cmd.startswith("-"):
        banner(cmd)
    if cmd in ("gui", "pick", "select", "review", "run"):
        preload_gui_libs()

    if cmd == "gui":
        gui_select(load_config())
    elif cmd == "pick":
        gui_pick(load_config())
    elif cmd == "select":
        gui_select(load_config())
    elif cmd == "review":
        review(load_config())
    elif cmd == "run":
        run(load_config())
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Ctrl+C in the console: a normal way to stop, not an error
        bar_done()
        say("\nstopped (Ctrl+C)")
