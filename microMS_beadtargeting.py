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

# GUI progress bars (GuiProgress) register here; each bar_* call below
# hands them the current Bar, or None when it is done. Called on the
# main thread only, which is where tk must be touched.
_BAR_HOOKS: list = []


def _bar_notify(bar) -> None:
    for hook in list(_BAR_HOOKS):
        try:
            hook(bar)
        except Exception:                 # a dead window must not stop a run
            _BAR_HOOKS.remove(hook)


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
        self.where = ""                 # output folder, see bar_where
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
    _bar_notify(_BAR)
    return _BAR


def bar_step(label: str) -> None:
    if _BAR is not None:
        _BAR.step(label)
        _bar_notify(_BAR)


def bar_tick(i: int, n: int) -> None:
    if _BAR is not None:
        _BAR.tick(i, n)
        _bar_notify(_BAR)


def bar_where(folder) -> None:
    """Tell the running bar which folder its files are written to; the
    window's progress bar shows it under the stage (GuiProgress)."""
    if _BAR is not None:
        _BAR.where = str(folder)
        _bar_notify(_BAR)


def bar_done() -> None:
    global _BAR
    if _BAR is not None:
        _BAR.done()
        _BAR = None
        _bar_notify(None)


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
    {"x_px": 2037.89, "y_px": 1494.87, "x_um": 21119.50, "y_um": -20632.80},
    {"x_px": 4859.69, "y_px": 1518.31, "x_um": 32404.30, "y_um": -20581.60},
    {"x_px": 6969.57, "y_px": 1424.54, "x_um": 40623.70, "y_um": -20104.40},
    {"x_px": 8911.65, "y_px": 1312.26, "x_um": 48345.80, "y_um": -19510.20},
    {"x_px": 11010.73, "y_px": 1448.30, "x_um": 56593.80, "y_um": -19867.40},
    {"x_px": 12423.04, "y_px": 1380.53, "x_um": 62233.30, "y_um": -19464.40},
    {"x_px": 13894.98, "y_px": 1486.25, "x_um": 68112.20, "y_um": -19801.10},
    {"x_px": 15380.48, "y_px": 1269.39, "x_um": 73975.80, "y_um": -18862.40},
    {"x_px": 16863.27, "y_px": 1572.99, "x_um": 79876.80, "y_um": -20052.80},
    {"x_px": 18145.46, "y_px": 1551.31, "x_um": 85038.30, "y_um": -19725.30},
    {"x_px": 18351.47, "y_px": 3240.11, "x_um": 85979.40, "y_um": -26510.90},
    {"x_px": 18402.98, "y_px": 4752.72, "x_um": 86156.10, "y_um": -32527.70},
    {"x_px": 18167.14, "y_px": 6129.78, "x_um": 85434.20, "y_um": -38068.80},
    {"x_px": 15963.29, "y_px": 6167.73, "x_um": 76605.90, "y_um": -38305.60},
    {"x_px": 14008.84, "y_px": 6390.02, "x_um": 688672.00, "y_um": -39338.20},
    {"x_px": 12290.21, "y_px": 6324.96, "x_um": 62101.90, "y_um": -39300.20},
    {"x_px": 10888.75, "y_px": 6395.44, "x_um": 56599.90, "y_um": -39562.70},
    {"x_px": 9462.89, "y_px": 6314.11, "x_um": 50888.60, "y_um": -39341.00},
    {"x_px": 7974.68, "y_px": 6287.01, "x_um": 44959.20, "y_um": -39358.60},
    {"x_px": 6288.59, "y_px": 6102.67, "x_um": 38308.20, "y_um": -38718.40},
    {"x_px": 4759.72, "y_px": 6053.88, "x_um": 32255.10, "y_um": -38647.40},
    {"x_px": 3247.12, "y_px": 6018.64, "x_um": 26132.50, "y_um": -38639.90},
    {"x_px": 1964.93, "y_px": 5820.76, "x_um": 21139.60, "y_um": -37890.70},
    {"x_px": 2030.40, "y_px": 4523.97, "x_um": 21277.20, "y_um": -32726.30},
    {"x_px": 1623.79, "y_px": 3279.10, "x_um": 19584.20, "y_um": -27824.70},
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
        # The 25-70 um radius clamp that used to guard "edge" placement
        # was removed on 2026-10-19: every ring sits at the measured or
        # drawn radius plus the offsets, whatever that radius is.

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

        # Shot pattern, picked with "Laser shot pattern..." in the
        # parameters window. The ring described above is ring 1 of
        # every pattern; the others add to it, none changes it.
        #
        # "ring"             ring 1 only (the original behaviour)
        # "two-rings"        ring 1 plus a second ring ring2-offset-um
        #                    further out, its count from the same
        #                    spacing rule at that radius, staggered
        #                    half a step
        # "two-rings-12"     as two-rings, ring 2 fixed at ring2-spots
        # "dense-ring"       ring 1 packed at the crater width, so the
        #                    craters touch and none overlaps
        # "dense-two-rings"  both rings packed at the crater width
        # "center"           one shot on the bead centre; the own-bead
        #                    crater check is skipped for it on purpose
        # "triple-graded"    three rings: ring 1 dense (craters touching),
        #                    ring 2 medium (the ring rule), ring 3 light
        #                    (light-spacing-factor x spot-spacing along
        #                    the arc, at least min-spots)
        # "triple-medium"    three rings, all medium
        # "triple-dense-inner"  rings 1 and 2 dense, ring 3 light
        "shot-pattern": "ring",
        "ring2-offset-um": 30,
        "ring2-spots": 12,
        "ring3-offset-um": 15,      # ring 3 beyond ring 2 (2026-10-19)
        "light-spacing-factor": 1.5,
    },

    # Laser footprint. Affects ONLY the crater-overlap check; it moves
    # no shot. UNVERIFIED -- confirm with the instrument operator.
    "focal-spot-um": 10,
    "beam-scan": True,
    "beam-scan-um": 20,

    # Drop a shot whose crater would overlap the bead itself, a
    # neighbour or an adjacent shot. OFF since 2026-10-19: the code only
    # chooses shot CENTRES -- ring 1 at edge-offset from the bead edge,
    # ring 2 at ring2-offset-um beyond it -- and overlapping craters are
    # accepted. The checks stay in place_shots for anyone who wants them.
    "enforce-bead-clearance": False,

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
        "threshold-step": 5,

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
        # 'Run and export map' sets this for one run: the scan with
        # the naming grid and every accepted bead labelled, in tiles,
        # plus a legend of bead coordinates (draw_bead_map)
        "export-map": False,

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
    # "drawn" (manual selection circle) or "broad" (broad method, halo
    # included): the circle IS the bead, so shot_radius trusts it
    source: str = ""
    shots: list = field(default_factory=list)
    name: str = ""      # grid name, e.g. "AC01" -- see name_beads


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
                            in ("1", "true", "yes"),
                            source=str(row.get("source", "") or "").strip()))
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
    ring: int = 1       # 1 = innermost ring (the centre shot counts as 1)
    shot: int = 1       # 1-based number of the shot within its ring


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

    # the bead's own radius, measured or drawn, with no clamp (2026-10-19)
    r = bead.diameter_um / 2.0
    return r + float(sp["edge-offset"])


def pattern_rings(bead: Bead, cfg: dict, R1: float,
                  angles1: list[float]) -> list[tuple[float, list[float]]]:
    """
    (radius, angles) of every ring the chosen shot-pattern places on
    one bead. Ring 1 is passed in exactly as place_shots always built
    it and is returned unchanged; the other patterns only add to it.

    Ring 2 sits ring2-offset-um further out. Its count comes from
    circular_pack at that radius (two-rings) or from ring2-spots
    (two-rings-12). A dense ring holds as many shots as fit with every
    crater clear of the next: the chord between neighbours, which is
    what the crater check measures, must reach the crater width
    (circular_pack spaces along the arc, whose chord falls short).
    A second ring is rotated so that none of its shots lands radially
    behind a ring-1 shot: by half of 360 / lcm(n1, n2), the smallest
    angle by which shots of the two rings can differ, which is the
    rotation that keeps every ring-2 shot as far as possible from the
    ring-1 shot nearest to it.
    """
    sp = cfg["shot-placement"]
    pattern = sp.get("shot-pattern", "ring")
    if pattern == "center":
        return [(0.0, [0.0])]
    if pattern not in ("two-rings", "two-rings-12",
                       "dense-ring", "dense-two-rings",
                       "triple-graded", "triple-medium", "triple-dense-inner"):
        return [(R1, angles1)]
    offset = float(sp.get("edge-offset", 15))
    rot = float(sp.get("rotation-offset-deg", 0.0))
    R2 = R1 + float(sp.get("ring2-offset-um", 30))
    R3 = R2 + float(sp.get("ring3-offset-um", 15))

    def packed(R: float, extra_rot: float) -> list[float]:
        # circular_pack works from the bead radius and adds the edge
        # offset itself, so hand it the ring radius less the offset.
        d = dict(cfg)
        d["shot-placement"] = dict(sp, **{"rotation-offset-deg": rot + extra_rot})
        return circular_pack(R - offset, d)

    def dense(R: float, extra_rot: float) -> list[float]:
        n = int(math.floor(math.pi / math.asin(min(1.0, crater / (2.0 * R)))))
        n = max(n, 1)
        return [rot + extra_rot + 360.0 * k / n for k in range(n)]

    def stagger(n1: int, n2: int) -> float:
        return 180.0 / (n1 * n2 // math.gcd(n1, n2))

    if pattern == "two-rings":
        n2 = len(packed(R2, 0.0))
        return [(R1, angles1), (R2, packed(R2, stagger(len(angles1), n2)))]
    if pattern == "two-rings-12":
        n2 = max(int(sp.get("ring2-spots", 12)), 1)
        st = rot + stagger(len(angles1), n2)
        return [(R1, angles1),
                (R2, [st + 360.0 * k / n2 for k in range(n2)])]
    crater = footprint_um(cfg)
    dense1 = dense(R1, 0.0)
    if pattern == "dense-ring":
        return [(R1, dense1)]
    if pattern in ("triple-graded", "triple-medium", "triple-dense-inner"):
        # three rings, each staggered against the one inside it
        def light(R: float, extra_rot: float) -> list[float]:
            spacing = (float(sp.get("spot-spacing", 60))
                       * float(sp.get("light-spacing-factor", 1.5)))
            n = max(int(math.floor(2 * math.pi * R / spacing)),
                    int(sp.get("min-spots", 4)))
            return [rot + extra_rot + 360.0 * k / n for k in range(n)]
        kinds = {"triple-graded": (dense, packed, light),
                 "triple-medium": (packed, packed, packed),
                 "triple-dense-inner": (dense, dense, light)}[pattern]
        rings, turn = [], 0.0
        for R, make in zip((R1, R2, R3), kinds):
            n = len(make(R, 0.0))
            if rings:
                turn += stagger(len(rings[-1][1]), n)
            rings.append((R, make(R, turn)))
        return rings
    n2 = len(dense(R2, 0.0))
    return [(R1, dense1), (R2, dense(R2, stagger(len(dense1), n2)))]


def place_shots(beads: list[Bead], cfg: dict,
                T: "Transform | None" = None) -> list[Shot]:
    fixed_angles = [float(a) for a in cfg["laser-shot-angles"]]
    dynamic = cfg["shot-placement"].get("dynamic-spots", False)
    pattern = cfg["shot-placement"].get("shot-pattern", "ring")
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
        # "ring" is exactly the (R, angles) above; the other patterns
        # add a ring or replace the set with the centre shot.
        rings = pattern_rings(b, cfg, R, angles)
        ring: list[Shot] = []
        # k is the ring number (1 = innermost), carried on the shot so
        # position names can say which ring a shot belongs to
        # j numbers the shots around that ring in placement order
        for R, a, k, j in [(R_, a_, k_, j_)
                           for k_, (R_, angs) in enumerate(rings, start=1)
                           for j_, a_ in enumerate(angs, start=1)]:
            rad = math.radians(a)
            # Pixel position too: microMS's writeXEO maps blobs from
            # pixels, and the map image is drawn in pixels.
            R_px = R / T_um_per_px if T_um_per_px else 0.0
            s = Shot(i, a,
                     b.x_um + R * math.cos(rad),
                     b.y_um + R * math.sin(rad),
                     b.x_px + R_px * math.cos(rad),
                     b.y_px + R_px * math.sin(rad),
                     ring=k, shot=j)

            # No software travel-limit check. The stage enforces its
            # own limits in hardware; a guessed coordinate window here
            # silently discarded entire target lists.
            if enforce and pattern != "center" \
                    and (R - crater / 2.0) < (b.diameter_um / 2.0):
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
        # Same switch as the checks above (enforce-bead-clearance).
        live = [s for s in ring if not s.dropped] if enforce else []
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
# BEAD NAMES  (grid cells, 2026-10-20)
#
# Accepted beads are named by a fixed MAP_COLS x MAP_ROWS grid laid
# over the scan (6 x 2: columns A-F across x, rows A-B down y):
# column letter from x (A = leftmost), row letter from y (A = top),
# then the bead's number inside that cell -- "AB01" is bead 01 of
# column A, row B. Cells are scan width / MAP_COLS by scan height /
# MAP_ROWS. Inside a cell the beads are numbered left to right in
# MAP_BANDS horizontal bands, top band first. Rejected and clumped
# beads get no name. A cell holding more than MAP_MAX_PER_CELL beads
# is only warned about.
# =====================================================================

MAP_COLS = 6                # grid columns across x: letters A..F
MAP_ROWS = 2                # grid rows down y: letters A..B
MAP_MAX_PER_CELL = 25       # warn when a cell holds more accepted beads
MAP_BANDS = 5               # numbering bands inside a cell (top first)
MAP_TILES = 10              # the full map is also written in this many tiles
MAP_TILE_OVERLAP = 0.05     # tiles overlap by this fraction of their size
# Label size follows the picture it is drawn on, so a tile or the
# overview stays readable when it is scaled to fit a screen: text
# height = picture width * MAP_LABEL_FRACTION, never under MAP_LABEL_PX.
MAP_LABEL_FRACTION = 1 / 130
MAP_LABEL_PX = 16
MAP_OVERVIEW_PX = 8000      # the whole-scan map is reduced to this longer side
MAP_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
RING_LETTERS = "XYZ"         # ring 1, 2, 3 in a position name (AB01X1)


def scan_size(scan: Path | None, beads: list[Bead]) -> tuple[int, int]:
    """(width, height) in pixels of the scan, or of the beads' extent
    when the scan is not available."""
    if scan is not None and scan.exists():
        try:
            import cv2
            img = cv2.imread(str(scan), cv2.IMREAD_UNCHANGED)
            if img is not None:
                return int(img.shape[1]), int(img.shape[0])
        except ImportError:
            pass
    if beads:
        return (int(math.ceil(max(b.x_px for b in beads))) + 1,
                int(math.ceil(max(b.y_px for b in beads))) + 1)
    return 1, 1


def name_beads(beads: list[Bead], width_px: int,
               height_px: int) -> tuple[float, float, int, int]:
    """
    Assign Bead.name to every accepted bead (see the section comment).
    Returns (cell width px, cell height px, columns, rows) so the map
    can draw the same grid.
    """
    for b in beads:
        b.name = ""
    cols, rows = MAP_COLS, MAP_ROWS
    cw = float(max(width_px, 1)) / cols
    ch = float(max(height_px, 1)) / rows
    acc = [b for b in beads if b.accepted]
    if not acc:
        return cw, ch, cols, rows
    xs = np.array([b.x_px for b in acc], float)
    ys = np.array([b.y_px for b in acc], float)
    ci = np.clip(np.floor(xs / cw).astype(int), 0, cols - 1)
    ri = np.clip(np.floor(ys / ch).astype(int), 0, rows - 1)
    groups: dict[tuple[int, int], list[Bead]] = {}
    for b, c, r in zip(acc, ci, ri):
        groups.setdefault((int(c), int(r)), []).append(b)
    band = ch / MAP_BANDS
    for (c, r), members in groups.items():
        members.sort(key=lambda b: (int((b.y_px - r * ch) // band), b.x_px))
        for k, b in enumerate(members, start=1):
            b.name = f"{MAP_LETTERS[c]}{MAP_LETTERS[r]}{k:02d}"
        if len(members) > MAP_MAX_PER_CELL:
            print(f"  WARNING: cell {MAP_LETTERS[c]}{MAP_LETTERS[r]} holds "
                  f"{len(members)} beads (more than {MAP_MAX_PER_CELL})")
    return cw, ch, cols, rows


# =====================================================================
# EXPORT -- CSV
# =====================================================================

def write_csv(path: Path, beads: list[Bead], shots: list[Shot],
              names: list[str] | None = None) -> None:
    """names: the position name of each shot, as written to the .xeo
    and .run (position_name). Without it the old B0001_A180 is used."""
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "position", "bead_name", "bead_id", "ring",
                    "shot", "angle_deg", "x_um", "y_um", "bead_x_um",
                    "bead_y_um",
                    "bead_diameter_um", "nn_um", "clumped"])
        for k, s in enumerate(shots):
            b = beads[s.bead_id]
            pos = (names[k] if names and k < len(names)
                   else f"B{s.bead_id:04d}_A{int(s.angle_deg):03d}")
            w.writerow([k, pos, b.name, s.bead_id, s.ring, s.shot,
                        f"{s.angle_deg:.1f}",
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
# The wrapper is CONFIRMED (2026-10-13) against a genuine .xeo from
# Dr Neumann's instrument, "MTP Slide Adapter II.xeo" ($Revision 1.6).
# It differs from microMS's brukerMapper header (Revision 1.5) in:
# RowsNumber 10 (not 100); MappingParameters mox/moy 0.0 (not 56.24 /
# 42.64) and alpha/beta 51750 (not 51.750); the MappingParameters and
# PlateSpots lines indented with four spaces, the AutoTeachSpots block
# with tabs only; </PlateType> unindented and followed by CRLF. The
# PlateSpot lines and the UnitCoord grid (C5 = -0.652174, 0.478261)
# are the same in both, so UnitCoord values are computed unchanged.
# =====================================================================

# FORMAT SPEC, reproduced verbatim from Dr Neumann's
# "MTP Slide Adapter II.xeo". flexMapper hands these to microMS's own
# writeXEO in place of brukerMapper's header and footer.
#
# The header is 12 lines, plus the <PlateSpots> line written per file,
# giving the 13 that microMS's own loadXEO skips. The footer is 12.

XEO_HEADER = [
    '<!-- $Revision: 1.6 $-->',
    '<PlateType>',
    '\t<GlobalParameters PlateTypeName="MTP Slide Adapter II" ProbeType="MTP"',
    '\t                  RowsNumber="10" ChipNumber="1" ChipsInRow="1"',
    '\t                  X_ChipOffsetSize="0" Y_ChipOffsetSize="0"',
    '\t                  HasDirectLabels="false" HasColRowLabels="true"',
    '\t                  HasNearNeighbourCalibrants="false"',
    '\t                  ProbeDiameterX="103.5" SampleDiameter="2"',
    '\t                  SamplePixelRadius="5" ZoomFactor="1"',
    '\t                  FirstCalibrant="TPX1" SecondCalibrant="TPX2" '
    'ThirdCalibrant="TPX3"',
    '\t                  />',
    '    <MappingParameters mox="0.0" moy="0.0" sinphi="0.000000" '
    'cosphi="1.000000" alpha="51750" beta="51750" tansigma="0.000000"/>',
]

# writeXEO writes this line itself, with a tab; flexMapper re-indents it
XEO_PLATESPOTS_INDENT = '    '

XEO_FOOTER = [
    '\t</PlateSpots>',
    '\t<AutoTeachSpots>',
    '\t\t<PlateSpot PositionIndex="0" PositionName="TPX1" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="0.550725"/>',
    '\t\t<PlateSpot PositionIndex="1" PositionName="TPX2" '
    'UnitCoord_X="0.729469" UnitCoord_Y="0.550725"/>',
    '\t\t<PlateSpot PositionIndex="2" PositionName="TPX3" '
    'UnitCoord_X="0.729469" UnitCoord_Y="0.057971"/>',
    '\t\t<PlateSpot PositionIndex="3" PositionName="TPX4" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="0.057971"/>',
    '\t\t<PlateSpot PositionIndex="4" PositionName="TPY1" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="-0.057971"/>',
    '\t\t<PlateSpot PositionIndex="5" PositionName="TPY2" '
    'UnitCoord_X="0.729469" UnitCoord_Y="-0.057971"/>',
    '\t\t<PlateSpot PositionIndex="6" PositionName="TPY3" '
    'UnitCoord_X="-0.729469" UnitCoord_Y="-0.550725"/>',
    '\t\t<PlateSpot PositionIndex="7" PositionName="TPY4" '
    'UnitCoord_X="0.729469" UnitCoord_Y="-0.550725"/>',
    '\t</AutoTeachSpots>',
    '</PlateType>',
]

# Named MTP grid positions and their UnitCoord fractions, from
# brukerMapper.py. FORMAT SPEC -- plate geometry, not a measurement.
# UnitCoord is a SIGNED fraction about the plate centre; the header's
# alpha and beta give the scale, 51.75 mm per unit on both axes
# (written 51750, in um, in Dr Neumann's reference .xeo).
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


def _flex_mapper():
    """
    timsTOF fleX coordinate mapper -- formerly flex_mapper.py, carried
    in unchanged so the tool is one script. microMS's brukerMapper is
    imported only when a .xeo is written.

    A concrete implementation of microMS's `brukerMapper` abstract base
    class, so `.xeo` files are written by microMS's own `writeXEO`.
    microMS is MIT licensed, copyright (c) 2016 troycomi. See
    microms/LICENSE.
    """
    import os
    import types

    HERE = os.path.dirname(os.path.abspath(__file__))
    if os.path.join(HERE, "microms") not in sys.path:
        sys.path.insert(0, os.path.join(HERE, "microms"))

    from CoordinateMappers import brukerMapper      # noqa: E402
    from ImageUtilities import blob                 # noqa: E402

    class flexMapper(brukerMapper.brukerMapper):
        """Coordinate mapper for the Bruker timsTOF fleX."""

        def __init__(self, coord_file=None):
            # brukerMapper.loadStagePoints() reads this during __init__,
            # so it has to be set first.
            self.motorCoordFilename = coord_file or os.path.join(
                HERE, "flexCoords.txt")
            self.instrumentExtension = ".xeo"
            self.instrumentName = "timsTOF fleX"
            super().__init__()

            # True on every Bruker mapper microMS ships: stage y counts
            # opposite to image y.
            self.reflectCoordinates = True

            # Dr Neumann's reference .xeo in place of brukerMapper's
            # Revision 1.5 header and footer (see XEO_HEADER)
            self.header = "\n".join(XEO_HEADER) + "\n"
            self.footer = "\n".join(XEO_FOOTER) + "\n"

        # -- required concrete methods ------------------------------------

        def isValidMotorCoord(self, instr):
            """Stage coordinates are two numbers separated by a space."""
            if instr is None or " " not in instr:
                return False
            try:
                a, b = instr.split(" ")[:2]
                float(a)
                float(b)
                return True
            except ValueError:
                return False

        def extractMotorPoint(self, inStr):
            if not self.isValidMotorCoord(inStr):
                return None
            a, b = inStr.split(" ")[:2]
            return (int(float(a)), int(float(b)))

        def loadInstrumentFile(self, filename):
            return self.loadXEO(filename)

        def saveInstrumentFile(self, filename, blobs, names=None):
            self.writeXEO(filename, blobs)
            # writeXEO puts a tab before <PlateSpots>; the reference
            # indents it with four spaces and uses CRLF line ends
            path = Path(filename)
            if path.exists():
                text = path.read_text().replace(
                    "\n\t<PlateSpots ",
                    "\n" + XEO_PLATESPOTS_INDENT + "<PlateSpots ", 1)
                # writeXEO names every spot x_<px>y_<py>; the .run is
                # matched to the .xeo by name, so the names handed in
                # (position_name, one per blob) replace them here
                if names:
                    text = rename_xeo_positions(text, names)
                path.write_text(text, newline="\r\n")


    def write_coord_file(path, calibration):
        """
        Write the tab-delimited file brukerMapper.loadStagePoints reads.

        calibration: [{"name": "C5", "x_um": ..., "y_um": ...}, ...]
        """
        with open(path, "w") as fh:
            for c in calibration:
                fh.write(f"{c['name']}\t{int(round(c['x_um']))}"
                         f"\t{int(round(c['y_um']))}\n")
        return path


    def make_blob(x_px, y_px, radius=1.0, group=None):
        """A microMS blob at a pixel position."""
        return blob.blob(x=x_px, y=y_px, radius=radius, circularity=1.0,
                         group=group)

    return types.SimpleNamespace(flexMapper=flexMapper,
                                 write_coord_file=write_coord_file,
                                 make_blob=make_blob)


def rename_xeo_positions(text: str, names: list[str]) -> str:
    """
    Replace the PositionName of each <PlateSpot> inside the
    <PlateSpots> block, in order, with the given names. The footer's
    AutoTeachSpots also use <PlateSpot> lines and are left alone.
    """
    import re
    out, k, inside = [], 0, False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("<PlateSpots"):
            inside = True
        elif s.startswith("</PlateSpots>"):
            inside = False
        elif inside and s.startswith("<PlateSpot ") and k < len(names):
            line = re.sub(r'PositionName="[^"]*"',
                          f'PositionName="{names[k]}"', line, count=1)
            k += 1
        out.append(line)
    if k != len(names):
        raise RuntimeError(f".xeo has {k} positions but {len(names)} "
                           f"names were given")
    return "\n".join(out) + ("\n" if text.endswith("\n") else "")


def write_xeo(prefix: Path, shots: list[Shot], beads: list[Bead],
              cfg: dict, M: "Transform | None" = None,
              names: list[str] | None = None) -> list[Path]:
    """
    Write .xeo files using microMS's own brukerMapper.writeXEO.

    names: one position name per shot (position_name). They replace
    writeXEO's x_<px>y_<py> names so the .xeo matches the .run.

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

    flex_mapper = _flex_mapper()

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
        s0 = (n - 1) * XEO_MAX_POSITIONS
        mapper.saveInstrumentFile(
            str(out), blobs,
            names[s0:s0 + len(chunk)] if names else None)
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
                   T: Transform, names: list[str] | None = None) -> Path:
    """Write a flexImaging target list in stage coordinates.
    names: the spot-name per shot (position_name); without it the
    old x<px>_y<py> is used."""
    rows = ["# X-pos Y-pos spot-name region"]
    region = str(cfg["output"].get("flex-region", "01"))
    negate = -1.0 if cfg["output"].get("flex-negate-y", True) else 1.0

    # Fiducials first, so flexImaging can register the slide itself.
    for i, f in enumerate(cfg["fiducials"]):
        rows.append(f"{f['x_um']:.0f} {negate * f['y_um']:.0f} "
                    f"fiducial{i} {region}")

    for k, sh in enumerate(shots):
        spot = (names[k] if names and k < len(names)
                else f"x{sh.x_px:.0f}_y{sh.y_px:.0f}")
        rows.append(f"{sh.x_um:.0f} {negate * sh.y_um:.0f} "
                    f"{spot} {region}")

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


def position_name(cfg: dict, index: int, shot,
                  beads: list[Bead] | None = None) -> str:
    """
    Name for one position, shared by the .xeo, the .run, the CSV and
    the flexImaging .txt.

    The .xeo and .run are matched by name alone, so whatever this
    returns must be used identically in both. Default (2026-10-20):
    <grid name><ring letter><shot number>, e.g. AB01X1 -- the bead's
    grid cell and number (name_beads), the ring the shot sits on as a
    letter (X = ring 1, Y = ring 2, Z = ring 3; RING_LETTERS) and the
    shot's number
    around that ring in placement order, from 1. No underscores.
    An output.position-name pattern overrides it; the fields are
    name, ringletter, shot, ring, angle, region, n, i, j, px, py, bead.
    """
    out = cfg["output"]
    bead = beads[shot.bead_id] if beads is not None else None
    name = (bead.name if bead is not None and bead.name
            else f"B{shot.bead_id:04d}")
    angle = int(round(shot.angle_deg)) % 360
    ring = int(getattr(shot, "ring", 1))
    pattern = out.get("position-name") or "{name}{ringletter}{shot}"
    return pattern.format(name=name, ring=ring,
                          ringletter=RING_LETTERS[min(ring, len(RING_LETTERS)) - 1],
                          shot=int(getattr(shot, "shot", 1)),
                          angle=angle,
                          region=out.get("region", 0),
                          n=index + 1,
                          i=shot.x_um / PLATE_UNIT_UM,   # plate units
                          j=shot.y_um / PLATE_UNIT_UM,
                          px=shot.x_px, py=shot.y_px,
                          bead=shot.bead_id)


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



def _map_annotate(canvas, beads: list[Bead], cw: float, ch: float,
                  cols: int, rows: int, f: float, label_px: float) -> None:
    """Draw the naming grid, its letters and the labelled beads onto
    canvas, whose pixels are the scan's scaled by f."""
    import cv2
    H, W = canvas.shape[:2]
    grid_col = (0, 200, 255)          # BGR: amber grid and letters
    bead_col = (60, 200, 60)          # green circles
    text_col = (255, 80, 0)           # blue labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    base = cv2.getTextSize("A", font, 1.0, 1)[0][1]    # px tall at scale 1
    scale = label_px / float(base)
    thick = max(1, int(round(scale)))
    lw = max(1, int(round(label_px / 8.0)))

    for c in range(1, cols):
        x = int(round(c * cw * f))
        cv2.line(canvas, (x, 0), (x, H - 1), grid_col, lw)
    for r in range(1, rows):
        y = int(round(r * ch * f))
        cv2.line(canvas, (0, y), (W - 1, y), grid_col, lw)
    big = scale * 2.5
    for c in range(cols):
        x = int(round((c + 0.5) * cw * f))
        cv2.putText(canvas, MAP_LETTERS[c], (x, int(base * big) + lw + 4),
                    font, big, grid_col, thick * 2, cv2.LINE_AA)
    for r in range(rows):
        y = int(round((r + 0.5) * ch * f))
        cv2.putText(canvas, MAP_LETTERS[r], (lw + 4, y), font, big,
                    grid_col, thick * 2, cv2.LINE_AA)

    for b in beads:
        x, y = int(round(b.x_px * f)), int(round(b.y_px * f))
        rad = max(int(round(b.diameter_px / 2.0 * f)) + 2, lw + 2)
        cv2.circle(canvas, (x, y), rad, bead_col, lw, cv2.LINE_AA)
        cv2.putText(canvas, b.name, (x + rad + 2, y - rad - 2), font,
                    scale, text_col, thick, cv2.LINE_AA)


def draw_bead_map(prefix: Path, beads: list[Bead], cfg: dict,
                  scan: Path | None, T: Transform, cw: float, ch: float,
                  cols: int, rows: int) -> list[Path]:
    """
    The bead map ('Run and export map'): the scan with the naming
    grid drawn over it, its column letters along the top and row
    letters down the left, and every accepted bead circled and
    labelled with its grid name. Written as

        <prefix>_map.png            the whole scan, reduced to
                                    MAP_OVERVIEW_PX on its longer side
        <prefix>_map_01.png ...     MAP_TILES equal full-resolution
                                    tiles (overlapping by
                                    MAP_TILE_OVERLAP so no label is
                                    cut), laid out to keep tiles close
                                    to square
        <prefix>_map_legend.txt     per grid cell, each bead's stage
                                    and pixel coordinates

    Labels are sized to the picture they sit on (MAP_LABEL_FRACTION).
    Returns the files written.
    """
    import cv2
    img = None
    if scan is not None and scan.exists():
        img = cv2.imread(str(scan), cv2.IMREAD_GRAYSCALE)
    if img is None:
        W, H = int(math.ceil(cols * cw)), int(math.ceil(rows * ch))
        img = np.full((max(H, 1), max(W, 1)), 255, np.uint8)
    H, W = img.shape[:2]
    acc = [b for b in beads if b.accepted and b.name]
    written = []

    # tiles: the (rows x cols) split of MAP_TILES whose tiles are
    # closest to square for this scan
    best = None
    for tr in range(1, MAP_TILES + 1):
        if MAP_TILES % tr:
            continue
        tc = MAP_TILES // tr
        score = abs(math.log((W / tc) / (H / tr)))
        if best is None or score < best[0]:
            best = (score, tr, tc)
    _, tr, tc = best
    tw, th = W / tc, H / tr
    ox, oy = tw * MAP_TILE_OVERLAP, th * MAP_TILE_OVERLAP

    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    label_px = max(MAP_LABEL_PX, tw * MAP_LABEL_FRACTION)
    _map_annotate(canvas, acc, cw, ch, cols, rows, 1.0, label_px)
    font = cv2.FONT_HERSHEY_SIMPLEX
    base = cv2.getTextSize("A", font, 1.0, 1)[0][1]
    hdr = label_px * 1.5 / base
    k = 0
    for r in range(tr):
        for c in range(tc):
            k += 1
            x0 = int(max(0, math.floor(c * tw - ox)))
            x1 = int(min(W, math.ceil((c + 1) * tw + ox)))
            y0 = int(max(0, math.floor(r * th - oy)))
            y1 = int(min(H, math.ceil((r + 1) * th + oy)))
            tile = canvas[y0:y1, x0:x1].copy()
            label = f"map {k}/{MAP_TILES}  px x {x0}-{x1}  y {y0}-{y1}"
            cv2.putText(tile, label, (6, int(base * hdr) + 6), font, hdr,
                        (0, 0, 255), max(1, int(round(hdr)) * 2), cv2.LINE_AA)
            tp = prefix.with_name(f"{prefix.name}_map_{k:02d}.png")
            cv2.imwrite(str(tp), tile)
            written.append(tp)
    del canvas

    # the whole scan, reduced, with labels sized for that picture
    f = min(1.0, MAP_OVERVIEW_PX / float(max(W, H)))
    small = (cv2.resize(img, (max(1, int(round(W * f))),
                              max(1, int(round(H * f)))),
                        interpolation=cv2.INTER_AREA) if f < 1.0 else img)
    overview = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    _map_annotate(overview, acc, cw, ch, cols, rows, f,
                  max(MAP_LABEL_PX, overview.shape[1] * MAP_LABEL_FRACTION))
    full = prefix.with_name(prefix.name + "_map.png")
    cv2.imwrite(str(full), overview)
    written.insert(0, full)

    # legend
    um_per_px = T.um_per_px if T is not None else 0.0
    lines = [f"Bead map legend -- {cols} x {rows} grid, cells "
             f"{cw:.0f} x {ch:.0f} px"
             + (f" ({cw * um_per_px:.0f} x {ch * um_per_px:.0f} um)"
                if um_per_px else ""),
             "Column letter = x (A leftmost), row letter = y (A top). "
             "Bead name = column + row + number inside the cell.",
             "Coordinates: stage um (x, y), then scan pixels in brackets.",
             f"Tiles: {tr} rows x {tc} columns of {MAP_TILES}, numbered "
             f"left to right, top to bottom.",
             ""]
    cells: dict[tuple[int, int], list[Bead]] = {}
    for b in acc:
        cells.setdefault((MAP_LETTERS.index(b.name[0]),
                          MAP_LETTERS.index(b.name[1])), []).append(b)
    for (c, r) in sorted(cells):
        members = sorted(cells[(c, r)], key=lambda b: b.name)
        parts = [f"bead {b.name} at x {b.x_um:.0f} um and y {b.y_um:.0f} um "
                 f"(px {b.x_px:.0f}, {b.y_px:.0f})" for b in members]
        lines.append(f"Region {MAP_LETTERS[c]} (x-axis) / {MAP_LETTERS[r]} "
                     f"(y-axis) contains {len(members)} beads: "
                     + "; ".join(parts))
    lp = prefix.with_name(prefix.name + "_map_legend.txt")
    lp.write_text("\n".join(lines) + "\n")
    written.append(lp)
    return written


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

    # centre the window on the beads it holds (the search above anchors
    # it at their corner, which put a lone bead in the top-left corner)
    hit = (xs >= bx) & (xs < bx + w) & (ys >= by) & (ys < by + h)
    if hit.any():
        bx, by = float(xs[hit].mean()) - w / 2, float(ys[hit].mean()) - h / 2
    bx = min(max(bx, 0), max(img.shape[1] - w, 0))
    by = min(max(by, 0), max(img.shape[0] - h, 0))
    x0, y0 = int(bx), int(by)
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


DIST_MAX_BEADS = 4          # beads measured in the distance picture


def draw_distances(path: Path, beads: list[Bead], cfg: dict, T: Transform,
                   scan: Path | None) -> bool:
    """
    Quick size measurement: a close-up of a few accepted beads with
    their shots, each bead labelled in thin black text with its radius
    in microns (centre to the drawn outline) and, per ring, the shot
    distance from the centre -- a line from the centre to the outline
    and to one live shot of each ring. The window is centred on the
    patch holding the most accepted beads, so nothing sits in a corner.
    Same rendering as draw_zoom; the other pictures are untouched.
    """
    try:
        import cv2
    except ImportError:
        return False
    if scan is None or not scan.exists():
        return False
    acc = [b for b in beads if b.accepted and b.shots]
    if not acc:
        return False
    img = cv2.imread(str(scan), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    z = cfg["output"]
    w = int(z.get("zoom-window-px", 700))
    h = int(w * 0.66)
    sc = int(z.get("zoom-scale", 3))
    xs = np.array([b.x_px for b in acc])
    ys = np.array([b.y_px for b in acc])

    # the window holding the most accepted beads, then centred on them
    best, bx, by = -1, float(xs.min()), float(ys.min())
    for cx in np.linspace(xs.min(), max(xs.max() - w, xs.min()), 40):
        for cy in np.linspace(ys.min(), max(ys.max() - h, ys.min()), 30):
            hit = ((xs >= cx) & (xs < cx + w) & (ys >= cy) & (ys < cy + h))
            if int(hit.sum()) > best:
                best, bx, by = int(hit.sum()), float(cx), float(cy)
    hit = ((xs >= bx) & (xs < bx + w) & (ys >= by) & (ys < by + h))
    cx, cy = float(xs[hit].mean()), float(ys[hit].mean())
    x0 = int(min(max(cx - w / 2, 0), max(img.shape[1] - w, 0)))
    y0 = int(min(max(cy - h / 2, 0), max(img.shape[0] - h, 0)))
    x1, y1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])

    vis = cv2.resize(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR),
                     ((x1 - x0) * sc, (y1 - y0) * sc),
                     interpolation=cv2.INTER_LANCZOS4)
    crater_px = footprint_um(cfg) / T.um_per_px

    def at(px, py):
        return int((px - x0) * sc), int((py - y0) * sc)

    font, fs, black = cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0)
    shown = sorted((b for b in acc if x0 < b.x_px < x1 and y0 < b.y_px < y1),
                   key=lambda b: math.hypot(b.x_px - cx, b.y_px - cy))
    for b in shown[:DIST_MAX_BEADS]:
        c = at(b.x_px, b.y_px)
        r_px = max(int(b.diameter_px / 2 * sc), 3)
        cv2.circle(vis, c, r_px, (0, 170, 0), 1)
        for sh in b.shots:
            px = T.um_to_px([sh.x_um, sh.y_um])[0]
            pt = at(px[0], px[1])
            rr = max(int(crater_px / 2 * sc), 3)
            if sh.dropped:
                cv2.circle(vis, pt, rr, (0, 165, 255), 1)
            else:
                cv2.circle(vis, pt, rr, (230, 120, 0), -1)
        # bead radius: centre to the outline, straight up
        cv2.line(vis, c, (c[0], c[1] - r_px), black, 1, cv2.LINE_AA)
        cv2.putText(vis, f"r {b.diameter_um / 2:.0f} um",
                    (c[0] + 4, c[1] - r_px - 6), font, fs, black, 1,
                    cv2.LINE_AA)
        # one live shot per ring (distinct distances from the centre),
        # each ring's label on its own side of the bead: ring 1 lower
        # right, ring 2 lower left, then upper right, upper left -- the
        # radius label has the top
        rings: dict = {}
        for sh in b.shots:
            if not sh.dropped:
                dist = round(math.hypot(sh.x_um - b.x_um, sh.y_um - b.y_um))
                rings.setdefault(dist, []).append(sh)
        sides = (30.0, 150.0, -30.0, -150.0)          # image degrees, y down
        for k, (dist, group) in enumerate(sorted(rings.items())):
            want = math.radians(sides[k % len(sides)])

            def off(sh):
                px = T.um_to_px([sh.x_um, sh.y_um])[0]
                a = math.atan2(px[1] - b.y_px, px[0] - b.x_px)
                return abs((a - want + math.pi) % (2 * math.pi) - math.pi)
            sh = min(group, key=off)
            px = T.um_to_px([sh.x_um, sh.y_um])[0]
            pt = at(px[0], px[1])
            cv2.line(vis, c, pt, black, 1, cv2.LINE_AA)
            # label just past the shot, on the far side from the centre
            rr = max(int(crater_px / 2 * sc), 3)
            dx, dy = pt[0] - c[0], pt[1] - c[1]
            n = max(math.hypot(dx, dy), 1.0)
            text = f"shot {dist:.0f} um"
            tw = cv2.getTextSize(text, font, fs, 1)[0][0]
            lx = int(pt[0] + dx / n * (rr + 6)) - (tw if dx < 0 else 0)
            ly = int(pt[1] + dy / n * (rr + 6)) + (12 if dy >= 0 else -4)
            cv2.putText(vis, text, (lx, ly), font, fs, black, 1, cv2.LINE_AA)

    cv2.putText(vis, f"green = bead outline   blue = shot   orange = dropped"
                     f"   black = distance from the bead centre, um",
                (6, 16), font, fs, black, 1, cv2.LINE_AA)
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

    state = {"key": None, "busy": False, "fast": False}
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
        # while a drag or a scroll is in progress: one level coarser (a
        # quarter of the pixels to colour and resample, so about 3x the
        # frame rate); the full-quality frame follows when it stops
        if state["fast"]:
            level = min(level + 1, len(pyr) - 1)

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

    def set_fast(on: bool) -> None:
        """attach_zoom / gui_attach_drag_pan: coarse frames while the
        view is moving, a full-quality one when it stops."""
        if state["fast"] != bool(on):
            state["fast"] = bool(on)
            refresh()
    fig.pyramid_fast = set_fast

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

FAST_SETTLE_MS = 150        # full-quality redraw this long after the last wheel notch


def pan_preview(fig):
    """
    Lag-free dragging: while the button is down the picture already on
    the tk canvas is slid by the mouse delta (a canvas move, about a
    millisecond) instead of re-rendering the figure (100-200 ms a frame
    at window size); the real frame is rendered once, on release.
    Returns (shift(dx, dy), reset()) -- shift takes screen deltas in
    matplotlib's convention (y up), reset puts the picture back before
    the limits are set and the figure redrawn. Both are no-ops on a
    backend without a tk canvas.
    """
    tkc = getattr(fig.canvas, "_tkcanvas", None)
    offset = [0, 0]

    def item():
        ids = tkc.find_withtag("all") if tkc is not None else ()
        return ids[0] if ids else None

    def shift(dx: float, dy: float) -> None:
        i = item()
        if i is None:
            return
        mx, my = int(round(dx)) - offset[0], -int(round(dy)) - offset[1]
        if mx or my:
            tkc.move(i, mx, my)
            offset[0] += mx
            offset[1] += my
            tkc.update_idletasks()

    def reset() -> None:
        i = item()
        if i is not None and (offset[0] or offset[1]):
            tkc.move(i, -offset[0], -offset[1])
        offset[0] = offset[1] = 0

    return shift, reset


def attach_zoom(fig, ax, step: float = 1.3, arrow_zoom=True):
    """Wire scroll zoom, middle-drag pan and keyboard shortcuts.

    Returns (zoom_in, zoom_out, fit) so the same actions can be bound
    to buttons. arrow_zoom False (or a callable returning False) leaves
    the up / down arrows to something else -- panning in the fiducial
    window, moving an unset circle in the bead window.
    """
    # the limits show_pyramid set (the whole image), before the window's
    # shape widens them; None if nothing is drawn yet
    home = {"lim": (ax.get_xlim(), ax.get_ylim()) if ax.images else None}
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

    # coarse frames while the wheel keeps turning; full quality
    # FAST_SETTLE_MS after the last notch (see show_pyramid.set_fast)
    fast = getattr(fig, "pyramid_fast", None)
    settle = {"id": None}

    def settled():
        settle["id"] = None
        if fast:
            fast(False)

    def on_scroll(ev):
        if ev.inaxes is not ax or ev.xdata is None:
            return
        if getattr(ev, "key", None) == "control":
            return                          # Ctrl+scroll: the circle tool's
        if fast:
            fast(True)
            widget = fig.canvas.get_tk_widget()
            if settle["id"] is not None:
                widget.after_cancel(settle["id"])
            settle["id"] = widget.after(FAST_SETTLE_MS, settled)
        # Zoom about the cursor so the feature under the pointer stays
        # put -- the behaviour every image viewer has.
        scale_about(ev.xdata, ev.ydata,
                    1 / step if ev.button == "up" else step)

    shift, reset = pan_preview(fig)

    def on_press(ev):
        if ev.button == 2 and ev.inaxes is ax:
            remember_home()
            pan["px"] = (ev.x, ev.y, ax.get_xlim(), ax.get_ylim())

    def apply_pan(ev):
        x0px, y0px, xl, yl = pan["px"]
        sx = (xl[1] - xl[0]) / max(ax.bbox.width, 1)
        sy = (yl[1] - yl[0]) / max(ax.bbox.height, 1)
        dx = -(ev.x - x0px) * sx
        dy = -(ev.y - y0px) * sy
        ax.set_xlim(xl[0] + dx, xl[1] + dx)
        ax.set_ylim(yl[0] + dy, yl[1] + dy)
        fig.canvas.draw_idle()

    def on_motion(ev):
        if pan["px"] is None or ev.x is None:
            return
        # slide the rendered picture; the frame is rendered on release
        shift(ev.x - pan["px"][0], ev.y - pan["px"][1])

    def on_release(ev):
        if ev.button == 2 and pan["px"] is not None:
            reset()
            if ev.x is not None:
                apply_pan(ev)
            pan["px"] = None

    def on_key(ev):
        arrows = arrow_zoom() if callable(arrow_zoom) else arrow_zoom
        if ev.key in ("+", "=") or (arrows and ev.key == "up"):
            zoom_in()
        elif ev.key in ("-", "_") or (arrows and ev.key == "down"):
            zoom_out()
        elif ev.key in ("f", "home", "0"):
            fit()

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("draw_event", lambda _e: remember_home())

    def on_resize(_ev):
        """Window resized or maximized: a view of the whole scan is
        fitted to the new shape again; a zoomed-in view keeps its
        centre and scale and simply shows more or less around it."""
        if home["lim"] is None:
            return
        (hx, hy), (x0, x1), (y0, y1) = (
            [sorted(v) for v in home["lim"]],
            sorted(ax.get_xlim()), sorted(ax.get_ylim()))
        if x0 <= hx[0] and x1 >= hx[1] and y0 <= hy[0] and y1 >= hy[1]:
            fit()

    fig.canvas.mpl_connect("resize_event", on_resize)
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


def results_tag(cfg: dict) -> str:
    """The results name as a file-name prefix: "test-quill_" or "" --
    every file in a results folder starts with it (2026-10-20)."""
    name = str(cfg["output"].get("run-name") or "").strip()
    return f"{name}_" if name else ""


def results_dir(cfg: dict, subfolder: str | None = None) -> Path:
    """
    HERE/<output.results-dir>, created on demand -- the home of every
    generated file, so outputs never mix with code and scans.
    """
    root = HERE / cfg["output"].get("results-dir", "RESULTS")
    if subfolder:
        # "<stamp>_<name>_<command>": the time stamp, the results name
        # asked at Save and review / run (if any), then review or run
        stamp, _, cmd = subfolder.partition(" ")
        name = str(cfg["output"].get("run-name") or "").strip()
        subfolder = "_".join(part for part in (stamp, name, cmd) if part)
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
    bar_start("run", 10)
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

    bar_step("naming beads")
    W, H = scan_size(scan, beads)
    cw, ch, cols, rows = name_beads(beads, W, H)
    names = [position_name(cfg, i, sh, beads) for i, sh in enumerate(ordered)]
    print(f"\nBead names     : {cols} x {rows} grid of {cw:.0f} x {ch:.0f} px "
          f"cells over {W} x {H} px")
    if names:
        print(f"  position names {names[0]} .. {names[-1]}")

    outdir = results_dir(cfg, timestamp() + " run")
    bar_where(outdir)
    prefix = outdir / (results_tag(cfg) + cfg["output"]["prefix"])
    print(f"\nOutput folder  : {outdir.relative_to(HERE)}")
    log(f"writing outputs with prefix {prefix}")
    csv_path = prefix.with_suffix(".csv")
    bar_step(f"writing {csv_path.name}")
    write_csv(csv_path, beads, ordered, names)
    print(f"\nWrote {csv_path.name}  ({len(ordered)} positions, stage um)")

    if cfg["output"].get("export-map", False):
        bar_step("rendering bead map")
        log("rendering bead map")
        for f in draw_bead_map(prefix, beads, cfg, scan, T, cw, ch, cols, rows):
            print(f"Wrote {f.name}")

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
        zp = prefix.with_name(results_tag(cfg) + "shot_placement_zoom.png")
        if draw_zoom(zp, beads, cfg, T, scan):
            print(f"Wrote {zp.name}")
        dp = prefix.with_name(results_tag(cfg) + "shot_distances.png")
        if draw_distances(dp, beads, cfg, T, scan):
            print(f"Wrote {dp.name}  (bead radius and shot distances, um)")

    bar_step("writing instrument files")
    if cfg["output"].get("write-flex-txt", True) and ordered:
        fp = prefix.with_name(prefix.name + "_flexImaging.txt")
        write_flex_txt(fp, ordered, cfg, T, names)
        print(f"Wrote {fp.name}  ({len(read_flex_txt(fp))} rows: "
              f"{len(cfg['fiducials'])} fiducials + {len(ordered)} targets, "
              f"stage um)")

    if cfg["output"].get("write-xeo", True) and ordered:
        M = fit_mtp(cfg)
        files = write_xeo(prefix, ordered, beads, cfg, M, names)
        for f in files:
            print(f"Wrote {f.name}  ({len(read_xeo(f))} positions)")

        if files and cfg["output"].get("write-run", True):
            # the same names as the .xeo, chunked the same way
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
    bar_where(outdir)

    # 'check' folded in: the registration report goes into the same
    # folder as the pictures, so one folder holds everything needed to
    # judge the run.
    import contextlib
    import io
    bar_step("writing check.txt")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        report_registration(cfg)
    tag = results_tag(cfg)
    (outdir / f"{tag}check.txt").write_text(buf.getvalue().lstrip("\n"))
    say(f"Wrote {(outdir / f'{tag}check.txt').relative_to(HERE)}")

    png = outdir / f"{tag}review.png"
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

    zp = outdir / f"{tag}review_zoom.png"
    bar_step("rendering review_zoom.png")
    if draw_zoom(zp, beads, cfg, T, scan):
        say(f"Wrote {zp.relative_to(HERE)}  (close-up of the densest "
            f"patch of selected beads)")
    dp = outdir / f"{tag}review_distances.png"
    if draw_distances(dp, beads, cfg, T, scan):
        say(f"Wrote {dp.relative_to(HERE)}  (bead radius and shot "
            f"distances, um)")
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


def gui_dropdown(parent, var, options: list, command=None, tips=None):
    """Light grey raised dropdown, sized to the longest option so it
    never changes width. Hovered entries highlight navy/white. A None
    in `options` is a thin separator line. `tips` {option: text} shows
    the text in a small box beside the menu while that entry is
    highlighted (mouse over it, before the click)."""
    width = max(len(o) for o in options if o) + 2
    mb = tk.Menubutton(parent, textvariable=var, width=width, font=F_BOX,
                       bg=GUI_BOX, fg=GUI_TXT, activebackground=GUI_BOX,
                       activeforeground=GUI_TXT, relief="raised", bd=2,
                       anchor="w", indicatoron=False, highlightthickness=0)
    menu = tk.Menu(mb, tearoff=0, bg=GUI_BOX, fg=GUI_TXT,
                   activebackground=GUI_NAVY, activeforeground=GUI_WHITE,
                   font=F_BOX, relief="raised", bd=2)
    for o in options:
        if o is None:
            menu.add_separator()
        else:
            menu.add_radiobutton(label=o, variable=var, value=o,
                                 command=command)
    mb.config(menu=menu)
    if tips:
        tip = {"win": None}

        def hide(_e=None):
            if tip["win"] is not None:
                tip["win"].destroy()
                tip["win"] = None

        def select(_e=None):
            hide()
            try:
                i = menu.index("active")
                text = tips.get(menu.entrycget(i, "label"), "") if i is not None else ""
            except tk.TclError:
                text = ""
            if not text:
                return
            tip["win"] = tk.Toplevel(menu, bg=GUI_DARK)
            tip["win"].overrideredirect(True)
            tk.Label(tip["win"], text=text, font=F_SMALL, bg=GUI_LAVENDER,
                     fg=GUI_TXT, justify="left", padx=6, pady=3).pack(
                         padx=1, pady=1)
            # directly left of the dropdown, level with the highlighted
            # entry: the menu drops below the button, so its position is
            # the button's (a posted menu reports none of its own)
            tip["win"].update_idletasks()
            try:
                dy = menu.yposition(i)
            except tk.TclError:
                dy = 0
            x = mb.winfo_rootx() - tip["win"].winfo_reqwidth() - 6
            y = mb.winfo_rooty() + mb.winfo_height() + dy
            tip["win"].geometry(f"+{max(x, 0)}+{y}")

        menu.bind("<<MenuSelect>>", select)
        menu.bind("<Unmap>", hide)
        menu.bind("<Leave>", hide)
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


def gui_fill_screen(win, w: int, h: int, maximize: bool = True) -> None:
    """Size an image window: w x h clamped to the screen, then (by
    default) maximized. The image area follows every later resize or
    full screen -- see gui_image_canvas."""
    sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
    win.geometry(f"{min(w, sw)}x{min(h, sh - px(80))}")
    if not maximize:
        return
    try:
        win.state("zoomed")                   # Windows, macOS
    except tk.TclError:
        try:
            win.attributes("-zoomed", True)   # Linux / X11
        except tk.TclError:
            pass


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
        ax.set_aspect("equal", adjustable="datalim")
        _quiet_datalim_note()
    return fig, ax, canvas


def _quiet_datalim_note() -> None:
    """With adjustable="datalim" matplotlib logs "Ignoring fixed x
    limits to fulfill fixed data aspect..." on every zoom -- widening
    the view is exactly what is wanted, so drop that one message."""
    import logging
    lg = logging.getLogger("matplotlib.axes._base")
    if not any(getattr(f, "_datalim_note", False) for f in lg.filters):
        f = logging.Filter()
        f.filter = lambda rec: "fulfill fixed data aspect" not in rec.getMessage()
        f._datalim_note = True
        lg.addFilter(f)


def gui_attach_drag_pan(fig, ax, widget=None, when=None, button=1):
    """Dragging with `button` moves the view. Same pixel-delta method
    as the middle-drag pan in attach_zoom, so the image follows the
    mouse exactly instead of accelerating away: the rendered picture
    slides under the mouse (pan_preview) and the frame is rendered once
    on release. `when(ev)` can veto a drag (the box tool may share the
    button)."""
    drag = {"px": None}
    shift, reset = pan_preview(fig)

    def on_press(ev):
        if (ev.button == button and ev.inaxes is ax
                and (when is None or when(ev))):
            drag["px"] = (ev.x, ev.y, ax.get_xlim(), ax.get_ylim())
            if widget is not None:
                widget.config(cursor="fleur")

    def on_motion(ev):
        if drag["px"] is None or ev.x is None:
            return
        shift(ev.x - drag["px"][0], ev.y - drag["px"][1])

    def on_release(ev):
        if ev.button == button and drag["px"] is not None:
            reset()
            x0px, y0px, xl, yl = drag["px"]
            drag["px"] = None
            if ev.x is not None:
                sx = (xl[1] - xl[0]) / max(ax.bbox.width, 1)
                sy = (yl[1] - yl[0]) / max(ax.bbox.height, 1)
                dx = -(ev.x - x0px) * sx
                dy = -(ev.y - y0px) * sy
                ax.set_xlim(xl[0] + dx, xl[1] + dx)
                ax.set_ylim(yl[0] + dy, yl[1] + dy)
                fig.canvas.draw_idle()
            if widget is not None:
                widget.config(cursor="")

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)
    return drag


class GuiOverlay:
    """
    Instant shapes drawn on the Tk canvas ABOVE the matplotlib picture,
    for anything that follows the mouse: the rubber-band boxes and the
    pending circle. Nothing of matplotlib runs while they move -- a
    full figure draw costs 100-200 ms and even blitting needs one such
    frame at the start -- so the shape appears and follows the pointer
    at once; the real redraw happens once, when the gesture ends.
    Positions are data coordinates, converted through the axes
    transform, so update() can put them back after a zoom or pan.
    """

    def __init__(self, fig, ax):
        self.fig, self.ax = fig, ax
        self.tkc = getattr(fig.canvas, "_tkcanvas", None)
        self.items: dict = {}
        self.shapes: dict = {}
        if self.tkc is not None:
            fig.canvas.mpl_connect("draw_event", lambda _e: self.update())

    def _xy(self, x: float, y: float):
        """Data -> Tk canvas pixels (Tk's y runs down)."""
        px, py = self.ax.transData.transform((x, y))
        return float(px), float(self.fig.bbox.height - py)

    def rect(self, x0, y0, x1, y1, color: str, dash=None, fill: bool = True):
        self.shapes["rect"] = (x0, y0, x1, y1, color, dash, fill)
        self.update()

    def circle(self, cx, cy, r, color: str):
        self.shapes["circle"] = (cx, cy, r, color)
        self.update()

    def clear(self) -> None:
        self.shapes.clear()
        if self.tkc is not None:
            for ids in self.items.values():
                for i in ids:
                    self.tkc.delete(i)
        self.items.clear()

    def update(self) -> None:
        """(Re)draw every shape at its data position."""
        if self.tkc is None:
            return
        for key, shape in self.shapes.items():
            if key == "rect":
                x0, y0, x1, y1, color, dash, fill = shape
                a, b = self._xy(x0, y0), self._xy(x1, y1)
                coords = (min(a[0], b[0]), min(a[1], b[1]),
                          max(a[0], b[0]), max(a[1], b[1]))
                if key not in self.items:
                    kw = {"outline": color, "width": 1}
                    if dash:
                        kw["dash"] = dash
                    if fill:
                        kw.update(fill=color, stipple="gray25")
                    self.items[key] = [self.tkc.create_rectangle(*coords, **kw)]
                else:
                    self.tkc.coords(self.items[key][0], *coords)
            elif key == "circle":
                cx, cy, r, color = shape
                c = self._xy(cx, cy)
                e = self._xy(cx + r, cy)
                rp = abs(e[0] - c[0])
                oval = (c[0] - rp, c[1] - rp, c[0] + rp, c[1] + rp)
                h = (c[0] - 4, c[1], c[0] + 4, c[1])
                v = (c[0], c[1] - 4, c[0], c[1] + 4)
                if key not in self.items:
                    self.items[key] = [
                        self.tkc.create_oval(*oval, outline=color, width=1,
                                             dash=(2, 2)),
                        self.tkc.create_line(*h, fill=color, width=1),
                        self.tkc.create_line(*v, fill=color, width=1)]
                else:
                    o, lh, lv = self.items[key]
                    self.tkc.coords(o, *oval)
                    self.tkc.coords(lh, *h)
                    self.tkc.coords(lv, *v)
        for ids in self.items.values():       # above the picture, which a
            for i in ids:                     # window resize re-creates
                self.tkc.tag_raise(i)
        self.tkc.update_idletasks()


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
        self.overlay = GuiOverlay(fig, ax)  # the band while dragging
        fig.canvas.mpl_connect("button_press_event", self._press)
        fig.canvas.mpl_connect("motion_notify_event", self._motion)
        fig.canvas.mpl_connect("button_release_event", self._release)

    def active(self, ev) -> bool:
        return not bool(self.mode_var.get())

    def _press(self, ev):
        if ev.button != 1 or ev.inaxes is not self.ax or not self.active(ev):
            return
        self.start = (ev.xdata, ev.ydata)
        self.widget.config(cursor="crosshair")
        self.overlay.rect(ev.xdata, ev.ydata, ev.xdata, ev.ydata, "#1f77b4")

    def _motion(self, ev):
        if self.start is None or ev.xdata is None:
            return
        x0, y0 = self.start
        self.overlay.rect(x0, y0, ev.xdata, ev.ydata, "#1f77b4")

    def _release(self, ev):
        if self.start is None or ev.button != 1:
            return
        x0, y0 = self.start
        self.start = None
        self.widget.config(cursor="")
        self.overlay.clear()
        if ev.xdata is None or abs(ev.xdata - x0) < 8 or abs(ev.ydata - y0) < 8:
            return                         # a click, not a box
        self.box = (min(x0, ev.xdata), max(x0, ev.xdata),
                    min(y0, ev.ydata), max(y0, ev.ydata))
        # the box itself is a patch in data coordinates (it moves with
        # the view and stays until cleared): one full draw, after the
        # gesture
        self.rect.set_bounds(self.box[0], self.box[2],
                             self.box[1] - self.box[0], self.box[3] - self.box[2])
        self.rect.set_visible(True)
        self.fig.canvas.draw_idle()
        if self.on_change:
            self.on_change(self.box)

    def clear(self):
        self.box = None
        self.rect.set_visible(False)
        self.fig.canvas.draw_idle()
        if self.on_change:
            self.on_change(None)


GUI_PROGRESS = GUI_NAVY      # progress bars: fallback block colour
# the blocks run through viridis from left (yellow) to right (dark blue)
GUI_PROGRESS_VIRIDIS = ("#fde725", "#5ec962", "#21918c", "#3b528b", "#440154")


def gui_viridis(t: float) -> str:
    """Colour at position t (0-1) along the viridis ramp above."""
    stops = GUI_PROGRESS_VIRIDIS
    t = min(max(float(t), 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a = [int(stops[i][k:k + 2], 16) for k in (1, 3, 5)]
    b = [int(stops[i + 1][k:k + 2], 16) for k in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(round(x + (y - x) * f)) for x, y in zip(a, b))
GUI_PROGRESS_BLOCK = 10      # block width, px (2 px gap), like the Win95 control
GUI_PROGRESS_SHARE = 0.75    # the trough takes this much of the width, centred


def gui_run_in_thread(win, work, done) -> None:
    """
    Run `work()` -- pipeline code that touches no tk widget -- in a
    worker thread while the window keeps its event loop, so the
    progress bars can move during a long OpenCV or matplotlib call;
    then `done(result, error)` on the main thread. A SystemExit raised
    by the pipeline arrives as `error`.
    """
    import threading
    box: dict = {}

    def target():
        try:
            box["result"] = work()
        except BaseException as e:            # SystemExit included
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()

    def poll():
        if t.is_alive():
            win.after(100, poll)
            return
        done(box.get("result"), box.get("error"))

    win.after(100, poll)


class GuiProgress:
    """
    A progress bar in a window, fed by the console Bar through
    _BAR_HOOKS: it shows the bars whose title is in `titles` ("analyze",
    or "review" and "run") and empties itself when they finish.

    The fill follows TIME: each stage is weighted by how long it took
    the last time this command ran (SAVES/progress_times.json), the
    fill inside a stage advances with the clock, and the label gives
    the seconds left. The first run of a command weights the stages
    equally and shows no estimate. The bar is redrawn by its own timer
    (POLL_MS) on the main thread, so it keeps moving while the pipeline
    works in a thread (gui_run_in_thread); the hook itself only takes
    notes and may be called from any thread.
    """

    TIMES_NAME = "progress_times.json"      # in SAVES/
    POLL_MS = 100

    def __init__(self, parent, titles: tuple, height: int = 14):
        import time
        self.titles, self.height = titles, height
        # Windows 95 look: the stage with the time left above a sunken
        # trough that fills with navy blocks, and the folder being written
        # under it (drawn in _draw). `height` is the trough's.
        self.line_h = px(15)
        self.top = self.line_h + px(2)            # where the trough starts
        self.widget = tk.Canvas(parent, height=self.top + height + self.line_h,
                                bg=GUI_BG, highlightthickness=0)
        self.text = self.widget.create_text(0, self.line_h // 2, anchor="n",
                                            text="", font=F_SMALL, fill=GUI_TXT)
        self.where = self.widget.create_text(0, self.top + height + self.line_h // 2,
                                             anchor="n", text="", font=F_SMALL,
                                             fill="#3a3a3a")
        self.trough = self.widget.create_rectangle(0, 0, 0, 0,
                                                   fill=GUI_BOX, width=0)
        self.bevel = (self.widget.create_line(0, 0, 0, 0, fill=GUI_DARK),
                      self.widget.create_line(0, 0, 0, 0, fill=GUI_WHITE))
        self.blocks: list = []
        self.fill = None                 # kept for callers that read it
        self.widget.bind("<Configure>", lambda _e: self._draw(*self._last))
        self._last = (0.0, "", "")
        self._time = time
        self._run = None                 # the bar being followed, see _note
        _BAR_HOOKS.append(self.update)
        self.widget.bind("<Destroy>", self._gone, add="+")
        self.widget.after(self.POLL_MS, self._poll)

    def _gone(self, _e=None):
        if self.update in _BAR_HOOKS:
            _BAR_HOOKS.remove(self.update)

    @staticmethod
    def _load_times() -> dict:
        import json
        try:
            return json.loads((SAVES_DIR / GuiProgress.TIMES_NAME).read_text())
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _save_times(title: str, durations: list) -> None:
        import json
        data = GuiProgress._load_times()
        data[title] = [round(float(d), 3) for d in durations]
        try:
            gui_saves_dir()
            (SAVES_DIR / GuiProgress.TIMES_NAME).write_text(json.dumps(data, indent=1))
        except OSError:
            pass

    def update(self, bar) -> None:
        """The hook: note the bar and its stage starts. No tk here --
        the pipeline may call it from its worker thread."""
        if bar is not None and bar.title in self.titles:
            self._note(bar)

    def _note(self, bar) -> None:
        now = self._time.time()
        if self._run is None or self._run["bar"] is not bar:
            hist = [max(float(w), 0.01)
                    for w in (self._load_times().get(bar.title) or [])]
            if hist and len(hist) != bar.steps:
                mean = sum(hist) / len(hist)      # a command declares more
                hist = (hist + [mean] * bar.steps)[:bar.steps]   # steps than it reports
            self._run = {"bar": bar, "starts": [], "hist": hist}
        starts = self._run["starts"]
        while len(starts) < bar.i:            # a new stage began
            starts.append(now)

    def _fraction(self, bar) -> tuple:
        """(fill 0-1, seconds left or None) from the stage weights."""
        now = self._time.time()
        starts, hist = self._run["starts"], self._run["hist"]
        weights = hist or [1.0] * bar.steps
        total = float(sum(weights))
        if not bar.i:
            return 0.0, (total if hist else None)
        done = sum(weights[:bar.i - 1])
        w = weights[bar.i - 1]
        if bar.frac is not None:            # the stage reports its own progress
            inside = bar.frac * w
        else:                               # the clock, short of the stage's share
            inside = min(now - starts[bar.i - 1], 0.95 * w)
        frac = min(max((done + inside) / total, 0.0), 1.0)
        left = max(total - done - (now - starts[bar.i - 1]), 0.0) if hist else None
        return frac, left

    def _draw(self, frac: float, label: str, where: str = "") -> None:
        self._last = (frac, label, where)
        c, h, t = self.widget, self.height, self.top
        w = max(c.winfo_width(), 1)
        tw = int(w * GUI_PROGRESS_SHARE)          # the trough, centred
        x0, x1 = (w - tw) // 2, (w - tw) // 2 + tw
        c.coords(self.trough, x0, t, x1, t + h)
        c.coords(self.bevel[0], x1 - 1, t, x0, t, x0, t + h - 1)          # dark: top, left
        c.coords(self.bevel[1], x0, t + h - 1, x1 - 1, t + h - 1, x1 - 1, t)  # light: bottom, right
        # the blocks: as many as fit in the filled part
        step = GUI_PROGRESS_BLOCK + 2
        n_full = max(int((tw - 4) / step), 1)     # blocks in a full trough
        n = int((tw - 4) * min(max(frac, 0.0), 1.0) / step)
        while len(self.blocks) < n:
            i = len(self.blocks)
            self.blocks.append(c.create_rectangle(
                x0 + 2 + i * step, t + 3, x0 + 2 + i * step + GUI_PROGRESS_BLOCK,
                t + h - 3, fill=GUI_PROGRESS, width=0))
        for i, b in enumerate(self.blocks):      # follow a resize; viridis
            c.coords(b, x0 + 2 + i * step, t + 3,   # by position in the trough
                     x0 + 2 + i * step + GUI_PROGRESS_BLOCK, t + h - 3)
            c.itemconfig(b, fill=gui_viridis(i / max(n_full - 1, 1)))
        c.coords(self.text, w // 2, 1)
        c.coords(self.where, w // 2, t + h + 2)
        while len(self.blocks) > n:
            c.delete(self.blocks.pop())
        c.itemconfig(self.text, text=label)
        c.itemconfig(self.where, text=where)

    def _poll(self) -> None:
        try:
            if not self.widget.winfo_exists():
                self._gone()
                return
        except tk.TclError:
            self._gone()
            return
        bar = _BAR
        if bar is not None and bar.title in self.titles:
            self._note(bar)
            frac, left = self._fraction(bar)
            label = f"{bar.title}: {bar.label}  {frac * 100:3.0f}%"
            if left is not None:
                label += (f"   about {left:.0f} s left" if left >= 1
                          else "   finishing...")
            where = getattr(bar, "where", "")
            self._draw(frac, label, f"writing to  {where}" if where else "")
        elif self._run is not None:
            # finished: remember how long each stage took, then clear
            starts = self._run["starts"] + [self._time.time()]
            self._save_times(self._run["bar"].title,
                             [b - a for a, b in zip(starts, starts[1:])])
            self._run = None
            self._draw(0.0, "", "")
        self.widget.after(self.POLL_MS, self._poll)


GUI_PENDING = "#1f77b4"      # a drawn circle not yet accepted (blue)
GUI_ARROW_STEP_PX = 1        # one arrow press moves it one image pixel
GUI_SCROLL_STEP_PX = 1       # one Ctrl+scroll notch changes its radius this much
GUI_ARROW_PAN = 0.10         # fiducial window: one press pans 10 % of the view


class GuiCircleTool:
    """
    Manual selection: click the bead centre and a circle of the
    parameters window's bead size (default_r() pixels) appears there --
    a thin dotted blue outline with a cross at the centre, no fill, so
    the bead stays visible. While it is pending:
      * the arrow keys move it GUI_ARROW_STEP_PX image pixels per press;
      * press anywhere on the image and drag: the radius grows by the
        drag's component away from the centre and shrinks toward it
        (a global radius control -- no need to grab the outline);
      * Enter sets it: on_enter(x, y, diameter_px) -- by default on_set,
        which makes a new accepted bead, drawn like every other one;
      * Esc drops it.
    A fresh click starts a new circle only while `mode_var` (Manual
    selection) is on and `blocked()` (the zoom box) is not. begin() opens
    the same editing on an existing bead -- the double-click of Manual
    accept / resizing -- with its own on_enter; `editing` is then True
    whatever mode_var says.
    """

    def __init__(self, fig, ax, mode_var, on_set, status, um_per_px,
                 blocked=lambda: False, default_r=lambda: 10.0):
        from matplotlib.patches import Circle
        self.fig, self.ax, self.mode_var = fig, ax, mode_var
        self.on_set, self.status, self.um_per_px = on_set, status, um_per_px
        self.blocked, self.default_r = blocked, default_r
        self.on_enter = on_set
        self.centre = None                  # (x, y) while pending
        self.r = 0.0
        self.editing = False                # begin() on an existing bead
        self.drag = None                    # (press x, press y, r0, ux, uy)
        self.ring = Circle((0, 0), 1, fill=False, ec=GUI_PENDING, lw=0.8,
                           ls=":", visible=False)
        ax.add_patch(self.ring)
        self.cross = ax.plot([], [], "+", color=GUI_PENDING, ms=8, mew=0.8,
                             visible=False)[0]
        self.overlay = GuiOverlay(fig, ax)  # the pending circle, instant
        fig.canvas.mpl_connect("button_press_event", self._press)
        fig.canvas.mpl_connect("motion_notify_event", self._motion)
        fig.canvas.mpl_connect("button_release_event", self._release)
        fig.canvas.mpl_connect("key_press_event", self._key)
        fig.canvas.mpl_connect("scroll_event", self._scroll)

    @property
    def pending(self):
        return self.centre

    def _scroll(self, ev):
        """Ctrl+scroll: the pending circle grows or shrinks by
        GUI_SCROLL_STEP_PX per notch (plain scroll still zooms)."""
        if self.centre is None or getattr(ev, "key", None) != "control":
            return
        self.r = max(self.r + (GUI_SCROLL_STEP_PX if ev.button == "up"
                               else -GUI_SCROLL_STEP_PX), 1.0)
        self._show()

    def _show(self):
        # the ring and cross artists are kept for the no-Tk backends; on
        # Tk the overlay draws the circle instantly instead
        self.ring.center = self.centre
        self.ring.set_radius(max(self.r, 0.5))
        self.cross.set_data([self.centre[0]], [self.centre[1]])
        if self.overlay.tkc is not None:
            self.overlay.circle(self.centre[0], self.centre[1],
                                max(self.r, 0.5), GUI_PENDING)
        else:
            self.ring.set_visible(True)
            self.cross.set_visible(True)
            self.fig.canvas.draw_idle()
        d_um = 2 * self.r * self.um_per_px()
        self.status(f"circle at x {self.centre[0]:.0f}  y {self.centre[1]:.0f}"
                    f"   diameter {d_um:.0f} um ({2 * self.r:.0f} px)"
                    "   --   arrows move it, press-drag anywhere or "
                    "Ctrl+scroll resizes it, Enter sets it, Esc drops it")

    def drop(self, why: str = ""):
        self.centre, self.r, self.drag = None, 0.0, None
        self.editing, self.on_enter = False, self.on_set
        self.overlay.clear()
        if self.ring.get_visible():
            self.ring.set_visible(False)
            self.cross.set_visible(False)
            self.fig.canvas.draw_idle()
        if why:
            self.status(why)

    def begin(self, x: float, y: float, r: float, on_enter) -> None:
        """Edit an existing bead: its circle becomes the pending one."""
        self.centre, self.r, self.editing = (x, y), max(r, 0.5), True
        self.on_enter = on_enter
        self._show()

    def _press(self, ev):
        if (ev.button != 1 or ev.inaxes is not self.ax or ev.xdata is None
                or self.blocked()):
            return
        if self.centre is not None:
            # resize: the radius follows the drag along the line from
            # the centre through the press point (outward = bigger)
            ux, uy = ev.xdata - self.centre[0], ev.ydata - self.centre[1]
            n = math.hypot(ux, uy)
            ux, uy = (ux / n, uy / n) if n > 1e-9 else (1.0, 0.0)
            self.drag = (ev.xdata, ev.ydata, self.r, ux, uy)
            return
        if not self.mode_var.get():
            return
        self.centre, self.r = (ev.xdata, ev.ydata), float(self.default_r())
        self._show()

    def _motion(self, ev):
        if self.drag is None or ev.xdata is None:
            return
        x0, y0, r0, ux, uy = self.drag
        self.r = max(r0 + (ev.xdata - x0) * ux + (ev.ydata - y0) * uy, 1.0)
        self._show()

    def _release(self, ev):
        if ev.button == 1 and self.drag is not None:
            self.drag = None

    def _key(self, ev):
        if self.pending is None:
            return
        step = {"left": (-1, 0), "right": (1, 0),
                "up": (0, -1), "down": (0, 1)}.get(ev.key)
        if step:
            # image y runs downward, so up is -y
            self.centre = (self.centre[0] + step[0] * GUI_ARROW_STEP_PX,
                           self.centre[1] + step[1] * GUI_ARROW_STEP_PX)
            self._show()
        elif ev.key == "enter":
            (x, y), r, done = self.centre, self.r, self.on_enter
            self.drop()
            done(x, y, 2 * r)
        elif ev.key == "escape":
            self.drop("circle dropped")


class GuiZoomBox:
    """One-shot zoom box: while `var` is on, a left-drag draws a dashed
    rectangle and the view zooms to it (widened to the window's shape so
    the image is not stretched); then `var` turns itself off."""

    def __init__(self, fig, ax, var, on_done=None):
        from matplotlib.patches import Rectangle
        self.fig, self.ax, self.var, self.on_done = fig, ax, var, on_done
        self.start = None
        self.rect = Rectangle((0, 0), 0, 0, fill=False, ec="#000080",
                              lw=1.5, ls="--", visible=False)
        ax.add_patch(self.rect)
        self.overlay = GuiOverlay(fig, ax)  # the band while dragging
        fig.canvas.mpl_connect("button_press_event", self._press)
        fig.canvas.mpl_connect("motion_notify_event", self._motion)
        fig.canvas.mpl_connect("button_release_event", self._release)

    def _press(self, ev):
        if (ev.button != 1 or not self.var.get() or ev.inaxes is not self.ax
                or ev.xdata is None):
            return
        self.start = (ev.xdata, ev.ydata)
        self.overlay.rect(ev.xdata, ev.ydata, ev.xdata, ev.ydata, "#000080",
                          dash=(4, 2), fill=False)

    def _motion(self, ev):
        if self.start is None or ev.xdata is None:
            return
        x0, y0 = self.start
        self.overlay.rect(x0, y0, ev.xdata, ev.ydata, "#000080",
                          dash=(4, 2), fill=False)

    def _release(self, ev):
        if self.start is None or ev.button != 1:
            return
        x0, y0 = self.start
        self.start = None
        self.overlay.clear()
        if ev.xdata is not None and abs(ev.xdata - x0) >= 2 \
                and abs(ev.ydata - y0) >= 2:
            # keep the view's aspect: grow the shorter side of the box
            xl, yl = self.ax.get_xlim(), self.ax.get_ylim()
            aspect = abs(yl[1] - yl[0]) / max(abs(xl[1] - xl[0]), 1e-9)
            cx, cy = (x0 + ev.xdata) / 2, (y0 + ev.ydata) / 2
            w, h = abs(ev.xdata - x0), abs(ev.ydata - y0)
            if h / w > aspect:
                w = h / aspect
            else:
                h = w * aspect
            self.ax.set_xlim(*sorted((cx - w / 2, cx + w / 2),
                                     reverse=xl[0] > xl[1]))
            self.ax.set_ylim(*sorted((cy - h / 2, cy + h / 2),
                                     reverse=yl[0] > yl[1]))
        self.var.set(False)
        self.fig.canvas.draw_idle()
        if self.on_done:
            self.on_done()


def gui_arrow_pan(win, fig, ax, frac: float = GUI_ARROW_PAN) -> None:
    """Arrow keys pan the view by `frac` of its width / height, whatever
    has the keyboard focus -- except a text box, where the arrows keep
    moving the text cursor."""
    def pan(dx: float, dy: float):
        def handler(_e=None):
            if isinstance(win.focus_get(), (tk.Entry, tk.Spinbox, tk.Text)):
                return None
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            ddx, ddy = dx * abs(x1 - x0), dy * abs(y1 - y0)
            ax.set_xlim(x0 + ddx, x1 + ddx)
            ax.set_ylim(y0 + ddy, y1 + ddy)
            fig.canvas.draw_idle()
            return "break"
        return handler
    # image y runs downward: Up shows what is above, i.e. smaller y
    for key, dx, dy in (("Left", -frac, 0), ("Right", frac, 0),
                        ("Up", 0, -frac), ("Down", 0, frac)):
        win.bind(f"<{key}>", pan(dx, dy))


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


# ---- laser shot pattern pictures ----------------------------------------

# the patterns, in the order the "Laser shot pattern..." tiles show
# them: CONFIG value, tile name, one-line description
GUI_PATTERNS = [
    # row 1: one ring
    ("ring", "Ring", "one ring just off the edge (original)"),
    ("dense-ring", "Dense ring", "one ring, spots touching"),
    ("center", "Center", "one shot on the bead centre"),
    # row 2: two rings
    ("two-rings", "Two rings", "a second ring further out, same spacing rule"),
    ("two-rings-12", "Two rings, 12", "second ring fixed at 12 shots"),
    ("dense-two-rings", "Dense two rings", "both rings, spots touching"),
    # row 3: three rings
    ("triple-graded", "Triple: dense / medium / light",
     "ring 1 packed to the bead, ring 2 medium, ring 3 light"),
    ("triple-medium", "Triple: medium x3", "three rings at medium spacing"),
    ("triple-dense-inner", "Triple: dense / dense / light",
     "rings 1 and 2 packed, ring 3 light"),
]
GUI_TWO_RING = ("two-rings", "two-rings-12", "dense-two-rings")
GUI_THREE_RING = ("triple-graded", "triple-medium", "triple-dense-inner")
GUI_PATTERN_IN = 1.5        # side of a pattern tile in the dialog, inches (2.0 until 2026-10-19)
GUI_PATTERN_SMALL_IN = 1.0  # the picture in the parameters window


def gui_pattern_canvas(parent, size_in: float = GUI_PATTERN_IN):
    """A square matplotlib figure, `size_in` inches a side, embedded
    in tk. Returns (fig, ax, canvas, widget)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    fig = Figure(figsize=(size_in, size_in), dpi=int(round(96 * GUI_SCALE)),
                 facecolor=GUI_BOX)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set_axis_off()
    canvas = FigureCanvasTkAgg(fig, master=parent)
    w = canvas.get_tk_widget()
    w.config(bg=GUI_BOX, highlightthickness=0, relief="sunken", bd=2)
    return fig, ax, canvas, w


def gui_pattern_draw(ax, pattern: str, bead_um: float, cfg: dict) -> int:
    """One bead of `bead_um` and the shots `pattern` puts on it, placed
    by place_shots itself so the picture is what run() will fire. A
    shot the crater checks drop is drawn hollow. Returns the number of
    live shots."""
    from matplotlib.patches import Circle
    c = dict(cfg)
    c["shot-placement"] = dict(cfg["shot-placement"], **{"shot-pattern": pattern})
    bead = Bead(0.0, 0.0, 0.0, 0.0, 0.0, float(bead_um), accepted=True)
    shots = place_shots([bead], c)
    r = float(bead_um) / 2.0
    crater = footprint_um(c)
    ax.clear()
    ax.set_axis_off()
    ax.add_patch(Circle((0.0, 0.0), r, fc="#505050", ec="black", lw=1))
    reach = r
    for s in shots:
        reach = max(reach, math.hypot(s.x_um, s.y_um) + crater / 2.0)
        if s.dropped:
            ax.add_patch(Circle((s.x_um, s.y_um), crater / 2.0, fill=False,
                                ec="#d62728", ls=":", lw=1))
        else:
            ax.add_patch(Circle((s.x_um, s.y_um), crater / 2.0, fc="#d62728",
                                ec="none", alpha=0.85))
            ax.plot(s.x_um, s.y_um, "+", color="white", ms=4, mew=0.8)
    lim = reach * 1.08
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    return sum(not s.dropped for s in shots)


def gui_pattern_dialog(parent, v: dict, cfg: dict) -> None:
    """Tiles of every laser shot pattern for the bead size in the
    parameters window. Clicking a tile selects it and closes the box;
    the parameters window's own picture follows the variable."""
    dlg = tk.Toplevel(parent, bg=GUI_BG)
    dlg.title("Laser shot pattern")
    dlg.resizable(False, False)
    gui_title_strip(dlg, "Laser shot pattern -- click one")
    body = tk.Frame(dlg, bg=GUI_BG, padx=14, pady=10)
    body.pack(fill="both", expand=True)
    bead_um = gui_float(v["bead_um"], float(cfg["bead-diameter"]))
    current = v["shot_pattern"].get()

    def choose(key: str):
        v["shot_pattern"].set(key)
        dlg.destroy()

    for i, (key, name, desc) in enumerate(GUI_PATTERNS):
        chosen = key == current
        bg = GUI_NAVY if chosen else GUI_BG
        cell = tk.Frame(body, bg=bg, padx=4, pady=4)
        cell.grid(row=i // 3, column=i % 3, padx=6, pady=6)
        fig, ax, canvas, w = gui_pattern_canvas(cell)
        n = gui_pattern_draw(ax, key, bead_um, cfg)
        canvas.draw()
        w.pack()
        title = gui_label(cell, f"{name}  ({n} shots)", bg=bg,
                          fg=GUI_WHITE if chosen else GUI_TXT)
        title.pack(anchor="w", pady=(3, 0))
        note = gui_small(cell, desc, bg=bg,
                         fg=GUI_WHITE if chosen else "#3a3a3a")
        note.pack(anchor="w")
        for widget in (cell, w, title, note):
            widget.config(cursor="hand2")
            widget.bind("<Button-1>", lambda _e, k=key: choose(k), add="+")

    bf = tk.Frame(body, bg=GUI_BG)
    bf.grid(row=(len(GUI_PATTERNS) + 2) // 3, column=0, columnspan=3,
            sticky="e", pady=(8, 0))
    gui_small(bf, f"drawn for a {bead_um:g} um bead and a "
                  f"{footprint_um(cfg):g} um laser").pack(side="left", padx=(0, 12))
    gui_button(bf, "Cancel", dlg.destroy, width=8).pack(side="left")
    dlg.bind("<Escape>", lambda _e: dlg.destroy())
    dlg.transient(parent)
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
    parent.wait_window(dlg)


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
    "broad_stringency": "100",      # broad method only: 100 = as found, lower = stricter
    "laser_offset": "15",           # first ring offset from the bead edge, um; seeded from CONFIG
    "ring2_offset": "15",           # second ring offset from ring 1, um
    "ring3_offset": "15",           # third ring offset from ring 2, um
    "laser_um": "30",               # laser spot diameter, um; seeded from CONFIG's footprint
    "shot_pattern": "ring",         # seeded from CONFIG shot-placement
    "acq_method": "",               # .run acqMethod; seeded from CONFIG
    "run_directory": "",            # .run directory; seeded from CONFIG
    "sample_name": "",              # .run sampleName; seeded from CONFIG
}

# stored with the save, no effect on detection yet (marked * in the window)
GUI_NO_EFFECT = ("image_type", "microscope_slide", "microscope_zoom",
                 "matrix")

# strict method: fraction of ALL objects found that is kept as beads,
# best fit first (size closest to nominal, then most isolated)
GUI_STRICT_KEEP = 0.05

# bead isolation window: a bead's own dark footprint reaches
# FACTOR x its measured radius + MARGIN px from its centre; anything
# dark past that and inside the isolation window is a neighbour. A
# scanner blurs a bead into a halo about twice the radius the detector
# reports (89 lone beads on the 7 um/px scan: 95 % within 2.0 x, all
# within 2.4 x); the margin covers the rest and the specks.
GUI_BODY_FACTOR = 2.0
GUI_BODY_MARGIN_PX = 4

# default / threshold / strict: the sweep also reports LIGHT blobs
# (filterByColor is off so that inverted beads pass at all), and a
# glare spot the size of a bead goes through as one. An object must be
# at least this many grey levels darker than its surroundings, on
# average over its disc. Light spots measure 0.0; the faintest real
# bead seen so far 5.9.
GUI_MIN_CONTRAST = 2

# broad method: a lenient pass for faint beads (no matrix). A pixel is
# "something" when it differs from the flat-field background by this
# many noise standard deviations, darker OR lighter; the noise is
# measured on the box itself (robust MAD). 3 keeps the specks of pure
# noise apart while a bead's faint rim and cap still register.
GUI_BROAD_SIGNIF = 3.0
# ...and a blob is "remotely round" when its area fills at least this
# fraction of the smallest circle around it (1.0 a disc, 0.5 about a
# 2:1 ellipse, a line near 0). On the 261008 no-matrix scans, blobs
# below 0.55 were almost all pieces of the scanner streaks. Lowered to
# 0.50 on 2026-10-19: a pale bead whose cap and crescent have just been
# joined measures 0.53-0.60 (its brighter neighbours 0.56-0.79).
GUI_BROAD_FILL = 0.50
# ...and the stringency slider of the bead window (100 = everything the
# steps above found, as before). Below 100 a blob must also SPAN at
# least GUI_BROAD_SWING x (1 - stringency / 100) noise SDs from its
# darkest to its lightest pixel of the flat field -- the threshold-step
# idea: how many grey levels the blob covers. Measured on the 261008
# test3038 scan (x 17500-18130, y 5260-5615): the two beads span 47
# and 49 SD (dark core plus bright halo), the strongest speck 18, the
# faint ring 5. So 75 drops the faint ones, 50 and below keep the beads
# only.
GUI_BROAD_SWING = 40.0

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
    v["laser_offset"].set(f"{float(cfg['shot-placement'].get('edge-offset', 15)):g}")
    v["laser_um"].set(f"{footprint_um(cfg):g}")
    v["shot_pattern"].set(str(cfg["shot-placement"].get("shot-pattern", "ring")))
    run_hdr = cfg["output"].get("run") or {}
    v["acq_method"].set(str(run_hdr.get("acqMethod", "")))
    v["run_directory"].set(str(run_hdr.get("directory", "")))
    v["sample_name"].set(str(run_hdr.get("sampleName", "")))
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
    cfg["shot-placement"]["shot-pattern"] = v["shot_pattern"].get()
    sp = cfg["shot-placement"]
    sp["edge-offset"] = gui_float(v["laser_offset"], float(sp.get("edge-offset", 15)))
    sp["ring2-offset-um"] = gui_float(v["ring2_offset"],
                                      float(sp.get("ring2-offset-um", 30)))
    sp["ring3-offset-um"] = gui_float(v["ring3_offset"],
                                      float(sp.get("ring3-offset-um", 15)))
    # the laser diameter box IS the footprint: dense packing and the
    # drawn spots follow it (CONFIG's focal spot + beam scan otherwise)
    cfg["focal-spot-um"] = gui_float(v["laser_um"], footprint_um(cfg))
    cfg["beam-scan"] = False
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
        self.run_name = ""               # results name, asked before review / run
        self.busy = False                # a review / run / analyze is working
        self.T = None
        self.on_beads_changed = None     # the bead window hooks this
        self.filter = None               # trained filter dict (gui_filter_load)

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
    tol = float(cfg.get("manual-selection", {}).get("match-radius-px", 12))

    def add_new(found, extra):
        """Objects of `extra` not already in `found` (same position)."""
        if found and extra:
            tree = cKDTree(np.array([[f[0], f[1]] for f in found]))
            dd, _ = tree.query(np.array([[f[0], f[1]] for f in extra]))
            extra = [f for f, d_ in zip(extra, dd) if d_ > tol]
        return found + list(extra)

    if method == "default":
        found = gui_dark_only(g, d, _detect_blobdetector(g, d))   # sweep
    elif method == "quick":
        found = list(_detect_flatfield(g, d))
    elif method == "broad":
        found = gui_detect_broad(g, d)     # lenient; accepted by analyze
    elif method == "trained":
        found = gui_detect_trained(g, d, d["trained-filter"])
    else:
        # strict: both detectors, objects combined (an object seen by
        # both counts once); gui_refilter then keeps only the best few
        found = add_new(list(_detect_flatfield(g, d)),
                        gui_dark_only(g, d, _detect_blobdetector(g, d)))

    # Neighbour pass: clusters and debris wider than the size cap never
    # reach the object list (the threshold sweep does not report them
    # at all), so the isolation filter could not see them and a bead
    # beside a clump passed as isolated. Find them with the flat-field
    # detector at a wide cap and add them as clumps: never accepted,
    # purple in the window, but present for the isolation filter.
    wide = dict(d)
    wide["max-diameter-px"] = max(int(d.get("max-diameter-px", 30)) * 4, 120)
    big = [(x, y, size, True) for x, y, size, cl in _detect_flatfield(g, wide)
           if size > float(d.get("max-diameter-px", 30)) or cl]
    found = add_new(found, big)
    beads = [Bead(x + x0, y + y0, size, clumped=cl) for x, y, size, cl in found]

    # Bead isolation window (from the centre): the isolation filter only
    # knows object CENTRES, so a bead touching the edge of a cluster, or
    # a hair, passed whenever that object's centroid was far enough away.
    # Measure the clearance to the nearest dark pixel instead; gui_refilter
    # folds it into nn_um so the native filters reject it. Only the
    # window asks for it (clear-cap-px is set by analyze, not CONFIG).
    if d.get("clear-cap-px"):
        for b, (clear, clump) in zip(beads, gui_clearance(
                g, d, found, float(d["clear-cap-px"]))):
            b.clear_px, b.clear_clump = clear, clump
    return beads


def gui_dark_only(g, d: dict, found: list) -> list:
    """Keep the objects of the threshold sweep that are darker than
    their surroundings (see GUI_MIN_CONTRAST): mean of the flat-field
    image over the measured disc, the measure _detect_flatfield
    thresholds. Light blobs -- glare, bright specks -- drop out."""
    import cv2
    found = list(found)
    if not found:
        return found
    k = int(d.get("background-kernel-px", 101))
    if k % 2 == 0:
        k += 1
    sub = cv2.subtract(g, cv2.medianBlur(g, k))
    out = []
    for x, y, size, cl in found:
        r = max(int(round(size / 2)), 1)
        ix, iy = int(round(x)), int(round(y))
        wx, wy = max(ix - r, 0), max(iy - r, 0)
        win = sub[wy:iy + r + 1, wx:ix + r + 1]
        disc = np.zeros(win.shape, np.uint8)
        cv2.circle(disc, (ix - wx, iy - wy), r, 255, -1)
        if cv2.mean(win, disc)[0] >= GUI_MIN_CONTRAST:
            out.append((x, y, size, cl))
    return out


def gui_detect_broad(g, d: dict) -> list:
    """
    broad method: every faint, roughly round blob whose size is inside
    the size window (d["broad-diameter-px"] = (lo, hi) in pixels). For
    scans without matrix, where a bead is only a faint ring, a pale cap
    or a dark crescent a few grey levels off the background.

    1. flat field: subtract a large median, as _detect_flatfield does,
       smooth a little, and mark every pixel that differs from the
       background by GUI_BROAD_SIGNIF noise SDs -- darker or lighter;
    2. join one bead's light and dark parts (closing, half of the
       smallest bead -- a third left the pale beads' cap and crescent
       apart, 2026-10-19), fill holes so a ring counts whole, and drop
       everything thinner than about half the smallest bead (opening)
       -- scanner streaks, hairs, specks;
    3. keep each remaining blob that fills at least GUI_BROAD_FILL of
       its enclosing circle and whose equivalent diameter is inside a
       relaxed size window: the loose pre-filter every method uses
       (half the smallest bead to twice the largest, as in
       gui_size_limits_px), its upper end stretched again by
       GUI_BODY_FACTOR -- without matrix a bead shows as a dark core in
       a bright scanner halo and the blob is the whole of it. The
       diameter reported is the whole blob, halo included, so the shots
       land outside the halo. The kernels in step 2 follow the bead
       window itself.

    Returns (x, y, diameter_px, False) like the other detectors; the
    bead window marks these as accepted, to be checked by hand.
    """
    import cv2
    h, w = g.shape[:2]
    dmin, dmax = (float(v) for v in d["broad-diameter-px"])
    acc_lo = 0.5 * dmin                                  # relaxed window
    acc_hi = max(2.0 * dmax * GUI_BODY_FACTOR, acc_lo + 1)   # + the halo

    def odd(v):
        return max(3, 2 * int(round(v / 2)) + 1)

    k = min(max(2 * int(1.5 * acc_hi / 4) + 1, 3), 255)
    bg = cv2.resize(cv2.medianBlur(cv2.resize(
        g, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA),
        k), (w, h), interpolation=cv2.INTER_LINEAR)
    step, pad = 1024, 4

    def flat_strips():
        """The smoothed flat field, 1024 rows at a time, so a
        whole-slide box never holds a float copy of the scan."""
        for y0 in range(0, h, step):
            a, b = max(y0 - pad, 0), min(y0 + step + pad, h)
            f = cv2.GaussianBlur(g[a:b].astype(np.float32)
                                 - bg[a:b].astype(np.float32), (0, 0), 1.0)
            yield y0, f[y0 - a:y0 - a + min(step, h - y0)]

    # noise SD, robust (MAD), from every 4th pixel of the flat field
    smp = np.concatenate([f[::4, ::4].ravel() for _, f in flat_strips()])
    sd = max(float(np.median(np.abs(smp - np.median(smp)))) * 1.4826, 0.25)
    del smp
    m = np.zeros((h, w), np.uint8)
    for y0, f in flat_strips():
        m[y0:y0 + len(f)] = (np.abs(f) > GUI_BROAD_SIGNIF * sd) * np.uint8(255)
    # stringency slider: required dark-to-light swing, 0 at 100 %
    need = GUI_BROAD_SWING * (1.0 - float(d.get("broad-stringency", 100)) / 100.0)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (odd(dmin / 2),) * 2))
    flood = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(flood, None, (0, 0), 255)
    m |= cv2.bitwise_not(flood[1:-1, 1:-1])                 # holes
    del flood
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (odd(0.6 * dmin),) * 2))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    side = np.maximum(stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT])
    keep = np.nonzero((np.arange(n) > 0) & (side >= 0.5 * acc_lo)
                      & (side <= 2 * acc_hi))[0]
    out = []
    for c, i in enumerate(keep):
        bar_tick(c + 1, len(keep))
        bx, by = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        bw_, bh_ = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        x0, y0 = max(bx - 2, 0), max(by - 2, 0)
        comp = (lab[y0:by + bh_ + 2, x0:bx + bw_ + 2] == i).astype(np.uint8)
        cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            continue
        cnt = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        if area <= 0:
            continue
        dd = 2 * math.sqrt(area / math.pi)
        if not (acc_lo <= dd <= acc_hi):
            continue                                # outside the size window
        _, rad = cv2.minEnclosingCircle(cnt)
        if rad <= 0 or area / (math.pi * rad * rad) < GUI_BROAD_FILL:
            continue                                # not even roughly round
        if need > 0:
            # the flat field over this blob only, as flat_strips made it
            sl = (slice(y0, y0 + comp.shape[0]), slice(x0, x0 + comp.shape[1]))
            fl = cv2.GaussianBlur(g[sl].astype(np.float32)
                                  - bg[sl].astype(np.float32), (0, 0), 1.0)
            vals = fl[comp > 0]
            if (float(vals.max()) - float(vals.min())) / sd < need:
                continue                            # too faint for the slider
        mo = cv2.moments(cnt)
        out.append((mo["m10"] / mo["m00"] + x0, mo["m01"] / mo["m00"] + y0,
                    dd, False))
    del bg
    return out


# ---- trained filters (FilterTrainer/filter_trainer.py) ------------------
# A trained filter is a .json written by the trainer, loaded through the
# "upload filter template..." entry of the bead-matching dropdown. Two
# kinds:
#   "rerank"    the broad pass at 100 % finds the candidates; each one is
#               scored 0..1 by a logistic model fitted to the circles the
#               user drew (features below); the stringency slider keeps
#               the candidates scoring at least 1 - slider / 100, so 100
#               keeps everything the broad pass found, as it does for
#               broad, and 50 keeps the model's own decision (score >= 0.5)
#   "template"  the average flat-field patch of the circled beads;
#               normalised cross-correlation finds the beads, and the
#               slider keeps the peaks correlating at least 1 - slider / 100
# Both programs call these same functions, so the trainer sees exactly
# the candidates and features the bead window will.

GUI_FILTER_FORMAT = "microMS_beadtargeting blobfilter 1"

# the features of gui_blob_features, in order; a filter trained with a
# different list is refused
GUI_FEATURE_NAMES = ("size", "core", "rim", "halo", "surround", "contrast",
                     "swing", "spread", "fill", "edge", "symmetry", "round")
# ring radii of the features as multiples of the candidate's radius: the
# inner disc, the whole disc, the halo (GUI_BODY_FACTOR, the measured
# halo extent) and a surround one radius further out
GUI_FEATURE_INNER = 0.5
GUI_FEATURE_SURROUND = GUI_BODY_FACTOR + 1.0


def gui_broad_field(g, d: dict):
    """The broad method's background and noise SD, computed exactly as
    gui_detect_broad does (its lines are repeated here so that function
    stays untouched): (bg uint8 like g, sd float)."""
    import cv2
    h, w = g.shape[:2]
    dmin, dmax = (float(v) for v in d["broad-diameter-px"])
    acc_lo = 0.5 * dmin
    acc_hi = max(2.0 * dmax * GUI_BODY_FACTOR, acc_lo + 1)
    k = min(max(2 * int(1.5 * acc_hi / 4) + 1, 3), 255)
    bg = cv2.resize(cv2.medianBlur(cv2.resize(
        g, (max(w // 4, 1), max(h // 4, 1)), interpolation=cv2.INTER_AREA),
        k), (w, h), interpolation=cv2.INTER_LINEAR)
    step, pad = 1024, 4
    parts = []
    for y0 in range(0, h, step):
        a, b = max(y0 - pad, 0), min(y0 + step + pad, h)
        f = cv2.GaussianBlur(g[a:b].astype(np.float32)
                             - bg[a:b].astype(np.float32), (0, 0), 1.0)
        parts.append(f[y0 - a:y0 - a + min(step, h - y0)][::4, ::4].ravel())
    smp = np.concatenate(parts)
    sd = max(float(np.median(np.abs(smp - np.median(smp)))) * 1.4826, 0.25)
    return bg, sd


def gui_flat_window(g, bg, sd: float, x0: int, y0: int, x1: int, y1: int):
    """The flat field of g[y0:y1, x0:x1] in noise-SD units, smoothed as
    gui_detect_broad smooths it."""
    import cv2
    f = cv2.GaussianBlur(g[y0:y1, x0:x1].astype(np.float32)
                         - bg[y0:y1, x0:x1].astype(np.float32), (0, 0), 1.0)
    return f / sd


def gui_blob_features(g, d: dict, cands: list, field=None) -> np.ndarray:
    """
    One row of GUI_FEATURE_NAMES per candidate (x, y, diameter_px, ...),
    measured on the flat field in noise-SD units (negative = darker than
    the background) over rings around the candidate's centre, radius r:
      size      diameter over the middle of the broad size window
      core      mean inside GUI_FEATURE_INNER x r
      rim       mean between the inner disc and r
      halo      mean between r and GUI_BODY_FACTOR x r
      surround  mean between the halo and GUI_FEATURE_SURROUND x r
      contrast  mean of the disc minus the surround
      swing     max minus min inside the disc (the stringency measure)
      spread    standard deviation inside the disc
      fill      fraction of the disc beyond GUI_BROAD_SIGNIF (something)
      edge      mean gradient magnitude on the rim ring
      symmetry  |left - right| + |top - bottom| of the disc means
      round     area / enclosing circle of the thresholded blob at the
                centre (GUI_BROAD_FILL's measure), 0 when there is none
    `field` = (bg, sd) from gui_broad_field, computed here when None.
    """
    import cv2
    if not cands:
        return np.zeros((0, len(GUI_FEATURE_NAMES)), np.float64)
    bg, sd = field if field is not None else gui_broad_field(g, d)
    h, w = g.shape[:2]
    dmin, dmax = (float(v) for v in d["broad-diameter-px"])
    nominal = max(0.5 * (dmin + dmax), 1.0)
    rows = []
    for c in cands:
        x, y, dia = float(c[0]), float(c[1]), float(c[2])
        r = max(dia / 2.0, 1.0)
        R = int(math.ceil(GUI_FEATURE_SURROUND * r)) + 2
        x0, y0 = max(int(round(x)) - R, 0), max(int(round(y)) - R, 0)
        x1, y1 = min(int(round(x)) + R + 1, w), min(int(round(y)) + R + 1, h)
        f = gui_flat_window(g, bg, sd, x0, y0, x1, y1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dx, dy = xx - x, yy - y
        rho = np.hypot(dx, dy)
        inner = rho <= GUI_FEATURE_INNER * r
        disc = rho <= r
        rim = disc & ~inner
        halo = (rho > r) & (rho <= GUI_BODY_FACTOR * r)
        surround = (rho > GUI_BODY_FACTOR * r) & (rho <= GUI_FEATURE_SURROUND * r)

        def mean(mask):
            return float(f[mask].mean()) if mask.any() else 0.0

        fd = f[disc] if disc.any() else np.zeros(1, np.float32)
        gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        sym = (abs(mean(disc & (dx < 0)) - mean(disc & (dx >= 0)))
               + abs(mean(disc & (dy < 0)) - mean(disc & (dy >= 0))))
        # roundness of the "something" blob under the centre
        m = (np.abs(f) > GUI_BROAD_SIGNIF).astype(np.uint8)
        rnd = 0.0
        if m.any():
            n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            cy, cx = int(round(y)) - y0, int(round(x)) - x0
            i = int(lab[min(max(cy, 0), m.shape[0] - 1),
                        min(max(cx, 0), m.shape[1] - 1)])
            if i == 0 and disc.any():
                # centre on background: the largest component in the disc
                ids, cnt = np.unique(lab[disc], return_counts=True)
                keep = ids > 0
                i = int(ids[keep][np.argmax(cnt[keep])]) if keep.any() else 0
            if i > 0:
                comp = (lab == i).astype(np.uint8)
                cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_NONE)
                if cs:
                    cnt_ = max(cs, key=cv2.contourArea)
                    area = cv2.contourArea(cnt_)
                    _, rad = cv2.minEnclosingCircle(cnt_)
                    if rad > 0:
                        rnd = float(area / (math.pi * rad * rad))
        rows.append([dia / nominal, mean(inner), mean(rim), mean(halo),
                     mean(surround), mean(disc) - mean(surround),
                     float(fd.max() - fd.min()), float(fd.std()),
                     float((np.abs(fd) > GUI_BROAD_SIGNIF).mean()),
                     float(grad[rim].mean()) if rim.any() else 0.0,
                     sym, rnd])
    return np.array(rows, np.float64)


def gui_filter_load(path) -> dict:
    """Read a trained filter; raises ValueError when it is not one."""
    import json
    with open(path, encoding="utf-8") as fh:
        filt = json.load(fh)
    if not isinstance(filt, dict) or filt.get("format") != GUI_FILTER_FORMAT:
        raise ValueError("not a microMS_beadtargeting filter file")
    if filt.get("kind") == "rerank":
        if tuple(filt.get("features", ())) != GUI_FEATURE_NAMES:
            raise ValueError("this filter was trained on a different "
                             "feature set; train it again")
        for k in ("mu", "sd", "w", "b"):
            if k not in filt:
                raise ValueError(f"filter is missing {k!r}")
    elif filt.get("kind") == "template":
        for k in ("template", "bead_px", "um_per_px"):
            if k not in filt:
                raise ValueError(f"filter is missing {k!r}")
    else:
        raise ValueError(f"unknown filter kind {filt.get('kind')!r}")
    filt["path"] = str(path)
    return filt


def gui_filter_score(filt: dict, X: np.ndarray) -> np.ndarray:
    """Logistic score 0..1 of each feature row under a rerank filter."""
    if len(X) == 0:
        return np.zeros(0, np.float64)
    mu, sd = np.array(filt["mu"], np.float64), np.array(filt["sd"], np.float64)
    w, b = np.array(filt["w"], np.float64), float(filt["b"])
    z = ((X - mu) / sd) @ w + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def gui_filter_cutoff(d: dict) -> float:
    """The slider as a score / correlation cutoff: 100 keeps all."""
    p = min(max(float(d.get("broad-stringency", 100)), 1.0), 100.0)
    return 1.0 - p / 100.0


def gui_detect_template(g, d: dict, filt: dict, all_peaks: bool = False) -> list:
    """
    template kind: normalised cross-correlation of the trained patch
    against the flat field, in strips like gui_detect_broad. A peak is a
    local maximum within one bead diameter that correlates at least the
    slider cutoff (every positive peak when all_peaks). The template is
    rescaled when this scan's um/px differs from the training scan's.
    Returns (x, y, diameter_px, False[, corr]) -- corr only with all_peaks.
    """
    import cv2
    h, w = g.shape[:2]
    t = np.array(filt["template"], np.float32)
    scale = float(filt["um_per_px"]) / float(d.get("um-per-px", filt["um_per_px"]))
    side = max(3, 2 * int(round(t.shape[0] * scale / 2)) + 1)
    if side != t.shape[0]:
        t = cv2.resize(t, (side, side), interpolation=cv2.INTER_LINEAR)
    bead_px = float(filt["bead_px"]) * scale
    if h < side or w < side:
        return []
    bg, sd = gui_broad_field(g, d)
    cut = 0.0 if all_peaks else gui_filter_cutoff(d)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                  (max(3, int(bead_px) | 1),) * 2)
    step, pad = 1024, side
    out = []
    for y0 in range(0, h, step):
        a, b = max(y0 - pad, 0), min(y0 + step + pad, h)
        if b - a < side:
            continue
        f = gui_flat_window(g, bg, sd, 0, a, w, b)
        cc = cv2.matchTemplate(f, t, cv2.TM_CCOEFF_NORMED)
        peak = (cc >= cv2.dilate(cc, k)) & (cc > max(cut, 1e-6))
        ys, xs = np.nonzero(peak)
        for yy, xx in zip(ys, xs):
            cy = yy + side // 2 + a
            if not (y0 <= cy < min(y0 + step, h)):
                continue                       # belongs to another strip
            cx = xx + side // 2
            rec = (float(cx), float(cy), bead_px, False)
            out.append(rec + (float(cc[yy, xx]),) if all_peaks else rec)
    del bg
    return out


def gui_detect_trained(g, d: dict, filt: dict) -> list:
    """The trained branch of gui_detect_region: see the note above."""
    if filt["kind"] == "template":
        return gui_detect_template(g, d, filt)
    cands = gui_detect_broad(g, {**d, "broad-stringency": 100})
    X = gui_blob_features(g, d, cands)
    s = gui_filter_score(filt, X)
    cut = gui_filter_cutoff(d)
    return [c for c, sc in zip(cands, s) if sc >= cut]


def gui_window_px(state, cfg: dict) -> tuple:
    """The size window of the parameters window -- average bead size
    +/- deviation -- in pixels at the current um/px."""
    um_px = gui_transform(state, cfg).um_per_px
    nominal = float(cfg["bead-diameter"])
    dev = nominal * float(cfg.get("bead-diameter-tolerance", 0.35))
    return max(nominal - dev, 0.0) / um_px, (nominal + dev) / um_px


def gui_clearance(g, d: dict, found: list, cap_px: float) -> list:
    """
    Distance, in pixels, from each object's centre to the nearest dark
    pixel that is not the object's own body. Returns one
    (clear_px, foreign_is_clump) per object; clumps get (inf, False)
    because they are never accepted anyway.

    "Dark" is the flat-field mask of _detect_flatfield, same recipe and
    same thresholds, so hairs and clusters of any size count. The own
    body is the mask component under the centre, out to GUI_BODY_FACTOR
    measured radii + GUI_BODY_MARGIN_PX (the scanner's halo). Beyond
    that, or in another component, every dark pixel is foreign -- a
    touching neighbour is part of the same component but lies outside
    the body. The search stops at cap_px: farther than that is inf,
    i.e. clear.
    """
    import cv2
    k = int(d.get("background-kernel-px", 101))
    if k % 2 == 0:
        k += 1
    sub = cv2.subtract(g, cv2.medianBlur(g, k))
    _, bw = cv2.threshold(sub, int(d.get("threshold", 6)), 255,
                          cv2.THRESH_BINARY)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, lab, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    # Specks below the detectors' own size floor are background.
    a_min = math.pi * (float(d.get("min-diameter-px", 3)) / 2) ** 2
    area = stats[:, cv2.CC_STAT_AREA]
    drop = area < a_min
    drop[0] = True
    lab[drop[lab]] = 0
    h, w = lab.shape
    cap = int(math.ceil(cap_px))

    out = []
    for x, y, size, cl in found:
        ix, iy = int(round(x)), int(round(y))
        if cl or not (0 <= ix < w and 0 <= iy < h):
            out.append((float("inf"), False))
            continue
        body_px = GUI_BODY_FACTOR * 0.5 * size + GUI_BODY_MARGIN_PX
        one_bead = math.pi * body_px ** 2      # larger than this = clump
        wx0, wy0 = max(ix - cap, 0), max(iy - cap, 0)
        win = lab[wy0:min(iy + cap + 1, h), wx0:min(ix + cap + 1, w)]
        ys, xs = np.nonzero(win)
        if len(xs) == 0:
            out.append((float("inf"), False))
            continue
        dist = np.hypot(xs + wx0 - x, ys + wy0 - y)
        labs = win[ys, xs]
        own = int(lab[iy, ix])
        if own == 0:
            # centre not dark (a pale or ring-shaped bead): take the
            # nearest dark pixel inside the measured radius as the body
            near = dist <= 0.5 * size
            if near.any():
                own = int(labs[near][np.argmin(dist[near])])
        foreign = (labs != own) | (dist > body_px)
        if not foreign.any():
            out.append((float("inf"), False))
            continue
        j = int(np.argmin(np.where(foreign, dist, np.inf)))
        out.append((float(dist[j]),
                    bool(labs[j] == own or area[labs[j]] > one_bead)))
    return out


def gui_merge(state: GuiState, new: list, cfg: dict) -> int:
    """Add objects not already known (boxes may overlap). A clump that
    an earlier pass left where this pass found a bead gives way to the
    bead: the neighbour sweep marks whatever a stricter slider rejected
    as a clump, and without this a looser slider could never bring the
    bead back on the same box. Everything added is appended, so the
    last `n` objects are the new ones (n returned)."""
    tol = float(cfg.get("manual-selection", {}).get("match-radius-px", 12))
    if state.beads and new:
        tree = cKDTree(np.array([[b.x_px, b.y_px] for b in state.beads]))
        d, j = tree.query(np.array([[b.x_px, b.y_px] for b in new]))
        keep, swapped = [], []
        for b, dd, jj in zip(new, d, j):
            if dd > tol:
                keep.append(b)
            elif (state.beads[jj].clumped and not b.clumped
                  and not state.beads[jj].manual and jj not in swapped):
                swapped.append(int(jj))            # the clump gives way
                keep.append(b)
        for jj in sorted(swapped, reverse=True):
            state.beads.pop(jj)
        new = keep
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
    for b in beads:
        # nearest dark pixel (gui_clearance), not just nearest centre
        if hasattr(b, "clear_px"):
            b.nn_um = min(b.nn_um, b.clear_px * T.um_per_px)
    shape_filter(beads, cfg)
    gui_mark_near_clump(beads, float(cfg["min-bead-separation"]))
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


def gui_mark_near_clump(beads: list, min_sep_um: float) -> None:
    """Window only: a bead that failed isolation with a clump inside
    the isolation window is drawn purple with the clumps; one failed
    by anything else (a contaminant, another bead) stays red. The
    filters are untouched -- this reads their result and looks the
    neighbours up again."""
    for b in beads:
        b.near_clump = False
    if len(beads) < 2 or min_sep_um <= 0:
        return
    pts = np.array([[b.x_um, b.y_um] for b in beads], float)
    tree = cKDTree(pts)
    for i, b in enumerate(beads):
        if b.reject_category == "not isolated":
            b.near_clump = getattr(b, "clear_clump", False) or any(
                beads[j].clumped
                for j in tree.query_ball_point(pts[i], min_sep_um)
                if j != i)


def gui_bead_category(b) -> str:
    if b.accepted:
        return "accepted"
    if b.clumped or getattr(b, "near_clump", False):
        return "clumped"
    return "rejected"


def gui_bead_colour(b) -> str:
    if b.accepted:
        return GUI_COL["accepted"]
    if b.clumped or getattr(b, "near_clump", False):
        return GUI_COL["clumped"]
    if b.reject_category == "manual":
        return GUI_COL["manual"]
    return GUI_COL["rejected"]


# ---- saves ----------------------------------------------------------------

def gui_beads_to_json(beads: list) -> list:
    rows = [{"x_px": round(b.x_px, 2), "y_px": round(b.y_px, 2),
             "diameter_px": round(b.diameter_px, 2),
             "clumped": bool(b.clumped), "manual": b.manual,
             "source": b.source}
            for b in beads]
    for b, row in zip(beads, rows):
        if hasattr(b, "clear_px") and math.isfinite(b.clear_px):
            row["clear_px"] = round(b.clear_px, 2)
            row["clear_clump"] = bool(b.clear_clump)
    return rows


def gui_beads_from_json(rows: list) -> list:
    out = []
    for r in rows:
        b = Bead(float(r["x_px"]), float(r["y_px"]), float(r["diameter_px"]),
                 clumped=bool(r.get("clumped", False)))
        b.manual = str(r.get("manual", "") or "")
        b.source = str(r.get("source", "") or "")
        if r.get("clear_px") is not None:
            b.clear_px = float(r["clear_px"])
            b.clear_clump = bool(r.get("clear_clump", False))
        out.append(b)
    return out


def gui_clean_name(name: str) -> str:
    """A typed name as a file-name piece: letters, digits, - and _."""
    return "".join(c if c.isalnum() or c in "-_ " else "_"
                   for c in name).strip().replace(" ", "_")


def gui_ask_run_name(state: GuiState, win) -> bool:
    """Before Save and review / Save and run: the name the results
    folder (<name>_<stamp> review) and the save file carry, with the
    last one offered again. False if the box was cancelled."""
    name = simpledialog.askstring(
        "Results name", "Name for these results (optional):",
        parent=win, initialvalue=state.run_name)
    if name is None:
        return False
    state.run_name = gui_clean_name(name)
    return True


def gui_save(state: GuiState, win, ask_name: bool = False):
    """Write SAVES/<name>_<timestamp>.json (the results name) -- the
    same file again while the name is unchanged. Returns the path or
    None."""
    import json
    if ask_name:
        if not gui_ask_run_name(state, win):
            return None
    wanted = (state.run_name + "_" if state.run_name else "")
    if state.save_path is None or ask_name or \
            not state.save_path.stem.startswith(wanted) or \
            (not state.run_name and "_" in state.save_path.stem[17:]):
        stem = wanted + timestamp()
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
        w.writerow(["x_px", "y_px", "diameter_px", "clumped", "source"])
        for b in state.beads:
            w.writerow([f"{b.x_px:.2f}", f"{b.y_px:.2f}",
                        f"{b.diameter_px:.2f}", int(b.clumped), b.source])
    save_manual([(b.x_px, b.y_px, "accept" if b.accepted else "reject")
                 for b in state.beads])
    cfg["input"]["beads"] = str(csv_path)
    return csv_path


def gui_size_limits_px(state: GuiState, cfg: dict) -> dict:
    """Tie the detectors' pixel pre-filter to the size window in the
    parameters window: half of (size - deviation) up to double
    (size + deviation), converted with the current um/px. The margin
    keeps debris near the beads visible to the isolation filter."""
    T = gui_transform(state, cfg)
    nominal = float(cfg["bead-diameter"])
    dev = nominal * float(cfg.get("bead-diameter-tolerance", 0.35))
    lo_um, hi_um = max(nominal - dev, 1.0) * 0.5, (nominal + dev) * 2.0
    cfg["detection"]["min-diameter-px"] = max(2, int(lo_um / T.um_per_px))
    cfg["detection"]["max-diameter-px"] = max(
        cfg["detection"]["min-diameter-px"] + 2,
        int(round(hi_um / T.um_per_px)))
    return cfg


def gui_run_cfg(state: GuiState, fids: list) -> dict:
    import copy
    cfg = gui_size_limits_px(state,
                             gui_apply_params(copy.deepcopy(state.cfg), state.v))
    cfg["fiducials"] = [dict(f) for f in fids]
    run_hdr = cfg["output"].setdefault("run", {})
    if state.v["acq_method"].get().strip():
        run_hdr["acqMethod"] = state.v["acq_method"].get().strip()
    if state.v["run_directory"].get().strip():
        run_hdr["directory"] = state.v["run_directory"].get().strip()
    if state.v["sample_name"].get().strip():
        run_hdr["sampleName"] = state.v["sample_name"].get().strip()
    if state.path is not None:
        cfg["input"]["scan"] = str(state.path)
    cfg["output"]["review-show"] = False
    cfg["output"]["run-name"] = state.run_name
    if state.beads:
        gui_refilter(state)
        gui_write_run_inputs(state, cfg)
    return cfg


def gui_review_into(state: GuiState, layer, status, master=None) -> None:
    """review() on the window's objects, in a worker thread so the
    window and its progress bar stay alive; then the planned shots are
    drawn on the canvas and the review pictures open in a pop-up."""
    if state.busy:
        status("busy -- wait for the current step to finish")
        return
    fids, real = gui_fit_fiducials(state)
    cfg = gui_run_cfg(state, fids)

    def done(result, err):
        state.busy = False
        if err is not None:
            bar_done()
            status(f"review stopped: {err}")
            return
        beads, shots, T = result
        live = sum(not s.dropped for s in shots)
        layer.show_shots(beads, shots, T, cfg)
        status(f"review: {live} shots on {sum(b.accepted for b in beads)} beads"
               + ("" if real else "  (provisional scale, no fiducials yet)")
               + "  -- pictures and check.txt in RESULTS")
        # the folder review() just wrote: the newest "... review"
        folders = sorted((HERE / cfg["output"].get("results-dir", "RESULTS")).glob(
            "*_review"), key=lambda p: p.stat().st_mtime)
        if folders and master is not None:
            gui_review_window(master, folders[-1])

    state.busy = True
    status("review running...")
    if master is None:                      # no window to keep alive
        try:
            done(review(cfg), None)
        except SystemExit as e:
            done(None, e)
        return
    gui_run_in_thread(master, lambda: review(cfg), done)


def gui_open_folder(folder: Path) -> None:
    """Show the folder in the file manager (Explorer on Windows)."""
    import os
    import platform
    import subprocess
    try:
        if hasattr(os, "startfile"):                 # Windows
            os.startfile(str(folder))
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(folder)])
        else:
            subprocess.Popen(["xdg-open", str(folder)])
    except OSError as e:
        say(f"could not open {folder}: {e}")


def gui_run_done(parent, folder) -> None:
    """'Run done' pop-up, then the results folder is opened."""
    dlg = tk.Toplevel(parent, bg=GUI_BG)
    dlg.title("Run done")
    dlg.resizable(False, False)
    gui_title_strip(dlg, "Run done")
    body = tk.Frame(dlg, bg=GUI_BG, padx=16, pady=12)
    body.pack(fill="both", expand=True)
    gui_label(body, "Run done.", font=F_BIG).pack(anchor="w")
    gui_label(body, "Target files (.xeo, .run, flexImaging txt, csv, "
                    "pictures) are in:", wraplength=px(460),
              justify="left").pack(anchor="w", pady=(4, 0))
    gui_small(body, str(folder) if folder else "RESULTS",
              wraplength=px(460), justify="left").pack(anchor="w")
    gui_button(body, "OK", dlg.destroy, width=8).pack(anchor="e",
                                                       pady=(12, 0))
    dlg.bind("<Return>", lambda _e: dlg.destroy())
    dlg.transient(parent)
    if folder is not None:
        gui_open_folder(folder)


def gui_review_window(master, folder: Path):
    """Pop-up with the review pictures: review.png (accepted beads and
    their shots) and review_zoom.png (close-up, every category), one at
    a time, with the usual zoom and drag."""
    import cv2
    pics = [p for name in ("review.png", "review_zoom.png", "review_distances.png")
            for p in sorted(folder.glob(f"*{name}")) if p.name.endswith(name)]
    if not pics:
        return None
    win = tk.Toplevel(master, bg=GUI_BG)
    win.title(f"microMS_beadtargeting -- review  {folder.name}")
    gui_fill_screen(win, px(1100), px(760), maximize=False)
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

    # -- type of image + drop-down image settings -----------------------
    panel_open = [False]

    def set_panel(open_: bool):
        panel_open[0] = open_
        if open_:
            panel.grid()
            bar_btn.config(text="^   Image Settings   ^")
        else:
            panel.grid_remove()
            bar_btn.config(text="v   Image Settings   v")

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
    prow1 = tk.Frame(panel, bg=GUI_LAVENDER)
    prow1.pack(anchor="w")
    gui_label(prow1, "Type of image *", bg=GUI_LAVENDER).pack(side="left")
    gui_dropdown(prow1, v["image_type"],
                 ["highres scanner", "beta"]).pack(side="left", padx=(6, 0))
    prow2 = tk.Frame(panel, bg=GUI_LAVENDER)
    prow2.pack(anchor="w", pady=(4, 0))
    gui_label(prow2, "scale", bg=GUI_LAVENDER).pack(side="left")
    gui_box(prow2, v["um_per_px"], width=6).pack(side="left", padx=(6, 4))
    gui_small(prow2, "um/px  -- used for the isolation filter until "
                     "fiducials are picked", bg=GUI_LAVENDER).pack(
                         side="left")
    set_panel(False)

    # -- bead size -------------------------------------------------------
    bead_row = row[0]
    f = line("Average bead size + deviation")
    gui_box(f, v["bead_um"], 5).pack(side="left")
    gui_small(f, "um").pack(side="left", padx=(3, 10))
    gui_small(f, "+/-").pack(side="left", padx=(0, 3))
    gui_box(f, v["bead_dev_um"], 4).pack(side="left")
    gui_small(f, "um").pack(side="left", padx=(3, 0))

    # -- isolation window ------------------------------------------------
    f = line(widget=gui_check(body, v["isolation_on"],
                              "Bead isolation window (from the center)"))
    gui_box(f, v["isolation_um"], 5).pack(side="left")
    gui_small(f, "um").pack(side="left", padx=(3, 0))

    # -- max points ------------------------------------------------------
    f = line("Max number of points")
    gui_box(f, v["max_points"], 6).pack(side="left")

    # -- laser shot pattern ----------------------------------------------
    # right of the bead size / isolation / max points rows
    pf = tk.Frame(body, bg=GUI_BG)
    # an empty 4th row takes the picture's extra height, so the three
    # rows keep their normal spacing
    body.rowconfigure(row[0], weight=1)
    row[0] += 1
    pf.grid(row=bead_row, column=1, rowspan=4, sticky="ne", pady=4)
    gui_label(pf, "Laser shot pattern").pack(anchor="w", pady=(0, 2))
    p_fig, p_ax, p_canvas, p_w = gui_pattern_canvas(pf, GUI_PATTERN_SMALL_IN)
    p_w.pack(side="left", anchor="w")
    pinfo = tk.Frame(pf, bg=GUI_BG)
    pinfo.pack(side="left", anchor="w", padx=(10, 0))
    p_name = gui_small(pinfo, "")
    p_name.pack(anchor="w", pady=(2, 0))
    def cfg_now() -> dict:
        """CONFIG with the window's values (laser offsets) applied, so
        the pattern pictures show what run() will fire."""
        import copy
        return gui_apply_params(copy.deepcopy(state.cfg), v)

    gui_button(pinfo, "Laser shot pattern...",
               lambda: gui_pattern_dialog(win, v, cfg_now())).pack(
                   anchor="w", pady=(4, 0))
    pattern_names = {k: name for k, name, _ in GUI_PATTERNS}

    def redraw_pattern(*_a):
        key = v["shot_pattern"].get()
        if key not in pattern_names:
            key = "ring"
        n = gui_pattern_draw(p_ax, key,
                             gui_float(v["bead_um"],
                                       float(state.cfg["bead-diameter"])),
                             cfg_now())
        p_canvas.draw_idle()
        p_name.config(text=f"{pattern_names[key]}: {n} shots per bead")

    v["shot_pattern"].trace_add("write", redraw_pattern)
    v["bead_um"].trace_add("write", redraw_pattern)
    redraw_pattern()

    # -- laser diameter: dense packing and the drawn spots follow it ------
    f = line("Laser diameter")
    gui_box(f, v["laser_um"], 5).pack(side="left")
    gui_small(f, "um  -- spot size; dense rings pack spots this far apart").pack(
        side="left", padx=(3, 0))

    # -- laser offset: first ring from the bead edge ------------------------
    f = line("Laser offset")
    gui_box(f, v["laser_offset"], 5).pack(side="left")
    gui_small(f, "um  -- first ring offset from the bead edge").pack(
        side="left", padx=(3, 0))

    # -- second and third ring offsets, shown for the patterns that have
    # them. Each row has two linked boxes: the offset from the ring inside
    # it, and "from bead center" = laser offset + every ring offset up to
    # it (15 / 15 -> 30 / 15 -> 45). Editing any box updates the others.
    def ring_row(label: str, inner: str, var, total):
        r = row[0]
        lbl = gui_label(body, label)
        lbl.grid(row=r, column=0, sticky="w", pady=4)
        frame = tk.Frame(body, bg=GUI_BG)
        frame.grid(row=r, column=1, sticky="w", padx=(18, 0), pady=4)
        row[0] += 1
        gui_small(frame, f"from {inner}").pack(side="left")
        gui_box(frame, var, 5).pack(side="left", padx=(3, 3))
        gui_small(frame, "um").pack(side="left", padx=(0, 10))
        gui_small(frame, "from bead center").pack(side="left")
        gui_box(frame, total, 5).pack(side="left", padx=(3, 3))
        gui_small(frame, "um").pack(side="left")
        return lbl, frame

    r2_total, r3_total = tk.StringVar(win, ""), tk.StringVar(win, "")
    r2_lbl, r2_frame = ring_row("Second ring offset", "ring 1",
                                v["ring2_offset"], r2_total)
    r3_lbl, r3_frame = ring_row("Third ring offset", "ring 2",
                                v["ring3_offset"], r3_total)
    linking = [False]

    def offs():
        return (gui_float(v["laser_offset"], 0), gui_float(v["ring2_offset"], 0),
                gui_float(v["ring3_offset"], 0))

    def totals_from_rings(*_a):
        if linking[0]:
            return
        linking[0] = True
        try:
            a, b, c = offs()
            r2_total.set(f"{a + b:g}")
            r3_total.set(f"{a + b + c:g}")
        finally:
            linking[0] = False

    def r2_from_total(*_a):
        if linking[0]:
            return
        linking[0] = True
        try:
            a, b, c = offs()
            b = gui_float(r2_total, 0) - a
            v["ring2_offset"].set(f"{b:g}")
            r3_total.set(f"{a + b + c:g}")
        finally:
            linking[0] = False

    def r3_from_total(*_a):
        if linking[0]:
            return
        linking[0] = True
        try:
            a, b, c = offs()
            v["ring3_offset"].set(f"{gui_float(r3_total, 0) - a - b:g}")
        finally:
            linking[0] = False

    def rings_shown(*_a):
        if not r2_lbl.winfo_exists():
            return
        pat = v["shot_pattern"].get()
        for shown, lbl, frame in ((pat in GUI_TWO_RING + GUI_THREE_RING, r2_lbl, r2_frame),
                                  (pat in GUI_THREE_RING, r3_lbl, r3_frame)):
            if shown:
                lbl.grid()
                frame.grid()
            else:
                lbl.grid_remove()
                frame.grid_remove()

    for key in ("laser_offset", "ring2_offset", "ring3_offset"):
        v[key].trace_add("write", totals_from_rings)
        v[key].trace_add("write", redraw_pattern)
    v["laser_um"].trace_add("write", redraw_pattern)
    r2_total.trace_add("write", r2_from_total)
    r3_total.trace_add("write", r3_from_total)
    v["shot_pattern"].trace_add("write", rings_shown)
    totals_from_rings()
    rings_shown()

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
    gui_fill_screen(win, px(1280), px(800))
    gui_title_strip(win, "Bead selection   --   "
                         f"{state.path.name if state.path else 'no scan'}")

    # bottom bar, packed first so the image can never squeeze it out
    bottom = tk.Frame(win, bg=GUI_BG, padx=8, pady=6)
    bottom.pack(fill="x", side="bottom")
    # save and review / run progress: a green bar along the bottom edge
    win.progress = GuiProgress(bottom, ("review", "run"), height=px(18))
    win.progress.widget.pack(side="bottom", fill="x", pady=(4, 0))
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
    # the method is the remembered one (last_settings / the loaded save);
    # global_sweep / flat_field are old settings without a control and
    # no longer force it back to default on every visit (2026-10-19)
    if v["method"].get() == "trained" and state.filter is None:
        v["method"].set("default")          # no filter loaded in this run
    if v["method"].get() not in ("default", "quick", "strict", "broad",
                                 "trained"):
        v["method"].set("default")          # a save made under the old names
    upload_entry = "upload filter template..."
    # what each method does: shown beside the menu while the entry is
    # highlighted (mouse over it, before the click)
    gui_dropdown(side, v["method"],
                 ["default", "quick", "strict", "broad", None,
                  upload_entry],
                 tips={"default": "global threshold sweep",
                       "quick": "flat-field subtraction",
                       "strict": "both combined, keep the best 5 %",
                       "broad": "any faint round blob in the size window,\n"
                                "all accepted -- reject by hand; the\n"
                                "stringency slider keeps only the stronger\n"
                                "blobs (100 = all of them)",
                       upload_entry: "a filter trained with FilterTrainer;\n"
                                     "broad's candidates re-scored (or the\n"
                                     "trained template matched), the slider\n"
                                     "keeps the stronger ones (100 = all)"}
                 ).pack(anchor="w", pady=(2, 4), fill="x")
    # the loaded filter's name, under the dropdown while it is in use
    filter_lbl = gui_small(side, "", justify="left", anchor="w")

    def upload_filter() -> None:
        """The last dropdown entry: pick a FilterTrainer .json."""
        start = HERE / "FilterTrainer" / "filters"
        p = filedialog.askopenfilename(
            parent=win, title="Open a trained filter",
            initialdir=str(start if start.is_dir() else HERE),
            filetypes=[("trained filter", "*_blobfilter.json"),
                       ("json", "*.json"), ("all files", "*.*")])
        prev = "trained" if state.filter is not None else "default"
        if not p:
            v["method"].set(prev)
            return
        try:
            state.filter = gui_filter_load(Path(p))
        except (OSError, ValueError) as e:
            set_status(f"filter not loaded: {e}")
            v["method"].set(prev)
            return
        v["method"].set("trained")
        set_status(f"filter loaded: {Path(p).name} ({state.filter['kind']})")
    # broad stringency slider: shown only while the broad method is
    # selected (see GUI_BROAD_SWING)
    trow = tk.Frame(side, bg=GUI_BG)
    trow.pack(anchor="w", fill="x", pady=(0, 8))
    gui_label(trow, "stringency").pack(anchor="w")
    srow = tk.Frame(trow, bg=GUI_BG)
    srow.pack(anchor="w", fill="x")
    # a wide slider with a big grip, the mouse wheel steps it, and a
    # box to type the value (1-100)
    sbar = tk.Scale(srow, variable=v["broad_stringency"], from_=1, to=100,
                    orient="horizontal", resolution=1, length=px(170),
                    width=px(18), sliderlength=px(30),
                    font=F_SMALL, bg=GUI_BG, fg=GUI_TXT,
                    troughcolor=GUI_BOX, activebackground=GUI_LIGHT,
                    highlightthickness=0, bd=1, sliderrelief="raised")
    sbar.pack(side="left")
    sbox_var = tk.StringVar(win, v["broad_stringency"].get())
    sbox = gui_box(srow, sbox_var, 4)
    sbox.pack(side="left", padx=(6, 2))
    gui_small(srow, "%").pack(side="left")

    def slider_wheel(e):
        sbar.set(min(max(int(float(sbar.get())) + (1 if e.delta > 0 else -1),
                         1), 100))
        return "break"
    sbar.bind("<MouseWheel>", slider_wheel)

    def sbox_apply(_e=None):
        try:
            val = min(max(int(round(float(sbox_var.get()))), 1), 100)
        except ValueError:
            val = int(float(v["broad_stringency"].get() or 100))
        v["broad_stringency"].set(str(val))
        sbox_var.set(str(val))
    sbox.bind("<Return>", sbox_apply)
    sbox.bind("<FocusOut>", sbox_apply)

    def sbox_follow(*_a):
        if sbox.winfo_exists() and win.focus_get() is not sbox:
            try:
                sbox_var.set(f"{int(float(v['broad_stringency'].get()))}")
            except ValueError:
                pass
    v["broad_stringency"].trace_add("write", sbox_follow)
    analyze_btn = gui_button(side, "Analyze box", lambda: analyze(),
                             font=F_BIG)
    analyze_btn.pack(fill="x", pady=(4, 0))

    def step_state(*_a) -> None:
        if trow.winfo_exists():             # the trace outlives the window
            m = v["method"].get()
            if m == upload_entry:
                win.after(1, upload_filter)  # the dialog, after the menu closes
                return
            if m in ("broad", "trained"):
                trow.pack(anchor="w", fill="x", pady=(0, 8), before=analyze_btn)
            else:
                trow.pack_forget()
            if m == "trained" and state.filter is not None:
                filter_lbl.config(text=f"filter: {Path(state.filter['path']).name}"
                                       f"\n({state.filter['kind']})")
                filter_lbl.pack(anchor="w", fill="x", before=analyze_btn)
            else:
                filter_lbl.pack_forget()
    v["method"].trace_add("write", step_state)
    step_state()
    # progress of the analysis, under the button (see GuiProgress)
    win.analyze_bar = GuiProgress(side, ("analyze",), height=px(18))
    win.analyze_bar.widget.pack(fill="x", pady=(3, 0))
    gui_small(side, "left-drag a box on the image, then analyze;\n"
                    "the box stays -- Clear box empties it so the same\n"
                    "area can be analyzed again with another method",
              justify="left", anchor="w").pack(anchor="w", fill="x",
                                               pady=(2, 8))
    # Reject box was dropped on 2026-10-20 (reject by hand instead);
    # Clear all blobs is the File menu's "Clear all beads" as a button
    for t, fn in (("Accept box", lambda: set_many("accept")),
                  ("Clear box", lambda: clear_box()),
                  ("Clear all blobs", lambda: clear_beads())):
        gui_button(side, t, fn).pack(fill="x", pady=2)
    draw_mode = tk.BooleanVar(win, False)
    gui_toggle(side, draw_mode, "Manual selection",
               command=lambda: draw_changed()).pack(fill="x", pady=(2, 0))
    gui_small(side, "click a bead centre: a bead-size dotted circle;\n"
                    "arrows move it, press-drag anywhere resizes it,\n"
                    "Enter sets it (green), Esc drops it",
              justify="left", anchor="w").pack(anchor="w", fill="x",
                                               pady=(2, 0))
    gui_hrule(side).pack(fill="x", pady=6)
    manual_mode = tk.BooleanVar(win, False)
    gui_toggle(side, manual_mode, "Manual accept / resizing",
               command=lambda: mode_changed()).pack(fill="x")
    gui_small(side, "while on: left-click a bead to flip it;\n"
                    "double-click one to move / resize it, Enter sets it",
              justify="left", anchor="w").pack(anchor="w", fill="x",
                                               pady=(2, 6))
    gui_hrule(side).pack(fill="x", pady=6)
    gui_label(side, "Show", font=F_TITLE).pack(anchor="w")
    show_vars = {}          # kept here: an unreferenced tk variable is
    for key, t, col in (("accepted", "accepted (green)", "#1e7d1e"),
                        ("clumped", "clumped / near a clump (purple)",
                         "#5b3a9e"),
                        ("rejected", "rejected / near a contaminant (red)",
                         "#b01c1c")):
        # only the accepted beads are drawn until a box is ticked; these
        # are window switches, never saved
        show_vars[key] = tk.BooleanVar(win, key == "accepted")   # GC'd
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
    zoombox = tk.BooleanVar(win, False)
    # left-drag draws the analysis box only when no other tool has it
    box.active = lambda ev: not (manual_mode.get() or draw_mode.get()
                                 or zoombox.get())
    def um_per_px_now() -> float:
        return gui_transform(state, gui_apply_params(
            copy.deepcopy(state.cfg), v)).um_per_px

    circle = GuiCircleTool(fig, ax, draw_mode, lambda x, y, d: set_circle(x, y, d),
                           set_status, um_per_px_now,
                           blocked=lambda: zoombox.get(),
                           # the parameters window's bead size, as a radius
                           default_r=lambda: gui_float(
                               v["bead_um"], float(state.cfg["bead-diameter"]))
                           / um_per_px_now() / 2.0)
    # matplotlib keeps only weak references to the tools' callbacks
    win.tools = (circle, GuiZoomBox(fig, ax, zoombox,
                                    on_done=lambda: zoom_box_done()))
    zoom_fns = (None, None, None)
    if state.img is not None:
        zoom_fns = attach_zoom(fig, ax,
                               arrow_zoom=lambda: circle.pending is None)
        gui_attach_drag_pan(fig, ax, widget, button=3)
    widget.config(cursor="crosshair")

    # the actions of the manual modes, in a box over the bottom of the
    # image while one of them is on
    hint = tk.Label(main, font=F_SMALL, bg=GUI_LAVENDER, fg=GUI_TXT,
                    justify="left", padx=8, pady=4, relief="raised", bd=2)
    HINT_DRAW = ("MANUAL SELECTION      click a bead centre = circle of the bead size      "
                 "arrow keys = move 1 px      press-drag anywhere = resize "
                 "(away from the centre bigger, toward it smaller)      "
                 "Ctrl+scroll = resize      Enter = set it (green)      Esc = drop it")
    HINT_FLIP = ("MANUAL ACCEPT / RESIZING      left-click a bead = flip accept / reject      "
                 "double-click = move / resize it: arrow keys, press-drag, "
                 "Ctrl+scroll      Enter = set it      Esc = cancel")

    def show_hint(text: str | None) -> None:
        if text:
            hint.config(text=text, wraplength=max(widget.winfo_width() - 40, 300))
            hint.place(in_=widget, relx=0.5, rely=1.0, y=-10, anchor="s")
        else:
            hint.place_forget()

    def draw_changed() -> None:
        """Manual selection and manual accept / reject share the left
        button, so one switches the other off."""
        if draw_mode.get():
            manual_mode.set(False)
            circle.drop()
            widget.config(cursor="crosshair")
            show_hint(HINT_DRAW)
            set_status("manual selection: click a bead centre for a "
                       "bead-size circle; arrows move, press-drag resizes, "
                       "Enter sets it")
        else:
            circle.drop()
            mode_changed()

    def set_circle(x: float, y: float, d: float) -> None:
        """Enter on a blue circle: it becomes an accepted bead (a manual
        pick, green), replacing any object whose centre is inside it."""
        from matplotlib.patches import Circle
        inside = [i for i, b in enumerate(state.beads)
                  if math.hypot(b.x_px - x, b.y_px - y) <= d / 2]
        b = Bead(x, y, d)
        b.source = "drawn"                # shots sit off the drawn circle
        gui_set_manual(b, "accept")
        if layer.beads is state.beads and len(layer.patches) == len(state.beads):
            # one circle in, the replaced ones out -- redrawing every
            # bead takes seconds on a full slide
            for i in reversed(inside):
                state.beads.pop(i)
                layer.patches.pop(i).remove()
            state.beads.append(b)
            c = Circle((x, y), max(d / 2, 4), fill=False, lw=1.3,
                       ec=gui_bead_colour(b))
            ax.add_patch(c)
            layer.patches.append(c)
            gui_refilter(state)
            layer.refresh()
            update_counts()
        else:
            state.beads = [o for i, o in enumerate(state.beads)
                           if i not in inside] + [b]
            redraw_all()
        set_status(f"circle accepted: x {x:.0f}  y {y:.0f}, {d:.0f} px"
                   + (f" -- replaced {len(inside)} object(s) inside it"
                      if inside else ""))

    def zoom_box_on() -> None:
        if zoombox.get():
            set_status("zoom box: left-drag around the area to zoom to")
            widget.config(cursor="crosshair")
        else:
            zoom_box_done()

    def zoom_box_done() -> None:
        widget.config(cursor="hand2" if manual_mode.get() else "crosshair")
        set_status("zoomed to the box -- Zoom reset (or f) shows the whole "
                   "scan" if not zoombox.get() else "")

    def mode_changed() -> None:
        on = manual_mode.get()
        if on and draw_mode.get():
            draw_mode.set(False)
        circle.drop()                      # also ends a double-click edit
        widget.config(cursor="hand2" if on else "crosshair")
        show_hint(HINT_FLIP if on else None)
        set_status("manual mode: left-click a bead to flip accept / reject, "
                   "double-click to move / resize it"
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
        if method == "trained" and state.filter is None:
            set_status("no filter loaded: pick 'upload filter template...'")
            return
        if state.busy:
            set_status("busy -- wait for the current step to finish")
            return
        bar_start("analyze", 3)
        bar_step(f"detecting ({method})")
        cfg = gui_size_limits_px(state,
                                 gui_apply_params(copy.deepcopy(state.cfg), v))
        if method in ("broad", "trained"):
            cfg["detection"]["broad-diameter-px"] = gui_window_px(state, cfg)
            cfg["detection"]["broad-stringency"] = min(
                max(gui_float(v["broad_stringency"], 100), 1), 100)
        if method == "trained":
            cfg["detection"]["trained-filter"] = state.filter
            cfg["detection"]["um-per-px"] = gui_transform(state, cfg).um_per_px
        # how far gui_clearance looks, in px: 3 x the isolation window
        cfg["detection"]["clear-cap-px"] = max(
            3 * float(cfg["min-bead-separation"])
            / gui_transform(state, cfg).um_per_px, 100)
        region = tuple(box.box)               # as drawn when Analyze was pressed

        def detected(found, err):
            """Back on the main thread: merge, filter, draw."""
            state.busy = False
            if err is not None:
                bar_done()
                set_status(f"analyze stopped: {err}")
                return
            added = gui_merge(state, found, cfg)      # appended at the end
            if method in ("broad", "trained"):
                # the lenient pass: everything it found starts accepted, as a
                # manual pick, and is checked by hand (clumps stay clumps)
                for b in state.beads[len(state.beads) - added:]:
                    if not b.clumped:
                        gui_set_manual(b, "accept")
                        b.source = "broad"    # shots sit off the whole blob
            state.boxes.append(region)
            bar_step("filtering")
            gui_refilter(state)
            bar_step("drawing")
            layer.rebuild(state.beads)
            bar_done()
            update_counts()
            set_status(f"{method}: {len(found)} objects in the box, {added} new"
                       f"   |   accepted {sum(b.accepted for b in state.beads)}"
                       f" of {len(state.beads)}")

        # the detection itself in a worker thread, so the window and its
        # progress bar keep moving through the long OpenCV calls
        state.busy = True
        gui_run_in_thread(win, lambda: gui_detect_region(state.img, region,
                                                         cfg, method), detected)

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

    flip_timer = {"id": None}

    def flip(i: int) -> None:
        flip_timer["id"] = None
        b = state.beads[i]
        gui_set_manual(b, "reject" if b.accepted else "accept")
        gui_refilter(state)
        layer.refresh()
        update_counts()
        set_status(f"bead {i} -> {b.manual}")

    def resized(b, x: float, y: float, d: float) -> None:
        """Enter after a double-click edit: the bead takes the circle's
        centre and size; its decision is left as it was."""
        b.x_px, b.y_px, b.diameter_px = x, y, d
        if b in layer.beads:
            c = layer.patches[layer.beads.index(b)]
            c.center = (x, y)
            c.set_radius(max(d / 2, 4))
        gui_refilter(state)
        layer.refresh()
        update_counts()
        set_status(f"bead resized: x {x:.0f}  y {y:.0f}, {d:.0f} px  --  "
                   f"left-click flips, double-click resizes")

    def on_click(ev):
        if (ev.button != 1 or not manual_mode.get() or ev.inaxes is not ax
                or ev.xdata is None or zoombox.get()):
            return
        if not state.beads or circle.pending is not None:
            return                          # the circle tool has this press
        d = [math.hypot(b.x_px - ev.xdata, b.y_px - ev.ydata)
             if layer.visible(b) else float("inf") for b in state.beads]
        i = int(np.argmin(d))
        if not math.isfinite(d[i]) or d[i] > 40:
            return
        if flip_timer["id"] is not None:    # second press of a double-click
            win.after_cancel(flip_timer["id"])
            flip_timer["id"] = None
        if ev.dblclick:
            b = state.beads[i]
            circle.begin(b.x_px, b.y_px, b.diameter_px / 2,
                         lambda x, y, dd, b=b: resized(b, x, y, dd))
            return
        # a single click flips -- after a pause, so a double-click does not
        flip_timer["id"] = win.after(250, flip, i)

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
        if state.busy:
            set_status("busy -- wait for the current step to finish")
            return
        if gui_save(state, win, ask_name=True) is None:
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
    zbar = tk.Frame(bottom, bg=GUI_BG)
    zbar.pack(side="left")
    for text, fn in (("Zoom +", zoom_fns[0]), ("Zoom -", zoom_fns[1])):
        gui_button(zbar, text, fn, width=7).pack(side="left", padx=(0, 2))
    gui_toggle(zbar, zoombox, "Zoom box", command=zoom_box_on).pack(
        side="left", padx=(0, 2))
    gui_button(zbar, "Zoom reset", zoom_fns[2]).pack(side="left")

    state.on_beads_changed = redraw_all
    redraw_all()
    win.protocol("WM_DELETE_WINDOW", lambda: (setattr(state, "on_beads_changed", None), win.destroy()))
    return win


# ---- window 3: correlate fiducials (pick) ----------------------------

def gui_pick_window(master, state: GuiState):
    win = tk.Toplevel(master, bg=GUI_BG)
    win.title("microMS_beadtargeting -- pick")
    gui_fill_screen(win, px(1280), px(800))
    gui_menu_bar(win, {"About": about_menu_items(win)})
    gui_title_strip(win, "Correlate fiducials   --   "
                         f"{state.path.name if state.path else 'no scan'}")

    # bottom bar, packed first so the image can never squeeze it out
    bottom = tk.Frame(win, bg=GUI_BG, padx=8, pady=6)
    bottom.pack(fill="x", side="bottom")
    # save and review / run progress: a green bar along the bottom edge
    win.progress = GuiProgress(bottom, ("review", "run"), height=px(18))
    win.progress.widget.pack(side="bottom", fill="x", pady=(4, 0))
    status = gui_small(bottom, "right-click = set pending pixel     "
                               "Tab = next box     Ctrl+V = paste     "
                               "Enter = add     scroll = zoom     "
                               "left-drag / arrow keys = move", anchor="w")
    status.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))

    def set_status(text: str) -> None:
        status.config(text=text)

    # .run header paths, next to the save buttons; remembered in
    # last_settings.json
    runf = tk.Frame(bottom, bg=GUI_BG)
    runf.pack(side="left", anchor="w")
    for r, (lbl, key) in enumerate((("acqMethod", "acq_method"),
                                    ("directory", "run_directory"),
                                    ("sampleName", "sample_name"))):
        gui_small(runf, lbl, anchor="w").grid(row=r, column=0, sticky="w")
        e = gui_box(runf, state.v[key], 48)
        e.grid(row=r, column=1, sticky="w", padx=4, pady=1)
        e.bind("<FocusOut>", lambda _e: gui_settings_save(state.v))
        e.bind("<Return>", lambda _e: gui_settings_save(state.v))

    main = tk.Frame(win, bg=GUI_BG)
    main.pack(fill="both", expand=True, padx=8, pady=(8, 0))

    # left-hand fiducial table: header stays put, rows scroll (up to 100)
    side = tk.Frame(main, bg=GUI_BG, width=px(400))
    side.pack(side="left", fill="y", padx=(0, 8))
    side.pack_propagate(False)

    gui_label(side, "Fiducials", font=F_TITLE).pack(anchor="w")
    head = tk.Frame(side, bg=GUI_BG)
    head.pack(anchor="w", fill="x")
    COLW = (12, 9, 9, 4, 4, 2)              # characters per column (rows)
    heads = []
    for c, h in enumerate(("#   pixel", "stage X (um)", "stage Y (um)",
                           "hide", "", "")):
        heads.append(gui_small(head, h, anchor="w"))
        heads[-1].grid(row=0, column=c, padx=2, sticky="w" if c < 3 else "")

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
    pending = {"px": None, "art": [], "edit": None}

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
        gui_check(table, w["hide"], pady=0, padx=0,
                  command=lambda i=i, var=w["hide"]: toggle_hide(i, var)).grid(
                      row=r, column=3, padx=2)
        tk.Button(table, text="edit", font=F_TINY_B, bg=GUI_BTN, fg=GUI_TXT,
                  relief="raised", bd=1, width=4, padx=0, pady=0,
                  highlightthickness=0,
                  command=lambda i=i: start_edit(i)).grid(row=r, column=4,
                                                          padx=2)
        tk.Button(table, text="x", font=F_TINY_B, bg=GUI_BTN, fg=GUI_TXT,
                  relief="raised", bd=1, width=2, padx=0, pady=0,
                  highlightthickness=0,
                  command=lambda i=i: delete_fid(i)).grid(row=r, column=5,
                                                          padx=2)
        rows.append(w)

    def rebuild_table():
        for child in table.winfo_children():
            child.destroy()
        rows.clear()
        for i, f in enumerate(state.fids):
            add_row(i, f)
        # same pixel width per column in the header and the rows, so
        # the labels sit over their boxes and checkboxes
        table.update_idletasks()
        for c, hl in enumerate(heads):
            wd = hl.winfo_reqwidth() + 4
            if rows:
                wd = max(wd, table.grid_bbox(c, 0)[2])
            head.grid_columnconfigure(c, minsize=wd)
            table.grid_columnconfigure(c, minsize=wd)

    def start_edit(i):
        pending["edit"] = i
        set_status(f"right-click the new pixel for fiducial {i}")

    def toggle_hide(i, var):
        state.fids[i]["hide"] = bool(var.get())
        refit()

    def delete_fid(i):
        state.fids.pop(i)
        rebuild_table()
        refit()
        set_status(f"removed fiducial {i}")

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
        for b, fn in zip(zoom_btns, attach_zoom(fig, ax, arrow_zoom=False)):
            b.config(command=fn)
        gui_attach_drag_pan(fig, ax, widget)
        gui_arrow_pan(win, fig, ax)          # arrow keys move the view

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
                    f"{res.max():.1f} um")
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
        i = pending["edit"]
        if i is not None and i < len(state.fids):
            pending["edit"] = None
            state.fids[i]["x_px"] = float(ev.xdata)
            state.fids[i]["y_px"] = float(ev.ydata)
            rebuild_table()
            refit()
            set_status(f"fiducial {i} moved to ({ev.xdata:.0f}, {ev.ydata:.0f})")
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
        if state.busy:
            set_status("busy -- wait for the current step to finish")
            return
        save_fiducials(active)
        gui_settings_save(state.v)
        say(f"Wrote {len(active)} fiducials to {CONFIG_PATH.name}")
        if not any(b.accepted for b in state.beads):
            set_status(f"fiducials saved; no beads selected -- go back to "
                       f"bead selection, or use '{what}' from the console "
                       f"for whole-scan detection")
            return
        if not gui_ask_run_name(state, win):
            set_status("cancelled")
            return
        if what == "review":
            gui_review_into(state, layer, set_status, win)
            return
        cfg = gui_run_cfg(state, active)
        if what == "map":
            cfg["output"]["export-map"] = True

        def done(_result, err):
            state.busy = False
            if err is not None:
                bar_done()
                set_status(f"run stopped: {err}")
                return
            folders = sorted((HERE / cfg["output"].get("results-dir",
                                                        "RESULTS")).glob("*_run"),
                             key=lambda p: p.stat().st_mtime)
            folder = folders[-1] if folders else None
            set_status(f"run done -- {folder.name if folder else 'RESULTS'}")
            gui_run_done(win, folder)

        state.busy = True
        set_status("run running...")
        gui_run_in_thread(win, lambda: run(cfg), done)

    gui_button(bottom, "Save and run  >", lambda: save_and("run"),
               font=F_BIG).pack(side="right")
    gui_button(bottom, "Run and export map", lambda: save_and("map"),
               font=F_BIG).pack(side="right", padx=(0, 10))
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
