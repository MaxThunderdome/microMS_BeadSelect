# CLAUDE.md

Instructions for Claude Code working in this repository.

## What this is

Image-guided MALDI-MSI targeting of SPPS resin beads on ITO slides, for a
Bruker timsTOF fleX. It detects beads in a scan of a matrix-coated slide,
keeps only isolated single beads, places laser shots on clean matrix just off
each bead edge, and exports coordinates for flexImaging / autoXecute.

<<<<<<< HEAD
One user-facing file: `microMS_beadtargeting.py`, holding the whole pipeline
and a `CONFIG` dict of every tunable parameter at the top.

## Hard constraints

**Do not split the pipeline into modules.** One script is the design, not an
accident. Add functions, not packages.

**Every tunable belongs in `CONFIG`.** If you add a magic number further down
the file, you have introduced a bug. Put it in `CONFIG` with a comment saying
what it does and when to change it.

**Never guess a physical constant.** These values are unmeasured and must stay
=======
Two user-facing files:

- `microMS_beadtargeting.py` — the whole pipeline, deliberately one file
- `laser_setup.yaml` — every tunable parameter

## Hard constraints

**Do not split the pipeline into modules.** A collaborating PI (Dr. Elizabeth
Neumann, UC Davis) runs this and should never have to open a `.py` file. One
script plus one config is the design, not an accident. Add functions, not
packages.

**Every tunable belongs in `laser_setup.yaml`.** If you add a magic number to
the source, you have introduced a bug. Read it from config with a sensible
default and document it in the YAML with a comment saying what it does and
when to change it.

**Never guess a physical constant.** Four values are unmeasured and must stay
>>>>>>> cb595a2987d1ea5efa53eee4d313a9cc4ef9826a
flagged as such until someone reads them off the instrument:

| value | status |
|---|---|
| `focal-spot-um: 10` | placeholder |
| `beam-scan-um: 20` | placeholder |
<<<<<<< HEAD
| adapter origin (`name-coordinates.x0-um` / `y0-um`) | unmeasured |
=======
| `.xeo` `PlateTypeName` header | unverified against a real fleX export |
>>>>>>> cb595a2987d1ea5efa53eee4d313a9cc4ef9826a
| `mtp_calibration` | empty; must be measured per instrument |

If asked to "just pick a reasonable default" for any of these, decline and
explain. A plausible-looking wrong value produces a file that loads cleanly and
fires the laser in the wrong place. Silence is safer than a guess here.

**Attribution rule.** The workflow ordering, similarity registration,
nearest-neighbour filter and click-training interaction follow microMS
(Comi et al. 2017, DOI 10.1007/s13361-017-1704-1). microMS carries an Illinois
copyright with no explicit licence, so behaviour is **reimplemented
independently**. Only `.xeo` format constants are reproduced verbatim, and they
are marked `FORMAT SPEC` in the source. Do not copy microMS source.

<<<<<<< HEAD
**Position names carry coordinates.** `R<region>X<x>Y<y>` encodes physical
adapter position in 10 µm units on both axes, confirmed by geometry against a
real run file — not sequence numbers. The `.run` and `.xeo` are matched by name
alone, so `position_name` is shared by both writers and must stay that way.

=======
>>>>>>> cb595a2987d1ea5efa53eee4d313a9cc4ef9826a
## Invariants that look like bugs but are not

**Filter order matters.** Clump screen, then isolation, then size. Isolation
runs against *every* detected object including debris that will later fail the
size filter. Reordering so shape comes first will delete the dust next to a
bead and let that bead falsely pass isolation. There is a regression test for
this.

**Clumps are marked, not deleted.** A clump stays in the object list so nearby
singles still fail isolation against it. Removing clumps from the list is the
same bug as above.

**Similarity transform, not affine.** An affine fit through exactly three
fiducials is exactly determined and reports a residual of zero, which tells the
operator nothing. The similarity fit leaves a real residual. Do not "improve"
this to affine.

<<<<<<< HEAD
**Shots are dropped only on genuine crater overlap.** Never trim shots to
satisfy a cosmetic clearance margin.

**There is no software travel-limit check, and none should be added.** A
`slide-bounds` window existed with invented values; on a real stage every shot
fell outside it and `run` wrote an empty target list and exited 0. The stage
enforces its own limits in hardware. If a bounds check is ever genuinely needed,
the numbers must come from the instrument, not from a plausible-looking
default.
=======
**Shots are dropped only on genuine crater overlap** or for leaving
`slide-bounds`. Never trim shots to satisfy a cosmetic clearance margin.
>>>>>>> cb595a2987d1ea5efa53eee4d313a9cc4ef9826a

**`UnitCoord_X/Y` in `.xeo` are plate fractions, not microns.** There are two
stacked transforms: scan px → stage µm (fiducials), then stage µm → plate
fraction (`mtp_calibration`). With `mtp_calibration` empty, `run` writes the
CSV and skips the `.xeo`. Keep that behaviour.

**The `.xeo` header is 13 lines and the footer is 12.** microMS reads positions
with `lines[13:-12]`. Changing the line count silently breaks interoperability.
`read_xeo` and a round-trip test exist to catch this.

## Known failure modes, already hit once

**Never call `input()` from a matplotlib callback.** It blocks the GUI event
loop; the window stops repainting and looks frozen, and on macOS it can
deadlock. Coordinate entry happens in-window via `TextBox`. This bug shipped
once — do not reintroduce it.

**Do not wire the matplotlib toolbar zoom.** It binds left-drag, which is the
box selector in `select`, and entering toolbar zoom mode silently swallows the
clicks that add or toggle beads. Zoom is scroll + middle-drag + buttons via
`attach_zoom`.

**Pan from pixel deltas, not data coordinates.** Reading `ev.xdata` while limits
are changing feeds the new limits into the next delta and the image accelerates
away under the cursor.

**`SimpleBlobDetector` defaults to `filterByColor=True, blobColor=0`** — dark
blobs. The image is inverted before detection, so beads are bright and the
default rejects every one of them. `filterByColor` is explicitly disabled.

**Detection performance.** `_count_cores` once built a full-ROI mask and
distance transform per component, making detection O(n × image) — 172 s on an
8000 px scan. It crops to each component's bounding box first: 1.5 s. Keep it
that way.

<<<<<<< HEAD
**cKDTree pads missing neighbours out of range.** `query(k=2)` on a tree with
one point returns index `len(points)` with distance `inf`. Any `k`-nearest
query must check the index bound and `isfinite` before indexing. A single
detected object hit this and crashed `run`.

**A least-squares residual cannot see a degenerate layout.** Duplicated or
collinear fiducials fit perfectly and report RMS 0 while registering badly.
`check_fiducial_geometry` catches both; do not treat a low residual alone as
proof of a good registration.
=======
**Interactive windows must not show the raw scan.** `imshow` on a full
75 Mpx scan costs ~750 ms per frame, because matplotlib colour-maps the
whole array on every draw and never clips it to the visible rectangle.
`attach_image_lod` hands it a decimated crop of the visible area
instead (~27 ms), dropping to 1:1 once you zoom in. It keeps the artist
in ORIGINAL scan pixels via `set_extent`, so `ev.xdata` is still a scan
pixel and click handlers need no change. Do not "simplify" this back to
a plain `imshow`.

**Three fiducials is a floor, not a target.** A similarity fit through
exactly 3 points leaves no spare, so `loo_residuals` returns None and the
operator sees an in-sample residual only. The picker allows up to
`max-fiducials` (15) and labels the Add button with the running count.
It also preloads the existing set, so it reports how many it loaded and
runs `registration_sanity` on them before the operator adds more -- a
stale set silently mixed with fresh picks is a bug that has shipped once.

**Commands write into `outputs/<command>_<stamp>/`.** `new_output_dir`
makes it. Nothing writes into the working directory any more, and a
re-run never overwrites the previous answer. `review` writes two
different things on purpose: `slide_overview.png` answers whether the
selection covers the right deposit, and the per-bead sheets answer
whether each bead's shots landed sensibly. Do not add shot angles to the
overview -- at slide scale the crater is sub-pixel and the marks would
be noise.

**`Detect in box` re-runs every filter over the union.** Isolation is a
property of the whole object list, so appending newly found objects to
an already-filtered list would leave them invisible to their
neighbours' isolation test. It also widens `detection.roi` on close via
`save_roi`, because `run` only looks inside the ROI and would otherwise
silently discard every override made outside it.
>>>>>>> cb595a2987d1ea5efa53eee4d313a9cc4ef9826a

**Apply edits once.** Two regressions in this file came from patch scripts that
double-applied or spliced across a region boundary, deleting whole functions
while leaving valid syntax. When editing programmatically, assert each anchor
matches exactly once, and run `pytest` plus `python microMS_beadtargeting.py
selftest` afterwards — the self-test does not open windows, so GUI code paths
need a separate read-through.

## Commands

```
python microMS_beadtargeting.py doctor    # environment check, run this first
python microMS_beadtargeting.py convert   # image -> TIFF
python microMS_beadtargeting.py pick      # click fiducials -> YAML
python microMS_beadtargeting.py select    # bead manual selection
python microMS_beadtargeting.py check     # registration quality only
<<<<<<< HEAD
=======
python microMS_beadtargeting.py review    # per-bead shot confirmation sheets
>>>>>>> cb595a2987d1ea5efa53eee4d313a9cc4ef9826a
python microMS_beadtargeting.py run       # detect, filter, shoot, export
python microMS_beadtargeting.py selftest  # synthetic end-to-end test

# -v / --verbose on any of them for a timed trace
pytest                                     # the test suite
```

## Style

Match what is there: standard library plus numpy/scipy/opencv/matplotlib, no
new dependencies without asking. Comments explain *why*, not *what* — the
existing ones document reasoning that is not recoverable from the code, and
that is the point of them. Keep them.

Prefer showing results in the terminal over writing extra files.

<<<<<<< HEAD
## Divergence from microMS

`DIVERGENCE.md` is the record of what uses microMS directly, what is ported to
match it exactly, and where this deliberately differs. Update it whenever you
change behaviour microMS also implements.

The one substantive algorithmic divergence is **blob finding**: microMS
thresholds a colour channel, this flat-field subtracts first, because bead
contrast here is under 10 grey levels with illumination drift. A global
threshold sweep returned 28,495 objects on the reference scan.

## Deliberate omissions

`MANUAL_COVERAGE.md` records what the microMS User Guide contains, what we
implement, what we deliberately skip, and what is genuinely missing. Read it
before adding a feature "because microMS has it" — several such features
(fluorescence filtering, histogram stratification, rectangular and hexagonal
packing, multiple blob lists) are omitted on purpose because this workflow
profiles the matrix halo around single isolated beads rather than sorting cell
populations or imaging each object.

The two acknowledged gaps worth implementing are a threshold view and manual
bead addition. Both are specified in that document.

=======
>>>>>>> cb595a2987d1ea5efa53eee4d313a9cc4ef9826a
## Current open work

- Measure `mtp_calibration` on the instrument; ask Dr. Neumann first for the
  `brukerMapper` motor-to-plate map her microMS install already has.
- Diff a real flexImaging `.xeo` export against this writer's output.
- Detection recall is poor at current slide contrast, and the clump screen
  over-fires as a result (549 of 1003 objects on the test slide). The manual
  selection window is the intended remedy; do not loosen
  `clump-core-fraction` to compensate, as that lets genuine pairs through.
- `distance-reference` should stay `center` until measured bead diameters are
  trustworthy. `edge` inherits threshold bias directly.
