# CLAUDE.md

Instructions for Claude Code working in this repository.

## What this is

Image-guided MALDI-MSI targeting of SPPS resin beads on ITO slides, for a
Bruker timsTOF fleX. It detects beads in a scan of a matrix-coated slide,
keeps only isolated single beads, places laser shots on clean matrix just off
each bead edge, and exports coordinates for flexImaging / autoXecute.

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
flagged as such until someone reads them off the instrument:

| value | status |
|---|---|
| `focal-spot-um: 10` | placeholder |
| `beam-scan-um: 20` | placeholder |
| `.xeo` `PlateTypeName` header | unverified against a real fleX export |
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

**Shots are dropped only on genuine crater overlap** or for leaving
`slide-bounds`. Never trim shots to satisfy a cosmetic clearance margin.

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
