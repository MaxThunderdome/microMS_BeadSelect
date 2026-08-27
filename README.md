# microMS_beadtargeting

Dependencies: 

pip install "numpy>=1.24" "scipy>=1.10" "opencv-python>=4.8" "matplotlib>=3.7"

---

## Commands

python microMS_beadtargeting.py doctor    # environment check
python microMS_beadtargeting.py convert   # image -> TIFF 
python microMS_beadtargeting.py pick      # click fiducials -> saved here  **close window to complete selection**
python microMS_beadtargeting.py select    # bead manual selection          **close window to complete selection**
python microMS_beadtargeting.py check     # registration quality only
python microMS_beadtargeting.py review    # show planned shots, no export
python microMS_beadtargeting.py run       # detect, filter, shoot, export
python microMS_beadtargeting.py selftest  # synthetic end-to-end test

---
## Outline

Image-guided MALDI-MSI targeting of SPPS resin beads on ITO slides, for a
Bruker timsTOF fleX.

One file. `microMS_beadtargeting.py` holds the pipeline and, at the top, a
`CONFIG` dict with every tunable parameter.

The workflow ordering, the point-based similarity registration, the
nearest-neighbour distance filter and the fiducial click-training interaction
all follow microMS:

> Comi TJ, Neumann EK, Do TD, Sweedler JV. *microMS: A Python Platform for
> Image-Guided Mass Spectrometry Profiling.* J. Am. Soc. Mass Spectrom. 2017,
> 28(9), 1919–1928. DOI 10.1007/s13361-017-1704-1

## Before the scan

Scribe fiducials **before** matrix application. The same physical marks have to
appear in the scan you register from and be drivable to on the stage.

Use **X-shaped etched marks**, not strokes. A line intersection can be
localised to a pixel; the centre of a wavy stroke cannot. Three minimum, in an
asymmetric arrangement — two along one edge, one offset perpendicular. Symmetry
admits a rotated mis-assignment that still produces a clean-looking transform.
Four or more unlocks leave-one-out cross validation, which is the only honest
error estimate available.

Registration error dominates final targeting accuracy. It is worth more
attention than anything downstream.

---

## 1. 'python microMS_beadtargeting.py convert'

Scan slide. Save as **TIFF**. Default hardcoded name: "slide01.tif". JPEG is rejected but converter included

```
python microMS_beadtargeting.py convert slide01.jpg
```

Colour input is converted to greyscale, 16-bit is scaled to 8-bit, and the
result is written next to the source — or into the script folder if the source
directory is read-only.

## 2. `python microMS_beadtargeting.py pick`

Opens the scan. Coordinate entry is **in the window**, not the terminal.

**close window to continue** once fiducials are entered

| action | effect |
|---|---|
| right-click on the image | set the pending pixel (gold X) |
| type into stage x / stage y | the measured stage reading |
| Tab | switch between the two boxes |
| Ctrl+V | paste; a copied pair like `18601.5, -20310.8` fills both boxes |
| **Add fiducial** | commit the pair |
| **Remove nearest** | delete the fiducial nearest the last right-click |
| **Reset** | clear the list |
| close the window | write `FIDUCIALS` into the source file |

Enter in the stage y box also commits, so entry can be keyboard-only.

Mock values (2 slides)
top left: 18601.5 , -20310.8    (x, y)
top right: 86083.1, -20161.0
bottom left: 18646.7, -69830.8
bottom right: 86124.7, -69700.2 

**upgraded for performation** Preload high resolution window so zooming does not have to recalculate each box

Both interactive windows zoom the same way:

| action | effect |
|---|---|
| scroll wheel | zoom about the cursor |
| middle-drag | pan |
| `+` / `-` | zoom about the centre |
| `f` | fit — back to the whole image |
| **Zoom +** / **Zoom -** / **Fit** | the same, as buttons |


The worst-fitting fiducial is drawn red and live RMS sits in the title, so a
mistyped coordinate shows up while you are still in the window. Only the
`FIDUCIALS` block is rewritten; everything around it is untouched, and
registration is reported on close.

## 3. `python microMS_beadtargeting.py selection`

Auto filtering is a starting point, not a verdict. At low contrast the detector
both merges real singles into false clumps and lets ragged pairs through, so the
operator gets the final say.

| colour | meaning |
|---|---|
| green | accepted |
| red | rejected by a filter |
| purple | screened as a clump |
| blue, thick | manually overridden |

| action | effect |
|---|---|
| right-click a bead | toggle accept/reject |
| drag a box | select a region |
| **Accept box** | accept every bead inside |
| **Reject box** | reject every bead inside |
| **Clear box** | drop the region selection |
| **Reset** | discard all manual overrides |
| checkboxes | show or hide each colour |
| close the window | write `manual_selection.csv` |

The three checkboxes control what is drawn. On a crowded slide the red and
purple circles bury the green ones — 549 of 1003 objects were clumps on the
reference scan — so unticking them is the only practical way to see what will
actually be acquired.

Hiding is display only. A hidden bead keeps its accept/reject state, still
counts as an isolation neighbour, and is still exported. But it is **not
clickable and not caught by the box tools**, so you cannot toggle something you
cannot see. The title lists any hidden category so a filtered view is never
mistaken for the whole picture.

The box tools are the fast path — draw around a debris field or a dense patch
and reject the lot in one click, rather than clicking a hundred beads.

Zoom with the scroll wheel, pan with middle-drag, `f` to fit. On an 8000 px scan
you will want to zoom in before judging individual beads — at full extent a bead
is a single pixel.

Overrides are stored by **pixel position, not index**, because detection indices
shift the moment any detection parameter changes. On reload each override is
matched to the nearest detected object within `match-radius-px`; anything with
no match is reported rather than silently dropped.

## 4. `python microMS_beadtargeting.py check`

Registration only. Reports RMS and per-fiducial residuals, recovered µm/px,
rotation, reflection flag, and leave-one-out CV at ≥4 fiducials.

Two things to look at. The recovered scale should match your scanner's known
µm/px. The reflection flag should match what you physically expect — a slide
scanned face-down, or a stage whose y counts opposite to the image, produces a
genuine mirror, and `allow-reflection: true` lets the fit absorb it silently.

With exactly three fiducials there is no cross-validation, and the in-sample
residual is a lower bound on true error, not an estimate of it.

**Do not reuse fiducials across sessions once the slide has been removed and
reinserted.** The microMS guide is explicit about this: repositioning the sample
shows up as a systematic error at every target, and because `FIDUCIALS` persists
in the source file, reusing them is the path of least resistance. Re-pick after
any remount.

## 5. `python microMS_beadtargeting.py review`

Once the selection looks right, preview it:

`review` fits the registration, applies every filter plus your manual
selection, places the shots, and draws them on the scan -- green accepted
beads, blue shot craters, orange dotted for dropped shots. Each review
gets its own `RESULTS/<date>_<time> review/` folder holding `review.png`
plus `review_zoom.png`, a close-up of the densest patch of selected beads
(a window also opens unless `output.review-show` is false). It exports nothing; `run` is still the only
command that writes target files.

## 6. `python microMS_beadtargeting.py run`

Everything `run` writes lands in `RESULTS/<date>_<time> run/`, a fresh
timestamped folder per run, so no acquisition package ever overwrites an
earlier one. (`flexCoords.txt` stays beside the script; it doubles as an
`mtp_calibration` input.)

The pipeline, in this order:

1. **Loose detection.** Deliberately permissive.
2. **Clump screen.** Beads that physically touch merge into one connected
   component, so a clump arrives as a single object with nothing near it and
   passes isolation cleanly. Its centroid is not a bead centre and its measured
   diameter is meaningless. Three tests catch it — outline aspect ratio, hull
   solidity, and counting distance-transform cores inside the component — and
   any one firing marks it. A clump is **marked, not deleted**, so nearby single
   beads still fail isolation against it.
3. **Isolation filter** at `min-bead-separation`, run against *every* detected
   object including debris. The ordering is the point: a bead sitting beside a
   speck of dust is not isolated, and shape-filtering first would delete the
   dust and let the bead falsely pass.
4. **Size filter** on what survives.
5. **Manual overrides** from `manual_selection.csv`, if present.
6. **Shot placement**, one per angle in `laser-shot-angles`.
7. **Validation** — slide bounds, crater vs. own bead, crater vs. neighbouring
   object, crater vs. adjacent shot.
8. **Serpentine ordering** and export.

