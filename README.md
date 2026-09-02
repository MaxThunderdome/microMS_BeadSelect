# microMS_beadtargeting

Dependencies: 
```bash
pip install "numpy>=1.24" "scipy>=1.10" "opencv-python>=4.8" "matplotlib>=3.7"
```
tkinter (the windows) ships with python.org Python on Windows.

---
```bash
python microMS_beadtargeting.py gui       # windows: parameters -> beads -> fiducials
python microMS_beadtargeting.py select    # same as gui (starts at the parameters window)
python microMS_beadtargeting.py pick      # fiducial window only, on the CONFIG scan
python microMS_beadtargeting.py review    # planned shots + check.txt, no export
python microMS_beadtargeting.py run       # detect, filter, shoot, export

```
Code Logic... (updated by hand_08/31/26 by Dan Parker)

1) Created mockup of the microMS code from scratch.... Cons: mathematical recreation of functions in microMS. Pros: soft start pf frontend python user interface... 
2) Restart from microMS remote git repo... scrapped backend and implemented native featue
3) Modeled the code to explicitly include features from publication including BrukerMapper, writeXeo, Gui constraints, distance filtering, circular patterning. Designed for use with FleX MALDI MSI
4) Modified blob parameters to further target beads.  Circular blob identification kept native and modified to include dynamic bead sizing
5) Locating targets.... Scan image then "autoblob". *Select* region of interest option included. Removed threshold global sweep and replaced with "flat field subtraction" to account for the fixed thresholds of bead diameters. 
6) Filtering of targets... Filter conditions... beads isolated with 150um from other beads or clumps of beads or other visible contaminants
7) Patterning targets.... circular patterning was selected over rectangle or hexagonal packed to accommodate for bead shape. Minimum target-to-target distance and offset circumference (60um from bead center = 45um radius + 15um from edge of bead [assuming 10um laser]). Max number of targets bead size dependant (6 for 90um separated at 60, 120, 180, 240, 300 and 360 degrees)
8) Coordinate transform.... point-based similarity registration between microMS and Bruker instrumental software. Code accommodates 3-15 fiducials. MTP Lide II assignment... B8, B10, B12, C8, C10, C12, etc. Worst fit to fiducials highlighted in red. Accuracy of point based similarity registration. Adjust fiducials until the red marked fiducial remains red even after accurate rejiggering.

## Commands

python microMS_beadtargeting.py gui       # windows: parameters -> beads -> fiducials
python microMS_beadtargeting.py select    # same as gui (starts at the parameters window)
python microMS_beadtargeting.py pick      # fiducial window only, on the CONFIG scan
python microMS_beadtargeting.py review    # planned shots + check.txt, no export
python microMS_beadtargeting.py run       # detect, filter, shoot, export

Add `-v` to any command for a timed trace. Long steps show a progress bar in
the console.

---
## Outline

Image-guided MALDI-MSI targeting of SPPS resin beads on ITO slides, for a
Bruker timsTOF fleX.

One file. `microMS_beadtargeting.py` holds the pipeline and, at the top, a
`CONFIG` dict with every tunable parameter. The windows (the `WINDOWED FLOW`
section near the end of the file) drive that same pipeline; they do not
reimplement it.

The workflow ordering, the point-based similarity registration, the
nearest-neighbour distance filter and the fiducial click-training interaction
all follow microMS:

> Comi TJ, Neumann EK, Do TD, Sweedler JV. *microMS: A Python Platform for
> Image-Guided Mass Spectrometry Profiling.* J. Am. Soc. Mass Spectrom. 2017,
> 28(9), 1919–1928. DOI 10.1007/s13361-017-1704-1

microMS source is vendored, unmodified, under `microms/` (the `.xeo` files are
written by its own `brukerMapper`); `flex_mapper.py` is the timsTOF fleX mapper
it does not ship. See `ATTRIBUTION.md`.

Repository layout:

```
microMS_beadtargeting.py   the pipeline, CONFIG at the top, windows at the end
flex_mapper.py             timsTOF fleX coordinate mapper (microMS subclass)
microms/                   vendored microMS, never edited
tests/test_pipeline.py     pytest suite, no instrument needed
SAVES/                     bead selections and last-used parameters (created on use)
RESULTS/                   one timestamped folder per review / run (created on use)
manual_selection.csv       accept/reject decisions that review and run obey
```

## Lab order of operations

1. Glue beads to the slide.
2. Scan the slide (TIFF; a .jpg is offered for conversion in the window).
3. `select` — pick parameters, draw boxes on the scan, analyze them, decide.
   Either review the pictures now, spray matrix and come back, or go straight
   on to the fiducials.
4. Load the slide into the timsTOF, find the fiducials, enter their stage
   readings in the fiducial window.
5. Save and run — `.xeo` and `.run` land in `RESULTS/`.

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

## 1. `python microMS_beadtargeting.py gui` — parameters window

Opens on the last-used values (`SAVES/last_settings.json`); **File → Restore
default values** resets them. **File → Save / Save As / Load** work here and in
the bead window.

| control | effect |
|---|---|
| **Upload image...** | choose the scan. A .jpg/.png is offered for conversion to .tif (Yes / Exit); the .tif is written next to the original and used as the slide image |
| Type of image, Matrix | recorded with the save (no effect on detection yet, marked `*`) |
| v Image Settings v | drops down the microscope zoom and the provisional scale (µm/px) used by the isolation filter until fiducials exist |
| Location of targets | preselects the bead-matching method (Global threshold sweep → default, flat field subtraction → quick) |
| Average bead size + deviation | the size window; read the measured sizes off the bead window's info box and set this to match your beads |
| Isolation window | `min-bead-separation`; untick to skip the isolation filter |
| Max number of points | cap on accepted beads (best size fit first, then most isolated) |
| **Continue to bead selection** | loads the scan and opens the bead window |

## 2. Bead selection window

Starts as a blank canvas: nothing is detected until you draw a box.

| action | effect |
|---|---|
| left-drag | draw a box (the box stays until replaced) |
| right-drag | move the view |
| scroll wheel, `+` / `-`, `f` | zoom about the cursor / centre, fit |
| **Analyze box** | find beads inside the box with the chosen method, then run the pipeline's isolation and size filters over everything found so far |
| **Accept box** / **Reject box** | override every visible bead inside the box |
| **Clear box** | remove the objects inside the box, keep the box — pick another method and analyze the same area again |
| **Manual accept / reject** (toggle) | while on, a left-click flips the bead under the cursor |
| checkboxes | show or hide each colour |
| File → Discard manual overrides / Clear all beads | as named |

Methods:

| method | what it does |
|---|---|
| default | global threshold sweep (OpenCV blob detector) |
| quick | flat-field subtraction then connected components |
| strict | both, combined; then only the best-fitting 5 % of everything found is kept (size closest to nominal, then most isolated) |
| threshold | default, with the threshold step set on the slider under the dropdown (smaller steps find fainter beads and take longer; the slider is live only for this method) |

| colour | meaning |
|---|---|
| green | accepted |
| red | rejected: near a contaminant or another bead, outside the size window, or over the max number of points |
| purple | a clump, or a bead rejected because a clump sits inside its isolation window |
| blue, thick | manually overridden |

The info box shows accepted / total, overrides, boxes analysed, and the
measured size (min, median, average, max in µm) of the accepted beads and of
the contaminants in the current box — use it to set the bead size window.

Two ways out:

- **Save and review (no matrix yet)** — saves `SAVES/<timestamp>[_name].json`
  (parameters, scan path, boxes, every object with its decision), runs
  `review` on exactly these beads, and draws the planned shots on the canvas so
  you can zoom around for hairs and contaminants. Later: reopen, **File → Load**,
  continue.
- **Continue to correlate fiducials (ready for MSI)** — opens the fiducial
  window.

Both refuse with "no beads selected" if nothing is accepted.

Detection is done on the box you draw with the pipeline's own detectors; the
isolation and size filters, the clump screen and the manual overrides are the
same functions `run` uses. Overrides are stored by **pixel position, not
index**, so they survive re-analysis.

## 3. `python microMS_beadtargeting.py pick` — fiducial window

Also reached from the bead window. Coordinate entry is **in the window**.

| action | effect |
|---|---|
| right-click on the image | set the pending pixel (gold X) |
| type into stage x / stage y | the measured stage reading |
| Tab | switch between the two boxes |
| Ctrl+V | paste; a copied pair like `18601.5, -20310.8` fills both boxes |
| Enter in stage y, or **Add fiducial** | commit the pair |
| **Remove nearest** | delete the fiducial nearest the last right-click |
| **Reset** | clear the list |
| table on the left | every fiducial; edit X / Y and press Enter to re-fit; **hide** leaves one out of the fit without deleting it; **x** deletes it |
| **Save and run** | write the (non-hidden) fiducials into `FIDUCIALS` in the source file, then `run` |
| **Save and review** | write the fiducials, then `review`, drawn in the window |

The worst-fitting fiducial is drawn red and the live RMS, worst residual and
µm/px sit under the table. Hidden fiducials are grey. Only the `FIDUCIALS`
block of the source file is rewritten. Three are needed; with no bead
selection loaded, the buttons save the fiducials and stop.

Mock values (2 slides)
top left: 18601.5 , -20310.8    (x, y)
top right: 86083.1, -20161.0
bottom left: 18646.7, -69830.8
bottom right: 86124.7, -69700.2 

**Do not reuse fiducials across sessions once the slide has been removed and
reinserted.** Repositioning the sample shows up as a systematic error at every
target, and because `FIDUCIALS` persists in the source file, reusing them is
the path of least resistance. Re-pick after any remount.

## 4. `python microMS_beadtargeting.py review`

Fits the registration, applies every filter plus the manual selection, places
the shots, and draws them on the scan. Each review gets its own
`RESULTS/<date>_<time> review/` folder holding:

- `check.txt` — the registration report: RMS and per-fiducial residuals,
  recovered µm/px, rotation, reflection flag, leave-one-out CV at ≥4 fiducials.
  The recovered scale should match your scanner's known µm/px; the reflection
  flag should match what you physically expect.
- `review.png` — the accepted beads and their shots (blue craters, orange
  dotted for dropped shots).
- `review_zoom.png` — a close-up of the densest patch, with every category
  drawn (red rejects, purple clumps included).

It exports nothing; `run` is still the only command that writes target files.
From the windows, review runs on the beads you selected; from the console it
detects over the whole scan.

## 5. `python microMS_beadtargeting.py run`

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
7. **Validation** — crater vs. own bead, crater vs. neighbouring
   object, crater vs. adjacent shot.
8. **Serpentine ordering** and export.

From the windows, `run` receives the selected beads through `input.beads`
(`SAVES/<stem>_beads.csv`) and `manual_selection.csv`, so it shoots exactly
what the window showed. A console `run` afterwards follows those same
decisions until `manual_selection.csv` is cleared or replaced.
