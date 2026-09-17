# microMS_beadtargeting

Dependencies (Python 3.10 or newer):
```bash
pip install "numpy>=1.24" "scipy>=1.10" "opencv-python>=4.8" "matplotlib>=3.7"
```
tkinter (the windows) ships with python.org Python on Windows.
The test suite (`tests/test_pipeline.py`) additionally needs `pip install pytest`; run `pytest` from the repository folder.

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
