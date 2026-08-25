# microMS User Guide — coverage analysis

What our pipeline takes from the microMS User Guide, what it deliberately
leaves out, and what it is missing that it should have.

Source: microMS User Guide, written by Troy Comi, Sweedler group, University of
Illinois — <https://neuroproteomics.scs.illinois.edu/Site/microms/microMS_UserGuide.pdf>

Software citation:

> Comi TJ, Neumann EK, Do TD, Sweedler JV. *microMS: A Python Platform for
> Image-Guided Mass Spectrometry Profiling.* J. Am. Soc. Mass Spectrom. 2017,
> 28(9), 1919–1928. DOI 10.1007/s13361-017-1704-1

Read 10 and 14 August 2026; re-read in full 18 August 2026 to produce this
document.

---

## 1. What the guide confirms

These were design decisions made before or while reading the guide, and the
guide validates them. Worth recording, because each one looks arbitrary
otherwise.

| Our behaviour | Guide |
|---|---|
| Isolation filter runs against **all** detected objects, before shape filtering | Loose blob finding first, then distance filtering, specifically so clustered neighbours are not skipped |
| `min-bead-separation: 150` exceeds the ablation footprint | The distance filter should exceed probe size to prevent contamination during acquisition; the value depends on stage accuracy, probe size and suspected analyte delocalisation during matrix application |
| Similarity transform: translation, rotation, uniform scale, optional reflection | Point-based similarity registration covering translation, rotation and scaling, with limited reflection support and no correction for skewed perspectives |
| Effective radius = √(area/π); circularity = 4π·area/perimeter² | Identical definitions |
| Worst-fitting fiducial drawn red, RMS shown live | Fiducial localisation error estimated per mark, worst drawn in red, reselect until it is no longer the worst |
| Four or more fiducials preferred | Error falls as 1/√(number of fiducials); surrounding the target area with fiducials gives the best accuracy |
| X-shaped etched fiducial marks recommended over the current wavy strokes | Etched X markers used successfully; the line intersection can be located accurately and resists distortion between imaging systems, particularly after matrix application |
| `.xeo` files split at 400 positions | 400 points is a software maximum, and microMS splits automatically |
| `mtp_calibration` — a second transform from stage µm to plate fraction | Bruker instruments report 2D stage positions, but automatic acquisition is directed by scaled fractions of the whole plate, so an additional transformation is required |
| Shot positions placed around the bead exterior at an offset from the circumference | Circular packing, with a minimum separation, a maximum target count, and an offset from the circumference; negative offsets place targets inside the blob |

Two further points from the guide that sharpen our own caveats:

**Pixelated perimeters inflate circularity.** The guide notes circularity reads
higher than a true geometric value because the perimeter is counted in pixels.
Relevant to our over-firing clump screen — the circularity arm of that test is
measuring something slightly optimistic.

**The plate map drifts.** The motor-to-fraction values rarely change, but
adjusting the stage or using a different set of teaching points moves them, and
they should be recalibrated occasionally. So `mtp_calibration` is not strictly a
one-time measurement — treat it as stable but re-check it periodically.

---

## 2. What we deliberately leave out

None of this is needed for the bead workflow. Listed so nobody wonders whether
it was overlooked.

**Installation and image plumbing.** Windows and Linux setup, openslide, ndpi
and bigTiff handling, image decimation, WASD/QE navigation, the `c#.tif`
channel-naming convention. GUI mechanics with no analogue here.

**Multichannel and fluorescence.** Up to nine channels with brightfield
required as channel one; fluorescence intensity histograms with per-channel
offsets and colour selection; filtering out dim blobs by intensity. We are
brightfield only.

**Histogram stratification.** The interactive population histogram, high and
low pass filters, "Pick Extremes" to select the largest and smallest *n*
objects, and saving histogram divisions as separate lists. Built for splitting
a cell population into subgroups; we want one homogeneous population and report
percentiles in the terminal instead.

**Rectangular and hexagonal packing.** Both generate a grid of positions over
each blob to acquire a mass spectral image of it. We profile the matrix halo
around the bead, not the bead itself.

**Multiple blob lists.** microMS supports up to ten lists, with filtering and
patterning spawning child lists. We have one list and a manual override file.

**Supporting new instruments.** Roughly thirty pages implementing
`coordinateMapper` and `brukerMapper` for two fictional instruments. Only
relevant if someone has to write or repair a timsTOF fleX mapper for microMS
itself.

**Direct instrument control.** A lab-built Zaber xyz stage for liquid
extraction. Not applicable.

---

## 3. What we are missing and should add

Ordered by how much each would help on the current slides.

### 3.1 Threshold view — highest value

microMS can display which pixels pass the current threshold, grouped and
coloured against a dark background, and middle-clicking an object reports its
area and circularity. The guide presents this as the way to work out suitable
blob parameters and to see why a given object was excluded.

This is the direct diagnostic for our worst open problem. Measured bead
diameter is threshold-dependent — the same beads measured 63, 70 and 136 µm
under different `threshold` and `background-kernel-px` settings, against a real
diameter near 90 µm — and right now the only feedback is the median-diameter
warning after a full run. A threshold view would make the correct setting
visible in seconds.

**Implementation:** a `threshold` subcommand rendering the post-flat-field
binary mask over the scan, with per-component area and diameter on click.

### 3.2 Manual addition of beads

microMS allows blobs to be added by hand: a click places one at a default
radius, and click-and-drag sets the circumference explicitly. Shift-click
removes.

Our `select` window can only toggle beads the detector already found. Recall on
the current slides is poor — plenty of clearly visible beads are never
detected — so there is no way to recover them at all. This is a real
functional gap, not a convenience.

**Implementation:** in `bead_manual_selection`, a drag gesture that creates a
`Bead` at the drawn radius, marked `manual`, written to
`manual_selection.csv` with its radius so `run` can reconstruct it.

### 3.3 Polygon regions of interest

microMS supports hand-drawn rectangular and polygonal ROIs, either restricting
blob finding or filtering an existing list to retain or remove what falls
inside.

We have a single rectangle in the YAML. It cannot follow the slide edge, and it
cannot exclude the fiducial strokes without also cutting a band of good beads.
Box-reject in `select` covers the rectangular remove case only.

**Implementation:** a polygon in `detection.roi`, plus a drawn-polygon reject
in `select`.

### 3.4 Recording blob-finding provenance

microMS writes the blob-finding parameters, the ROI, and the ordered list of
filters applied into the saved blob list file, explicitly as a record of how
that list was produced.

Our `targets.csv` records positions and per-bead measurements but not the
settings that produced them. Six months on there is no way to tell which
threshold generated a given target list.

**Implementation:** a comment header in `targets.csv` carrying the detection
block, fiducial residuals, and counts at each filter stage.

### 3.5 Travelling-salesperson path ordering

microMS optimises stage travel with a TSP approximation bounded to three
minutes, and notes that simple top-to-bottom, left-to-right ordering makes the
stage travel roughly twice as far.

Our serpentine ordering is essentially that unoptimised alternative. At 156
positions this is negligible. It would matter at thousands.

**Implementation:** low priority. Serpentine is fine at current scale.

### 3.6 A registration caveat to document

The guide advises against reusing a saved registration once the sample has been
removed and reinserted, because repositioning appears as a systematic error at
every target.

Our fiducials live in `laser_setup.yaml` and are trivially reusable, so the
temptation is built in. This belongs in the README as an explicit warning.

### 3.7 Fiducial guidance we have not written down

Two further points from the guide worth adding to the slide-prep section:

- Smaller fiducials localise more precisely, but must stay large enough to find
  on the instrument camera.
- Setting the default blob radius to the probe size makes targets that are too
  close together easier to spot.

---

## 4. The MTP calibration file

The guide settles what to ask Dr. Neumann for.

`brukerMapper` requires a motor-coordinate file, and the guide gives its
format: tab-delimited text, the first column an `.xeo` position on the slide
adapter, the next two columns the corresponding x and y coordinates. It lives
in the directory containing the coordinate mappers. The values must be present
before the mapper will initialise, so if her microMS runs on the timsTOF, this
file exists on her machine.

The guide also confirms the slide adapter `.xeo` geometry file defines the
coordinate set that `brukerMapper` references — which supports, without
confirming, our `MTP Slide Adapter II` header.

### Settled by the reference run file

`Imaging_Run.run` (AutoExecute 7.6.6.0, 922 611 positions across 7 regions,
supplied by Dr. Neumann) confirms:

- `baseGeometry="MTP Slide Adapter II"` — our `.xeo` `PlateTypeName` string is
  correct. Previously a guess.
- The `.run` carries **no coordinates**. It is an ordered list of position
  *names* plus acquisition settings, and resolves coordinates through
  `geometry="<stem>"` → `<stem>.xeo`. So a targeting run needs **both** files,
  written together, with names matching exactly.
- Position naming: `R<region:02d>X<x>Y<y>`, where **X and Y are physical
  adapter coordinates in whole units**, not sequence numbers. One unit is
  10 µm, the same on both axes. The evidence:

  | observation | at 10 µm/unit | matches |
  |---|---|---|
  | two Y bands 2623 units apart | 26.2 mm | slide pitch on a 2-slide adapter |
  | X spans 5294 units | 52.9 mm | tissue inboard of a 75 mm slide |
  | each region 448 × 295 units | 4.48 × 2.95 mm | a mouse kidney section |
  | `sampleName` = `kidneyslides34` | slides 3 and 4 | the two Y bands, holding 3 and 4 regions |

  The scale is inferred from geometry, not documented. Confirm against a real
  `.xeo` before acquiring.

- **Both axes share one scale.** This settles the anisotropy question raised in
  §3 of the MTP notes: a uniform-scale similarity fit is the right model for
  stage µm → adapter units, and the residual warning should stay quiet on real
  data.
- The matching geometry is named `Imaging_Run`, so **`Imaging_Run.xeo` exists in
  the same folder on her machine.** That is the single file still needed.

**Still to verify on the instrument:**

1. Get her motor-coordinate file, and ask which instrument and adapter mounting
   it was measured on.
2. Get `Imaging_Run.xeo` — the geometry paired with the run file already
   supplied. It is the last piece: it shows the real `UnitCoord` values and the
   XML wrapper, which settles both the format and the plate-fraction question
   below.
3. Confirm whether `UnitCoord_X/Y` share a single scale or are normalised
   independently to plate width and height. Our stacked fit assumes one scale
   and reports a residual if that is wrong; a non-square adapter would break it.
4. Measure focal spot and beam scan width, replacing the 10 µm / 20 µm
   placeholders.
5. Confirm whether autoXecute random walk is enabled in the acquisition method.
6. Confirm the `.run` `type` attribute for discrete profiling. The reference
   uses `FastImaging`, but that run rastered tissue; we fire discrete positions.

---

## 5. Summary

The pipeline reproduces the parts of microMS the bead workflow needs, and the
guide independently confirms the non-obvious choices — particularly the filter
ordering and the stacked plate-fraction transform.

The omissions are all population-sorting, multichannel, or instrument-authoring
features that do not apply.

The genuine gaps are **threshold view** and **manual bead addition**. Both
address measured weaknesses of the current slides rather than hypothetical
ones: threshold view attacks the diameter-bias problem that currently forces
`distance-reference: center`, and manual addition recovers beads that detection
misses entirely.
