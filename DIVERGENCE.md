# Divergence from microMS

Where this pipeline follows microMS, where it substitutes something else, and
why. Kept current — if you change behaviour that microMS also implements, note
it here.

Reference: Comi TJ, Neumann EK, Do TD, Sweedler JV. *microMS: A Python Platform
for Image-Guided Mass Spectrometry Profiling.* J. Am. Soc. Mass Spectrom. 2017,
28(9), 1919–1928. Source: <https://github.com/troycomi/microMS> (MIT).

---

## 1. Uses microMS directly

Vendored unmodified under `microms/`, called at runtime.

| what | where |
|---|---|
| `.xeo` header, footer, `<PlateSpot .../>` line | `brukerMapper.writeXEO` |
| `.xeo` reading | `brukerMapper.loadXEO`, `lines[13:-12]` |
| named MTP grid fractions | `brukerMapper.MTPMapX` / `MTPMapY` |
| motor → plate-fraction map | `brukerMapper.loadStagePoints`, `motorToMTP` |
| point-based similarity registration | `coordinateMapper._PBSR` |
| `blob` objects passed to the writer | `ImageUtilities.blob` |

`flex_mapper.flexMapper` adds only the five concrete methods `brukerMapper`
requires. The public release ships no timsTOF fleX mapper.

## 2. Ported to match microMS exactly

Reimplemented here, but reproducing microMS's algorithm rather than an
alternative.

**Circular packing** — `blobList.circularPackPoints`. Shot count follows bead
radius:

```
maxR = maxSpots * spacing / 2pi - offset
minR = minSpots * spacing / 2pi - offset

radius > maxR  ->  maxSpots
radius < minR  ->  minSpots        (spacing ignored)
otherwise      ->  floor(2pi * (radius + offset) / spacing)
```

Verified against the source: a 36 or 60 µm bead gets 4 shots, a 90 µm gets 6, a
165 µm gets 10. Setting `max-spots == min-spots` restores a fixed count, and
`dynamic-spots: False` uses `laser-shot-angles` verbatim.

**Distance filtering** — `blobList.distanceFilter`. Same rule: a blob fails if
any neighbour is closer than the cutoff, and both members of a too-close pair
fail. microMS bins the area into subregions for speed; a cKDTree query gives the
same answer and also yields the nearest-neighbour distance, which is kept for
the CSV.

**Effective radius and circularity** — `sqrt(area/pi)` and `4*pi*area/perimeter^2`,
the same definitions. The *values* differ because the mask differs; see §3.

**Filter ordering** — loose detection, then distance filtering, then shape.
Deliberate, and the reason is in the guide: filtering shape first deletes the
debris beside a bead and the bead then falsely passes isolation.

## 3. Deliberate divergence

### Blob finding — the main one

| | microMS | here |
|---|---|---|
| method | threshold a colour channel, group adjacent pixels above it | subtract a large-kernel median, then connected components |
| assumes | bright objects on a dark, roughly uniform background | drifting background, very low contrast |

microMS's approach suits fluorescence images, where cells are bright on black.
It does not survive a brightfield scan of a matrix-coated slide: bead-to-matrix
contrast is under 10 grey levels and both illumination and matrix thickness
drift across the field. A global threshold sweep on the reference scan returned
**28,495 objects**, almost all mounting-hardware speckle.

Flat-field subtraction removes the drift first, so the threshold only has to
separate beads from local background. `detection.method: "blob"` still selects
OpenCV's `SimpleBlobDetector` global threshold sweep, closest to microMS's
approach, if the sample ever warrants it.

Consequence: measured radius and circularity are not numerically comparable to
microMS's on the same image, even though the formulas match.

### Clump screening — added, not in microMS

microMS relies on circularity to reject unresolved objects (its "blob 2"). At
this contrast that is not enough — touching beads merge into components that
still read as fairly round, and the guide itself notes pixelated perimeters make
circularity read *higher* than the true value. Three tests are used instead:
outline aspect ratio, hull solidity, and counting distance-transform cores. Any
one firing marks the object.

A clump is **marked, not deleted**, so a nearby single bead still fails isolation
against it.

### Shot placement reference

microMS's circular packing places targets at `radius + offset`, scaling with the
measured blob. This does the same by default (`distance-reference: edge`).

Added here: because that scaling derives shot distance from the measured radius,
the crater-overlap check compares that radius against itself and can never fail.
A mis-measured bead is therefore placed wrongly and silently. `suspect_radius`
counts accepted beads whose diameter differs from `bead-diameter` by more than
`suspect-diameter-tolerance`, which must be tighter than
`bead-diameter-tolerance` or nothing can be both accepted and suspect.

`distance-reference: center` is the alternative: a fixed distance regardless of
measured size. It has the opposite failure mode — a mis-measured bead fails the
clearance check loudly instead of being placed quietly.

### Image display

microMS decimates plain TIFFs to keep large scans workable. This does the same
thing more aggressively: `show_pyramid` keeps the scan at several resolutions
and, on every zoom, crops the visible rectangle out of the level that matches
the screen.

Choosing a level alone is not enough — matplotlib processes the whole array on
every draw and clips what falls outside the axes, so a deep zoom costs the same
as the full view. Cropping is what makes it fast. On the 8000x6039 reference
scan this took eight zoom steps from 26.9 s to 6.3 s, and a single deep-zoom
redraw to 0.14 s.

`extent` always tracks the crop in full-resolution coordinates, so clicks,
fiducials and bead positions are unaffected by which level is showing.

### Path ordering

microMS uses a TSP approximation bounded to three minutes. This uses serpentine
ordering, which the flexImaging manual notes makes the stage travel roughly
twice as far. Negligible at 36–156 positions; it would matter at thousands.

### Target lists

microMS keeps up to ten, spawning children through filtering and patterning.
This keeps one list plus a manual-override file, since the goal is a single
homogeneous population rather than sorted subgroups.

### Registration reporting

microMS highlights the worst fiducial. This does that too, and adds RMS,
per-fiducial residuals, recovered µm/px, a reflection flag, leave-one-out cross
validation at ≥4 fiducials, and a degenerate-geometry check for duplicated or
collinear fiducials — which fit perfectly and register badly.

## 4. Not implemented

Present in microMS, not needed here: multichannel and fluorescence filtering,
the population histogram with high/low-pass filters and "pick extremes",
rectangular and hexagonal packing, polygon ROIs, image decimation and openslide,
direct instrument control, and the other coordinate mappers.

## 5. Added here, no microMS equivalent

The `.run` autoXecute writer, the 400-position `.xeo` split (that lives in
`solarixMapper`, not `brukerMapper`), the manual selection window with box tools
and per-category visibility, the QC overlay and shot-placement zoom, the TIFF
converter, `doctor`, and the test suite.
