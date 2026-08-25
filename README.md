# microMS_beadtargeting

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

Algorithms and interactions are reimplemented independently. Only the `.xeo`
header and footer constants are reproduced, because interoperability requires
the exact strings; they are marked `FORMAT SPEC` in the source. microMS carries
an Illinois copyright with no explicit licence, so nothing else is copied.

```
pip install -r requirements.txt
```

Repository layout:

```
microMS_beadtargeting.py   the pipeline, with CONFIG at the top
tests/test_pipeline.py     pytest suite (22 tests, no instrument needed)
CLAUDE.md                  constraints and invariants for Claude Code
ATTRIBUTION.md             microMS citation and the copy boundary
MANUAL_COVERAGE.md         what we take from the microMS guide, and what we skip
DIVERGENCE.md              every point where this differs from microMS
```

Python 3.10 or newer, and four packages. `requirements.txt` lists which command
needs which — `check` needs no image libraries at all.

---

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

## 1. Scan the matrix-coated slide

Sample face up. Save as **TIFF**. JPEG is rejected.

If the scanner only gives you JPEG or PNG:

```
python microMS_beadtargeting.py convert scan.jpg
```

Colour input is converted to greyscale, 16-bit is scaled to 8-bit, and the
result is written next to the source — or into the script folder if the source
directory is read-only.

| flag | effect |
|---|---|
| `--microms` | invert and add the `_c1` suffix, for **microMS** |
| `--invert` | invert only |
| `--downsample=0.5` | halve both dimensions |

**Do not feed a `--microms` file to this pipeline.** microMS thresholds for
bright objects so its input must be pre-inverted; this pipeline inverts
internally at detection time. Inverting twice puts you back where you started
with no error message. Convert with no flags for this tool, `--microms` for
hers.

`--downsample` changes µm/px by the same factor. The registration will recover
the new scale correctly, but any pixel figure you wrote down by hand is now
wrong.

Note the scanner's µm/px. You never type it in, but you will compare it against
the scale the registration recovers, and a mismatch means something is wrong.

## 2. Edit `CONFIG`

At the top of `microMS_beadtargeting.py`. Set `input["scan"]`. Under
`detection`, keep `invert: True` if beads are darker than the matrix background
— they usually are; thresholding is for *bright* objects, so without inverting
it finds background instead of beads. Set `detection["roi"]` to the slide you
want, in pixels, or dark mounting hardware inverts to bright and floods the
object list.

Check `bead-diameter`, `min-bead-separation` and `shot-placement`. Leave
`output["write-xeo"]` False until MTP calibration is measured.

**There is no software travel-limit check.** An earlier version had a
`slide-bounds` window with values I invented; on a real stage every shot fell
outside it and the target list came out empty with no error. The stage enforces
its own limits in hardware, so that check is gone rather than re-guessed.

## 3. Record fiducial stage coordinates

On the instrument, drive to each scribed mark and write down the stage reading
in µm. This is manual and it sets your accuracy ceiling.

## 4. `python microMS_beadtargeting.py pick`

Opens the scan. Coordinate entry is **in the window**, not the terminal.

| action | effect |
|---|---|
| right-click on the image | set the pending pixel (gold X) |
| type into stage x / stage y | the measured stage reading |
| **Add fiducial** | commit the pair |
| **Remove nearest** | delete the fiducial nearest the last right-click |
| **Reset** | clear the list |
| close the window | write `FIDUCIALS` into the source file |

Enter in the stage y box also commits, so entry can be keyboard-only.

Both interactive windows zoom the same way:

| action | effect |
|---|---|
| scroll wheel | zoom about the cursor |
| middle-drag | pan |
| `+` / `-` | zoom about the centre |
| `f` | fit — back to the whole image |
| **Zoom +** / **Zoom -** / **Fit** | the same, as buttons |

The matplotlib toolbar's own zoom is deliberately not wired: it binds left-drag,
which is already the box selector in `select`, and entering toolbar zoom mode
silently swallows the clicks that add or toggle beads. Scroll and middle-drag
stay clear of both mouse buttons.


The worst-fitting fiducial is drawn red and live RMS sits in the title, so a
mistyped coordinate shows up while you are still in the window. Only the
`FIDUCIALS` block is rewritten; everything around it is untouched, and
registration is reported on close.

If matplotlib is on a non-interactive backend the command exits with a message
rather than opening nothing — check with:

```
python -c "import matplotlib; print(matplotlib.get_backend())"
```

`Agg` cannot open a window. `TkAgg`, `QtAgg` and `MacOSX` are fine.

## 5. `python microMS_beadtargeting.py check`

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

## 6. `python microMS_beadtargeting.py select` — bead manual selection

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

## 7. `python microMS_beadtargeting.py run`

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

Outputs `targets.csv` (stage µm, in firing order), `targets_overlay.png`,
`shot_placement_zoom.png`, and `targets_NNN.xeo` if MTP calibration is
populated.

`shot_placement_zoom.png` auto-centres on the densest patch of accepted beads at
3× magnification. It is the picture to check before committing an acquisition —
it shows whether shots actually land on clean matrix just off each bead edge.

The overlay is the fastest way to catch a bad detection before you are standing
at the instrument: green accepted, red rejected, filled blue shots, orange
dotted dropped shots.

## 8. Load into flexImaging / autoXecute

See the `.xeo` gate below.

## 9. Burn-mark validation

Fire the list on a sacrificial slide, rescan, measure the offset from intended
bead centres. Nothing upstream validates the physical chain — the transform can
be numerically perfect and still be off if stage calibration or fiducial
teaching is wrong. Do this before any real sample.

---

## Shot placement: edge vs. centre

`shot-placement.distance-reference` chooses how the shot radius is measured.

**`edge` (default)** — each bead's own measured radius plus `edge-offset`.
Adapts to real bead size. Shot-to-shot spacing then differs between beads of
different size. `min-radius` / `max-radius` clamp the measured radius so one
badly-measured bead cannot fling shots across the slide.

**`center`** — a fixed `laser-distance` from every bead centre regardless of
measured size. Uniform spacing; a smaller bead gets a larger real gap to its
edge.

Edge is the default because measured bead diameters on the current slides
cluster near 80 µm rather than the nominal 90, so a fixed 60 µm from centre
would sit ~20 µm off the edge rather than the intended 15.

---

## Registration notes

The transform is a **similarity** fit — uniform scale, rotation, optional
reflection, translation — not affine. An affine fit through exactly three
points is exactly determined and reports a residual of zero, which tells you
nothing about registration quality. The similarity fit still leaves a real one.

Shots are dropped **only on genuine crater overlap** or for falling outside
`slide-bounds`. Nothing is trimmed to satisfy a cosmetic clearance margin.

---

## Open gates

**`mtp_calibration` must be measured before any `.xeo` is usable.**
`UnitCoord_X/Y` in a `.xeo` are fractions of the plate, not microns, so there
is a second registration stacked on the first:

```
scan pixels --[fiducials]--> stage µm --[MTP calibration]--> plate fraction
```

Fit it from three named MTP grid positions: drive to each on the instrument and
record the stage µm, then pair each with its `UnitCoord` read out of a **real
`.xeo` exported from flexImaging**. Spread them across the plate. While the
list is empty the CSV is still written and the `.xeo` is skipped, rather than
emitting a plausible-looking file that fires in the wrong place.

The fit reports its own residual. A similarity fit assumes `UnitCoord_X` and
`UnitCoord_Y` share one scale; if they are instead normalised independently to
plate width and height, a non-square plate breaks that assumption and the
residual goes large. Three points cannot distinguish the two cases any other
way, so it warns rather than absorbing it.

**The `.xeo` header is UNVERIFIED.** It declares `MTP Slide Adapter II`. Diff a
genuine flexImaging export against a file from this writer before acquiring — a
header mismatch is the most likely failure and a diff makes it obvious. Do not
assume the Slide Adapter II uses the same A1/C20 position naming as a standard
384-spot MTP; open a real export and look.

**`focal-spot-um: 10` and `beam-scan-um: 20` are placeholders.** They affect
only the crater-overlap check, so wrong values silently change which shots are
dropped. They do not move any shot. Confirm with the instrument operator.

**Confirm whether autoXecute random walk is enabled** in the acquisition
method. It moves the actual firing position away from the coordinate you gave
it, and at 15 µm off the bead edge there is not much room.

---

## Diagnosing a problem

Start here when something will not run:

```
python microMS_beadtargeting.py doctor
```

It checks Python version, all five packages, the matplotlib backend, config
validity, fiducial and MTP counts, and whether the scan exists and actually
decodes — then prints a pass/fail line. It catches the JPEG-instead-of-TIFF
case and the `Agg`-backend case, which between them cause most startup
failures.

Add `-v` to any command for a timed trace:

```
python microMS_beadtargeting.py run -v
```

Every line is flushed immediately, so a crash mid-run leaves the trace intact
rather than losing it to buffering. If detection returns nothing, the run
prints the four settings worth checking, in order.

## Commands

```
python microMS_beadtargeting.py doctor    # environment check
python microMS_beadtargeting.py convert   # image -> TIFF
python microMS_beadtargeting.py pick      # click fiducials -> YAML
python microMS_beadtargeting.py select    # bead manual selection
python microMS_beadtargeting.py check     # registration quality only
python microMS_beadtargeting.py run       # detect, filter, shoot, export
python microMS_beadtargeting.py selftest  # synthetic end-to-end test

# -v / --verbose works on all of them

pytest                                    # full regression suite
```
