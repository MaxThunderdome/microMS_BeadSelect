# Attribution

This project follows microMS:

> Comi TJ, Neumann EK, Do TD, Sweedler JV. *microMS: A Python Platform for
> Image-Guided Mass Spectrometry Profiling.* J. Am. Soc. Mass Spectrom. 2017,
> 28(9), 1919-1928. DOI 10.1007/s13361-017-1704-1

Cite this paper in any publication using this software.

## Licence

**microMS is MIT licensed** — `MIT (c) 2016 troycomi`,
<https://github.com/troycomi/microMS>.

An earlier version of this file claimed microMS carried an Illinois copyright
with no explicit licence, and the code was written to avoid reuse on that
basis. That was wrong. MIT permits reuse with attribution, so the constraint
does not apply.

What is reproduced here from microMS, all under MIT with the copyright notice
retained:

- the `.xeo` header and footer constants, from `brukerMapper.py`
- the `<PlateSpot .../>` line format and the `x_<X>y_<Y>` position naming
- `MTP_MAP_X` / `MTP_MAP_Y`, the named MTP grid fractions
- the flexImaging spot-list format, from `flexImagingSolarix.py`

The detection, filtering, clump screening and shot placement here are original.

## What is NOT in microMS

The public release ships four coordinate mappers: ultrafleXtreme, solariX
(plus a flexImaging variant), oMALDI and a Zaber stage. **There is no timsTOF
fleX mapper.** Anyone running microMS against a fleX is using a locally
written one.

That matters less than it sounds. `brukerMapper` handles the whole `.xeo`
path, and the only instrument-specific pieces are:

1. `motorCoordFilename` — the stage coordinates of named MTP positions
2. `isValidMotorCoord` / `extractMotorPoint` — how the stage reports a
   coordinate as text
3. `reflectCoordinates` — `True` on every Bruker mapper shipped

Item 1 is a measurement anyone can take at the instrument. Items 2 and 3 only
matter inside microMS's own GUI, not here.
