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

**microMS source is vendored, unmodified, under `microms/`** — the minimum
subset needed to write `.xeo` files with its own `brukerMapper.writeXEO`:

    CoordinateMappers/  brukerMapper, coordinateMapper
    ImageUtilities/     blob
    GUICanvases/        GUIConstants

`flex_mapper.py` supplies the concrete `flexMapper` subclass microMS does not
ship, since the public release has no timsTOF fleX mapper. It provides only
`motorCoordFilename`, `isValidMotorCoord`, `extractMotorPoint`,
`loadInstrumentFile` and `saveInstrumentFile`. The `.xeo` header and footer,
the MTP grid fractions, the motor-to-plate-fraction map and the point-based
similarity registration all come from microMS unchanged.

The detection, clump screening, filtering, shot placement, `.run` writer and
400-position split are original.

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
