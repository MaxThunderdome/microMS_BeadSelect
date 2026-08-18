# Attribution

The workflow ordering, the point-based similarity registration, the
nearest-neighbour distance filter and the fiducial click-training interaction
in this project follow microMS:

> Comi TJ, Neumann EK, Do TD, Sweedler JV. *microMS: A Python Platform for
> Image-Guided Mass Spectrometry Profiling.* J. Am. Soc. Mass Spectrom. 2017,
> 28(9), 1919–1928. DOI 10.1007/s13361-017-1704-1

Cite this paper in any publication using this software.

## What is and is not copied

microMS carries a University of Illinois copyright with **no explicit
licence**. Accordingly:

- All algorithms and interactions here are **reimplemented independently**
  from the published description and observed behaviour. No microMS source is
  copied.
- The only exception is the `.xeo` header and footer constants, which are
  reproduced because interoperability requires the exact strings. They are
  marked `FORMAT SPEC` in `microMS_beadtargeting.py`.

Anyone extending this code should keep that boundary. If you need microMS
behaviour that is not here, describe it and reimplement it — do not paste.

## Licence

Not yet chosen. Settle this before publishing the repository, and note that
the microMS relationship above constrains what can be claimed about
derivation.
