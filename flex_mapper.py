"""
flex_mapper.py -- timsTOF fleX coordinate mapper.

A concrete implementation of microMS's `brukerMapper` abstract base
class. This exists so that `.xeo` geometry files are written by
microMS's own `writeXEO`, not by a reimplementation of it.

microMS ships mappers for the ultrafleXtreme, solariX, oMALDI and a
Zaber stage. There is no timsTOF fleX mapper in the public release,
so this supplies the small set of concrete methods `brukerMapper`
requires:

    motorCoordFilename    stage coordinates of named MTP positions
    isValidMotorCoord     is a string a valid stage coordinate
    extractMotorPoint     parse one
    loadInstrumentFile    read a .xeo back
    saveInstrumentFile    write one

Everything else -- the .xeo header and footer, the MTP grid fractions,
the motor-to-plate-fraction map and the point-based similarity
registration -- comes from `brukerMapper` and `coordinateMapper`
unchanged.

microMS is MIT licensed, copyright (c) 2016 troycomi. See
microms/LICENSE.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "microms"))

from CoordinateMappers import brukerMapper      # noqa: E402
from ImageUtilities import blob                 # noqa: E402


class flexMapper(brukerMapper.brukerMapper):
    """Coordinate mapper for the Bruker timsTOF fleX."""

    def __init__(self, coord_file=None):
        # brukerMapper.loadStagePoints() reads this during __init__,
        # so it has to be set first.
        self.motorCoordFilename = coord_file or os.path.join(
            HERE, "flexCoords.txt")
        self.instrumentExtension = ".xeo"
        self.instrumentName = "timsTOF fleX"
        super().__init__()

        # True on every Bruker mapper microMS ships: stage y counts
        # opposite to image y.
        self.reflectCoordinates = True

    # -- required concrete methods ------------------------------------

    def isValidMotorCoord(self, instr):
        """Stage coordinates are two numbers separated by a space."""
        if instr is None or " " not in instr:
            return False
        try:
            a, b = instr.split(" ")[:2]
            float(a)
            float(b)
            return True
        except ValueError:
            return False

    def extractMotorPoint(self, inStr):
        if not self.isValidMotorCoord(inStr):
            return None
        a, b = inStr.split(" ")[:2]
        return (int(float(a)), int(float(b)))

    def loadInstrumentFile(self, filename):
        return self.loadXEO(filename)

    def saveInstrumentFile(self, filename, blobs):
        self.writeXEO(filename, blobs)


def write_coord_file(path, calibration):
    """
    Write the tab-delimited file brukerMapper.loadStagePoints reads.

    calibration: [{"name": "C5", "x_um": ..., "y_um": ...}, ...]
    """
    with open(path, "w") as fh:
        for c in calibration:
            fh.write(f"{c['name']}\t{int(round(c['x_um']))}"
                     f"\t{int(round(c['y_um']))}\n")
    return path


def make_blob(x_px, y_px, radius=1.0, group=None):
    """A microMS blob at a pixel position."""
    return blob.blob(x=x_px, y=y_px, radius=radius, circularity=1.0,
                     group=group)
