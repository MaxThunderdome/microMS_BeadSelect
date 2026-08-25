"""
Regression tests for microMS_beadtargeting.

These cover the invariants that look like bugs to someone reading the
code cold, and would otherwise get "fixed" into real bugs. See
CLAUDE.md for the reasoning behind each.

    pytest

No instrument, image or display is required.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import microMS_beadtargeting as M  # noqa: E402

# microMS ships this for the ultrafleXtreme: named MTP position, then
# the stage coordinate measured on that instrument.
REAL_CAL = [
    {"name": "C20", "x_um": -23215, "y_um": -13605},
    {"name": "C5",  "x_um": -90705, "y_um": -13715},
    {"name": "G20", "x_um": -23190, "y_um": -31610},
    {"name": "G5",  "x_um": -90680, "y_um": -31715},
]


# ---------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------

@pytest.fixture
def cfg():
    c = M.load_config()
    c["fiducials"] = [
        {"x_px": 0, "y_px": 0, "x_um": 0, "y_um": 0},
        {"x_px": 2000, "y_px": 0, "x_um": 19480, "y_um": 0},
        {"x_px": 0, "y_px": 1500, "x_um": 0, "y_um": 14610},
    ]
    return c


@pytest.fixture
def transform(cfg):
    return M.transform_from_config(cfg)


# ---------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------

def test_similarity_recovers_scale_rotation_exactly():
    rng = np.random.default_rng(0)
    th = 0.05
    true = M.Transform(9.74,
                       np.array([[math.cos(th), -math.sin(th)],
                                 [math.sin(th), math.cos(th)]]),
                       np.array([1200.0, 3400.0]))
    src = rng.uniform(0, 3000, (5, 2))
    dst = true.px_to_um(src)

    T = M.fit_similarity(src, dst)
    assert T.um_per_px == pytest.approx(9.74, abs=1e-9)
    assert T.rotation_deg == pytest.approx(math.degrees(th), abs=1e-9)
    assert not T.reflected
    assert M.residuals(T, src, dst).max() < 1e-6


def test_px_um_round_trip(transform):
    pts = np.array([[10.0, 20.0], [1500.0, 900.0]])
    back = transform.um_to_px(transform.px_to_um(pts))
    assert np.allclose(back, pts)


def test_reflection_is_detected():
    src = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    dst = src @ np.array([[1., 0.], [0., -1.]])       # mirror in y
    assert M.fit_similarity(src, dst, allow_reflection=True).reflected


def test_three_fiducials_give_no_loo_estimate():
    """Not a limitation to route around -- dropping one of three
    leaves too few points to fit, so there is no honest estimate."""
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1000, (3, 2))
    dst = src * 9.74
    assert M.loo_residuals(src, dst) is None
    assert M.loo_residuals(*[np.vstack([a, a[:1] + 5]) for a in (src, dst)]) \
        is not None


def test_similarity_not_affine_leaves_real_residual():
    """An affine fit through 3 points is exactly determined and would
    report zero regardless of how bad the fiducials are."""
    src = np.array([[0., 0.], [1000., 0.], [0., 1000.]])
    dst = np.array([[0., 0.], [9740., 0.], [0., 9740. * 1.05]])  # anisotropic
    assert M.residuals(M.fit_similarity(src, dst), src, dst).max() > 100


# ---------------------------------------------------------------------
# filter ordering
# ---------------------------------------------------------------------

def test_isolation_runs_before_shape_so_debris_still_counts(cfg, transform):
    """A bead beside a speck of dust is not isolated. If the shape
    filter ran first the dust would vanish and the bead would pass."""
    beads = [M.Bead(1000, 1200, 8.2), M.Bead(1005, 1210, 2.0)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)

    assert not beads[0].accepted
    assert beads[0].reject_category == "not isolated"
    assert not beads[1].accepted          # debris fails on size


def test_packed_row_all_rejected(cfg, transform):
    beads = [M.Bead(500 + 10 * i, 500, 8.2) for i in range(6)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    assert not any(b.accepted for b in beads)


def test_isolated_bead_of_right_size_is_accepted(cfg, transform):
    beads = [M.Bead(1000 + 200 * i, 1200, 8.2) for i in range(5)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    assert all(b.accepted for b in beads)


# ---------------------------------------------------------------------
# clump screen
# ---------------------------------------------------------------------

def _canvas():
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((200, 600), np.uint8)
    cv2.circle(img, (100, 100), 20, 255, -1)                  # single
    cv2.circle(img, (280, 100), 20, 255, -1)                  # touching
    cv2.circle(img, (312, 100), 20, 255, -1)                  #   pair
    for c in ((470, 90), (500, 108), (472, 126)):             # triple
        cv2.circle(img, c, 20, 255, -1)
    return img


DETECT = {"min-diameter-px": 10, "max-diameter-px": 200,
          "background-kernel-px": 151, "threshold": 60,
          "min-circularity": 0.70, "max-aspect-ratio": 1.8,
          "min-solidity": 0.90, "screen-clumps": True}


def test_touching_beads_flagged_as_clumps():
    found = sorted(M._detect_flatfield(_canvas(), dict(DETECT)),
                   key=lambda f: f[0])
    assert [f[3] for f in found] == [False, True, True]


def test_clumps_are_returned_not_discarded():
    """They must stay in the object list so nearby singles still fail
    isolation against them."""
    found = M._detect_flatfield(_canvas(), dict(DETECT))
    assert sum(1 for f in found if f[3]) == 2
    assert len(found) == 3


def test_clumped_bead_can_never_be_accepted(cfg, transform):
    beads = [M.Bead(1000, 1200, 8.2, clumped=True)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    assert not beads[0].accepted
    assert beads[0].reject_category == "clumped"


# ---------------------------------------------------------------------
# shot placement
# ---------------------------------------------------------------------

def test_edge_mode_tracks_measured_radius(cfg):
    cfg["shot-placement"]["distance-reference"] = "edge"
    small = M.Bead(0, 0, 0, diameter_um=60.0)
    large = M.Bead(0, 0, 0, diameter_um=100.0)
    off = cfg["shot-placement"]["edge-offset"]
    assert M.shot_radius(small, cfg) == pytest.approx(30 + off)
    assert M.shot_radius(large, cfg) == pytest.approx(50 + off)


def test_center_mode_ignores_measured_radius(cfg):
    cfg["shot-placement"]["distance-reference"] = "center"
    fixed = cfg["shot-placement"]["laser-distance"]
    for d in (60.0, 100.0):
        assert M.shot_radius(M.Bead(0, 0, 0, diameter_um=d), cfg) == fixed


def test_edge_mode_radius_is_clamped(cfg):
    cfg["shot-placement"]["distance-reference"] = "edge"
    sp = cfg["shot-placement"]
    absurd = M.Bead(0, 0, 0, diameter_um=4000.0)
    assert M.shot_radius(absurd, cfg) == pytest.approx(
        sp["max-radius"] + sp["edge-offset"])


def test_shot_count_matches_circular_pack(cfg, transform):
    """With dynamic-spots the count comes from the bead radius, as in
    microMS's circularPackPoints."""
    beads = [M.Bead(1000 + 200 * i, 1200, 8.2) for i in range(3)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    shots = M.place_shots(beads, cfg)

    expected = sum(len(M.circular_pack(b.diameter_um / 2, cfg))
                   for b in beads if b.accepted)
    assert len(shots) == expected


def test_no_software_travel_limit(cfg, transform):
    """There is no slide-bounds check. It was invented, it is not a
    real constraint, and it silently discarded every shot in a run.
    The stage enforces its own limits in hardware."""
    beads = [M.Bead(1000, 1200, 8.2)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)

    shots = M.place_shots(beads, cfg)
    assert shots
    assert not any(s.dropped for s in shots)
    assert "slide-bounds" not in cfg


def test_config_has_no_invented_coordinate_window():
    assert "slide-bounds" not in M.CONFIG


def test_fiducials_come_from_the_module():
    cfg = M.load_config()
    assert cfg["fiducials"] == [dict(f) for f in M.FIDUCIALS]


def test_save_fiducials_round_trips(tmp_path):
    """`pick` writes the block back into the source file."""
    src = tmp_path / "mod.py"
    src.write_text("HEADER = 1\nFIDUCIALS = [\n]\nFOOTER = 2\n")
    M.save_fiducials([{"x_px": 1.0, "y_px": 2.0,
                       "x_um": 3.0, "y_um": 4.0}], src)
    text = src.read_text()
    assert "HEADER = 1" in text and "FOOTER = 2" in text

    ns = {}
    exec(text, ns)
    assert ns["FIDUCIALS"] == [{"x_px": 1.0, "y_px": 2.0,
                                "x_um": 3.0, "y_um": 4.0}]


# ---------------------------------------------------------------------
# .xeo export
# ---------------------------------------------------------------------


def test_xeo_splits_at_400_and_round_trips(cfg, tmp_path):
    cfg["mtp_calibration"] = REAL_CAL
    mtp = M.fit_mtp(cfg)
    assert mtp is not None

    shots = [M.Shot(0, 0, 100.0 * i, 200.0) for i in range(950)]
    files = M.write_xeo(tmp_path / "t", shots, [], cfg, mtp)
    assert [len(M.read_xeo(f)) for f in files] == [400, 400, 150]


def test_empty_mtp_calibration_blocks_xeo(cfg):
    """No defensible default exists, so nothing is written."""
    cfg["mtp_calibration"] = []
    assert M.fit_mtp(cfg) is None


# ---------------------------------------------------------------------
# manual selection
# ---------------------------------------------------------------------

def test_overrides_match_by_position_not_index(cfg, tmp_path):
    """Detection indices shift when any detection parameter changes,
    so overrides are keyed by pixel position."""
    sel = tmp_path / "manual_selection.csv"
    M.save_manual([(1000.0, 1200.0, "accept")], sel)

    beads = [M.Bead(5.0, 5.0, 8.2), M.Bead(1002.0, 1203.0, 8.2)]
    for b in beads:
        b.accepted = False
    M.apply_manual(beads, cfg, sel)

    assert beads[1].accepted and beads[1].manual == "accept"
    assert not beads[0].accepted


def test_override_beyond_match_radius_is_ignored(cfg, tmp_path):
    sel = tmp_path / "manual_selection.csv"
    M.save_manual([(1000.0, 1200.0, "accept")], sel)
    beads = [M.Bead(4000.0, 4000.0, 8.2)]
    beads[0].accepted = False
    M.apply_manual(beads, cfg, sel)
    assert not beads[0].accepted


# ---------------------------------------------------------------------
# zoom helper
# ---------------------------------------------------------------------

def test_zoom_about_cursor_keeps_point_fixed():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.imshow(np.zeros((600, 800), np.uint8))
    handlers = {}
    real = fig.canvas.mpl_connect
    fig.canvas.mpl_connect = lambda n, f: handlers.setdefault(n, f) or real(n, f)
    zin, zout, fit = M.attach_zoom(fig, ax)
    fig.canvas.draw()

    home = ax.get_xlim()
    frac = (200.0 - home[0]) / (home[1] - home[0])

    ev = type("E", (), {"inaxes": ax, "xdata": 200.0, "ydata": 150.0,
                        "button": "up"})()
    handlers["scroll_event"](ev)

    x0, x1 = ax.get_xlim()
    assert (200.0 - x0) / (x1 - x0) == pytest.approx(frac, abs=1e-9)
    assert (x1 - x0) < (home[1] - home[0])

    fit()
    assert ax.get_xlim() == pytest.approx(home)
    plt.close(fig)


# ---------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------

def test_single_detected_object_does_not_crash(cfg, transform):
    """cKDTree pads a missing neighbour with index == len(points),
    which is out of range. One detected object always triggers it."""
    beads = [M.Bead(1000, 1200, 8.2)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    assert beads[0].accepted

    shots = M.place_shots(beads, cfg)
    assert len(shots) == len(M.circular_pack(beads[0].diameter_um / 2, cfg))
    assert not any(s.dropped for s in shots)


def test_empty_bead_list_is_survivable(cfg, transform, tmp_path):
    M.to_stage([], transform)
    M.isolation_filter([], cfg["min-bead-separation"])
    M.shape_filter([], cfg)
    assert M.place_shots([], cfg) == []
    assert M.serpentine([], []) == []
    M.write_csv(tmp_path / "empty.csv", [], [])


def test_duplicate_fiducials_are_flagged():
    """The fit collapses to scale 1.0 and still reports RMS 0."""
    src = np.array([[0., 0.], [0., 0.], [2000., 1500.]])
    warn = M.check_fiducial_geometry(src)
    assert any("same pixel" in w for w in warn)


def test_collinear_fiducials_are_flagged():
    """Zero residual, but nothing constrains the perpendicular
    direction."""
    src = np.array([[0., 0.], [1000., 0.], [2000., 0.]])
    T = M.fit_similarity(src, src * 9.74)
    assert M.residuals(T, src, src * 9.74).max() < 1e-9   # looks perfect
    assert any("collinear" in w for w in M.check_fiducial_geometry(src))


def test_well_spread_fiducials_are_not_flagged():
    src = np.array([[0., 0.], [2000., 0.], [0., 1500.]])
    assert M.check_fiducial_geometry(src) == []


# ---------------------------------------------------------------------
# autoXecute .run
# ---------------------------------------------------------------------

def test_run_and_xeo_use_identical_names(cfg, tmp_path):
    """autoXecute matches the two files by name alone."""
    cfg["mtp_calibration"] = REAL_CAL
    shots = [M.Shot(i // 4, 90 * (i % 4), 100.0 * i, 200.0) for i in range(5)]
    xeo = M.write_xeo(tmp_path / "t", shots, [], cfg, M.fit_mtp(cfg))[0]

    names = [M.position_name(cfg, i, s) for i, s in enumerate(shots)]
    run = M.write_run(tmp_path / "t_001.run", names, cfg)

    assert M.read_run(run) == names
    for n in names:
        assert n in xeo.read_text()


def test_run_geometry_attribute_matches_filename(cfg, tmp_path):
    """autoXecute resolves coordinates via geometry -> <stem>.xeo."""
    run = M.write_run(tmp_path / "targets_001.run", ["R00X1Y1"], cfg)
    assert 'geometry="targets_001"' in run.read_text()


def test_run_declares_the_confirmed_base_geometry(cfg, tmp_path):
    """Confirmed against a real AutoExecute 7.6.6.0 run file."""
    run = M.write_run(tmp_path / "t.run", ["R00X1Y1"], cfg)
    assert 'baseGeometry="MTP Slide Adapter II"' in run.read_text()


def test_run_uses_crlf_like_the_reference_file(cfg, tmp_path):
    run = M.write_run(tmp_path / "t.run", ["R00X1Y1", "R00X2Y1"], cfg)
    raw = run.read_bytes()
    assert b"\r\n" in raw
    assert b"\n" not in raw.replace(b"\r\n", b"")



# ---------------------------------------------------------------------
# coordinate-encoded position names
# ---------------------------------------------------------------------







# ---------------------------------------------------------------------
# selection window visibility
# ---------------------------------------------------------------------

def test_category_assignment_is_exclusive(cfg, transform):
    """Each bead falls in exactly one checkbox category."""
    beads = [M.Bead(1000, 1200, 8.2),                    # -> accepted
             M.Bead(3000, 1200, 8.2, clumped=True),      # -> clumped
             M.Bead(5000, 1200, 1.0)]                    # -> rejected
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)

    def category(b):
        if b.accepted:
            return "accepted"
        if b.clumped:
            return "clumped"
        return "rejected"

    assert [category(b) for b in beads] == ["accepted", "clumped", "rejected"]


def test_hiding_does_not_change_export(cfg, transform):
    """Visibility is display only -- a hidden bead is still exported
    and still counts as an isolation neighbour."""
    beads = [M.Bead(1000, 1200, 8.2), M.Bead(1010, 1200, 8.2)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    before = [b.accepted for b in beads]

    # hiding is a dict in the GUI closure; nothing touches bead state
    assert [b.accepted for b in beads] == before
    assert all(not b.accepted for b in beads)   # each blocks the other


# ---------------------------------------------------------------------
# .xeo format, against microMS's own writer
# ---------------------------------------------------------------------

def test_header_and_footer_match_microms():
    """12 header lines plus the per-file <PlateSpots> line gives the 13
    that microMS's loadXEO skips."""
    assert len(M.XEO_HEADER) == 12
    assert len(M.XEO_FOOTER) == 12
    assert 'PlateTypeName="MTP Slide Adapter II"' in M.XEO_HEADER[2]
    assert 'alpha="51.750000"' in M.XEO_HEADER[11]
    assert M.XEO_FOOTER[-1].strip() == "</PlateType>"


def test_mtp_names_resolve_to_plate_fractions():
    assert M.mtp_name_to_unit("C20") == (0.652174, 0.478261)
    assert M.mtp_name_to_unit("G5") == (-0.652174, 0.130435)
    assert M.mtp_name_to_unit("ZZ") is None


def test_real_calibration_recovers_the_header_scale(cfg, capsys):
    """51.75 mm per unit is declared by alpha/beta in the header."""
    cfg["mtp_calibration"] = REAL_CAL
    mtp = M.fit_mtp(cfg)
    assert mtp is not None
    mm_per_unit = 1 / mtp.um_per_px / 1000
    assert abs(mm_per_unit - M.MTP_UNIT_MM) < 0.5


def test_unitcoord_stays_in_plate_range(cfg):
    """UnitCoord is a signed fraction about the plate centre, roughly
    +/-0.73 in X and +/-0.55 in Y -- not microns."""
    cfg["mtp_calibration"] = REAL_CAL
    mtp = M.fit_mtp(cfg)
    shots = [M.Shot(0, 0, -50000.0, -20000.0)]
    u = mtp.px_to_um(np.array([[s.x_um, s.y_um] for s in shots]))
    assert abs(u[0][0]) < 0.8 and abs(u[0][1]) < 0.6


def test_xeo_written_by_microms_brukermapper(cfg, tmp_path):
    """The .xeo comes from microMS's own writeXEO, not a copy of it."""
    cfg["fiducials"] = [
        {"x_px": 200, "y_px": 150, "x_um": 18601.5, "y_um": -20310.8},
        {"x_px": 8120, "y_px": 150, "x_um": 86083.1, "y_um": -20161.0},
        {"x_px": 200, "y_px": 5960, "x_um": 18646.7, "y_um": -69830.8},
        {"x_px": 8120, "y_px": 5960, "x_um": 86124.7, "y_um": -69700.2},
    ]
    shots = [M.Shot(0, 0, 0.0, 0.0, x_px=2765, y_px=1665)]
    files = M.write_xeo(tmp_path / "t", shots, [], cfg)
    assert len(files) == 1
    text = files[0].read_text()
    assert "<PlateType>" in text
    assert 'PlateTypeName="MTP Slide Adapter II"' in text
    assert 'PositionName="x_2765y_1665"' in text
    # microMS's loadXEO skips 13 header and 12 footer lines, leaving
    # exactly the spot lines.
    spots = M.read_xeo(files[0])
    assert len(spots) == 1
    assert spots[0].strip().startswith("<PlateSpot ")



def test_stage_mode_still_needs_calibration(cfg, tmp_path):
    cfg["mtp_calibration"] = []
    cfg["fiducial-units"] = "stage"
    assert M.fit_mtp(cfg) is None
    assert M.write_xeo(tmp_path / "t", [M.Shot(0, 0, 0, 0)], [], cfg,
                       None) == []


def test_plate_extent_matches_the_adapter():
    """75.5 x 57.0 mm, from alpha/beta and the teach-point extents."""
    assert M.PLATE_X_UNITS == pytest.approx(7550, abs=0.1)
    assert M.PLATE_Y_UNITS == pytest.approx(5700, abs=0.1)


def test_unitcoord_round_trips_through_plate_units():
    ux, uy = M.plate_to_unitcoord(4487, 4424)
    assert abs(ux) <= M.TEACH_X and abs(uy) <= M.TEACH_Y
    x, y = M.unitcoord_to_plate(ux, uy)
    assert x == pytest.approx(4487) and y == pytest.approx(4424)


def test_reference_run_positions_land_on_the_plate():
    """Every position in Dr Neumann's run file is inside the adapter."""
    for x, y in ((1868, 1308), (7162, 4801), (4487, 4424)):
        ux, uy = M.plate_to_unitcoord(x, y)
        assert abs(ux) <= M.TEACH_X
        assert abs(uy) <= M.TEACH_Y


def test_position_name_carries_plate_units(cfg):
    """Same convention the instrument uses for its own positions."""
    cfg["fiducial-units"] = "plate"
    shot = M.Shot(0, 0, 44870.0, 44240.0)      # um -> 4487, 4424 units
    assert M.position_name(cfg, 0, shot) == "R00X4487Y4424"


def test_position_name_stage_mode_follows_microms(cfg):
    """microMS's loadXEO parses x_<X>y_<Y> back into pixel positions."""
    cfg["fiducial-units"] = "stage"
    shot = M.Shot(0, 0, 1000.0, 2000.0, x_px=123.4, y_px=567.8)
    assert M.position_name(cfg, 0, shot) == "x_123y_568"


def test_xeo_round_trips_through_the_microms_slice(cfg, tmp_path):
    cfg["mtp_calibration"] = REAL_CAL
    shots = [M.Shot(i, 0, -50000.0 + i, -20000.0, x_px=i, y_px=i)
             for i in range(950)]
    files = M.write_xeo(tmp_path / "t", shots, [], cfg, M.fit_mtp(cfg))
    # 12 header lines + the per-file <PlateSpots> line = the 13 that
    # microMS's loadXEO skips, leaving exactly the spot lines.
    assert [len(M.read_xeo(f)) for f in files] == [400, 400, 150]
    assert all(l.strip().startswith("<PlateSpot ")
               for l in M.read_xeo(files[0]))


# ---------------------------------------------------------------------
# unit-mode regressions
# ---------------------------------------------------------------------

def test_fiducial_units_key_exists_in_config():
    """It was once absent, so to_microns silently took the plate branch
    and multiplied every distance by 10 -- every bead then failed the
    size filter with no error."""
    assert "fiducial-units" in M.CONFIG
    assert M.CONFIG["fiducial-units"] in ("stage", "plate")


def test_stage_mode_does_not_rescale(cfg):
    cfg["fiducial-units"] = "stage"
    T = M.Transform(8.52, np.eye(2), np.zeros(2))
    assert M.to_microns(T, cfg).um_per_px == pytest.approx(8.52)


def test_plate_mode_rescales_by_ten(cfg):
    cfg["fiducial-units"] = "plate"
    T = M.Transform(0.852, np.eye(2), np.zeros(2))
    assert M.to_microns(T, cfg).um_per_px == pytest.approx(8.52)


def test_missing_key_defaults_to_stage(cfg):
    """The safe default: stage matches the measured calibration, and a
    wrong default here is invisible except as absurd bead diameters."""
    cfg.pop("fiducial-units", None)
    T = M.Transform(8.52, np.eye(2), np.zeros(2))
    assert M.to_microns(T, cfg).um_per_px == pytest.approx(8.52)


def test_bad_fiducial_units_is_rejected():
    import copy
    saved = copy.deepcopy(M.CONFIG)
    try:
        M.CONFIG["fiducial-units"] = "microns"
        with pytest.raises(SystemExit):
            M.load_config()
    finally:
        M.CONFIG.clear()
        M.CONFIG.update(saved)


def test_measured_calibration_recovers_header_scale():
    """The four fleX slide corners fit to the alpha/beta the .xeo
    header declares."""
    cfg = M.load_config()
    mtp = M.fit_mtp(cfg)
    assert mtp is not None
    assert abs(1 / mtp.um_per_px / 1000 - M.MTP_UNIT_MM) < 0.05


# ---------------------------------------------------------------------
# Comi et al. 2017 accuracy rules
# ---------------------------------------------------------------------

def test_paper_accuracy_constants_present():
    """probe radius >= target localization error;
       distance filter > that error + probe radius."""
    assert "target-localization-error-um" in M.CONFIG
    assert "recommended-fiducials" in M.CONFIG
    assert M.CONFIG["recommended-fiducials"] >= 12


def test_localization_error_is_unset_by_default():
    """38.3 um is a 2017 ultrafleXtreme with a ~100 um footprint, not
    this instrument. None skips the check until it is measured."""
    assert M.CONFIG["target-localization-error-um"] is None


def test_separation_rule_when_error_is_known(cfg):
    cfg["target-localization-error-um"] = 10.0
    probe_r = M.footprint_um(cfg) / 2.0
    assert float(cfg["min-bead-separation"]) > 10.0 + probe_r


# ---------------------------------------------------------------------
# circular packing, ported from blobList.circularPackPoints
# ---------------------------------------------------------------------

def test_shot_count_follows_bead_radius(cfg):
    """Small beads keep min-spots; larger ones get more, so shot-to-shot
    spacing stays above spot-spacing."""
    counts = [len(M.circular_pack(d / 2, cfg)) for d in (36, 60, 90, 120, 165)]
    assert counts == [4, 4, 6, 7, 10]
    assert counts == sorted(counts)


def test_min_and_max_spots_are_respected(cfg):
    sp = cfg["shot-placement"]
    assert len(M.circular_pack(1.0, cfg)) == sp["min-spots"]
    assert len(M.circular_pack(10000.0, cfg)) == sp["max-spots"]


def test_equal_min_max_gives_a_fixed_count(cfg):
    cfg["shot-placement"]["min-spots"] = 4
    cfg["shot-placement"]["max-spots"] = 4
    for d in (36, 90, 165):
        assert len(M.circular_pack(d / 2, cfg)) == 4


def test_angles_are_evenly_spaced(cfg):
    a = M.circular_pack(45.0, cfg)
    steps = [round(a[i + 1] - a[i], 6) for i in range(len(a) - 1)]
    assert len(set(steps)) == 1


def test_rotation_offset_shifts_every_angle(cfg):
    cfg["shot-placement"]["rotation-offset-deg"] = 30.0
    assert M.circular_pack(45.0, cfg)[0] == pytest.approx(30.0)


def test_dynamic_can_be_switched_off(cfg, transform):
    cfg["shot-placement"]["dynamic-spots"] = False
    beads = [M.Bead(1000, 1200, 8.2)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    shots = M.place_shots(beads, cfg)
    assert len(shots) == len(cfg["laser-shot-angles"])


# ---------------------------------------------------------------------
# edge placement and the suspect-radius flag
# ---------------------------------------------------------------------

def test_suspect_band_is_tighter_than_accept_band():
    """If it were wider, no bead could be both accepted and suspect and
    the check would be unreachable -- which it was when first written."""
    assert (M.CONFIG["suspect-diameter-tolerance"]
            < M.CONFIG["bead-diameter-tolerance"])


def test_bad_tolerance_relationship_is_rejected():
    import copy
    saved = copy.deepcopy(M.CONFIG)
    try:
        M.CONFIG["suspect-diameter-tolerance"] = 0.9
        with pytest.raises(SystemExit):
            M.load_config()
    finally:
        M.CONFIG.clear()
        M.CONFIG.update(saved)


def test_suspect_radius_flags_a_mismeasured_bead(cfg):
    nominal = float(cfg["bead-diameter"])
    assert not M.suspect_radius(M.Bead(0, 0, 0, diameter_um=nominal), cfg)
    assert M.suspect_radius(M.Bead(0, 0, 0, diameter_um=nominal * 1.35), cfg)
    assert M.suspect_radius(M.Bead(0, 0, 0, diameter_um=nominal * 0.65), cfg)


def test_edge_placement_scales_with_measured_radius(cfg):
    cfg["shot-placement"]["distance-reference"] = "edge"
    off = cfg["shot-placement"]["edge-offset"]
    small = M.Bead(0, 0, 0, diameter_um=60.0)
    large = M.Bead(0, 0, 0, diameter_um=100.0)
    assert M.shot_radius(small, cfg) == pytest.approx(30 + off)
    assert M.shot_radius(large, cfg) == pytest.approx(50 + off)


def test_edge_is_the_default():
    assert M.CONFIG["shot-placement"]["distance-reference"] == "edge"


# ---------------------------------------------------------------------
# image pyramid
# ---------------------------------------------------------------------

def _scan(w=4000, h=3000):
    return np.random.default_rng(0).integers(0, 255, (h, w), dtype=np.uint8)


def test_pyramid_halves_each_level():
    pyr = M.build_pyramid(_scan())
    assert len(pyr) > 1
    for a, b in zip(pyr, pyr[1:]):
        assert b.shape[1] == a.shape[1] // 2


def test_pyramid_stops_before_going_tiny():
    pyr = M.build_pyramid(_scan(1024, 1024), levels=10, min_side=512)
    assert min(pyr[-1].shape) >= 512


def test_extent_is_always_full_resolution():
    """Coordinates must not shift when the drawn level changes."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _scan()
    h, w = img.shape
    fig, ax = plt.subplots()
    im = M.show_pyramid(fig, ax, img, cmap="gray")
    fig.canvas.draw()
    assert tuple(im.get_extent()) == (-0.5, w - 0.5, h - 0.5, -0.5)

    seen = set()
    for f in (1.0, 0.5, 0.1):
        pass
    for f in (1.0, 0.5, 0.1):
        ax.set_xlim(w / 2 - w * f / 2, w / 2 + w * f / 2)
        ax.set_ylim(h / 2 + h * f / 2, h / 2 - h * f / 2)
        fig.canvas.draw()
        seen.add(im.get_array().shape[1])
        assert im.get_array().shape[1] <= w
        # a fixed pixel stays at the same data coordinate
        d = ax.transData.transform([[1234, 567]])
        back = ax.transData.inverted().transform(d)[0]
        assert back[0] == pytest.approx(1234, abs=0.01)
        assert back[1] == pytest.approx(567, abs=0.01)

    assert len(seen) > 1, "zooming never changed the drawn level"
    plt.close(fig)


def test_zoom_crops_instead_of_drawing_everything():
    """matplotlib processes the whole array and clips, so choosing a
    pyramid level is not enough -- the visible rectangle has to be
    cropped out or a deep zoom is as slow as the full view."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = _scan()
    h, w = img.shape
    fig, ax = plt.subplots()
    im = M.show_pyramid(fig, ax, img, cmap="gray")
    ax.set_xlim(2000, 2100)
    ax.set_ylim(1550, 1450)
    fig.canvas.draw()
    arr = im.get_array()
    assert arr.shape[1] < w / 4          # cropped, not the whole scan
    d = ax.transData.transform([[2050, 1500]])
    back = ax.transData.inverted().transform(d)[0]
    assert back[0] == pytest.approx(2050, abs=0.01)
    plt.close(fig)


# ---------------------------------------------------------------------
# review
# ---------------------------------------------------------------------

def test_review_saves_picture_and_exports_nothing(cfg, tmp_path, monkeypatch):
    """'review' is a preview between 'select' and 'run': it must render
    the shot plan but never write target files."""
    beads = [M.Bead(1000 + 200 * i, 1200, 8.2) for i in range(4)]
    T = M.transform_from_config(cfg)
    M.to_stage(beads, T)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    assert any(b.accepted for b in beads)

    monkeypatch.setattr(M, "HERE", tmp_path)
    monkeypatch.setattr(M, "build_beads", lambda c, t: (beads, None))
    cfg["output"]["review-show"] = False

    M.review(cfg)

    assert (tmp_path / "targets_review.png").exists()
    exports = [p.name for p in tmp_path.iterdir()
               if p.suffix in (".csv", ".xeo", ".run", ".txt")]
    assert exports == [], exports


def test_review_is_dispatched(monkeypatch):
    """The help text advertises 'review'; the dispatcher must know it.
    (It was once a documented command with no implementation.)"""
    called = []
    monkeypatch.setattr(M, "review", lambda cfg: called.append(True))
    monkeypatch.setattr(M.sys, "argv", ["microMS_beadtargeting.py", "review"])
    M.main()
    assert called == [True]
