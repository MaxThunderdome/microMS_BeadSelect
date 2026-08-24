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


def test_one_shot_per_angle(cfg, transform):
    beads = [M.Bead(1000 + 200 * i, 1200, 8.2) for i in range(3)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    shots = M.place_shots(beads, cfg)
    assert len(shots) == len(cfg["laser-shot-angles"]) * 3


def test_shots_outside_slide_bounds_are_dropped(cfg, transform):
    cfg["slide-bounds"] = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
    beads = [M.Bead(1000, 1200, 8.2)]
    M.to_stage(beads, transform)
    M.isolation_filter(beads, cfg["min-bead-separation"])
    M.shape_filter(beads, cfg)
    shots = M.place_shots(beads, cfg)
    assert shots and all(s.dropped for s in shots)
    assert all("bounds" in s.drop_reason for s in shots)


# ---------------------------------------------------------------------
# .xeo export
# ---------------------------------------------------------------------

def test_header_and_footer_line_counts_are_locked():
    """microMS reads positions with lines[13:-12]."""
    assert len(M.XEO_HEADER) == 13
    assert len(M.XEO_FOOTER) == 12


def test_xeo_splits_at_400_and_round_trips(cfg, tmp_path):
    cfg["mtp_calibration"] = [
        {"name": "A", "x_um": 0, "y_um": 0, "unit_x": 0.0, "unit_y": 0.0},
        {"name": "B", "x_um": 75000, "y_um": 0, "unit_x": 1.0, "unit_y": 0.0},
        {"name": "C", "x_um": 0, "y_um": 25000, "unit_x": 0.0,
         "unit_y": 0.3333},
    ]
    mtp = M.fit_mtp(cfg)
    assert mtp is not None

    shots = [M.Shot(0, 0, 100.0 * i, 200.0) for i in range(950)]
    files = M.write_xeo(tmp_path / "t", shots, [], mtp, cfg)
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
    assert len(shots) == len(cfg["laser-shot-angles"])
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
    cfg["mtp_calibration"] = [
        {"name": "A", "x_um": 0, "y_um": 0, "unit_x": 0.0, "unit_y": 0.0},
        {"name": "B", "x_um": 75000, "y_um": 0, "unit_x": 1.0, "unit_y": 0.0},
        {"name": "C", "x_um": 0, "y_um": 25000, "unit_x": 0.0,
         "unit_y": 0.3333},
    ]
    shots = [M.Shot(i // 4, 90 * (i % 4), 100.0 * i, 200.0) for i in range(5)]
    xeo = M.write_xeo(tmp_path / "t", shots, [], M.fit_mtp(cfg), cfg)[0]

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


def test_position_name_pattern_is_configurable(cfg):
    cfg["output"]["position-name"] = "B{bead:04d}_A{angle:03d}"
    assert M.position_name(cfg, 0, M.Shot(29, 90, 0, 0)) == "B0029_A090"


# ---------------------------------------------------------------------
# coordinate-encoded position names
# ---------------------------------------------------------------------

def test_names_encode_adapter_coordinates(cfg):
    """X and Y in a position name are physical, not sequential."""
    cfg["output"]["name-coordinates"] = {
        "enabled": True, "unit-um": 10, "x0-um": 0.0, "y0-um": 0.0,
        "flip-x": False, "flip-y": False}
    shot = M.Shot(0, 0, 44870.0, 44240.0)      # 10 um units -> 4487, 4424
    assert M.position_name(cfg, 0, shot) == "R00X4487Y4424"


def test_adapter_origin_offsets_the_name(cfg):
    cfg["output"]["name-coordinates"] = {
        "enabled": True, "unit-um": 10, "x0-um": 1000.0, "y0-um": 2000.0,
        "flip-x": False, "flip-y": False}
    assert M.stage_to_adapter(cfg, 44870.0, 44240.0) == (4387, 4224)


def test_axis_flip_is_supported(cfg):
    cfg["output"]["name-coordinates"] = {
        "enabled": True, "unit-um": 10, "x0-um": 0.0, "y0-um": 0.0,
        "flip-x": False, "flip-y": True}
    assert M.stage_to_adapter(cfg, 1000.0, 1000.0) == (100, -100)


def test_missing_adapter_origin_refuses(cfg):
    """A guessed origin writes names that fire somewhere else."""
    cfg["output"]["name-coordinates"] = {"enabled": True, "unit-um": 10}
    with pytest.raises(SystemExit):
        M.stage_to_adapter(cfg, 1000.0, 1000.0)


def test_sequential_naming_still_available(cfg):
    cfg["output"]["name-coordinates"] = {"enabled": False}
    assert M.position_name(cfg, 0, M.Shot(0, 0, 999.0, 999.0)) == "R00X1Y1"
