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


def test_synthetic_fiducials_are_flagged(cfg):
    """
    A fiducial set generated from an exact transform fits perfectly.
    Real clicks never do, because a similarity fit through 3 points is
    overdetermined. Shipping such a set is the dangerous case: it
    reports a flawless registration and fires in the wrong place.
    """
    src = np.array([[400.0, 3300.0], [7700.0, 3300.0], [400.0, 5650.0]])
    dst = src * 9.74 + np.array([0.0, 0.0])
    T = M.fit_similarity(src, dst)
    res = M.residuals(T, src, dst)
    assert res.max() < 1e-6                      # fits exactly, as set up
    notes = M.registration_sanity(src, dst, res, cfg)
    assert any("SYNTHETIC" in n for n in notes)


def test_hand_picked_fiducials_are_not_flagged(cfg):
    """The guard must stay quiet on a plausible hand-picked set."""
    src = np.array([[400.0, 3300.0], [7700.0, 3300.0], [400.0, 5650.0]])
    dst = src * 9.74
    dst = dst + np.array([[3.0, -2.0], [-2.0, 4.0], [1.0, 3.0]])  # click noise
    T = M.fit_similarity(src, dst)
    res = M.residuals(T, src, dst)
    assert not M.registration_sanity(src, dst, res, cfg)


def test_collinear_fiducials_are_flagged(cfg):
    """Three marks along one slide edge cannot constrain rotation."""
    src = np.array([[400.0, 3300.0], [4000.0, 3302.0], [7700.0, 3299.0]])
    dst = src * 9.74 + np.array([[2.0, -1.0], [-1.0, 2.0], [1.0, 1.0]])
    T = M.fit_similarity(src, dst)
    res = M.residuals(T, src, dst)
    notes = M.registration_sanity(src, dst, res, cfg)
    assert any("collinear" in n for n in notes)


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
    files = M.write_xeo(tmp_path / "t", shots, [], mtp)
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

def _lod_axes(img, cfg, plain=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_axes([0.05, 0.16, 0.92, 0.79])
    if plain:
        ax.imshow(img, cmap="gray")
    else:
        M.attach_image_lod(ax, img, cfg)
    return fig, ax


def test_lod_extent_matches_plain_imshow_when_undecimated(cfg):
    """
    The decimated display must stay in ORIGINAL scan pixels or every
    click in 'pick' and 'select' lands somewhere else. When the image
    fits the budget there is no decimation, so the artist must sit at
    exactly the extent plain imshow would have used. Comparing against
    matplotlib itself keeps this honest -- checking the extent by
    inverting through that same extent would pass no matter what.
    """
    import matplotlib.pyplot as plt
    img = np.zeros((300, 400), np.uint8)
    fig_a, ax_a = _lod_axes(img, cfg)
    fig_b, ax_b = _lod_axes(img, cfg, plain=True)
    try:
        assert ax_a.images[0].get_array().shape == img.shape   # step == 1
        assert (ax_a.images[0].get_extent()
                == pytest.approx(ax_b.images[0].get_extent()))
    finally:
        plt.close(fig_a)
        plt.close(fig_b)


def test_lod_landmark_lands_on_its_own_scan_pixel(cfg):
    """
    Drive a real cursor position through matplotlib's transforms and
    ask the artist what is under it. transData is built from the axes
    limits, which are true scan pixels, so this is independent of the
    extent arithmetic under test.
    """
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import MouseEvent

    rng = np.random.default_rng(0)
    img = rng.integers(0, 60, size=(3000, 4000), dtype=np.uint8)
    img[1234, 2345] = 255
    assert (img == 255).sum() == 1

    fig, ax = _lod_axes(img, cfg)
    try:
        ax.set_xlim(2345 - 100, 2345 + 100)
        ax.set_ylim(1234 + 75, 1234 - 75)
        fig.canvas.draw()
        art = ax.images[0]
        left, right = art.get_extent()[0], art.get_extent()[1]
        assert (right - left) / art.get_array().shape[1] == 1   # 1:1 here

        sx, sy = ax.transData.transform((2345.0, 1234.0))
        ev = MouseEvent("motion_notify_event", fig.canvas, sx, sy)
        assert art.get_cursor_data(ev) == 255
    finally:
        plt.close(fig)


def test_lod_decimates_when_zoomed_out(cfg):
    """A full-slide view must be decimated to the configured budget."""
    import matplotlib.pyplot as plt
    img = np.zeros((7551, 10002), np.uint8)
    fig, ax = _lod_axes(img, cfg)
    try:
        budget = float(cfg.get("display-max-megapixels", 4.0)) * 1e6
        arr = ax.images[0].get_array()
        assert arr.size <= budget, (arr.size, budget)
        assert arr.size < img.size / 4
        left, right, bottom, top = ax.images[0].get_extent()
        assert left < 1 and right > 10002 - 10
        assert top < 1 and bottom > 7551 - 10
    finally:
        plt.close(fig)


def test_lod_restores_limits_when_autoscale_is_on(cfg):
    """
    set_extent re-autoscales whenever autoscale is on, which would snap
    the view back to the whole slide on every refresh and make zooming
    impossible. attach_image_lod turns autoscale off AND puts the
    limits back; this checks the second guard by re-enabling the first
    one's failure mode.
    """
    import matplotlib.pyplot as plt
    img = np.zeros((3000, 4000), np.uint8)
    fig, ax = _lod_axes(img, cfg)
    try:
        # The refresh runs on the limit-changed callback, not on draw,
        # and a plain set_xlim would switch autoscale off before it
        # fires. auto=True keeps it on so set_extent really does try to
        # rescale, which is the case the restore exists for.
        ax.set_autoscale_on(True)          # undo the belt, leave the braces
        ax.set_xlim(1000, 1200, auto=True)
        ax.set_ylim(900, 700, auto=True)
        fig.canvas.draw()
        assert ax.get_xlim() == pytest.approx((1000, 1200))
        assert ax.get_ylim() == pytest.approx((900, 700))
    finally:
        plt.close(fig)


def test_review_sheet_covers_every_accepted_bead(cfg, transform, tmp_path):
    """
    The confirmation sheet exists to catch the ONE bead whose shots went
    wrong, so it must paginate rather than sample. With a per-sheet cap
    of 4 and 9 accepted beads that is 3 sheets, not one truncated one.
    """
    import cv2
    scan = tmp_path / "scan.png"
    img = np.full((600, 600), 200, np.uint8)
    beads = []
    for i in range(9):
        x, y = 60 + (i % 3) * 200, 60 + (i // 3) * 200
        cv2.circle(img, (x, y), 5, 90, -1)
        b = M.Bead(x_px=float(x), y_px=float(y), diameter_px=10.0,
                   diameter_um=90.0, nn_um=500.0, accepted=True)
        beads.append(b)
    cv2.imwrite(str(scan), img)

    c = dict(cfg)
    c["review"] = {"panel-px": 60, "panel-scale": 2, "columns": 2,
                   "max-panels-per-sheet": 4}
    for i, b in enumerate(beads):
        b.x_um, b.y_um = transform.px_to_um([[b.x_px, b.y_px]])[0]
    M.place_shots(beads, c)

    old = M.REVIEW_DIR
    M.REVIEW_DIR = tmp_path / "image confirmation"
    try:
        pages = M.draw_shot_review(beads, c, transform, scan, "2026-01-02 03:04:05")
    finally:
        M.REVIEW_DIR = old

    assert len(pages) == 3, [p.name for p in pages]      # 4 + 4 + 1
    assert all(p.exists() and p.stat().st_size > 0 for p in pages)
    # filesystem-safe, chronologically sortable
    assert all(" " not in p.name and ":" not in p.name for p in pages)
    assert all("2026-01-02_030405" in p.name for p in pages)


def test_review_writes_nothing_without_accepted_beads(cfg, transform, tmp_path):
    """No selection means nothing to confirm -- and no empty sheet."""
    import cv2
    scan = tmp_path / "scan.png"
    cv2.imwrite(str(scan), np.full((200, 200), 200, np.uint8))
    beads = [M.Bead(x_px=50.0, y_px=50.0, diameter_px=10.0, accepted=False)]
    old = M.REVIEW_DIR
    M.REVIEW_DIR = tmp_path / "image confirmation"
    try:
        assert M.draw_shot_review(beads, cfg, transform, scan,
                                  "2026-01-02 03:04:05") == []
        assert not M.REVIEW_DIR.exists()
    finally:
        M.REVIEW_DIR = old


def _gui_backend():
    """Name of an importable interactive backend, or None."""
    for mod, name in (("PyQt5", "QtAgg"), ("PySide6", "QtAgg"),
                      ("tkinter", "TkAgg")):
        try:
            __import__(mod)
            return name
        except ImportError:
            continue
    return None


def test_max_fiducials_is_a_ceiling_not_a_target(cfg):
    """
    3 is the MINIMUM for a similarity fit, not the goal: with exactly 3
    there is no spare point, so leave-one-out cannot run. The shipped
    ceiling must leave room to go past 3.
    """
    assert cfg["max-fiducials"] >= 4
    src = np.array([[0, 0], [2000, 0], [0, 1500], [2000, 1500]], float)
    dst = src * 9.74
    assert M.loo_residuals(src[:3], dst[:3]) is None      # 3 -> no estimate
    assert M.loo_residuals(src, dst) is not None          # 4 -> real one


@pytest.mark.skipif(
    not _gui_backend(), reason="fiducial picker needs a GUI backend")
def test_picker_stops_at_max_fiducials(cfg, tmp_path, monkeypatch):
    """Drives the real picker; the Add button must refuse past the cap."""
    import matplotlib
    matplotlib.use(_gui_backend(), force=True)
    import matplotlib.pyplot as plt
    import matplotlib.widgets as W
    from matplotlib.backend_bases import MouseEvent
    import cv2

    scan = tmp_path / "scan.png"
    cv2.imwrite(str(scan), np.full((800, 900), 200, np.uint8))

    seen = {"buttons": [], "boxes": []}
    B, T = W.Button, W.TextBox
    monkeypatch.setattr(W, "Button", lambda ax, label, **k: (
        seen["buttons"].append(B(ax, label, **k)) or seen["buttons"][-1]))
    monkeypatch.setattr(W, "TextBox", lambda ax, label, **k: (
        seen["boxes"].append(T(ax, label, **k)) or seen["boxes"][-1]))
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(M, "save_fiducials", lambda f, **k: None)

    c = dict(cfg)
    c["fiducials"] = []
    c["max-fiducials"] = 5
    c["input"] = dict(c["input"])
    c["input"]["scan"] = str(scan)
    M.pick_fiducials(c)

    add_btn, bx, by = seen["buttons"][0], seen["boxes"][0], seen["boxes"][1]
    fig = add_btn.ax.figure
    img_ax = fig.axes[0]
    for i in range(8):
        x, y = 100 + (i % 4) * 150, 100 + (i // 4) * 200
        ev = MouseEvent("button_press_event", fig.canvas, 0, 0, button=3)
        ev.inaxes = img_ax
        ev.xdata, ev.ydata = float(x), float(y)
        fig.canvas.callbacks.process("button_press_event", ev)
        bx.set_val(str(x * 9.74))
        by.set_val(str(y * 9.74))
        add_btn._observers.process("clicked", None)

    assert add_btn.label.get_text().endswith("5/5")
    plt.close("all")


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
