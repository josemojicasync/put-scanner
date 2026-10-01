"""Tests de la fase 6: segmentación geométrica de aansluitingen (abertura -> tubo por bandas),
anchuras variables con la altura y referencias de profundidad ambiguas.

Geometría sintética con medidas conocidas; ningún número procede de un scan real.
Ejecutar:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import analysis as A  # noqa: E402
import synthetic as S  # noqa: E402

STEP = 0.009
_cache = {}


def run(name, factory):
    if name not in _cache:
        _cache[name] = A.analyze(factory())
    return _cache[name]


def within(tc, m, truth, msg=""):
    tc.assertIsNotNone(m["value"], f"{msg}: UNKNOWN, se esperaba un valor")
    tc.assertLessEqual(abs(m["value"] - truth), m["U"], f"{msg}: {m['value']:.4f} ± {m['U']:.4f} vs real {truth}")


def single(tc, res):
    tc.assertEqual(len(res["connections"]), 1, f"se esperaba 1 aansluiting, hay {len(res['connections'])}")
    return res["connections"][0]


class TestOpeningVsPipe(unittest.TestCase):
    def test_opening_diameter_differs_from_pipe(self):
        """Agujero Ø340 en una pared de 12 cm, tubo Ø250 detrás: dos geometrías, no se promedian."""
        c = single(self, run("open_vs_pipe", lambda: S.rect_put(step=STEP, pipes=[dict(
            wall="+Y", z=0.8, d=0.25, hole_d=0.34, thickness=0.12, length=0.7)])))
        within(self, c["diameter"], 0.25, "D tubo")
        self.assertIsNotNone(c["opening_diameter"]["value"], "la abertura en la cara debe medirse")
        self.assertGreater(c["opening_diameter"]["value"], c["diameter"]["value"] + 0.05,
                           "la abertura (Ø340) no debe confundirse con el tubo (Ø250)")
        labels = {b["label"] for b in c["bands"]}
        self.assertIn("OPENING_RING", labels)
        used_t = (c["points"][c["used_mask"]] - c["c0"]) @ c["axis"]
        self.assertGreater(np.percentile(used_t, 2), 0.09, "el diámetro no debe usar puntos del espesor de la pared")

    def test_collar_larger_than_pipe(self):
        """Collar que sobresale 2,5 cm delante de la pared alrededor de un tubo Ø300."""
        c = single(self, run("collar", lambda: S.rect_put(step=STEP, pipes=[dict(
            wall="-X", z=0.9, d=0.30, length=0.6, collar=dict(width=0.06, protrusion=0.025))])))
        within(self, c["diameter"], 0.30, "D con collar")


class TestObliquePipe(unittest.TestCase):
    def test_oblique_pipe(self):
        """Tubo Ø200 a 25° de la normal de la pared: la sección debe cortarse perpendicular al eje real."""
        c = single(self, run("oblique", lambda: S.rect_put(step=STEP, pipes=[dict(
            wall="+Y", z=1.0, d=0.20, yaw_deg=25, length=0.8)])))
        within(self, c["diameter"], 0.20, "D oblicuo")
        dev = c["direction_m"]
        self.assertEqual(dev["method"], "band_centers_axis")
        self.assertLessEqual(abs(abs(dev["value"]) - 25), max(dev["U"] or 0, 3.0), f"desviación {dev['value']:.1f}°")


class TestOcclusions(unittest.TestCase):
    def test_partially_under_water(self):
        c = single(self, run("water", lambda: S.rect_put(step=STEP, pipes=[dict(
            wall="+Y", z=0.6, d=0.315, water_z=0.52, water_rough=0.005, water_slope=1.0)])))
        self.assertTrue(c["lower_pipe_occluded"])
        self.assertNotEqual(c["bob"]["status"], A.MEASURED, "bajo el agua no hay geometría directa del fondo")
        within(self, c["crown"], 0.6 + 0.1575, "kruin con agua")
        if c["diameter"]["value"] is not None:
            within(self, c["diameter"], 0.315, "D con agua")

    def test_back_fill(self):
        """Tapa transversal (relleno de reconstrucción) a 35 cm: se detecta y se excluye del ajuste."""
        c = single(self, run("fill", lambda: S.rect_put(step=STEP, pipes=[dict(
            wall="+Y", z=0.8, d=0.30, length=0.6, cap_t=0.35)])))
        self.assertTrue(c["possible_reconstruction_fill"])
        within(self, c["diameter"], 0.30, "D con tapa")
        trans_used = (c["transverse_mask"] & c["used_mask"]).sum()
        self.assertEqual(trans_used, 0, "los puntos de la tapa no deben usarse en el diámetro")


class TestFalseCandidates(unittest.TestCase):
    def test_flat_relief_behind_wall(self):
        """Rebaje plano de 3,5 cm detrás de la cara, sin tubo: no es una aansluiting."""
        r = run("relief", lambda: S.rect_put(step=STEP, reliefs=[dict(
            wall="-X", u0=-0.10, u1=0.05, z0=0.5, z1=0.8, depth=0.035)]))
        self.assertEqual(r["connections"], [])
        kinds = [o.get("kind") for o in r["put"]["openings"]] + [x["kind"] for x in r["put"]["rejected_candidates"]]
        self.assertTrue(any(k in ("relief", "WALL_RELIEF") for k in kinds), f"relieve no identificado: {kinds}")

    def test_relief_below_pipe_is_not_a_second_connection(self):
        """Ranura plana bajo un tubo real, tocando su abertura: una sola aansluiting y sin diámetro para la ranura."""
        r = run("relief_below", lambda: S.rect_put(step=STEP, pipes=[dict(wall="+Y", z=0.95, d=0.30, length=0.6)],
                                                  reliefs=[dict(wall="+Y", u0=-0.08, u1=0.08, z0=0.30, z1=0.80, depth=0.035)]))
        c = single(self, r)
        within(self, c["diameter"], 0.30, "D del tubo real")


class TestBands(unittest.TestCase):
    def test_stable_bands(self):
        c = single(self, run("stable", lambda: S.rect_put(step=STEP, noise=0.002, pipes=[dict(
            wall="+Y", z=0.8, d=0.315, length=0.6)])))
        pipe = [b for b in c["bands"] if b["label"] == "PIPE"]
        self.assertGreaterEqual(len(pipe), 5)
        D = [2 * b["r"] for b in pipe]
        self.assertLess(max(D) - min(D), 0.012, f"bandas no estables: {np.round(D, 3)}")
        self.assertEqual(c["diameter"]["status"], A.MEASURED)
        within(self, c["diameter"], 0.315, "D bandas estables")
        self.assertGreater(c["pipe_length_observed"], 0.3)

    def test_incompatible_bands_unknown(self):
        """Tubo cónico (Ø250 -> Ø400): ningún tramo de radio constante -> diámetro UNKNOWN, sin promediar."""
        c = single(self, run("cone", lambda: S.rect_put(step=STEP, pipes=[dict(
            wall="+Y", z=0.9, d=0.25, d_end=0.40, length=0.5)])))
        self.assertEqual(c["diameter"]["status"], A.UNKNOWN)
        self.assertIsNone(c["diameter"]["value"])


class TestWallsAndReferences(unittest.TestCase):
    def test_converging_walls(self):
        """Paredes X convergentes: 800 mm abajo -> 780 mm arriba."""
        r = run("taper", lambda: S.rect_put(step=STEP, taper_x=-0.02))
        par = run("parallel", lambda: S.rect_put(step=STEP))
        wx = r["put"]["width_x"]
        prof = wx["profile"]
        self.assertTrue(prof["significant"])
        self.assertLess(prof["variation"], -0.01)
        zmid = prof["z"][1]
        within(self, wx, 0.8 - 0.02 * zmid / 2.5, "anchura a media altura")
        self.assertGreater(wx["U"], par["put"]["width_x"]["U"], "la variación con la altura debe aumentar U")
        self.assertIn("niet parallel", " ".join(r["quality"]["warnings"]))
        self.assertFalse(par["put"]["width_x"]["profile"]["significant"])

    def test_bottom_two_surfaces(self):
        r = run("floor2", lambda: S.rect_put(step=STEP, floor_step=0.04))
        p = r["put"]
        self.assertGreaterEqual(len(p["bottom_surface_candidates"]), 2)
        self.assertEqual(p["depth"]["status"], A.ESTIMATED)
        self.assertGreaterEqual(p["depth"]["U"], 0.035, "la separación entre superficies debe entrar en U")

    def test_ground_two_surfaces(self):
        r = run("ground2", lambda: S.rect_put(step=STEP, ground_step=0.04))
        p = r["put"]
        self.assertGreaterEqual(len(p["reference_surface_candidates"]), 2)
        self.assertEqual(p["depth"]["status"], A.ESTIMATED)


if __name__ == "__main__":
    unittest.main()
