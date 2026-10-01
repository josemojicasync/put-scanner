"""Tests con geometría sintética conocida.

Comprueban tres cosas:
  1. con geometría suficiente se mide correctamente (el valor real cae dentro de ±U);
  2. la incertidumbre aumenta cuando la geometría empeora;
  3. sin información suficiente el resultado es UNKNOWN (value None), nunca un número inventado.

Ejecutar:  python -m unittest discover -s tests -v
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import analysis as A  # noqa: E402
import synthetic as S  # noqa: E402
from scan_result import ScanResult, build_scan_result  # noqa: E402

STEP = 0.009  # espaciado de puntos sintético (≈ Polycam LiDAR)
_cache = {}


def run(name, factory):
    if name not in _cache:
        _cache[name] = A.analyze(factory())
    return _cache[name]


def assert_within(tc, m, truth, msg=""):
    """El valor real debe caer dentro del intervalo ±U reportado."""
    tc.assertIsNotNone(m["value"], f"{msg}: se esperaba un valor, es UNKNOWN")
    tc.assertLessEqual(abs(m["value"] - truth), m["U"], f"{msg}: {m['value']:.4f} ± {m['U']:.4f} vs real {truth}")


def one_connection(tc, res):
    tc.assertEqual(len(res["connections"]), 1, "se esperaba exactamente una aansluiting")
    return res["connections"][0]


class TestRectangularPut(unittest.TestCase):
    def test_perfect(self):
        r = run("rect", lambda: S.rect_put(step=STEP))
        p = r["put"]
        self.assertEqual(p["shape"]["label"], "rechthoekig")
        self.assertEqual(p["shape"]["status"], A.MEASURED)
        assert_within(self, p["width_x"], 0.8, "X")
        assert_within(self, p["width_y"], 1.0, "Y")
        assert_within(self, p["depth"], 2.5, "diepte")
        for k in ("width_x", "width_y", "depth"):
            self.assertEqual(p[k]["status"], A.MEASURED)
        self.assertLess(abs(p["width_x"]["value"] - 0.8), 0.003)
        self.assertEqual(r["connections"], [])

    def test_noise_and_outliers(self):
        clean = run("rect", lambda: S.rect_put(step=STEP))["put"]
        noisy = run("rect_noisy", lambda: S.rect_put(step=STEP, noise=0.005, outliers=0.01))["put"]
        assert_within(self, noisy["width_x"], 0.8, "X ruido")
        assert_within(self, noisy["width_y"], 1.0, "Y ruido")
        assert_within(self, noisy["depth"], 2.5, "diepte ruido")
        # más ruido -> más incertidumbre estadística
        self.assertGreater(noisy["width_x"]["sigma_stat"], clean["width_x"]["sigma_stat"])

    def test_missing_wall(self):
        r = run("missing", lambda: S.rect_put(step=STEP, missing=("-X",)))
        p = r["put"]
        self.assertIn("-X", p["walls"]["missing"])
        self.assertNotEqual(p["width_x"]["status"], A.MEASURED, "sin pared -X la anchura X no puede ser MEASURED")
        if p["width_x"]["value"] is not None:
            assert_within(self, p["width_x"], 0.8, "X estimada")
        self.assertEqual(p["width_y"]["status"], A.MEASURED)
        self.assertIn("-X", " ".join(r["quality"]["warnings"]))
        self.assertIn(r["quality"]["items"]["Put"], ("POOR", "RESCAN"))

    def test_partial_wall(self):
        full = run("rect", lambda: S.rect_put(step=STEP))["put"]
        r = run("partial", lambda: S.rect_put(step=STEP, partial={"-X": 0.2}))
        p = r["put"]
        self.assertNotEqual(p["width_x"]["status"], A.MEASURED, "pared al 20 %: la anchura no debe ser MEASURED")
        assert_within(self, p["width_x"], 0.8, "X pared parcial")
        self.assertGreaterEqual(p["width_x"]["sigma_stat"], full["width_x"]["sigma_stat"])
        self.assertNotEqual(r["quality"]["items"]["Put"], "GOOD")

    def test_no_ground(self):
        r = run("noground", lambda: S.rect_put(step=STEP, ground=False))
        p = r["put"]
        self.assertEqual(p["depth"]["status"], A.ESTIMATED, "sin maaiveld la diepte solo puede estimarse")
        self.assertEqual(r["quality"]["items"]["Maaiveld"], "RESCAN")
        assert_within(self, p["width_x"], 0.8, "X sin maaiveld")


class TestRoundPut(unittest.TestCase):
    def test_round(self):
        r = run("round", lambda: S.round_put(D=1.2, step=STEP))
        p = r["put"]
        self.assertEqual(p["shape"]["label"], "rond")
        self.assertEqual(p["shape"]["status"], A.MEASURED)
        assert_within(self, p["diameter"], 1.2, "diameter put")
        self.assertEqual(p["diameter"]["status"], A.MEASURED)
        assert_within(self, p["depth"], 2.5, "diepte")
        self.assertLess(p["circularity"]["value"], 0.003)

    def test_round_noisy(self):
        r = run("round_noisy", lambda: S.round_put(D=1.2, step=STEP, noise=0.004, outliers=0.01))
        assert_within(self, r["put"]["diameter"], 1.2, "diameter put ruido")


class TestPipes(unittest.TestCase):
    """Tubo Ø315 en la pared +Y, eje a 0,60 m sobre el bodem: kruin 0,7575, BOB 0,4425."""
    D, Z = 0.315, 0.6

    def pipe(self, name, arc, **kw):
        return run(name, lambda: S.rect_put(step=STEP, pipes=[dict(wall="+Y", z=self.Z, d=self.D, arc=arc)], **kw))

    def test_full_circle(self):
        c = one_connection(self, self.pipe("pipe360", (0, 360)))
        self.assertEqual(c["diameter"]["status"], A.MEASURED)
        assert_within(self, c["diameter"], self.D, "D 360°")
        assert_within(self, c["crown"], self.Z + self.D / 2, "kruin 360°")
        self.assertEqual(c["bob"]["status"], A.MEASURED, "con el tubo completo la BOB se observa directamente")
        assert_within(self, c["bob"], self.Z - self.D / 2, "BOB 360°")
        self.assertEqual(c["wall"], "+Y")

    def test_half_circle(self):
        full = one_connection(self, self.pipe("pipe360", (0, 360)))
        c = one_connection(self, self.pipe("pipe180", (0, 180)))
        assert_within(self, c["diameter"], self.D, "D 180°")
        self.assertGreater(c["diameter"]["U"], full["diameter"]["U"], "menos arco -> más incertidumbre")
        self.assertNotEqual(c["bob"]["status"], A.MEASURED, "sin la parte inferior la BOB no se observa")

    def test_arc_120(self):
        c = one_connection(self, self.pipe("pipe120", (30, 150)))
        self.assertNotEqual(c["diameter"]["status"], A.MEASURED, "arco < 180°: como mucho ESTIMATED")
        if c["diameter"]["value"] is not None:
            assert_within(self, c["diameter"], self.D, "D 120°")

    def test_arc_90_not_measured(self):
        c = one_connection(self, self.pipe("pipe90", (45, 135)))
        self.assertNotEqual(c["diameter"]["status"], A.MEASURED)
        assert_within(self, c["crown"], self.Z + self.D / 2, "kruin 90°")  # la kruin sí se ve

    def test_shallow_arc_unknown(self):
        """40° de arco: la curvatura no está determinada -> UNKNOWN, sin número."""
        c = one_connection(self, self.pipe("pipe40", (70, 110)))
        self.assertEqual(c["diameter"]["status"], A.UNKNOWN)
        self.assertIsNone(c["diameter"]["value"])
        self.assertIsNone(c["bob"]["value"])
        self.assertEqual(run("pipe40", None)["quality"]["items"][c["id"]], "RESCAN")

    def test_noise_outliers(self):
        c = one_connection(self, self.pipe("pipe_noisy", (0, 360), noise=0.004, outliers=0.01))
        assert_within(self, c["diameter"], self.D, "D con ruido")
        assert_within(self, c["crown"], self.Z + self.D / 2, "kruin con ruido")

    def test_water_hides_invert(self):
        """Agua/banket rugoso e inclinado tapa el tercio inferior del tubo: el diámetro no debe corromperse
        y la BOB no puede ser MEASURED (no se ve). Regresión: el banket se tomaba como fondo del tubo."""
        c = one_connection(self, run("pipe_water", lambda: S.rect_put(
            step=STEP, pipes=[dict(wall="+Y", z=self.Z, d=self.D, arc=(0, 360), water_z=self.Z - 0.08,
                                   water_rough=0.005, water_slope=1.0)])))
        if c["diameter"]["value"] is not None:
            assert_within(self, c["diameter"], self.D, "D con agua")
        self.assertNotEqual(c["bob"]["status"], A.MEASURED, "la BOB está bajo el agua: no se observa")
        assert_within(self, c["crown"], self.Z + self.D / 2, "kruin con agua")
        warnings = " ".join(run("pipe_water", None)["quality"]["warnings"])
        self.assertIn("verborgen", warnings)

    def test_direction_is_not_assumed_measured(self):
        """Si la dirección no se observa, debe ser ESTIMATED con método assumed_wall_normal."""
        c = one_connection(self, run("pipe_short", lambda: S.rect_put(
            step=STEP, pipes=[dict(wall="+Y", z=self.Z, d=self.D, arc=(0, 360), length=0.08)])))
        self.assertEqual(c["direction_m"]["method"], "assumed_wall_normal")
        self.assertEqual(c["direction_m"]["status"], A.ESTIMATED)
        self.assertEqual(c["slope"]["status"], A.UNKNOWN)


class TestArcMath(unittest.TestCase):
    """_measure_arc directamente: la incertidumbre crece al reducir el arco y acaba en UNKNOWN."""

    def measure(self, arc, noise=0.002):
        sz = S.pipe_section(d=0.315, arc=arc, noise=noise)
        return A._measure_arc(sz, np.random.default_rng(0), floor_sigma=0.001, top_z=3.0, top_sigma=0.001)

    def test_uncertainty_grows_as_arc_shrinks(self):
        U = []
        for arc in [(0, 360), (0, 180), (30, 150), (45, 135)]:
            d = self.measure(arc)["diameter"]
            U.append(d["U"] if d["value"] is not None else d["attempt_U"])
        self.assertEqual(U, sorted(U), f"U debería crecer: {U}")

    def test_too_short_arc_unknown(self):
        res = self.measure((75, 105))
        self.assertEqual(res["diameter"]["status"], A.UNKNOWN)
        self.assertIsNone(res["diameter"]["value"])

    def test_nominal_is_separate(self):
        res = self.measure((30, 150))
        self.assertIn("nominal", res)
        self.assertNotIn(res["diameter"]["method"], ("nominal",))
        for n in res["nominal"]:
            self.assertIn(n, A.NOMINALS_MM)


class TestScanResult(unittest.TestCase):
    def test_json_roundtrip(self):
        r = run("pipe360", None)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "fake.ply"
            src.write_bytes(b"ply")
            sr = build_scan_result(r, src, 1.0)
            out = Path(tmp) / "result.json"
            sr.save(out)
            data = json.loads(out.read_text(encoding="utf-8"))
            back = ScanResult.load(out)
        self.assertEqual(back.to_dict(), data)
        self.assertEqual(back.put.width_x_mm.status, "MEASURED")
        self.assertAlmostEqual(back.put.width_x_mm.value, 800, delta=3)
        self.assertEqual(back.connections[0].diameter_mm.unit, "mm")
        for m in (back.put.width_x_mm, back.connections[0].diameter_mm):
            for key in ("value", "uncertainty", "confidence", "status", "method"):
                self.assertTrue(hasattr(m, key))


class TestRegressionRealScan(unittest.TestCase):
    """Caso de regresión con el scan real (solo si existe en este PC; no se modifica)."""
    path = ROOT / "data" / "scan.ply"

    @unittest.skipUnless((ROOT / "data" / "scan.ply").is_file(), "data/scan.ply no disponible")
    def test_scan_ply(self):
        r = A.load_and_analyze(self.path)
        p = r["put"]
        self.assertEqual(p["shape"]["label"], "rechthoekig")
        # valores de referencia de la fase 3 (792 / 805 mm, diepte ~2,68 m): deben seguir dentro de ±U
        assert_within(self, p["width_x"], 0.792, "X regresión")
        assert_within(self, p["width_y"], 0.805, "Y regresión")
        assert_within(self, p["depth"], 2.68, "diepte regresión")
        self.assertEqual(len(r["connections"]), 2)
        for c in r["connections"]:
            self.assertNotEqual(c["diameter"]["status"], A.MEASURED, "arcos ~100-120°: nunca MEASURED")
            # FASE 6: antes se exigía kruin MEASURED. El diagnóstico (fase 5) y el perfil de corona por bandas
            # mostraron que ese valor era el borde del agujero en la pared o el límite de visibilidad (la "corona"
            # baja 13-17° con la profundidad). Invariante nueva: una kruin MEASURED no puede ser el borde de la
            # abertura (debe estar por debajo de su borde superior); si no se observa, UNKNOWN con motivo.
            if c["crown"]["status"] == A.MEASURED:
                self.assertLess(c["crown"]["value"], c["opening_top"]["value"])
            else:
                self.assertEqual(c["crown"]["status"], A.UNKNOWN)
                self.assertTrue(c["crown"].get("reason"))


if __name__ == "__main__":
    unittest.main()
