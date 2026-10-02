"""Separación best_estimate (mejor valor geométrico calculado) / measurement (medida aceptada por las reglas).
La medida aceptada no cambia nunca; best_estimate solo deja de ocultar valores ya calculados.
Ejecutar:  python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import analysis as A  # noqa: E402
import scan_result as SR  # noqa: E402
import synthetic as S  # noqa: E402


class TestBestEstimateUnit(unittest.TestCase):
    def test_accepted_measurement_is_its_own_best(self):
        m = A.meas(0.337, 0.025, A.LAAG, "robust_3d_cylinder_fit", A.ESTIMATED)
        b = A.best_estimate(m)
        self.assertEqual((b["value"], b["U"], b["reliability"], b["accepted"]), (0.337, 0.025, A.ESTIMATED, True))

    def test_computed_but_not_certifiable_is_uncertain(self):
        """Un intento calculado (U > R10) aparece como UNCERTAIN; la medida sigue siendo UNKNOWN sin valor."""
        m = A.unknown("partial_arc_circle_fit", reason="U te groot", attempt_value=0.167, attempt_U=0.045)
        b = A.best_estimate(m)
        self.assertEqual(b["reliability"], A.UNCERTAIN)
        self.assertFalse(b["accepted"])
        self.assertAlmostEqual(b["value"], 0.167)
        self.assertAlmostEqual(b["U"], 0.045)
        self.assertIsNone(m["value"])
        self.assertEqual(m["status"], A.UNKNOWN)

    def test_no_calculation_is_not_available(self):
        b = A.best_estimate(A.unknown("crown_band_profile", reason="límite de visibilidad"))
        self.assertEqual(b["reliability"], A.NOT_AVAILABLE)
        self.assertIsNone(b["value"])
        self.assertIsNone(b["U"])

    def test_candidate_without_finite_uncertainty_ignored(self):
        b = A.best_estimate(A.unknown("x"), dict(value=0.3, U=float("inf"), method="c"), dict(value=0.3, U=None, method="d"))
        self.assertEqual(b["reliability"], A.NOT_AVAILABLE)

    def test_smallest_relative_uncertainty_wins(self):
        m = A.unknown("partial_arc_circle_fit", attempt_value=0.30, attempt_U=0.09)
        b = A.best_estimate(m, dict(value=0.26, U=0.035, method="robust_3d_cylinder_fit"))
        self.assertEqual(b["method"], "robust_3d_cylinder_fit")

    def test_cylinder_only_if_shape_valid(self):
        """El cilindro 3D rechazado por FORMA (cono, RMS) no es una mejor estimación; rechazado solo por U, sí."""
        base = dict(diameter=A.unknown("pipe_bands"))
        ok = A.connection_best(dict(base, cylinder_fit=dict(ok_fit=True, shape_ok=True, D=0.258, U=0.068)))
        self.assertEqual(ok["diameter"]["reliability"], A.UNCERTAIN)
        self.assertAlmostEqual(ok["diameter"]["value"], 0.258)
        bad = A.connection_best(dict(base, cylinder_fit=dict(ok_fit=True, shape_ok=False, D=0.36, U=0.02)))
        self.assertEqual(bad["diameter"]["reliability"], A.NOT_AVAILABLE)


class TestBestEstimateEndToEnd(unittest.TestCase):
    def _json(self, pipe):
        r = A.analyze(S.rect_put(step=0.009, noise=0.003, pipes=[pipe]))
        self.assertEqual(len(r["connections"]), 1)
        return json.loads(SR.build_scan_result(r, "synthetic.ply", 0.0).to_json())["connections"][0]

    def test_unknown_diameter_keeps_measurement_but_exposes_best(self):
        """Tubo corto y parcial: measurement.diameter UNKNOWN sin número; best_estimate con valor ± U, UNCERTAIN."""
        c = self._json(dict(wall="+Y", z=0.6, d=0.315, arc=(40, 140), length=0.12))
        self.assertIsNone(c["diameter_mm"]["value"])
        self.assertEqual(c["diameter_mm"]["status"], "UNKNOWN")
        b = c["best_estimate"]["diameter_mm"]
        self.assertEqual(b["reliability"], "UNCERTAIN")
        self.assertFalse(b["accepted"])
        self.assertIsNotNone(b["value"])
        self.assertIsNotNone(b["uncertainty"])

    def test_measured_best_equals_measurement(self):
        c = self._json(dict(wall="+Y", z=0.6, d=0.315, arc=(0, 360)))
        b, m = c["best_estimate"]["diameter_mm"], c["diameter_mm"]
        self.assertEqual(m["status"], "MEASURED")
        self.assertEqual((b["value"], b["uncertainty"], b["reliability"], b["accepted"]),
                         (m["value"], m["uncertainty"], "MEASURED", True))

    def test_cone_gives_no_best_diameter(self):
        """Cono: no hay ajuste geométrico válido -> NOT_AVAILABLE (no se inventa un valor)."""
        c = self._json(dict(wall="+Y", z=0.6, d=0.28, d_end=0.40, arc=(20, 160)))
        self.assertIsNone(c["diameter_mm"]["value"])
        self.assertEqual(c["best_estimate"]["diameter_mm"]["reliability"], "NOT_AVAILABLE")
        self.assertIsNone(c["best_estimate"]["diameter_mm"]["value"])


if __name__ == "__main__":
    unittest.main()
