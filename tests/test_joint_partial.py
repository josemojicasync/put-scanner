"""Tests del fallback de radio común para tubos parcialmente visibles (fase 7: _joint_partial_pipe_fit).

Se generan superficies de tubo sintéticas por bandas (arcos parciales), se forman las bandas con el mismo
código del pipeline (connections._bands) y se prueba el ajuste conjunto. Ningún número procede de un scan real.
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
import connections as C  # noqa: E402
import synthetic as S  # noqa: E402

AXIS = np.array([0.0, 1.0, 0.0])
C0 = np.array([0.0, 0.5, 0.8])
U_AX, V_AX = C._frame_for_axis(AXIS)


def tube(segments, arc, noise=0.0, step=0.012, seed=0, tilt_deg=0.0):
    """segments: [(t0, t1, r0, r1)] radio lineal entre r0 y r1. arc en grados (90° = kruin).
    tilt_deg: el eje REAL del tubo está inclinado respecto al eje supuesto (AXIS) en el plano vertical.
    Devuelve puntos y normales (radiales: perpendiculares al eje como en un tubo real)."""
    rng = np.random.default_rng(seed)
    th = np.radians(tilt_deg)
    a_true = AXIS * np.cos(th) + V_AX * np.sin(th)
    u_t, v_t = C._frame_for_axis(a_true)
    P, Nn = [], []
    for t0, t1, r0, r1 in segments:
        for t in np.arange(t0, t1, step):
            r = r0 + (r1 - r0) * (t - t0) / max(t1 - t0, 1e-9)
            ang = np.arange(np.radians(arc[0]), np.radians(arc[1]), step / r)
            radial = np.outer(np.cos(ang), u_t) + np.outer(np.sin(ang), v_t)
            P.append(C0 + a_true * t + r * radial)
            Nn.append(-radial)
    P, Nn = np.vstack(P), np.vstack(Nn)
    if noise:
        P = P + rng.normal(0, noise, P.shape)
    return P, Nn


def joint(P, Nn, seed=0):
    rows = C._bands(P, Nn, C0, AXIS, np.zeros(len(P), bool))
    jf = C._joint_partial_pipe_fit(rows, P, C0, AXIS, U_AX, V_AX, np.random.default_rng(seed),
                                   floor_s=0.001, top_z=3.0, top_s=0.001, sigma_axis=np.radians(1.0))
    return rows, jf


class TestJointPartialFit(unittest.TestCase):
    def test_A_partial_bands_same_cylinder(self):
        """Arcos parciales (130°) del mismo cilindro Ø300 con ruido: cada banda da su propio radio, el ajuste
        conjunto recupera el diámetro dentro de ±U y como máximo ESTIMATED."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.003)
        rows, jf = joint(P, Nn)
        self.assertTrue(jf["ok"], jf.get("reason"))
        d = jf["arc"]["diameter"]
        self.assertLessEqual(abs(d["value"] - 0.30), d["U"], f"{d['value']:.4f} ± {d['U']:.4f}")
        conn = dict(bob=None)
        C._apply_joint_fit(conn, jf, P, C0, U_AX, V_AX)
        self.assertEqual(conn["diameter"]["status"], A.ESTIMATED)
        self.assertEqual(conn["diameter"]["method"], "joint_partial_pipe_bands")

    def test_A2_tilted_cylinder_partial(self):
        """Cilindro Ø300 inclinado 6° respecto al eje supuesto, arcos parciales: el radio común con eje libre lo
        recupera (la prueba de cono NO debe rechazar una inclinación real) y como máximo ESTIMATED."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.003, tilt_deg=6.0)
        _, jf = joint(P, Nn)
        self.assertTrue(jf["ok"], jf.get("reason"))
        d = jf["arc"]["diameter"]
        self.assertLessEqual(abs(d["value"] - 0.30), d["U"], f"{d['value']:.4f} ± {d['U']:.4f}")

    def test_B_cone_rejected(self):
        """Tubo cónico Ø280 -> Ø400: no hay radio común -> sin diámetro."""
        P, Nn = tube([(0.0, 0.40, 0.14, 0.20)], arc=(25, 155), noise=0.002)
        _, jf = joint(P, Nn)
        self.assertFalse(jf["ok"], "un cono no debe aceptarse como cilindro")

    def test_C_two_incompatible_radii(self):
        """Dos superficies con radios distintos (Ø280 hasta 20 cm, Ø340 después) -> UNKNOWN, no se promedian."""
        P, Nn = tube([(0.0, 0.20, 0.14, 0.14), (0.20, 0.40, 0.17, 0.17)], arc=(25, 155), noise=0.002)
        _, jf = joint(P, Nn)
        self.assertFalse(jf["ok"], "dos radios distintos no deben combinarse en uno")

    def test_D_face_collar_does_not_drag_pipe(self):
        """Banda de la cara (0-5 cm) con radio de agujero/collar Ø380, tubo Ø300 detrás: la banda de la cara no
        participa y el diámetro del tubo no se arrastra hacia la abertura."""
        P, Nn = tube([(0.0, 0.05, 0.19, 0.19), (0.05, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.002)
        rows, jf = joint(P, Nn)
        self.assertTrue(jf["ok"], jf.get("reason"))
        for i in jf["bands"]:
            self.assertGreaterEqual(rows[i]["t0"], C.BAND_M - 1e-9, "la banda de la cara no debe usarse")
        d = jf["arc"]["diameter"]
        self.assertLessEqual(abs(d["value"] - 0.30), d["U"])
        self.assertLess(d["value"], 0.34, "el diámetro del tubo no debe acercarse al de la abertura")

    def test_E_water_hides_bottom_bob_unknown(self):
        """Agua oculta la parte inferior: con geometría lateral suficiente puede estimarse el diámetro, pero la BOB
        no se observa y no se deriva del círculo estimado."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(-20, 200), noise=0.003)   # 220° visibles, fondo oculto
        rows, jf = joint(P, Nn)
        self.assertTrue(jf["ok"], jf.get("reason"))
        conn = dict(bob=A.meas(0.1, 0.01, A.LAAG, "crown_minus_diameter", A.ESTIMATED))
        C._apply_joint_fit(conn, jf, P, C0, U_AX, V_AX)
        self.assertEqual(conn["bob"]["status"], A.UNKNOWN, "la BOB no debe derivarse del círculo estimado")
        self.assertIsNone(conn["bob"]["value"])
        self.assertEqual(conn["diameter"]["status"], A.ESTIMATED)

    def test_F_total_arc_too_small(self):
        """Arco total de 60°: la curvatura no determina el radio -> no se acepta."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(60, 120), noise=0.002)
        _, jf = joint(P, Nn)
        self.assertFalse(jf["ok"])

    def test_F2_only_two_bands(self):
        """Con solo 2 bandas detrás de la cara no hay comprobación leave-one-band-out -> no se acepta."""
        P, Nn = tube([(0.0, 0.15, 0.15, 0.15)], arc=(25, 155), noise=0.002)
        _, jf = joint(P, Nn)
        self.assertFalse(jf["ok"])
        self.assertIn("bandas", jf["reason"])


class TestJointNotUsedForGoodGeometry(unittest.TestCase):
    def test_G_full_stable_pipe_uses_normal_method(self):
        """Tubo completo y estable: el método principal (tramo estable) se mantiene, no el fallback."""
        r = A.analyze(S.rect_put(step=0.009, pipes=[dict(wall="+Y", z=0.6, d=0.315, arc=(0, 360))]))
        self.assertEqual(len(r["connections"]), 1)
        c = r["connections"][0]
        self.assertEqual(c["diameter"]["status"], A.MEASURED)
        self.assertNotEqual(c["diameter"]["method"], "joint_partial_pipe_bands")
        self.assertNotIn("joint_fit", c, "el fallback no debe ejecutarse si hay tramo estable")

    def test_deterministic(self):
        """El mismo PLY/nube produce exactamente el mismo resultado."""
        pcd = S.rect_put(step=0.009, noise=0.003, pipes=[dict(wall="+Y", z=0.6, d=0.315, arc=(20, 160))])
        a, b = A.analyze(pcd), A.analyze(pcd)
        for ca, cb in zip(a["connections"], b["connections"]):
            self.assertEqual(ca["diameter"]["value"], cb["diameter"]["value"])
            self.assertEqual(ca["diameter"]["status"], cb["diameter"]["status"])


if __name__ == "__main__":
    unittest.main()
