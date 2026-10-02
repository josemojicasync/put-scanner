"""Tests del ajuste de cilindro 3D robusto (fase 8: connections.pipe_cylinder_3d) y de la separación
visual_geometry (solo dibujo) / measurement (diámetro). Superficies sintéticas; ningún número procede de un scan real.
Ejecutar:  python -m unittest discover -s tests -v
"""
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import analysis as A  # noqa: E402
import connections as C  # noqa: E402
import scan_result as SR  # noqa: E402
import synthetic as S  # noqa: E402
from test_joint_partial import AXIS, C0, U_AX, V_AX, tube  # noqa: E402


def cyl(P, Nn, excl=None, seed=0):
    excl = np.zeros(len(P), bool) if excl is None else excl
    return C.pipe_cylinder_3d(P, Nn, C0, AXIS, excl, np.random.default_rng(seed), np.radians(1.0))


class TestCylinder3D(unittest.TestCase):
    def test_partial_cylinder_recovered(self):
        """Arco parcial de 130° de un Ø300: el cilindro 3D recupera el diámetro dentro de ±U, con perfil acotado
        que contiene el valor real."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.002)
        cf = cyl(P, Nn)
        self.assertTrue(cf["measured"], cf["reason"])
        self.assertLessEqual(abs(cf["D"] - 0.30), cf["U"], f"{cf['D']:.4f} ± {cf['U']:.4f}")
        self.assertTrue(cf["profile_bounded"])
        lo, hi = cf["profile_interval_mm"]
        self.assertTrue(lo <= 300 <= hi, cf["profile_interval_mm"])

    def test_insufficient_arc_not_accepted(self):
        """Arco de 30°: sin información de curvatura -> no se acepta como diámetro."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(75, 105), noise=0.002)
        cf = cyl(P, Nn)
        self.assertFalse(cf.get("measured"), "un arco de 30° no debe dar diámetro")

    def test_cone_rejected(self):
        """Cono Ø280 -> Ø400 visto solo por arriba: un eje libre podría absorberlo como cilindro inclinado;
        el radio propio de las bandas lo delata -> rechazado y sin forma de cilindro."""
        P, Nn = tube([(0.0, 0.40, 0.14, 0.20)], arc=(25, 155), noise=0.002)
        cf = cyl(P, Nn)
        self.assertFalse(cf.get("measured"), "un cono no debe aceptarse como cilindro")
        self.assertFalse(cf.get("shape_ok"))

    def test_collar_is_not_pipe(self):
        """Banda de la cara (0-5 cm) con radio de collar Ø380, tubo Ø300 detrás: el collar no arrastra el diámetro."""
        P, Nn = tube([(0.0, 0.05, 0.19, 0.19), (0.05, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.002)
        cf = cyl(P, Nn)
        self.assertTrue(cf["measured"], cf["reason"])
        self.assertLessEqual(abs(cf["D"] - 0.30), cf["U"])
        self.assertLess(cf["D"], 0.34)

    def test_tilted_axis(self):
        """Eje real inclinado 8° respecto al supuesto: el cilindro 3D recupera eje y diámetro."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.002, tilt_deg=8.0)
        cf = cyl(P, Nn)
        self.assertTrue(cf["measured"], cf["reason"])
        self.assertLessEqual(abs(cf["D"] - 0.30), cf["U"])
        self.assertLess(abs(cf["tilt_deg"] - 8.0), 2.0, f"inclinación {cf['tilt_deg']:.1f}°")
        self.assertTrue(cf["axis_tilt_significant"])

    def test_water_occlusion(self):
        """Agua oculta el fondo (superficie plana excluida): el diámetro se estima con la parte visible y la BOB
        queda UNKNOWN (no se deriva del cilindro)."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(-20, 200), noise=0.002)
        rng = np.random.default_rng(3)
        n = 600
        wz = -0.08
        half = np.sqrt(0.15 ** 2 - wz ** 2)
        W = C0 + np.outer(rng.uniform(0.0, 0.40, n), AXIS) + np.outer(rng.uniform(-half, half, n), U_AX) + wz * V_AX
        Pw, Nw = np.vstack([P, W]), np.vstack([Nn, np.tile(V_AX, (n, 1))])
        excl = np.r_[np.zeros(len(P), bool), np.ones(n, bool)]
        cf = cyl(Pw, Nw, excl)
        self.assertTrue(cf["measured"], cf["reason"])
        self.assertLessEqual(abs(cf["D"] - 0.30), cf["U"])
        conn = dict(bob=A.meas(0.1, 0.01, A.LAAG, "x", A.ESTIMATED), note="")
        C._apply_cylinder_fit(conn, cf)
        self.assertEqual(conn["bob"]["status"], A.UNKNOWN)
        self.assertIsNone(conn["bob"]["value"])
        self.assertEqual(conn["diameter"]["status"], A.ESTIMATED)

    def test_leave_one_band_out_stable(self):
        """Quitar una banda no cambia el diámetro más allá de ±U."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.002)
        cf = cyl(P, Nn)
        self.assertGreaterEqual(len(cf["lobo_D_mm"]), 3)
        for d in cf["lobo_D_mm"]:
            self.assertLessEqual(abs(d / 1000 - cf["D"]), cf["U"], cf["lobo_D_mm"])

    def test_interior_contamination_does_not_bias(self):
        """15 % de puntos DENTRO del tubo (sedimento/objetos a 1-4 cm de la pared): el diámetro no se sesga."""
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.002)
        rng = np.random.default_rng(5)
        n = int(0.15 * len(P))
        ang = rng.uniform(np.radians(25), np.radians(155), n)
        rr = 0.15 - rng.uniform(0.01, 0.04, n)
        rad = np.outer(np.cos(ang), U_AX) + np.outer(np.sin(ang), V_AX)
        X = C0 + np.outer(rng.uniform(0.0, 0.40, n), AXIS) + rr[:, None] * rad
        cf = cyl(np.vstack([P, X]), np.vstack([Nn, -rad]))
        self.assertTrue(cf.get("ok_fit"))
        self.assertLessEqual(abs(cf["D"] - 0.30), max(cf["U"], 0.006), f"{cf['D']:.4f} ± {cf['U']:.4f}")

    @unittest.expectedFailure
    def test_oval_partial_arc_honest_uncertainty(self):
        """LIMITACIÓN CONOCIDA (fase 8c): tubo ovalado (5 %: Ø315 horizontal, Ø285 vertical) visto solo por un arco
        de 130° en la parte baja. Lo correcto sería rechazarlo o que ±U cubriera el diámetro medio (300), pero una
        ovalidad SUAVE no es identificable desde un arco parcial: todos los sub-arcos dan el mismo radio sesgado y
        ni el jackknife angular la detecta. Cubrirla exige un supuesto explícito de ovalidad máxima (decisión
        pendiente). Si este test empieza a pasar, quitar expectedFailure."""
        rng = np.random.default_rng(7)
        P, Nn = [], []
        ah, av = 0.1575, 0.1425
        for t in np.arange(0.0, 0.40, 0.012):
            th = np.arange(np.radians(-155), np.radians(-25), 0.012 / 0.15)
            pt = np.outer(ah * np.cos(th), U_AX) + np.outer(av * np.sin(th), V_AX)
            nrm = np.outer(np.cos(th) / ah, U_AX) + np.outer(np.sin(th) / av, V_AX)
            P.append(C0 + AXIS * t + pt)
            Nn.append(-nrm / np.linalg.norm(nrm, axis=1)[:, None])
        P = np.vstack(P) + rng.normal(0, 0.002, (sum(len(p) for p in P), 3))
        cf = cyl(P, np.vstack(Nn))
        self.assertGreater(cf["sigma_ang_mm"], 0.0)
        if cf.get("measured"):
            self.assertLessEqual(abs(cf["D"] - 0.30), cf["U"], f"{cf['D']:.4f} ± {cf['U']:.4f}")

    def test_deterministic(self):
        P, Nn = tube([(0.0, 0.40, 0.15, 0.15)], arc=(25, 155), noise=0.002)
        a, b = cyl(P, Nn), cyl(P, Nn)
        self.assertEqual(a["D"], b["D"])
        self.assertEqual(a["U"], b["U"])


class TestVisualGeometry(unittest.TestCase):
    def test_unknown_diameter_still_drawn_visual_only(self):
        """Tubo parcial cuyo diámetro queda UNKNOWN: sigue habiendo geometría de tubo para dibujar, con radio
        marcado SOLO VISUAL."""
        P, Nn = tube([(0.0, 0.15, 0.15, 0.15)], arc=(25, 155), noise=0.002)   # corto: no da diámetro
        cf = cyl(P, Nn)
        self.assertFalse(cf.get("measured"))
        conn = dict(diameter=A.unknown("pipe_bands"), axis=AXIS, c0=C0, tube_length_observed=0.15, visible_len=0.15,
                    opening=dict(status="CONFIRMED", width=0.30, height=0.20))
        vg = C.visual_geometry(conn, SimpleNamespace(n=AXIS, o=C0), cf)
        self.assertTrue(vg["confirmed_pipe"])
        self.assertTrue(vg["visual_only"])
        self.assertIn("SOLO VISUAL", vg["note"])
        self.assertGreater(vg["visible_length"], 0.05)
        self.assertLessEqual(vg["visible_length"], 0.20, "no inventar longitud de tubo no observada")

    def test_rejected_shape_not_used_for_display(self):
        """Un ajuste 3D rechazado por forma (cono) no da el radio de dibujo."""
        P, Nn = tube([(0.0, 0.40, 0.14, 0.20)], arc=(25, 155), noise=0.002)
        cf = cyl(P, Nn)
        conn = dict(diameter=A.unknown("pipe_bands"), axis=AXIS, c0=C0, tube_length_observed=0.40, visible_len=0.40,
                    opening=dict(status="CONFIRMED", width=0.30, height=0.20))
        vg = C.visual_geometry(conn, SimpleNamespace(n=AXIS, o=C0), cf, band_radii=[0.15, 0.16])
        self.assertNotEqual(vg["display_radius_source"], "visual_only_3d_fit")
        self.assertTrue(vg["visual_only"])

    def test_end_to_end_unknown_tube_json(self):
        """Put sintética con un tubo corto y parcial: diámetro UNKNOWN en el JSON (sin número) pero geometría
        visual presente y marcada visual_only."""
        r = A.analyze(S.rect_put(step=0.009, noise=0.003,
                                 pipes=[dict(wall="+Y", z=0.6, d=0.315, arc=(40, 140), length=0.12)]))
        self.assertEqual(len(r["connections"]), 1)
        c = r["connections"][0]
        self.assertEqual(c["diameter"]["status"], A.UNKNOWN)
        d = json.loads(SR.build_scan_result(r, "synthetic.ply", 0.0).to_json())["connections"][0]
        self.assertIsNone(d["diameter_mm"]["value"])
        vg = d["visual_geometry"]
        self.assertTrue(vg["confirmed_pipe"])
        self.assertTrue(vg["visual_only"])
        self.assertGreater(vg["display_radius_mm"], 0)
        self.assertLessEqual(vg["visible_length_mm"], 200)


if __name__ == "__main__":
    unittest.main()
