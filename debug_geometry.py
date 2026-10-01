"""DEBUG GEOMETRY: visor 3D de diagnóstico sobre la nube REAL (herramienta aparte; no cambia scanner.py).

Uso:
    python debug_geometry.py data/scan.ply
    python debug_geometry.py "data/Put 15.ply"

Panel izquierdo: qué mostrar.
  Put:        MEASURE X/Y/DEPTH, WALL ±X/±Y, BOTTOM, GROUND, ORIENTATION
  Por conexión (A1, A2, ...):
    A#            resumen: superficie usada para el diámetro + eje + círculo final + contorno de la abertura
    A# OPENING    celdas de la abertura en la cara (cian), puntos de borde (amarillo), contorno ajustado,
                  puntos vistos detrás de la abertura (azul)
    A# BANDS      puntos de cada banda por etiqueta (verde PIPE, naranja OPENING_RING, rojo REJECTED, magenta FILL)
                  y el círculo ajustado de cada banda perpendicular al eje
    A# USED       SOLO los puntos que participan en el diámetro (verde) + círculo final (amarillo) + centro
    A# REJECTED   puntos de la región del tubo NO usados (rojo), bandas no PIPE
    A# WATER      superficie de agua/sedimento/banket excluida (azul) y su nivel
    A# FILL       superficies transversales al eje (tapa/relleno, escalón, cara) excluidas (magenta)
  O#  candidato de abertura NO confirmado (posible / rechazado / relieve) y su razón
  R#  grupo de puntos detrás de la pared descartado (relieve de pared / sin abertura) y su razón
Opciones: "Hele scan (grijs)", "Residuen kleuren" (residuo firmado ±2 cm en planos), "Verworpen punten".
Todo en el sistema local de la put (origen = centro de la cámara al nivel del bodem, Z arriba).
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import analysis as A
import diagnose as D

BG = [0.06, 0.07, 0.08, 1.0]
PANEL = gui.Color(0.1, 0.11, 0.12, 1.0)
GREEN, RED, ORANGE, BLUE, GREY = (0.3, 0.9, 0.4), (1.0, 0.3, 0.3), (1.0, 0.6, 0.1), (0.35, 0.55, 1.0), (0.45, 0.47, 0.5)
YELLOW, WHITE, MAGENTA, CYAN, PLANE = (1.0, 0.85, 0.2), (1, 1, 1), (1.0, 0.3, 1.0), (0.2, 0.95, 0.95), (0.5, 0.75, 1.0, 0.25)
LABEL_COLOR = {"PIPE": GREEN, "OPENING_RING": ORANGE, "REJECTED": RED, "FILL": MAGENTA}
TEXT_LINES = 160
SUBVIEWS = ["OPENING", "BANDS", "USED", "REJECTED", "WATER", "FILL"]


def pcd(P, color):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float).reshape(-1, 3)))
    p.paint_uniform_color(color)
    return p


def residual_colors(d, sat=0.02):
    f = np.clip(d / sat, -1, 1)
    col = np.zeros((len(d), 3))
    col[:, 0] = np.where(f > 0, 1, 1 + f)
    col[:, 1] = 1 - np.abs(f)
    col[:, 2] = np.where(f < 0, 1, 1 - f)
    return col


def plane_quad(n, c, pts, margin=0.03):
    n = n / np.linalg.norm(n)
    u = np.cross(n, [0, 0, 1.0]) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    q = pts - c
    a0, a1 = np.percentile(q @ u, [1, 99]) + [-margin, margin]
    b0, b1 = np.percentile(q @ v, [1, 99]) + [-margin, margin]
    corners = [c + u * a + v * b for a, b in [(a0, b0), (a1, b0), (a1, b1), (a0, b1)]]
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(corners),
                                  o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3], [0, 2, 1], [0, 3, 2]]))
    m.compute_vertex_normals()
    return m


def lines(points, pairs, color):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.asarray(points, float)),
                              o3d.utility.Vector2iVector(np.asarray(pairs)))
    ls.paint_uniform_color(color)
    return ls


def ring3d(center, r, u, v, color, n=96):
    t = np.linspace(0, 2 * np.pi, n + 1)
    p = center + r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    return lines(p, [[i, i + 1] for i in range(n)], color)


def sphere(c, r, color):
    s = o3d.geometry.TriangleMesh.create_sphere(r, 10)
    s.translate(c)
    s.paint_uniform_color(color)
    s.compute_vertex_normals()
    return s


def box(Q):
    bb = o3d.geometry.AxisAlignedBoundingBox(Q.min(axis=0), Q.max(axis=0))
    ls = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bb)
    return np.asarray(ls.points), np.asarray(ls.lines)


class DebugApp:
    def __init__(self, path):
        self.path = Path(path)
        print(f"Analizando {self.path.name} ...")
        self.res = A.load_and_analyze(self.path)
        self.sections, self.geom = D.diagnose(self.res)
        put = self.res["put"]
        self.conn_ids = [c["id"] for c in self.res["connections"]]
        self.open_items = [o for o in put.get("openings", []) if o["status"] != "CONFIRMED"]
        self.rej_items = put.get("rejected_candidates", [])
        app = gui.Application.instance
        app.initialize()
        self.f_mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE, point_size=12))
        w = self.window = app.create_window(f"DEBUG GEOMETRY - {self.path.name}", 1750, 1000)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background(BG)
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0.3, -0.5, -1.0])

        left = self.left = gui.ScrollableVert(3, gui.Margins(8, 8, 8, 8))
        left.background_color = PANEL
        left.add_child(gui.Label("DEBUG GEOMETRY"))
        self.chk_scan = gui.Checkbox("Hele scan (grijs)"); self.chk_scan.checked = True
        self.chk_res = gui.Checkbox("Residuen kleuren")
        self.chk_rej = gui.Checkbox("Verworpen punten"); self.chk_rej.checked = True
        for c in (self.chk_scan, self.chk_res, self.chk_rej):
            c.set_on_checked(lambda _: self.show(self.current))
            left.add_child(c)
        items = ["ALLES", "MEASURE X", "WALL +X", "WALL -X", "MEASURE Y", "WALL +Y", "WALL -Y",
                 "MEASURE DEPTH", "BOTTOM", "GROUND", "ORIENTATION"]
        for cid in self.conn_ids:
            items += [cid] + [f"{cid} {s}" for s in SUBVIEWS]
        items += [f"O{i + 1}" for i in range(len(self.open_items))] + [f"R{i + 1}" for i in range(len(self.rej_items))]
        for it in items:
            b = gui.Button(it)
            b.horizontal_padding_em, b.vertical_padding_em = 0.3, 0.1
            b.set_on_clicked(lambda it=it: self.show(it))
            left.add_child(b)

        right = self.right = gui.ScrollableVert(1, gui.Margins(10, 10, 10, 10))
        right.background_color = PANEL
        self.text = []
        for _ in range(TEXT_LINES):
            lbl = gui.Label(""); lbl.font_id = self.f_mono; lbl.visible = False
            right.add_child(lbl); self.text.append(lbl)
        for wd in (self.scene, left, right):
            w.add_child(wd)
        w.set_on_layout(self._layout)
        self.current = "ALLES"
        self.show("ALLES")

    def _layout(self, ctx):
        r = self.window.content_rect
        lw, rw = 170, 700
        self.left.frame = gui.Rect(r.x, r.y, lw, r.height)
        self.right.frame = gui.Rect(r.get_right() - rw, r.y, rw, r.height)
        self.scene.frame = gui.Rect(r.x + lw, r.y, max(r.width - lw - rw, 64), r.height)

    # ---------------------------------------------------------------- utilidades de escena
    def add(self, name, geom, kind="points", color=None, size=None):
        m = rendering.MaterialRecord()
        if kind == "points":
            m.shader, m.point_size = "defaultUnlit", size or 3
        elif kind == "line":
            m.shader, m.line_width = "unlitLine", size or 2
        elif kind == "glass":
            m.shader, m.base_color = "defaultLitTransparency", color or PLANE
        else:
            m.shader = "defaultUnlit"
        self._n += 1
        self.scene.scene.add_geometry(f"{name}_{self._n}", geom, m)

    def set_text(self, lines_):
        out = []
        subst = {"≥": ">=", "≤": "<=", "≈": "~", "−": "-", "Ø": "D", "¡": "!", "–": "-", "∥": "||", "²": "2", "³": "3",
                 "—": "-", "σ": "s"}
        for ln in lines_:
            for k, v in subst.items():
                ln = ln.replace(k, v)
            while len(ln) > 100:
                cut = ln.rfind(" ", 0, 100)
                cut = cut if cut > 20 else 100
                out.append(ln[:cut]); ln = "    " + ln[cut:].lstrip()
            out.append(ln)
        for lbl, t in zip(self.text, out + [None] * (TEXT_LINES - len(out))):
            lbl.visible = t is not None
            lbl.text = t or ""
        self.window.set_needs_layout()

    # ---------------------------------------------------------------- selección
    def show(self, item):
        self.current = item
        self.scene.scene.clear_geometry()
        self._n = 0
        L = self.res["local_points"]
        if self.chk_scan.checked:
            self.add("scan", pcd(L, GREY), size=1.5)
        g = self.geom
        focus_pts, view = None, None
        parts = item.split(" ", 1)
        if item == "ALLES":
            for k in ("WALL +X", "WALL -X", "WALL +Y", "WALL -Y", "BOTTOM", "GROUND"):
                self.draw_plane_region(k, g.get(k, {}))
            for cn in self.res["connections"]:
                self.draw_used(cn)
            text = D.report_text(self.path, self.sections).splitlines()
        elif item.startswith("WALL") or item in ("BOTTOM", "GROUND"):
            self.draw_plane_region(item, g.get(item, {}))
            text = self.sections.get(item, [item])
            if "n" in g.get(item, {}):
                focus_pts = g[item]["candidates"]
                view = -g[item]["n"] + [0, 0, 0.3] if item.startswith("WALL") else np.array([0.3, -0.4, 1.0])
        elif item in ("MEASURE X", "MEASURE Y", "MEASURE DEPTH"):
            keys = g[item].get("walls", ["BOTTOM", "GROUND"])
            for k in keys:
                self.draw_plane_region(k, g.get(k, {}))
            seg = g[item]["segment"]
            if seg is not None:
                self.add("seg", lines(list(seg), [[0, 1]], WHITE), "line", size=4)
                self.add("s0", sphere(seg[0], 0.015, WHITE), "mesh"); self.add("s1", sphere(seg[1], 0.015, WHITE), "mesh")
            text = self.sections[item] + sum(([""] + self.sections.get(k, []) for k in keys), [])
        elif item == "ORIENTATION":
            text = self.sections[item]
        elif parts[0] in self.conn_ids:
            cn = next(c for c in self.res["connections"] if c["id"] == parts[0])
            sub = parts[1] if len(parts) > 1 else "SUMMARY"
            self.draw_connection(cn, sub)
            text = [f"[{item}]  " + self.legend(sub), ""] + self.sections.get(parts[0], [parts[0]])
            focus_pts, view = cn["points"], -cn["axis"] + 0.35 * cn["s_axis"] + [0, 0, 0.35]
        elif item.startswith("O") and item[1:].isdigit():
            op = self.open_items[int(item[1:]) - 1]
            self.draw_opening(op, tag=item)
            text = self.sections.get(item, [item])
            fr = op["frame"]
            focus_pts = np.array([fr.point(u, z) for u, z in op["edge_uv"]]) if len(op["edge_uv"]) else None
            view = -fr.n + [0, 0, 0.2]
        else:  # R#
            rc = self.rej_items[int(item[1:]) - 1]
            Q = L[rc["indices"]]
            self.add("rej", pcd(Q, MAGENTA), size=4)
            if len(Q):
                self.add("box", lines(*box(Q), WHITE), "line")
            text = self.sections.get(item, [item])
            focus_pts = Q
        self.set_text(text)
        self.focus(focus_pts, view)
        self.window.post_redraw()

    @staticmethod
    def legend(sub):
        return {"SUMMARY": "verde = puntos usados para el diámetro, amarillo = círculo final, blanco = eje, cian = contorno abertura",
                "OPENING": "cian = celdas de la abertura en la cara, amarillo = borde, blanco = contorno ajustado, azul = visto detrás",
                "BANDS": "verde PIPE, naranja OPENING_RING, rojo REJECTED, magenta FILL; círculo de cada banda",
                "USED": "SOLO los puntos del diámetro (verde), círculo final (amarillo), centro (blanco)",
                "REJECTED": "rojo = puntos de la región del tubo no usados; círculos de bandas no PIPE",
                "WATER": "azul = agua/sedimento/banket excluido, plano = nivel detectado",
                "FILL": "magenta = superficies transversales al eje (tapa/relleno, escalón, cara) excluidas"}[sub]

    def focus(self, pts, view):
        L = self.res["local_points"]
        if pts is None or len(pts) < 3:
            pts = L
        view = np.array([0.8, -1.0, 0.7]) if view is None else np.asarray(view, float)
        bb = o3d.geometry.AxisAlignedBoundingBox(pts.min(axis=0) - 0.05, pts.max(axis=0) + 0.05)
        c = bb.get_center()
        view = view / np.linalg.norm(view)
        self.scene.setup_camera(45, bb, c)
        self.scene.look_at(c, c + view * max(np.linalg.norm(bb.get_extent()) * 1.3, 0.8), [0, 0, 1])

    # ---------------------------------------------------------------- planos (paredes / bodem / maaiveld)
    def draw_plane_region(self, key, geo):
        P = geo.get("candidates")
        if P is None or not len(P):
            return
        if "inlier_mask" not in geo:
            self.add(key + "c", pcd(P, RED), size=3)
            return
        inl = geo["inlier_mask"]
        if self.chk_res.checked:
            sel = np.ones(len(P), bool) if self.chk_rej.checked else inl
            p = pcd(P[sel], WHITE)
            p.colors = o3d.utility.Vector3dVector(residual_colors(geo["residuals"][sel]))
            self.add(key + "r", p, size=3)
        else:
            self.add(key + "i", pcd(P[inl], GREEN), size=3)
            if self.chk_rej.checked and (~inl).any():
                self.add(key + "x", pcd(P[~inl], RED), size=3)
        self.add(key + "q", plane_quad(geo["n"], geo["c"], P[inl]), "glass")
        alt = geo.get("alt")
        if alt:
            self.add(key + "a", pcd(alt["points"], ORANGE), size=4)
            self.add(key + "aq", plane_quad(alt["n"], alt["c"], alt["points"]), "glass", (1.0, 0.6, 0.1, 0.25))
        if "eval" in geo:
            self.add(key + "e", sphere(list(geo["eval"]), 0.02, WHITE), "mesh")

    # ---------------------------------------------------------------- aberturas y conexiones
    def draw_opening(self, op, tag):
        fr = op["frame"]
        cells = np.argwhere(op["mask"])
        if len(cells):
            u = fr.u_range[0] + (cells[:, 0] + 0.5) * 0.02
            z = fr.z_range[0] + (cells[:, 1] + 0.5) * 0.02
            self.add(tag + "cells", pcd(np.array([fr.point(a, b) for a, b in zip(u, z)]), CYAN), size=6)
        if len(op["edge_uv"]):
            self.add(tag + "edge", pcd(np.array([fr.point(a, b) for a, b in op["edge_uv"]]), YELLOW), size=5)
        circ = op.get("circle")
        if circ is not None:
            t = np.linspace(0, 2 * np.pi, 97)
            pts = [fr.point(circ["c"][0] + circ["r"] * np.cos(a), circ["c"][1] + circ["r"] * np.sin(a)) for a in t]
            self.add(tag + "circ", lines(pts, [[i, i + 1] for i in range(96)], WHITE), "line", size=3)
        if len(op["behind_idx"]):
            self.add(tag + "behind", pcd(self.res["local_points"][op["behind_idx"]], BLUE), size=3)

    def draw_used(self, cn):
        Q = cn["points"]
        self.add(cn["id"] + "u", pcd(Q[cn["used_mask"]], GREEN), size=3)
        if cn["diameter"]["value"] is not None:
            self.add(cn["id"] + "ring", ring3d(cn["center"], cn["radius"], cn["s_axis"], cn["v_axis"], YELLOW), "line", size=3)

    def draw_connection(self, cn, sub):
        Q, tag = cn["points"], cn["id"]
        a, u, v, c0 = cn["axis"], cn["s_axis"], cn["v_axis"], cn["c0"]
        tmax = max([b["t1"] for b in cn["bands"]], default=0.3)
        self.add(tag + "axis", lines([c0, c0 + a * tmax], [[0, 1]], WHITE), "line", size=2)
        if sub == "SUMMARY":
            self.draw_used(cn)
            circ = cn["opening"].get("circle")
            if circ is not None:
                fr = cn["opening"]["frame"]
                t = np.linspace(0, 2 * np.pi, 97)
                pts = [fr.point(circ["c"][0] + circ["r"] * np.cos(x), circ["c"][1] + circ["r"] * np.sin(x)) for x in t]
                self.add(tag + "oc", lines(pts, [[i, i + 1] for i in range(96)], CYAN), "line", size=2)
            if self.chk_rej.checked:
                rest = ~(cn["used_mask"])
                self.add(tag + "rest", pcd(Q[rest], (0.6, 0.35, 0.35)), size=2)
        elif sub == "OPENING":
            self.draw_opening(cn["opening"], tag + "op")
        elif sub == "BANDS":
            for i, b in enumerate(cn["bands"]):
                col = LABEL_COLOR.get(b["label"], RED)
                if len(b["idx"]):
                    self.add(f"{tag}b{i}", pcd(Q[b["idx"]], col), size=3)
                if "center" in b:
                    self.add(f"{tag}br{i}", ring3d(b["center"], b["r"], u, v, col), "line", size=2)
        elif sub == "USED":
            self.add(tag + "u", pcd(Q[cn["used_mask"]], GREEN), size=4)
            if cn["diameter"]["value"] is not None:
                self.add(tag + "ring", ring3d(cn["center"], cn["radius"], u, v, YELLOW), "line", size=3)
                self.add(tag + "c", sphere(cn["center"], 0.012, WHITE), "mesh")
        elif sub == "REJECTED":
            rest = ~(cn["used_mask"] | cn["water_mask"] | cn["transverse_mask"])
            self.add(tag + "x", pcd(Q[rest], RED), size=3)
            for i, b in enumerate(cn["bands"]):
                if b["label"] != "PIPE" and "center" in b:
                    self.add(f"{tag}br{i}", ring3d(b["center"], b["r"], u, v, LABEL_COLOR.get(b["label"], RED)), "line", size=2)
        elif sub == "WATER":
            W = Q[cn["water_mask"]]
            if len(W):
                self.add(tag + "w", pcd(W, BLUE), size=4)
                self.add(tag + "wq", plane_quad(np.array([0, 0, 1.0]), W.mean(axis=0), W), "glass", (0.3, 0.5, 1.0, 0.3))
        elif sub == "FILL":
            T = Q[cn["transverse_mask"]]
            if len(T):
                self.add(tag + "t", pcd(T, MAGENTA), size=4)
            for i, b in enumerate(cn["bands"]):
                if b["label"] == "FILL" and "center" in b:
                    self.add(f"{tag}f{i}", ring3d(b["center"], b["r"], u, v, MAGENTA), "line", size=3)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "data" / "scan.ply")
    DebugApp(path)
    gui.Application.instance.run()
