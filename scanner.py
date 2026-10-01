"""PUT SCANNER - app principal: abrir PLY -> Analyseren -> modelo técnico 3D con medidas."""
import argparse
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from analysis import (ESTIMATED, HOOG, LAAG, MEASURED, MIDDEL, ONZEKER, SCALE_REL_SIGMA, UNKNOWN,
                      AnalysisError, analyze, report)
from scan_result import build_scan_result

DEFAULT_SCAN = Path(__file__).parent / "data" / "scan.ply"
BG = [0.075, 0.085, 0.10, 1.0]
PANEL_BG = gui.Color(0.105, 0.115, 0.13, 1.0)
LEFT_W, RIGHT_W, ROWS = 230, 440, 170
C_TEXT, C_DIM, C_HEAD = (0.86, 0.88, 0.9), (0.52, 0.56, 0.6), (0.62, 0.74, 0.86)
C_WALL, C_EDGE, C_FLOOR = (0.45, 0.52, 0.6, 0.16), (0.6, 0.67, 0.75), (0.24, 0.27, 0.31, 1.0)
C_DIM_LINE, C_EXT_LINE = (0.88, 0.9, 0.92), (0.45, 0.49, 0.54)
C_PIPE = (0.82, 0.58, 0.28)
CONF_COLOR = {HOOG: (0.4, 0.78, 0.5), MIDDEL: (0.85, 0.78, 0.35), LAAG: (0.92, 0.6, 0.3), ONZEKER: (0.9, 0.38, 0.35)}
STATUS_COLOR = {MEASURED: CONF_COLOR[HOOG], ESTIMATED: CONF_COLOR[LAAG], UNKNOWN: CONF_COLOR[ONZEKER]}
DIM_OFFSET = 0.22   # distancia de las cotas a la geometría
PIPE_LEN = 0.55


# ------------------------------------------------------------ formato

def fmt(m, force_mm=False):
    """< 1000 mm -> mm; mayor -> m. Siempre con incertidumbre."""
    v, U = m["value"], m["U"]
    if abs(v) < 1.0 or force_mm:
        return f"{v * 1000:.0f} ± {U * 1000:.0f} mm"
    return f"{v:.3f} ± {U:.3f} m"


def short(v):
    return f"{v * 1000:.0f} mm" if abs(v) < 1.0 else f"{v:.2f} m"


def dim_text(m):
    """Texto de cota en el modelo 3D: '?' si UNKNOWN, '~' si ESTIMATED."""
    if m["value"] is None:
        return "? mm"
    return ("~" if m["status"] == ESTIMATED else "") + f"{m['value'] * 1000:.0f} mm"


CONF_NL = {"HIGH": HOOG, "MEDIUM": MIDDEL, "LOW": LAAG, "NONE": ONZEKER}
QUALITY_COLOR = {"GOOD": (0.4, 0.78, 0.5), "FAIR": (0.85, 0.78, 0.35), "POOR": (0.92, 0.6, 0.3),
                 "RESCAN": (0.9, 0.38, 0.35)}


def fmt_m(m):
    """Measurement del ScanResult -> texto. mm < 1000 en mm, mayores en m; grados; UNKNOWN sin número.
    ESTIMATED lleva el prefijo ~ para no confundirlo con una medida."""
    if m.value is None:
        return "UNKNOWN"
    if isinstance(m.value, str):
        return m.value
    pre = "~ " if m.status == ESTIMATED else ""
    U = m.uncertainty
    if m.unit == "deg":
        return pre + f"{m.value:.1f}°" + ("" if U is None else f" ± {U:.1f}°")
    if abs(m.value) >= 1000:
        return pre + f"{m.value / 1000:.3f}" + ("" if U is None else f" ± {U / 1000:.3f}") + " m"
    return pre + f"{m.value:.0f}" + ("" if U is None else f" ± {U:.0f}") + " mm"


def mrow(name, m):
    color = CONF_COLOR[ONZEKER] if m.value is None else (C_DIM if m.status == ESTIMATED else C_TEXT)
    return ("kv", name, fmt_m(m), CONF_NL.get(m.confidence, ""), color)


def wrap(text, width):
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])


# ------------------------------------------------------------ geometría del modelo

def lineset(points, pairs, color):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.asarray(points, float)),
                              o3d.utility.Vector2iVector(np.asarray(pairs)))
    ls.paint_uniform_color(color)
    return ls


def box_edges(cx, cy, hx, hy, z0, z1, color):
    c = [(cx + sx * hx, cy + sy * hy) for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    pts = [(x, y, z0) for x, y in c] + [(x, y, z1) for x, y in c]
    pairs = [[i, (i + 1) % 4] for i in range(4)] + [[i + 4, (i + 1) % 4 + 4] for i in range(4)] + [[i, i + 4] for i in range(4)]
    return lineset(pts, pairs, color)


def ring(center, radius, u, v, color, n=64, dashed=False):
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    p = center + radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    pairs = [[i, (i + 1) % n] for i in range(0, n, 2 if dashed else 1)]
    return lineset(p, pairs, color)


def mesh_box(x0, y0, z0, x1, y1, z1):
    m = o3d.geometry.TriangleMesh.create_box(x1 - x0, y1 - y0, z1 - z0)
    m.translate([x0, y0, z0])
    m.compute_vertex_normals()
    return m


def wall_panels(s, z0, t=0.012):
    """4 paneles finos (para poder ocultar la pared frontal en la doorsnede)."""
    cx, cy, hx, hy, z1 = s["cx"], s["cy"], s["hx"], s["hy"], s["z1"]
    return {"+X": mesh_box(cx + hx, cy - hy, z0, cx + hx + t, cy + hy, z1),
            "-X": mesh_box(cx - hx - t, cy - hy, z0, cx - hx, cy + hy, z1),
            "+Y": mesh_box(cx - hx, cy + hy, z0, cx + hx, cy + hy + t, z1),
            "-Y": mesh_box(cx - hx, cy - hy - t, z0, cx + hx, cy - hy, z1)}


def pipe_mesh(c, length=PIPE_LEN):
    """Cilindro de la conexión: empieza en la cara interior de la pared y sale hacia fuera."""
    d = c["direction"]
    m = o3d.geometry.TriangleMesh.create_cylinder(c["radius"], length, resolution=48)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, d)
    if np.linalg.norm(axis) > 1e-9:
        angle = np.arccos(np.clip(z @ d, -1, 1))
        m.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle), center=[0, 0, 0])
    m.translate(c["center"] + d * length / 2)
    m.compute_vertex_normals()
    return m


def cad_dimension(p0, p1, out, ext_from=None, arrow=0.035):
    """Cota tipo CAD: línea con flechas, líneas de referencia desde la geometría. Devuelve (líneas cota, líneas ref)."""
    p0, p1, out = (np.asarray(v, float) for v in (p0, p1, out))
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    pts = [p0, p1,
           p0, p0 + d * arrow + out * arrow * 0.4, p0, p0 + d * arrow - out * arrow * 0.4,
           p1, p1 - d * arrow + out * arrow * 0.4, p1, p1 - d * arrow - out * arrow * 0.4]
    dim = lineset(pts, [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]], C_DIM_LINE)
    ext = None
    if ext_from is not None:
        e = [np.asarray(q, float) for q in ext_from]
        pe = [e[0], p0 + out * 0.03, e[1], p1 + out * 0.03]
        ext = lineset(pe, [[0, 1], [2, 3]], C_EXT_LINE)
    return dim, ext


# ------------------------------------------------------------ fila del panel derecho

class Row:
    """Fila de 3 columnas (nombre | valor | nivel). Open3D GUI no permite quitar widgets: se reutilizan."""

    def __init__(self, parent):
        self.h = gui.Horiz(6)
        self.a, self.b, self.c = gui.Label(""), gui.Label(""), gui.Label("")
        for w in (self.a, self.b, self.c):
            self.h.add_child(w)
        parent.add_child(self.h)
        self.h.visible = False

    def set(self, a="", b="", c="", ca=C_TEXT, cb=C_TEXT, cc=C_TEXT, fa=0, fb=0, fc=0):
        for lbl, text, col, font in ((self.a, a, ca, fa), (self.b, b, cb, fb), (self.c, c, cc, fc)):
            lbl.text, lbl.text_color, lbl.font_id = text or "", gui.Color(*col), font
        self.h.visible = True


# ------------------------------------------------------------ aplicación

class PutScannerApp:
    def __init__(self, path, auto):
        app = gui.Application.instance
        app.initialize()
        self.f_title = app.add_font(gui.FontDescription(point_size=19))
        self.f_head = app.add_font(gui.FontDescription(point_size=14))
        self.f_mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE, point_size=13))
        self.f_small = app.add_font(gui.FontDescription(point_size=11))

        self.path, self.result, self.labels3d, self.busy = None, None, [], False
        self.mode, self.model_names, self.front_names = "3d", [], []
        w = self.window = app.create_window("PUT SCANNER", 1680, 960)
        em = w.theme.font_size

        # centro: vista 3D + barra de vistas
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background(BG)
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0.3, -0.5, -1.0])
        self.viewbar = gui.Horiz(4, gui.Margins(6, 4, 6, 4))
        self.viewbar.background_color = PANEL_BG
        for text, fn in [("3D", self.view_3d), ("Bovenaanzicht", self.view_top), ("Zijaanzicht", self.view_side),
                         ("Doorsnede", self.view_section), ("Reset camera", self.view_reset)]:
            b = gui.Button(text); b.horizontal_padding_em, b.vertical_padding_em = 0.6, 0.2
            b.set_on_clicked(fn); self.viewbar.add_child(b)
        self.chk_scan = gui.Checkbox("Toon scan"); self.chk_scan.set_on_checked(lambda _: self.apply_visibility())
        self.viewbar.add_fixed(8); self.viewbar.add_child(self.chk_scan)

        # panel izquierdo
        left = self.left = gui.Vert(6, gui.Margins(16, 16, 16, 16))
        left.background_color = PANEL_BG
        left.add_child(self._label("PUT SCANNER", C_HEAD, self.f_title))
        left.add_child(self._label("put-inmeting uit 3D-scan", C_DIM, self.f_small))
        left.add_fixed(em)
        b = gui.Button("Bestand openen"); b.set_on_clicked(self.on_open); left.add_child(b)
        b = gui.Button("Analyseren"); b.set_on_clicked(self.on_analyze); left.add_child(b)
        b = gui.Button("Opslaan (JSON)"); b.set_on_clicked(self.on_save); left.add_child(b)
        left.add_fixed(em * 0.5)
        self.lbl_file = self._label("Geen bestand", C_DIM, self.f_small); left.add_child(self.lbl_file)
        self.lbl_status = self._label("", C_TEXT, self.f_small); left.add_child(self.lbl_status)
        left.add_stretch()
        left.add_child(self._label("SCAN QUALITY", C_HEAD, self.f_head))
        self.q_rows = []
        for _ in range(4):
            lbl = self._label("", C_DIM, self.f_small); left.add_child(lbl); self.q_rows.append(lbl)
        left.add_fixed(em * 0.5)
        left.add_child(self._label(f"± = 2 sigma (95%), incl. {SCALE_REL_SIGMA * 100:.1f}%\nschaalonzekerheid (aanname)", C_DIM, self.f_small))

        # panel derecho: filas reutilizables
        right = self.right = gui.ScrollableVert(3, gui.Margins(18, 16, 14, 16))
        right.background_color = PANEL_BG
        self.rows = [Row(right) for _ in range(ROWS)]
        self.set_rows([("head", "RESULTATEN"), ("note", "Open een PLY en klik op Analyseren.")])

        for widget in (self.scene, self.viewbar, left, right):
            w.add_child(widget)
        w.set_on_layout(self._layout)

        if path:
            self.load(Path(path))
            if auto:
                self.on_analyze()

    # --- helpers de interfaz
    def _label(self, text, color, font=None):
        lbl = gui.Label(text); lbl.text_color = gui.Color(*color)
        if font is not None:
            lbl.font_id = font
        return lbl

    def _layout(self, ctx):
        r = self.window.content_rect
        self.left.frame = gui.Rect(r.x, r.y, LEFT_W, r.height)
        self.right.frame = gui.Rect(r.get_right() - RIGHT_W, r.y, RIGHT_W, r.height)
        # nunca un ancho <= 0 (ventana minimizada o muy estrecha): Filament aborta
        self.scene.frame = gui.Rect(r.x + LEFT_W, r.y, max(r.width - LEFT_W - RIGHT_W, 64), max(r.height, 64))
        pref = self.viewbar.calc_preferred_size(ctx, gui.Widget.Constraints())
        self.viewbar.frame = gui.Rect(r.x + LEFT_W + 12, r.y + 12, pref.width, pref.height)

    def set_rows(self, rows):
        """rows: ("head", t) | ("sub", t) | ("note", t) | ("gap",) | ("kv", nombre, valor, nivel[, color valor])."""
        M, H, S = self.f_mono, self.f_head, self.f_small
        for row, spec in zip(self.rows, rows + [None] * (ROWS - len(rows))):
            if spec is None:
                row.h.visible = False
                continue
            kind = spec[0]
            if kind == "head":
                row.set(spec[1], "", spec[2] if len(spec) > 2 else "", C_HEAD, C_TEXT,
                        STATUS_COLOR.get(spec[2], C_DIM) if len(spec) > 2 else C_DIM, H, 0, S)
            elif kind == "sub":
                row.set(spec[1], "", "", C_TEXT, fa=H)
            elif kind == "note":
                row.set(spec[1], "", "", C_DIM, fa=S)
            elif kind == "warn":
                row.set(spec[1], "", "", QUALITY_COLOR["POOR"], fa=S)
            elif kind == "gap":
                row.set(" ", fa=S)
            else:
                _, name, value, level = spec[:4]
                vcol = spec[4] if len(spec) > 4 else C_TEXT
                row.set(f"{name:<13}", f"{value:>18}", level or "", C_DIM, vcol,
                        CONF_COLOR.get(level, C_DIM), M, M, M)
        self.window.set_needs_layout()

    def status(self, text, color=C_TEXT):
        self.lbl_status.text = text
        self.lbl_status.text_color = gui.Color(*color)

    # --- escena
    def clear_scene(self):
        self.scene.scene.clear_geometry()
        for lbl in self.labels3d:
            self.scene.remove_3d_label(lbl)
        self.labels3d, self.model_names, self.front_names = [], [], []

    def add(self, name, geom, kind="lit", color=None, size=None, model=True, front=False):
        m = rendering.MaterialRecord()
        if kind == "line":
            m.shader, m.line_width = "unlitLine", size or 1.5
        elif kind == "points":
            m.shader, m.point_size = "defaultUnlit", size or 1.5
        elif kind == "glass":
            m.shader, m.base_color = "defaultLitTransparency", color
            m.base_roughness, m.base_reflectance = 0.9, 0.05
        else:
            m.shader, m.base_color = "defaultLit", color or (0.5, 0.5, 0.5, 1)
            m.base_roughness = 0.7
        self.scene.scene.add_geometry(name, geom, m)
        if model:
            self.model_names.append(name)
        if front:
            self.front_names.append(name)

    def label(self, pos, text, color=C_TEXT):
        lbl = self.scene.add_3d_label(np.asarray(pos, float), text)
        lbl.color = gui.Color(*color)
        self.labels3d.append(lbl)

    def apply_visibility(self):
        sc = self.scene.scene
        section = self.mode == "section"
        for name in ("scan", "scan_cut"):
            if sc.has_geometry(name):
                want = self.chk_scan.checked and ((name == "scan_cut") == section or not self.result)
                sc.show_geometry(name, want)
        for name in self.front_names:
            sc.show_geometry(name, not section)
        self.window.post_redraw()

    # --- acciones
    def on_open(self):
        dlg = gui.FileDialog(gui.FileDialog.OPEN, "PLY openen", self.window.theme)
        dlg.add_filter(".ply", "PLY-scan (.ply)")
        if self.path:
            dlg.set_path(str(self.path.parent))
        dlg.set_on_cancel(self.window.close_dialog)
        dlg.set_on_done(lambda p: (self.window.close_dialog(), self.load(Path(p))))
        self.window.show_dialog(dlg)

    def load(self, path):
        pcd = o3d.io.read_point_cloud(str(path))
        if len(pcd.points) == 0:
            self.status("Kan PLY niet lezen", CONF_COLOR[ONZEKER])
            return
        self.path, self.pcd, self.result = path, pcd, None
        self.lbl_file.text = f"{path.name}\n{len(pcd.points):,} punten"
        self.status("Klaar om te analyseren")
        self.clear_scene()
        preview = o3d.geometry.PointCloud(pcd.points)
        preview.paint_uniform_color([0.55, 0.6, 0.66])
        self.add("scan", preview, "points", size=1.5, model=False)
        self.chk_scan.checked = True
        for lbl in self.q_rows:
            lbl.text = ""
        self.set_rows([("head", "RESULTATEN"), ("note", "Klik op Analyseren.")])
        bbox = preview.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bbox, bbox.get_center())

    def on_analyze(self):
        if self.busy or not self.path:
            return
        self.busy = True
        self.status("Analyseren...", C_HEAD)
        pcd = self.pcd

        def work():
            t0 = time.time()
            try:
                res, err = analyze(pcd), None
            except AnalysisError as e:
                res, err = None, str(e)
            except Exception as e:  # cualquier fallo inesperado se muestra en la interfaz
                traceback.print_exc()
                res, err = None, f"{type(e).__name__}: {e}"
            dt = time.time() - t0
            gui.Application.instance.post_to_main_thread(self.window, lambda: self.on_done(res, err, dt))

        threading.Thread(target=work, daemon=True).start()

    def on_done(self, res, err, dt):
        self.busy = False
        if err:
            self.status(f"Analyse mislukt:\n{err}", CONF_COLOR[ONZEKER])
            return
        self.result = res
        self.scan_result = build_scan_result(res, self.path, dt)
        print(f"\n[{self.path.name}] análisis en {dt:.2f} s\n{report(res)}")
        self.status(f"Analyse klaar ({dt:.1f} s)", CONF_COLOR[HOOG])
        self.build_model(res)                 # geometría 3D: necesita puntos/planos del análisis
        self.fill_panel(self.scan_result)     # textos: solo del ScanResult
        self.fill_quality(self.scan_result)
        self.view_reset()

    # --- modelo limpio
    def build_model(self, res):
        put, rect = res["put"], res["rect"]
        ch, top = put["chamber"], put["top_z"]
        conns = res["connections"]
        self.clear_scene()

        # nube (opcional) en el sistema local; versión cortada para la doorsnede
        L = res["local_points"]
        for name, pts in (("scan", L), ("scan_cut", L[L[:, 1] > 0])):
            p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
            p.paint_uniform_color([0.62, 0.68, 0.74])
            self.add(name, p, "points", size=1.5, model=False)

        # put: paredes finas semitransparentes + aristas
        for i, s in enumerate(put["sections"]):
            z0 = 0.0 if s is ch else s["z0"]
            if rect:
                for key, m in wall_panels(s, z0).items():
                    self.add(f"wall{i}{key}", m, "glass", C_WALL, front=(key == "-Y"))
                self.add(f"edge{i}", box_edges(s["cx"], s["cy"], s["hx"], s["hy"], z0, s["z1"], C_EDGE), "line", size=1.5)
            else:
                m = o3d.geometry.TriangleMesh.create_cylinder(s["hx"], s["z1"] - z0, resolution=64, split=1)
                m.translate([s["cx"], s["cy"], (z0 + s["z1"]) / 2]); m.compute_vertex_normals()
                self.add(f"wall{i}", m, "glass", C_WALL)
                for j, zz in enumerate((z0, s["z1"])):
                    self.add(f"edge{i}_{j}", ring([s["cx"], s["cy"], zz], s["hx"], [1, 0, 0], [0, 1, 0], C_EDGE), "line", size=1.5)
        self.add("floor", mesh_box(-ch["hx"], -ch["hy"], -0.02, ch["hx"], ch["hy"], 0.0), "lit", C_FLOOR)
        g = max(ch["hx"], ch["hy"]) + 0.45
        self.add("ground", box_edges(0, 0, g, g, top, top, (0.35, 0.4, 0.38)), "line", size=1)

        # conexiones: medida -> sólida; estimada -> semitransparente; desconocida -> contorno de la abertura
        for i, c in enumerate(conns):
            st = c["status"]
            f = c["direction"][1] < -0.7  # conexión en la pared frontal: se oculta en la doorsnede
            if st in (MEASURED, ESTIMATED) and "radius" in c:
                u, v = c["s_axis"], np.array([0, 0, 1.0])
                if st == MEASURED:
                    self.add(f"pipe{i}", pipe_mesh(c), "lit", C_PIPE + (1.0,), front=f)
                    self.add(f"pring{i}", ring(c["center"], c["radius"], u, v, C_PIPE), "line", size=2, front=f)
                else:
                    self.add(f"pipe{i}", pipe_mesh(c), "glass", C_PIPE + (0.28,), front=f)
                    self.add(f"pring{i}", ring(c["center"], c["radius"], u, v, C_PIPE, dashed=True), "line",
                             size=1.5, front=f)
                    self.add(f"pring_out{i}", ring(c["center"] + c["direction"] * PIPE_LEN, c["radius"], u, v,
                                                   C_PIPE, dashed=True), "line", size=1.5, front=f)
                end = c["center"] + c["direction"] * (PIPE_LEN + 0.08) + [0, 0, c["radius"] * 0.3]
            else:
                # Pipe diameter UNKNOWN: show only the REAL observed wall-opening edge.
                # Never fabricate a rectangle from opening_w/opening_h.
                wc = c["wall_center"]
                op = c.get("opening", {})
                edge_uv = np.asarray(op.get("edge_uv", []), dtype=float)
                fr = op.get("frame")

                if fr is not None and len(edge_uv) >= 2:
                    center_uv = edge_uv.mean(axis=0)
                    rel = edge_uv - center_uv
                    ang = np.arctan2(rel[:, 1], rel[:, 0])
                    order = np.argsort(ang)
                    edge_uv = edge_uv[order]
                    ang = ang[order]

                    # edge_uv is in wall coordinates (u, z); WallFrame.point()
                    # maps those observations back to the correct local 3D wall.
                    pts = [fr.point(float(uv[0]), float(uv[1])) for uv in edge_uv]

                    # Connect only neighboring observed sectors. Missing arcs remain
                    # missing instead of being visually invented.
                    pairs = []
                    max_gap = np.deg2rad(20)
                    for j in range(len(pts) - 1):
                        if float(ang[j + 1] - ang[j]) <= max_gap:
                            pairs.append([j, j + 1])

                    wrap_gap = float((ang[0] + 2 * np.pi) - ang[-1])
                    if len(pts) >= 3 and wrap_gap <= max_gap:
                        pairs.append([len(pts) - 1, 0])

                    if pairs:
                        self.add(
                            f"open{i}",
                            lineset(pts, pairs, CONF_COLOR[ONZEKER]),
                            "line",
                            size=2,
                            front=f,
                        )

                end = wc + c["direction"] * 0.15
            self.label(end, f"A{i + 1}", STATUS_COLOR[st])

        self._dimensions(put, conns)

    def _dimensions(self, put, conns):
        """Cotas fuera de la geometría; se elige el lado/altura sin conexiones cerca."""
        ch, top = put["chamber"], put["top_z"]
        hx, hy, zt = ch["hx"], ch["hy"], ch["z1"]

        def conflict(side_axis, side, z):
            n = 0
            for c in conns:
                ref = c.get("center", c["wall_center"])
                r = c.get("radius", 0.15)
                if c["direction"][side_axis] * side > 0.7 and abs(ref[2] - z) < r + 0.2:
                    n += 1
            return n

        def pick(side_axis):
            cands = [(s, z) for z in (0.0, zt) for s in (-1, 1)]
            return min(cands, key=lambda sz: (conflict(side_axis, *sz), cands.index(sz)))

        # Binnenmaat X: paralela a X, en el lado ±Y
        s, z = pick(1)
        side_x = s
        y = s * (hy + DIM_OFFSET)
        dim, ext = cad_dimension([-hx, y, z], [hx, y, z], [0, s, 0], ext_from=[[-hx, s * hy, z], [hx, s * hy, z]])
        self.add("dimx", dim, "line", size=1.5); self.add("dimx_ext", ext, "line", size=1)
        self.label([-0.08, y + s * 0.07, z], dim_text(put["width_x"]), C_DIM_LINE)

        # Binnenmaat Y: paralela a Y, en el lado ±X
        s, z = pick(0)
        side_y = s
        x = s * (hx + DIM_OFFSET)
        dim, ext = cad_dimension([x, -hy, z], [x, hy, z], [s, 0, 0], ext_from=[[s * hx, -hy, z], [s * hx, hy, z]])
        self.add("dimy", dim, "line", size=1.5); self.add("dimy_ext", ext, "line", size=1)
        self.label([x + s * 0.07, -0.08, z], dim_text(put["width_y"]), C_DIM_LINE)

        # Diepte: vertical en la esquina opuesta a las cotas X/Y (sin solapes de texto)
        sx, sy = -side_y, -side_x
        p = np.array([sx * (hx + DIM_OFFSET), sy * (hy + DIM_OFFSET), 0])
        out = np.array([sx, sy, 0]) / np.sqrt(2)
        dim, ext = cad_dimension(p, p + [0, 0, top], out,
                                 ext_from=[[sx * hx, sy * hy, 0], [sx * hx * 1.0, sy * hy * 1.0, top]])
        self.add("dimz", dim, "line", size=1.5); self.add("dimz_ext", ext, "line", size=1)
        self.label(p + [0, 0, top / 2] + out * 0.06,
                   ("~" if put["depth"]["status"] == ESTIMATED else "") + f"{put['depth']['value']:.2f} m", C_DIM_LINE)

    # --- panel de resultados (lee SOLO del ScanResult: datos separados de la visualización)
    def fill_panel(self, sr):
        put, q = sr.put, sr.quality
        rows = [("head", "SCAN QUALITY")]
        for name, st in q.items.items():
            rows.append(("kv", name, st, "", QUALITY_COLOR.get(st, C_TEXT)))
        for wmsg in q.warnings:
            for i, line in enumerate(wrap(wmsg, 54)):
                rows.append(("warn", ("! " if i == 0 else "  ") + line))

        rows += [("gap",), ("head", "PUT"), mrow("Vorm", put.shape)]
        if put.shape.value == "rond":
            rows.append(mrow("Diameter", put.diameter_mm))
            rows.append(mrow("Circulariteit", put.circularity_mm))
        else:
            rows.append(mrow("Binnenmaat X", put.width_x_mm))
            rows.append(mrow("Binnenmaat Y", put.width_y_mm))
        rows.append(mrow("Diepte", put.depth_mm))
        if put.depth_mm.value is not None:
            rows.append(("kv", "", f"{put.depth_mm.value:.0f} ± {put.depth_mm.uncertainty:.0f} mm", "", C_DIM))
        rows.append(mrow("Oriëntatie", put.orientation))
        rows.append(("kv", "Buitenmaat", "niet zichtbaar", ONZEKER, C_DIM))
        if put.walls:
            rows += [("gap",), ("note", "WANDEN (RANSAC-vlakken)")]
            for wl in put.walls:
                txt = f"{wl.points} ptn  {wl.rms_mm:.1f} mm  {wl.coverage * 100:.0f}%" if wl.scanned else "NIET GESCAND"
                rows.append(("kv", f"  wand {wl.id}", txt, "", C_DIM if wl.scanned else CONF_COLOR[ONZEKER]))
            names = dict(parallel_x="parallel X", parallel_y="parallel Y", perpendicular="haaks X/Y")
            for k, v in put.wall_geometry.items():
                rows.append(("kv", f"  {names[k]}", f"{v:.2f}°", "", C_DIM))
        rows += [("gap",), ("note", "OPBOUW (van boven, indicatief)")]
        for s in reversed(put.sections):
            rows.append(("kv", f"  {s['z0_mm'] / 1000:.2f}-{s['z1_mm'] / 1000:.2f} m",
                         f"{s['size_x_mm']} x {s['size_y_mm']}", "", C_DIM))

        rows += [("gap",), ("head", f"AANSLUITINGEN ({len(sr.connections)})")]
        if not sr.connections:
            rows.append(("note", "Geen gevonden"))
        for c in sr.connections:
            rows += [("gap",), ("head", f"{c.id}  wand {c.wall}", c.diameter_mm.status)]
            rows.append(("kv", "Hoek", f"{c.angle_deg:.0f}°", ""))
            rows.append(mrow("Diameter", c.diameter_mm))
            if c.nominal_suggestions:
                rows.append(("kv", "  suggestie", " / ".join(f"Ø{n}" for n in c.nominal_suggestions) + "?", "", C_DIM))
            rows.append(("kv", "  zichtbaar", f"boog {c.visible_arc_deg:.0f}°  "
                         f"{c.observed_opening_mm['width']}x{c.observed_opening_mm['height']}", "", C_DIM))
            rows.append(mrow("Kruin", c.crown_mm))
            rows.append(mrow("BOB", c.invert_bob_mm))
            rows.append(mrow("Hart", c.center_height_mm))
            rows.append(mrow("BOB diepte", c.bob_depth_mm))
            if c.inner_height_mm.value is not None:
                rows.append(mrow("Binnenhoogte", c.inner_height_mm))
            rows.append(mrow("Richting", c.direction))
            rows.append(mrow("Helling", c.slope_deg))
        rows += [("gap",),
                 ("note", "Kruin/BOB/Hart: t.o.v. bodem. BOB diepte: t.o.v. maaiveld."),
                 ("note", "Hoek: rond de put, 0° = +X. Richting: afwijking van de wandnormaal."),
                 ("note", "~ = geschat (ESTIMATED) · UNKNOWN = niet meetbaar met deze scan"),
                 ("note", "Suggestie nominaal = GEEN meting.")]
        if len(rows) > ROWS:
            rows = rows[:ROWS - 1] + [("note", "… (zie JSON / console voor de rest)")]
        self.set_rows(rows)

    def fill_quality(self, sr):
        q = sr.quality
        texts = [f"Punten   {q.analysed_points:,} geanalyseerd / {q.total_points:,}",
                 f"Puntafstand   {q.point_spacing_mm:.1f} mm   ruis {q.noise_mm or 0:.1f} mm",
                 f"Wanddekking   {q.chamber_coverage * 100:.0f}%   bodem {q.bottom_coverage * 100:.0f}%",
                 f"Schaal   {q.scale['units']}, niet geverifieerd"]
        for lbl, t in zip(self.q_rows, texts):
            lbl.text = t

    def on_save(self):
        if not getattr(self, "scan_result", None):
            self.status("Eerst analyseren", CONF_COLOR[LAAG])
            return
        out = Path(__file__).parent / "results" / f"{self.path.stem}_scanresult.json"
        self.scan_result.save(out)
        self.status(f"Opgeslagen:\nresults/{out.name}", CONF_COLOR[HOOG])

    # --- cámaras (sistema local: Z arriba)
    def _box(self):
        if self.result:
            p = self.result["put"]; ch = p["chamber"]
            m = max(ch["hx"], ch["hy"]) + DIM_OFFSET + 0.15
            return o3d.geometry.AxisAlignedBoundingBox([-m, -m, -0.1], [m, m, p["top_z"] + 0.1])
        return self.scene.scene.bounding_box

    def _camera(self, direction, up, fov):
        """Vistas técnicas con FOV pequeño (casi ortográficas) manteniendo el control con el ratón."""
        b = self._box()
        c = b.get_center()
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        dist = np.linalg.norm(b.get_extent()) / 2 / np.tan(np.radians(fov) / 2) * 1.05
        self.scene.setup_camera(fov, b, c)
        cam = self.scene.scene.camera
        fr = self.scene.frame
        aspect = fr.width / fr.height if fr.width > 0 and fr.height > 0 else 1.5
        cam.set_projection(fov, aspect, 0.05, dist * 3, rendering.Camera.FovType.Vertical)
        self.scene.look_at(c, c + d * dist, up)
        self.window.post_redraw()

    def _set_mode(self, mode):
        self.mode = mode
        self.apply_visibility()

    def view_3d(self):
        self._set_mode("3d"); self._camera([1.0, -1.25, 0.75], [0, 0, 1], 40)

    def view_top(self):
        self._set_mode("3d"); self._camera([0, 0, 1], [0, 1, 0], 8)

    def view_side(self):
        self._set_mode("3d"); self._camera([0, -1, 0], [0, 0, 1], 8)

    def view_section(self):
        self._set_mode("section"); self._camera([0, -1, 0], [0, 0, 1], 8)

    def view_reset(self):
        self.chk_scan.checked = False
        self.view_3d()


def main():
    ap = argparse.ArgumentParser(description="PUT SCANNER")
    ap.add_argument("scan", nargs="?", default=DEFAULT_SCAN if DEFAULT_SCAN.is_file() else None)
    ap.add_argument("--auto", action="store_true", help="analizar automáticamente al abrir")
    args = ap.parse_args()
    PutScannerApp(args.scan, args.auto)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
