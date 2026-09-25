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

from analysis import HOOG, LAAG, MIDDEL, ONZEKER, AnalysisError, analyze, report

DEFAULT_SCAN = Path(__file__).parent / "data" / "scan.ply"
BG = [0.02, 0.03, 0.05, 1.0]
PANEL_BG = gui.Color(0.035, 0.05, 0.08, 1.0)
LEFT_W, RIGHT_W, ROWS = 210, 330, 90
C_TEXT, C_DIM, C_ACCENT = (0.8, 0.88, 0.95), (0.45, 0.55, 0.65), (0.35, 0.8, 1.0)
C_PUT_EDGE, C_DIM_LINE = (0.55, 0.75, 0.95), (0.9, 0.9, 0.9)
CONF_COLOR = {HOOG: (0.3, 0.9, 0.45), MIDDEL: (0.95, 0.85, 0.3), LAAG: (1.0, 0.55, 0.2), ONZEKER: (1.0, 0.3, 0.3)}


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


def circle_pts(center, radius, u, v, n=64):
    t = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return center + radius * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))


def ring(center, radius, u, v, color, n=64):
    p = circle_pts(center, radius, u, v, n)
    return lineset(p, [[i, (i + 1) % n] for i in range(n)], color)


def section_mesh(s, rect, z0):
    h = s["z1"] - z0
    if rect:
        m = o3d.geometry.TriangleMesh.create_box(2 * s["hx"], 2 * s["hy"], h)
        m.translate([s["cx"] - s["hx"], s["cy"] - s["hy"], z0])
    else:
        m = o3d.geometry.TriangleMesh.create_cylinder(s["hx"], h, resolution=48, split=1)
        m.translate([s["cx"], s["cy"], z0 + h / 2])
    m.compute_vertex_normals()
    return m


def pipe_mesh(c, length=0.6):
    """Cilindro de la conexión: empieza un poco dentro de la pared y sale hacia fuera."""
    d = c["direction"]
    m = o3d.geometry.TriangleMesh.create_cylinder(c["radius"], length, resolution=40)
    z = np.array([0, 0, 1.0])
    axis = np.cross(z, d)
    if np.linalg.norm(axis) > 1e-9:
        angle = np.arccos(np.clip(z @ d, -1, 1))
        m.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis / np.linalg.norm(axis) * angle), center=[0, 0, 0])
    m.translate(c["center"] + d * (length / 2 - 0.05))
    m.compute_vertex_normals()
    return m


def dimension(p0, p1, offset_dir, tick=0.04, color=C_DIM_LINE):
    """Línea de cota con marcas en los extremos; devuelve (lineset, punto para la etiqueta)."""
    p0, p1, o = np.asarray(p0, float), np.asarray(p1, float), np.asarray(offset_dir, float)
    pts = [p0, p1, p0 - o * tick, p0 + o * tick, p1 - o * tick, p1 + o * tick]
    return lineset(pts, [[0, 1], [2, 3], [4, 5]], color), (p0 + p1) / 2 + o * tick * 2


# ------------------------------------------------------------ aplicación

class PutScannerApp:
    def __init__(self, path, auto):
        app = gui.Application.instance
        app.initialize()
        self.f_title = app.add_font(gui.FontDescription(point_size=20))
        self.f_head = app.add_font(gui.FontDescription(point_size=15))
        self.f_mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE, point_size=14))

        self.path, self.result, self.labels3d, self.busy = None, None, [], False
        w = self.window = app.create_window("PUT SCANNER", 1600, 920)
        em = w.theme.font_size

        # centro: vista 3D + barra de vistas
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background(BG)
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0.4, -0.6, -1.0])
        self.viewbar = gui.Horiz(6, gui.Margins(8, 6, 8, 6))
        for text, fn in [("Bovenaanzicht", self.view_top), ("Zijaanzicht", self.view_side),
                         ("Reset camera", self.view_reset)]:
            b = gui.Button(text); b.set_on_clicked(fn); self.viewbar.add_child(b)

        # panel izquierdo
        left = self.left = gui.Vert(8, gui.Margins(14, 14, 14, 14))
        left.background_color = PANEL_BG
        left.add_child(self._label("PUT SCANNER", C_ACCENT, self.f_title))
        left.add_fixed(em * 0.5)
        b = gui.Button("Bestand openen"); b.set_on_clicked(self.on_open); left.add_child(b)
        self.btn_analyze = gui.Button("Analyseren"); self.btn_analyze.set_on_clicked(self.on_analyze)
        left.add_child(self.btn_analyze)
        left.add_fixed(em * 0.5)
        self.lbl_file = self._label("Geen bestand", C_DIM); left.add_child(self.lbl_file)
        self.lbl_status = self._label("", C_TEXT); left.add_child(self.lbl_status)
        left.add_fixed(em)
        left.add_child(self._label("WEERGAVE", C_DIM))
        self.chk_scan = gui.Checkbox("Toon scan"); self.chk_scan.set_on_checked(self.on_toggle); left.add_child(self.chk_scan)
        self.chk_model = gui.Checkbox("Toon model"); self.chk_model.checked = True
        self.chk_model.set_on_checked(self.on_toggle); left.add_child(self.chk_model)
        left.add_fixed(em)
        for s in ["Muis links: roteren", "Wiel: zoomen", "Ctrl + slepen: verschuiven"]:
            left.add_child(self._label(s, C_DIM))

        # panel derecho: filas reutilizables (Open3D GUI no permite quitar widgets)
        right = self.right = gui.ScrollableVert(4, gui.Margins(16, 14, 14, 14))
        right.background_color = PANEL_BG
        self.rows = []
        for _ in range(ROWS):
            lbl = gui.Label(""); lbl.visible = False; right.add_child(lbl); self.rows.append(lbl)
        self.set_rows([("RESULTATEN", C_DIM, self.f_head), ("Open een PLY en klik op Analyseren.", C_TEXT, None)])

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
        self.scene.frame = gui.Rect(r.x + LEFT_W, r.y, r.width - LEFT_W - RIGHT_W, r.height)
        pref = self.viewbar.calc_preferred_size(ctx, gui.Widget.Constraints())
        self.viewbar.frame = gui.Rect(r.x + LEFT_W + 10, r.y + 10, pref.width, pref.height)

    def set_rows(self, rows):
        """rows: lista de (texto, color, font o None). Texto vacío = separador."""
        for lbl, row in zip(self.rows, rows + [None] * (ROWS - len(rows))):
            lbl.visible = row is not None
            if row:
                text, color, font = row
                lbl.text = text or " "
                lbl.text_color = gui.Color(*color)
                lbl.font_id = font if font is not None else 0
        self.window.set_needs_layout()

    def status(self, text, color=C_TEXT):
        self.lbl_status.text = text
        self.lbl_status.text_color = gui.Color(*color)

    # --- escena
    def clear_scene(self):
        self.scene.scene.clear_geometry()
        for lbl in self.labels3d:
            self.scene.remove_3d_label(lbl)
        self.labels3d = []

    def add(self, name, geom, kind="lit", color=None, size=None):
        m = rendering.MaterialRecord()
        if kind == "line":
            m.shader, m.line_width = "unlitLine", size or 2
        elif kind == "points":
            m.shader, m.point_size = "defaultUnlit", size or 1.5
        elif kind == "glass":
            m.shader, m.base_color = "defaultLitTransparency", color
            m.base_roughness, m.base_reflectance = 0.8, 0.1
        else:
            m.shader, m.base_color = "defaultLit", color or (0.5, 0.5, 0.5, 1)
            m.base_roughness = 0.6
        self.scene.scene.add_geometry(name, geom, m)

    def label(self, pos, text, color=C_TEXT):
        lbl = self.scene.add_3d_label(np.asarray(pos, float), text)
        lbl.color = gui.Color(*color)
        self.labels3d.append(lbl)

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
        preview.paint_uniform_color([0.45, 0.55, 0.65])
        self.add("scan", preview, "points", size=1.5)
        self.chk_scan.checked = True
        self.set_rows([("RESULTATEN", C_DIM, self.f_head), ("Klik op Analyseren.", C_TEXT, None)])
        bbox = preview.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60, bbox, bbox.get_center())

    def on_analyze(self):
        if self.busy or not self.path:
            return
        self.busy = True
        self.status("Analyseren...", C_ACCENT)
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
            self.status(f"Analyse mislukt: {err}", CONF_COLOR[ONZEKER])
            return
        self.result = res
        print(f"\n[{self.path.name}] análisis en {dt:.1f} s\n{report(res)}")
        self.status(f"Analyse klaar ({dt:.1f} s)", CONF_COLOR[HOOG])
        self.build_model(res)
        self.fill_panel(res)
        self.view_reset()
        self.window.post_redraw()

    def on_toggle(self, _checked=None):
        sc = self.scene.scene
        if sc.has_geometry("scan"):
            sc.show_geometry("scan", self.chk_scan.checked)
        for name in self.model_names if self.result else []:
            sc.show_geometry(name, self.chk_model.checked)

    # --- modelo limpio
    def build_model(self, res):
        put, rect = res["put"], res["rect"]
        ch = put["chamber"]
        self.clear_scene()
        names = []

        def add(name, geom, *a, **k):
            self.add(name, geom, *a, **k); names.append(name)

        # nube (opcional) en el sistema local
        scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(res["local_points"]))
        scan.paint_uniform_color([0.5, 0.62, 0.75])
        self.add("scan", scan, "points", size=1.5)

        # tramos de la put: cámara desde el fondo, el resto apilado
        for i, s in enumerate(put["sections"]):
            z0 = 0.0 if s is ch else s["z0"]
            add(f"sec{i}", section_mesh(s, rect, z0), "glass", (0.35, 0.45, 0.6, 0.22))
            if rect:
                add(f"edge{i}", box_edges(s["cx"], s["cy"], s["hx"], s["hy"], z0, s["z1"], C_PUT_EDGE), "line", size=2)
            else:
                for zz in (z0, s["z1"]):
                    add(f"edge{i}_{zz:.2f}", ring([s["cx"], s["cy"], zz], s["hx"], [1, 0, 0], [0, 1, 0], C_PUT_EDGE), "line", size=2)
        floor = o3d.geometry.TriangleMesh.create_box(2 * ch["hx"], 2 * ch["hy"], 0.01)
        floor.translate([ch["cx"] - ch["hx"], ch["cy"] - ch["hy"], -0.01]); floor.compute_vertex_normals()
        add("floor", floor, "lit", (0.18, 0.22, 0.3, 1))

        # maaiveld
        g = max(ch["hx"], ch["hy"]) * 1.8
        add("ground", box_edges(0, 0, g, g, put["top_z"], put["top_z"], (0.3, 0.4, 0.3)), "line", size=1)
        self.label([g, -g, put["top_z"]], "maaiveld", (0.5, 0.7, 0.5))

        # cotas
        hx, hy, top = ch["hx"], ch["hy"], put["top_z"]
        ls, p = dimension([-hx, -hy - 0.12, 0], [hx, -hy - 0.12, 0], [0, 1, 0]); add("dimx", ls, "line", size=2)
        self.label(p + [0, -0.1, 0], f"X {put['width_x'] * 1000:.0f} mm", C_DIM_LINE)
        ls, p = dimension([hx + 0.12, -hy, 0], [hx + 0.12, hy, 0], [1, 0, 0]); add("dimy", ls, "line", size=2)
        self.label(p + [0.08, 0, 0], f"Y {put['width_y'] * 1000:.0f} mm", C_DIM_LINE)
        ls, p = dimension([hx + 0.12, -hy - 0.12, 0], [hx + 0.12, -hy - 0.12, top], [1, -1, 0]); add("dimz", ls, "line", size=2)
        self.label(p, f"diepte {put['depth']:.2f} m", C_DIM_LINE)
        if put["diameter"] is not None:
            self.label([0, 0, ch["z1"]], f"Ø {put['diameter'] * 1000:.0f} mm", C_DIM_LINE)

        # conexiones
        for i, c in enumerate(res["connections"]):
            col = CONF_COLOR[c["conf"]]
            if c["diameter"] is not None:
                add(f"pipe{i}", pipe_mesh(c), "lit", col + (1,))
                u = c["s_axis"]; v = np.array([0, 0, 1.0])
                add(f"pipering{i}", ring(c["center"], c["radius"], u, v, col), "line", size=3)
                txt = f"A{i + 1}  Ø {c['diameter'] * 1000:.0f} mm  {c['conf']}"
                pos = c["center"] + c["direction"] * 0.35 + [0, 0, c["radius"] + 0.05]
            else:  # sin medida fiable: solo la abertura visible, en rojo
                wc, u = c["wall_center"], c["s_axis"]
                w2, h2 = c["opening_w"] / 2, max(c["opening_h"], 0.02) / 2
                pts = [wc + u * a + [0, 0, b] for a, b in [(-w2, -h2), (w2, -h2), (w2, h2), (-w2, h2)]]
                add(f"open{i}", lineset(pts, [[0, 1], [1, 2], [2, 3], [3, 0]], col), "line", size=3)
                txt, pos = f"A{i + 1}  ONZEKER", wc + [0, 0, h2 + 0.08]
            self.label(pos, txt, col)

        # ejes locales
        axes = lineset([[0, 0, 0], [0.25, 0, 0], [0, 0.25, 0], [0, 0, 0.25]], [[0, 1], [0, 2], [0, 3]], (1, 1, 1))
        axes.colors = o3d.utility.Vector3dVector([[1, .3, .3], [.3, 1, .3], [.4, .5, 1]])
        add("axes", axes, "line", size=3)
        for p, t, c in [([0.28, 0, 0], "X", (1, .4, .4)), ([0, 0.28, 0], "Y", (.4, 1, .4)), ([0, 0, 0.28], "Z", (.5, .6, 1))]:
            self.label(p, t, c)

        self.model_names = names
        self.chk_scan.checked, self.chk_model.checked = False, True
        self.on_toggle()

    # --- panel de resultados
    def fill_panel(self, res):
        put, M, H = res["put"], self.f_mono, self.f_head
        mm = lambda v: f"{v * 1000:.0f} mm"
        rows = [("PUT", C_ACCENT, H),
                (f"Vorm       {put['shape']}", C_TEXT, M)]

        def meas(name, value, conf):
            rows.append((f"{name:<11}{value}", C_TEXT, M))
            rows.append((f"{'':<11}{conf}", CONF_COLOR[conf], M))

        if put["diameter"] is not None:
            meas("Diameter", mm(put["diameter"]), put["size_conf"])
            meas("Radius", mm(put["diameter"] / 2), put["size_conf"])
        else:
            rows.append(("Diameter   n.v.t. (rechthoekig)", C_DIM, M))
        meas("Depth", f"{put['depth']:.2f} m", put["depth_conf"])
        meas("Width X", mm(put["width_x"]), put["size_conf"])
        meas("Width Y", mm(put["width_y"]), put["size_conf"])
        rows.append((f"Buitenmaat onbekend", C_DIM, M))
        rows.append((f"{'':<11}{ONZEKER}", CONF_COLOR[ONZEKER], M))
        rows.append(("", C_DIM, None))
        rows.append(("Opbouw (van boven)", C_DIM, None))
        for s in reversed(put["sections"]):
            rows.append((f" {s['z0']:4.2f}-{s['z1']:4.2f} m  {2 * s['hx'] * 1000:.0f}x{2 * s['hy'] * 1000:.0f}", C_DIM, M))

        rows += [("", C_DIM, None), (f"AANSLUITINGEN ({len(res['connections'])})", C_ACCENT, H)]
        if not res["connections"]:
            rows.append(("Geen gevonden", C_DIM, None))
        for i, c in enumerate(res["connections"]):
            rows.append(("", C_DIM, None))
            rows.append((f"Aansluiting {i + 1}   [{c['conf']}]", CONF_COLOR[c["conf"]], None))
            if c["diameter"] is not None:
                rows.append((f"Diameter   {mm(c['diameter'])}", C_TEXT, M))
                rows.append((f"Radius     {mm(c['radius'])}", C_TEXT, M))
                rows.append((f"Boog       ~{c['coverage'] * 360:.0f}°  rms {c['fit_rms'] * 1000:.1f} mm", C_DIM, M))
                rows.append((f"BOB        {mm(c['invert_z'])} t.o.v. bodem", C_TEXT, M))
            else:
                rows.append(("Diameter   onzeker", CONF_COLOR[ONZEKER], M))
                rows.append((f"  {c['note']}", C_DIM, None))
            rows.append((f"Opening    {mm(c['opening_w'])} x {mm(c['opening_h'])}", C_TEXT, M))
            rows.append((f"Hoogte     {mm(c['height'])}", C_TEXT, M))
            rows.append((f"Diepte     {c['depth']:.2f} m", C_TEXT, M))
            rows.append((f"Hoek       {c['angle_deg']:.0f}°", C_TEXT, M))
            rows.append(("Richting   haaks op wand*", C_DIM, M))
        rows += [("", C_DIM, None),
                 ("Hoogte = as boven bodem", C_DIM, None), ("Diepte = as onder maaiveld", C_DIM, None),
                 ("Hoek = rond put, 0° = +X", C_DIM, None), ("* aangenomen, niet gemeten", C_DIM, None)]
        self.set_rows(rows[:ROWS])

    # --- cámaras (sistema local: Z arriba)
    def _box(self):
        if self.result:
            p = self.result["put"]; ch = p["chamber"]
            m = max(ch["hx"], ch["hy"]) + 0.3
            return o3d.geometry.AxisAlignedBoundingBox([-m, -m, -0.1], [m, m, p["top_z"] + 0.1])
        return self.scene.scene.bounding_box

    def _look(self, direction, up):
        b = self._box()
        c = b.get_center()
        self.scene.setup_camera(50, b, c)
        d = np.asarray(direction, float); d /= np.linalg.norm(d)
        self.scene.look_at(c, c + d * np.linalg.norm(b.get_extent()) * 1.2, up)

    def view_top(self):
        self._look([0, 0, 1], [0, 1, 0])

    def view_side(self):
        self._look([0, -1, 0], [0, 0, 1])

    def view_reset(self):
        self._look([1.0, -1.3, 0.8], [0, 0, 1])


def main():
    ap = argparse.ArgumentParser(description="PUT SCANNER")
    ap.add_argument("scan", nargs="?", default=DEFAULT_SCAN if DEFAULT_SCAN.is_file() else None)
    ap.add_argument("--auto", action="store_true", help="analizar automáticamente al abrir")
    args = ap.parse_args()
    PutScannerApp(args.scan, args.auto)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
