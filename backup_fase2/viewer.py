"""Estilo visual de Put Scanner: nube azul/cian, fondo oscuro, cuadrícula y ventana GUI con panel."""
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

from measurement import AXIS_NAMES, UP_AXIS

BG = [0.015, 0.025, 0.05]
PANEL_BG = gui.Color(0.03, 0.05, 0.09, 1.0)
LOW, HIGH = np.array([0.0, 0.06, 0.35]), np.array([0.0, 0.55, 0.9])  # azul profundo -> cian
C_TEXT, C_DIM = (0.75, 0.9, 1.0), (0.4, 0.55, 0.7)
C_CYAN, C_SEL, C_OUT = (0.3, 0.9, 1.0), (1.0, 0.85, 0.1), (1.0, 0.4, 0.1)
C_CIRCLE, C_CENTER = (0.2, 1.0, 0.3), (1.0, 1.0, 1.0)
PANEL_W = 330
GRID_STEP_M = 0.1


# ---------- geometría de estilo ----------

def scanner_cloud(points):
    """Copia en memoria de la nube coloreada azul->cian según la altura (el PLY no se toca)."""
    h = points[:, UP_AXIS]
    t = ((h - h.min()) / max(np.ptp(h), 1e-9))[:, None] ** 0.8
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.colors = o3d.utility.Vector3dVector(LOW * (1 - t) + HIGH * t)
    return pcd


def _to3d(a, b, up):
    """Convierte coordenadas (horizontal a, horizontal b, vertical) a XYZ según UP_AXIS."""
    h = [i for i in range(3) if i != UP_AXIS]
    p = np.zeros((len(a), 3))
    p[:, h[0]], p[:, h[1]], p[:, UP_AXIS] = a, b, up
    return p


def lines(points, pairs, color):
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.asarray(points, float)),
                              o3d.utility.Vector2iVector(np.asarray(pairs)))
    ls.paint_uniform_color(color)
    return ls


def make_grid(points):
    """Cuadrícula en el suelo (mínimo del eje vertical), cada GRID_STEP_M metros."""
    h = [i for i in range(3) if i != UP_AXIS]
    lo = np.floor(points[:, h].min(axis=0) / GRID_STEP_M) * GRID_STEP_M - GRID_STEP_M
    hi = np.ceil(points[:, h].max(axis=0) / GRID_STEP_M) * GRID_STEP_M + GRID_STEP_M
    floor = points[:, UP_AXIS].min()
    pts, pairs = [], []
    for a in np.arange(lo[0], hi[0] + 1e-9, GRID_STEP_M):
        pairs.append([len(pts), len(pts) + 1]); pts += [(a, lo[1]), (a, hi[1])]
    for b in np.arange(lo[1], hi[1] + 1e-9, GRID_STEP_M):
        pairs.append([len(pts), len(pts) + 1]); pts += [(lo[0], b), (hi[0], b)]
    pts = np.array(pts)
    return lines(_to3d(pts[:, 0], pts[:, 1], floor), pairs, (0.08, 0.16, 0.26))


def make_axes(points):
    """Ejes XYZ (rojo/verde/azul) desde la esquina inferior; devuelve (lineset, puntas)."""
    origin = points.min(axis=0)
    size = float(np.ptp(points, axis=0).max()) * 0.25
    tips = origin + np.eye(3) * size
    ls = o3d.geometry.LineSet(o3d.utility.Vector3dVector(np.vstack([origin, tips])),
                              o3d.utility.Vector2iVector([[0, 1], [0, 2], [0, 3]]))
    ls.colors = o3d.utility.Vector3dVector([[1, 0.25, 0.25], [0.3, 1, 0.3], [0.3, 0.5, 1]])
    return ls, tips


def sphere(center, radius, color):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(center)
    s.paint_uniform_color(color)
    return s


# ---------- ventana de selección (Open3D clásico) ----------

def style_legacy(vis):
    opt = vis.get_render_option()
    opt.background_color = np.array(BG)
    opt.point_size = 3.0
    opt.show_coordinate_frame = True


# ---------- ventana PUT SCANNER (Open3D GUI) ----------

class ScannerWindow:
    """Vista 3D + panel lateral de medidas."""

    def __init__(self, subtitle):
        app = gui.Application.instance
        app.initialize()
        self.f_title = app.add_font(gui.FontDescription(point_size=22))
        self.f_big = app.add_font(gui.FontDescription(point_size=18))
        self.f_mono = app.add_font(gui.FontDescription(gui.FontDescription.MONOSPACE, point_size=16))

        self.window = app.create_window(f"PUT SCANNER - {subtitle}", 1400, 860)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background(BG + [1.0])
        self.scene.scene.set_lighting(rendering.Open3DScene.LightingProfile.NO_SHADOWS, [0, 0, -1])

        self.panel = gui.Vert(4, gui.Margins(16, 16, 16, 16))
        self.panel.background_color = PANEL_BG
        self.text("PUT SCANNER", C_CYAN, self.f_title)
        self.text(subtitle.upper(), C_DIM)
        self.gap()

        self.window.add_child(self.scene)
        self.window.add_child(self.panel)
        self.window.set_on_layout(self._layout)
        self._n = 0

    def _layout(self, _ctx):
        r = self.window.content_rect
        self.scene.frame = gui.Rect(r.x, r.y, r.width - PANEL_W, r.height)
        self.panel.frame = gui.Rect(r.get_right() - PANEL_W, r.y, PANEL_W, r.height)

    # panel
    def text(self, s, color=C_TEXT, font=None):
        lbl = gui.Label(s)
        lbl.text_color = gui.Color(*color)
        if font is not None:
            lbl.font_id = font
        self.panel.add_child(lbl)

    def gap(self, h=10):
        self.panel.add_fixed(h)

    def section(self, s):
        self.gap()
        self.text(s, C_DIM)

    def row(self, key, value, color=C_TEXT):
        self.text(f"{key:<13}{value}", color, self.f_mono)

    def button(self, s, callback):
        b = gui.Button(s)
        b.set_on_clicked(callback)
        self.panel.add_child(b)

    # escena
    def add(self, geom, point_size=None, line_width=None):
        m = rendering.MaterialRecord()
        if line_width:
            m.shader, m.line_width = "unlitLine", line_width
        else:
            m.shader, m.point_size = "defaultUnlit", point_size or 2.5
        self._n += 1
        self.scene.scene.add_geometry(f"g{self._n}", geom, m)

    def label(self, pos, s, color=C_TEXT, scale=1.0):
        lbl = self.scene.add_3d_label(np.asarray(pos, float), s)
        lbl.color = gui.Color(*color)
        lbl.scale = scale

    def add_scan(self, points):
        self.add(scanner_cloud(points), point_size=2.5)
        self.add(make_grid(points), line_width=1)
        axes, tips = make_axes(points)
        self.add(axes, line_width=3)
        for tip, name, c in zip(tips, AXIS_NAMES, [(1, .4, .4), (.4, 1, .4), (.5, .6, 1)]):
            self.label(tip, name, c, 1.3)

    def focus(self, bbox, view_dir=None):
        """Encuadra bbox mirando desde view_dir (por defecto oblicuo), con el eje vertical hacia arriba."""
        self.scene.setup_camera(60, bbox, bbox.get_center())
        up = np.eye(3)[UP_AXIS]
        if view_dir is None:
            view_dir = np.array([0.8, -1.0, 0.0]) + up * 0.9
        view_dir = np.asarray(view_dir, float) / np.linalg.norm(view_dir)
        if abs(view_dir @ up) > 0.99:
            up = np.eye(3)[(UP_AXIS + 1) % 3]
        center = bbox.get_center()
        dist = np.linalg.norm(bbox.get_extent()) * 1.1
        self.scene.look_at(center, center + view_dir * dist, up)

    def run(self):
        gui.Application.instance.run()


def show_overview(points):
    """Fase 1 con estilo escáner: nube, cuadrícula, ejes y cotas generales."""
    w = ScannerWindow("Vista general")
    w.add_scan(points)
    mn, mx = points.min(axis=0), points.max(axis=0)
    dims = mx - mn
    for i, name in enumerate(AXIS_NAMES):
        p = mn.copy(); p[i] = (mn[i] + mx[i]) / 2
        w.label(p, f"{name} {dims[i]:.2f} m", C_DIM)

    w.section("ESCANEO")
    w.row("Puntos", f"{len(points):,}")
    for i, name in enumerate(AXIS_NAMES):
        w.row(f"Ancho {name}" if i != UP_AXIS else f"Alto {name}", f"{dims[i]:.3f} m")
    w.section("CONTROLES")
    for s in ["Arrastrar   rotar", "Rueda       zoom", "Ctrl+arr.   mover", "Esc/cerrar  salir"]:
        w.text(s, C_DIM, w.f_mono)
    w.focus(o3d.geometry.AxisAlignedBoundingBox(mn, mx))
    w.run()


def show_result(points, result, n_clicks):
    """Ventana de resultado: medidas en la vista 3D y en el panel lateral."""
    w = ScannerWindow("Análisis de tubería")
    w.add_scan(points)

    r, c, n, u, v = result["radius"], result["center"], result["normal"], result["u"], result["v"]
    sel = result["points"]
    ok = result["inliers"]
    for mask, color in [(ok, C_SEL), (~ok, C_OUT)]:
        if mask.any():
            p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(sel[mask]))
            p.paint_uniform_color(color)
            w.add(p, point_size=10)

    t = np.linspace(0, 2 * np.pi, 129)[:-1]
    ring = c + r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    w.add(lines(ring, [[i, (i + 1) % len(ring)] for i in range(len(ring))], C_CIRCLE), line_width=5)
    # diámetro dibujado y eje de la tubería
    w.add(lines([c - u * r, c + u * r], [[0, 1]], C_CIRCLE), line_width=2)
    w.add(lines([c - n * r * 2.5, c + n * r * 2.5], [[0, 1]], C_CENTER), line_width=3)
    w.add(sphere(c, max(r * 0.07, 0.004), C_CENTER))

    tilt = np.degrees(np.arccos(min(abs(n[UP_AXIS]), 1.0)))
    w.label(c + u * r * 1.15, f"Ø {r * 2000:.1f} mm", C_CIRCLE, 1.6)
    w.label(c - u * r * 2.6, f"C {c[0]:.3f} / {c[1]:.3f} / {c[2]:.3f} m", C_TEXT)
    w.label(c + n * r * 2.6, f"eje {tilt:.0f}° de la vertical", C_CENTER)

    w.text(f"Ø {r * 2000:.1f} mm", C_CIRCLE, w.f_title)
    w.section("MEDIDAS")
    w.row("Diámetro", f"{r * 2000:.1f} mm", C_CIRCLE)
    w.row("Radio", f"{r * 1000:.1f} mm")
    w.row("Ajuste RMS", f"{result['rms'] * 1000:.1f} mm")
    w.row("Puntos", f"{int(ok.sum())} / {len(sel)} ({n_clicks} clics)")
    w.section("CENTRO (m)")
    for i, name in enumerate(AXIS_NAMES):
        w.row(name, f"{c[i]:+.4f}")
    w.section("ORIENTACIÓN")
    w.row("Inclinación", f"{tilt:.1f}° vs {AXIS_NAMES[UP_AXIS]}")
    for part in result["orientation"].split("; "):
        for line in part.split(", "):
            w.text(line, C_DIM)
    w.section("LEYENDA")
    w.text("Escaneo (azul = bajo, cian = alto)", C_CYAN)
    w.text("Selección usada", C_SEL)
    w.text("Selección descartada", C_OUT)
    w.text("Círculo ajustado", C_CIRCLE)
    w.text("Centro y eje de la tubería", C_CENTER)
    w.gap()

    pipe_box = o3d.geometry.AxisAlignedBoundingBox(c - max(r * 2.5, 0.1), c + max(r * 2.5, 0.1))
    full_box = o3d.geometry.AxisAlignedBoundingBox(points.min(axis=0), points.max(axis=0))
    # mirar la boca desde el interior de la put, ligeramente de lado
    inward = n if n @ (points.mean(axis=0) - c) > 0 else -n
    pipe_dir = inward + 0.35 * u + 0.25 * v
    w.button("Enfocar tubería", lambda: w.focus(pipe_box, pipe_dir))
    w.button("Ver put completa", lambda: w.focus(full_box))
    w.focus(pipe_box, pipe_dir)
    w.run()
