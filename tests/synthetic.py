"""Geometría sintética de puts con medidas conocidas (metros), para tests.

Todas las nubes se giran y desplazan al final para comprobar que el análisis no depende de la
orientación del PLY.
"""
import numpy as np
import open3d as o3d


def _grid(u0, u1, v0, v1, step):
    u = np.arange(u0, u1 + 1e-9, step)
    v = np.arange(v0, v1 + 1e-9, step)
    U, V = np.meshgrid(u, v)
    return U.ravel(), V.ravel()


def _pipe_points(center, normal, along, d, arc, length, step):
    """Superficie interior de un tubo que sale de la pared: ángulos `arc` (grados; 90° = kruin)."""
    r = d / 2
    a0, a1 = np.radians(arc[0]), np.radians(arc[1])
    da = step / r
    ang = np.arange(a0, a1 + 1e-9, da)
    t = np.arange(0.0, length + 1e-9, step)
    A, T = np.meshgrid(ang, t)
    A, T = A.ravel(), T.ravel()
    up = np.array([0, 0, 1.0])
    return center + np.outer(np.cos(A) * r, along) + np.outer(np.sin(A) * r, up) + np.outer(T, normal)


def _finish(P, noise, outliers, yaw_deg, shift, rng):
    if noise > 0:
        P = P + rng.normal(0, noise, P.shape)
    if outliers > 0:
        lo, hi = P.min(axis=0), P.max(axis=0)
        P = np.vstack([P, rng.uniform(lo, hi, (int(outliers * len(P)), 3))])
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    P = P @ R.T + np.asarray(shift)
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))


def _tube(center, a, r0, r1, t0, t1, arc, step):
    """Superficie de un tubo a lo largo de `a` entre t0 y t1, radio lineal r0->r1 (cono si distintos).
    Ángulos del arco en el plano (lateral, arriba'): 90° = kruin."""
    lat = np.cross([0, 0, 1.0], a); lat /= np.linalg.norm(lat)
    up = np.cross(a, lat)
    pts = []
    for t in np.arange(t0, t1 + 1e-9, step):
        r = r0 + (r1 - r0) * (t - t0) / max(t1 - t0, 1e-9)
        ang = np.arange(np.radians(arc[0]), np.radians(arc[1]) + 1e-9, step / r)
        pts.append(center + a * t + np.outer(np.cos(ang) * r, lat) + np.outer(np.sin(ang) * r, up))
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _disc(center, a, r_in, r_out, step):
    """Anillo/disco plano perpendicular a `a` (tapa de relleno, escalón, collar)."""
    lat = np.cross([0, 0, 1.0], a); lat /= np.linalg.norm(lat)
    up = np.cross(a, lat)
    x, y = _grid(-r_out, r_out, -r_out, r_out, step)
    rr = np.hypot(x, y)
    m = (rr <= r_out) & (rr >= r_in)
    return center + np.outer(x[m], lat) + np.outer(y[m], up)


def rect_put(wx=0.8, wy=1.0, depth=2.5, step=0.012, noise=0.0, missing=(), partial=None, ground=True,
             pipes=(), outliers=0.0, yaw_deg=17.0, shift=(0.3, -0.2, -1.0), seed=0,
             taper_x=0.0, floor_step=0.0, ground_step=0.0, reliefs=()):
    """Put rectangular interior wx x wy, profundidad depth (bodem z=0, maaiveld z=depth).
    partial: {"-X": 0.3} conserva solo el 30 % inferior de esa pared.
    pipes: [dict(wall="+Y", offset=0.0, z=0.6, d=0.315, arc=(0, 360), length=0.5,
                 hole_d=None (abertura en la pared; por defecto = d), thickness=0.0 (espesor de pared: la abertura
                 sigue como cilindro de hole_d hasta ese espesor, luego empieza el tubo), yaw_deg=0 (tubo oblicuo),
                 cap_t=None (tapa/relleno a esa profundidad), d_end=None (tubo cónico),
                 collar=None | dict(width, protrusion) (anillo que sobresale delante de la pared),
                 water_z=None, water_rough=0, water_slope=0)]
    taper_x: diferencia de anchura X entre arriba y abajo (paredes X convergentes/divergentes).
    floor_step / ground_step: la mitad x>0 del bodem / maaiveld desplazada esa altura (dos superficies).
    reliefs: [dict(wall, u0, u1, z0, z1, depth)] rebaje plano detrás de la cara (ranura/nicho, no es tubo)."""
    rng = np.random.default_rng(seed)
    pts = []
    walls = {"+X": (np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), wx / 2, wy / 2),
             "-X": (np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), wx / 2, wy / 2),
             "+Y": (np.array([0, 1.0, 0]), np.array([1.0, 0, 0]), wy / 2, wx / 2),
             "-Y": (np.array([0, -1.0, 0]), np.array([1.0, 0, 0]), wy / 2, wx / 2)}

    def axis_of(p):
        n = walls[p["wall"]][0]
        y = np.radians(p.get("yaw_deg", 0.0))
        return np.array([n[0] * np.cos(y) - n[1] * np.sin(y), n[0] * np.sin(y) + n[1] * np.cos(y), 0.0])

    def wall_dist(key, z):
        n, _, dist, _ = walls[key]
        return dist + (taper_x / 2) * (z / depth) if key[1] == "X" else dist

    for key, (n, along, dist, half) in walls.items():
        if key in missing:
            continue
        u, v = _grid(-half, half, 0.0, depth, step)
        keep = np.ones(len(u), bool)
        if partial and key in partial:
            keep &= v < partial[key] * depth
        dd = np.array([wall_dist(key, z) for z in v])
        P = np.outer(dd, n) + np.outer(u, along) + np.outer(v, [0, 0, 1.0])
        for p in pipes:
            if p["wall"] == key:
                a = axis_of(p)
                c = n * dist + along * p.get("offset", 0.0) + np.array([0, 0, p["z"]])
                hole_r = p.get("hole_d", p["d"]) / 2
                keep &= np.linalg.norm(np.cross(P - c, a), axis=1) > hole_r   # elipse si el tubo es oblicuo
        for rf in reliefs:
            if rf["wall"] == key:
                inr = (u >= rf["u0"]) & (u <= rf["u1"]) & (v >= rf["z0"]) & (v <= rf["z1"])
                back = P[inr] + n * rf["depth"]          # fondo plano del rebaje, paralelo a la pared
                pts.append(back)
                keep &= ~inr
        pts.append(P[keep])
    x, y = _grid(-wx / 2, wx / 2, -wy / 2, wy / 2, step)
    pts.append(np.column_stack([x, y, np.where(x > 0, floor_step, 0.0)]))
    if ground:
        g = 0.8
        x, y = _grid(-wx / 2 - g, wx / 2 + g, -wy / 2 - g, wy / 2 + g, step * 1.5)
        out = (np.abs(x) > wx / 2 + 0.02) | (np.abs(y) > wy / 2 + 0.02)
        pts.append(np.column_stack([x[out], y[out], depth + np.where(x[out] > 0, ground_step, 0.0)]))
    for p in pipes:
        n, along, dist, _ = walls[p["wall"]]
        a = axis_of(p)
        center = n * dist + along * p.get("offset", 0.0) + np.array([0, 0, p["z"]])
        r, length, arc = p["d"] / 2, p.get("length", 0.5), p.get("arc", (0, 360))
        hole_r = p.get("hole_d", p["d"]) / 2
        th = p.get("thickness", 0.0)
        s = step * 0.8
        P = []
        if th > 0:
            P.append(_tube(center, a, hole_r, hole_r, 0.0, th, arc, s))       # agujero en la pared (espesor)
            if hole_r > r + 0.005:
                P.append(_disc(center + a * th, a, r, hole_r, s))             # escalón transversal agujero -> tubo
        t_end = p.get("cap_t") or length
        P.append(_tube(center, a, r, (p.get("d_end") or p["d"]) / 2, th, t_end, arc, s))
        if p.get("cap_t"):
            P.append(_disc(center + a * p["cap_t"], a, 0.0, r, s))            # tapa / relleno de reconstrucción
        if p.get("collar"):
            cl = p["collar"]
            P.append(_disc(center - n * cl["protrusion"], n, hole_r, hole_r + cl["width"], s))
        P = np.vstack(P)
        if p.get("water_z") is not None:
            # agua/banket dentro del tubo: superficie plana (rugosa, algo inclinada) que tapa la parte inferior
            wz = p["water_z"]
            P = P[P[:, 2] > wz]
            half = np.sqrt(max(r ** 2 - (wz - p["z"]) ** 2, 0.0))
            lat = np.cross([0, 0, 1.0], a); lat /= np.linalg.norm(lat)
            u, t = _grid(-half, half, 0.0, length, s)
            zz = wz + 0.05 * t * p.get("water_slope", 0.0) + rng.normal(0, p.get("water_rough", 0.0), len(u))
            P = np.vstack([P, center + np.outer(u, lat) + np.outer(t, a) + np.outer(zz - p["z"], [0, 0, 1.0])])
        pts.append(P)
    return _finish(np.vstack(pts), noise, outliers, yaw_deg, shift, rng)


def round_put(D=1.2, depth=2.5, step=0.012, noise=0.0, ground=True, outliers=0.0, yaw_deg=17.0,
              shift=(0.3, -0.2, -1.0), seed=0):
    rng = np.random.default_rng(seed)
    r = D / 2
    ang = np.arange(0, 2 * np.pi, step / r)
    z = np.arange(0, depth + 1e-9, step)
    A, Z = np.meshgrid(ang, z)
    pts = [np.column_stack([r * np.cos(A.ravel()), r * np.sin(A.ravel()), Z.ravel()])]
    x, y = _grid(-r, r, -r, r, step)
    inside = np.hypot(x, y) < r
    pts.append(np.column_stack([x[inside], y[inside], np.zeros(inside.sum())]))
    if ground:
        x, y = _grid(-r - 0.8, r + 0.8, -r - 0.8, r + 0.8, step * 1.5)
        out = np.hypot(x, y) > r + 0.02
        pts.append(np.column_stack([x[out], y[out], np.full(out.sum(), depth)]))
    return _finish(np.vstack(pts), noise, outliers, yaw_deg, shift, rng)


def pipe_section(d=0.315, arc=(0, 360), noise=0.0, outliers=0.0, step=0.006, seed=0):
    """Sección (s, z) de un tubo: puntos del arco visible, para probar _measure_arc directamente."""
    rng = np.random.default_rng(seed)
    r = d / 2
    a = np.arange(np.radians(arc[0]), np.radians(arc[1]) + 1e-9, step / r)
    a = np.repeat(a, 5)  # 5 puntos por ángulo (a lo largo del tubo)
    sz = np.column_stack([r * np.cos(a), 0.5 + r * np.sin(a)])
    if noise > 0:
        sz += rng.normal(0, noise, sz.shape)
    if outliers > 0:
        k = int(outliers * len(sz))
        sz = np.vstack([sz, rng.uniform(sz.min(axis=0) - 0.05, sz.max(axis=0) + 0.05, (k, 2))])
    return sz
