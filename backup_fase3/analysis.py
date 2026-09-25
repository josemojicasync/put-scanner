"""Análisis automático de una put (pozo de registro) a partir de un escaneo PLY.

Todo internamente en metros. Sistema local de la put:
  origen = centro de la cámara a la altura del fondo, Z = eje vertical (hacia arriba),
  X/Y = paredes (put rectangular) o proyección del X del PLY (put redonda).
"""
import numpy as np
import open3d as o3d

from measurement import fit_circle_2d, fit_circle_robust

VOXEL_M = 0.01
NORMAL_RADIUS_M = 0.05
SLICE_M = 0.05
WALL_NZ_MAX = 0.35          # |n·z| por debajo -> pared vertical
FLOOR_NZ_MIN = 0.85         # |n·z| por encima -> superficie horizontal
SECTION_TOL_M = 0.03        # cortes con tamaño dentro de esta tolerancia -> mismo tramo
RECT_SYMMETRY_MIN = 0.5     # fuerza de simetría de 90° de las normales para considerar rectangular
BEHIND_WALL_M = 0.03        # distancia mínima detrás de la pared para ser parte de una conexión
PIPE_CLUSTER_EPS_M = 0.05
PIPE_MIN_POINTS = 40

HOOG, MIDDEL, LAAG, ONZEKER = "HOOG", "MIDDEL", "LAAG", "ONZEKER"


class AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------- preprocesado

def preprocess(pcd):
    """Limpia, reduce y calcula normales. No modifica el objeto original."""
    p = o3d.geometry.PointCloud(pcd)
    p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    if len(p.points) > 30000:
        p = p.voxel_down_sample(VOXEL_M)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
    if len(p.points) < 1000:
        raise AnalysisError(f"Muy pocos puntos tras limpiar ({len(p.points)}).")
    return np.asarray(p.points).copy(), np.asarray(p.normals).copy()


# ---------------------------------------------------------------- orientación

def _least_normal_direction(normals):
    """Dirección con menor proyección de normales: eje de una estructura de paredes verticales."""
    w, v = np.linalg.eigh(normals.T @ normals / len(normals))
    return v[:, 0]


def estimate_frame(P, N):
    """Devuelve (origen provisional, R 3x3 con filas = ejes locales X,Y,Z, simetría rectangular)."""
    axis = _least_normal_direction(N)
    for _ in range(3):  # refinar solo con puntos de pared
        wall = np.abs(N @ axis) < WALL_NZ_MAX
        axis = _least_normal_direction(N[wall])

    # sentido: el extremo con superficies horizontales más extensas (terreno) es arriba
    t = P @ axis
    lo, hi = np.percentile(t, [2, 98])
    flat = np.abs(N @ axis) > FLOOR_NZ_MIN
    wall = np.abs(N @ axis) < WALL_NZ_MAX
    center = P[wall].mean(axis=0)
    radial = np.linalg.norm(np.cross(P - center, axis), axis=1)
    band = 0.15 * (hi - lo)
    spread_hi = np.percentile(radial[flat & (t > hi - band)], 90) if (flat & (t > hi - band)).sum() > 20 else 0
    spread_lo = np.percentile(radial[flat & (t < lo + band)], 90) if (flat & (t < lo + band)).sum() > 20 else 0
    if spread_lo > spread_hi:
        axis = -axis

    # X/Y: alinear con las paredes si la put es rectangular
    helper = np.eye(3)[np.argmin(np.abs(axis))]
    e1 = np.cross(axis, helper); e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    wall = np.abs(N @ axis) < WALL_NZ_MAX
    theta = np.arctan2(N[wall] @ e2, N[wall] @ e1)
    m4 = np.mean(np.exp(4j * theta))
    symmetry = float(abs(m4))
    if symmetry >= RECT_SYMMETRY_MIN:
        rot = np.angle(m4) / 4
    else:  # redonda: X local = X del PLY proyectado
        rot = np.arctan2(np.eye(3)[0] @ e2, np.eye(3)[0] @ e1)
    x = np.cos(rot) * e1 + np.sin(rot) * e2
    y = np.cross(axis, x)
    return center, np.vstack([x, y, axis]), symmetry


# ---------------------------------------------------------------- put

def _slice_shape(xy, rect):
    """Tamaño de un corte horizontal de pared. Rect: (cx, cy, hx, hy, rms). Redonda: (cx, cy, r, r, rms)."""
    if rect:
        lo, hi = np.percentile(xy, 3, axis=0), np.percentile(xy, 97, axis=0)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        d = np.minimum(np.abs(np.abs(xy[:, 0] - c[0]) - h[0]), np.abs(np.abs(xy[:, 1] - c[1]) - h[1]))
        return c[0], c[1], h[0], h[1], float(np.sqrt(np.mean(np.minimum(d, 0.05) ** 2)))
    cx, cy, r, _, rms = fit_circle_robust(xy)
    return cx, cy, r, r, rms


def _confidence(score_ok):
    """score_ok: lista de condiciones de más exigente a menos -> nivel."""
    for level, ok in zip([HOOG, MIDDEL, LAAG], score_ok):
        if ok:
            return level
    return ONZEKER


def detect_put(L, N_L, rect):
    """Segmenta la put en coordenadas locales (L = puntos, N_L = normales locales)."""
    wall = np.abs(N_L[:, 2]) < WALL_NZ_MAX
    z = L[:, 2]
    slices = []
    for z0 in np.arange(np.percentile(z, 0.5), np.percentile(z, 99.5), SLICE_M):
        m = wall & (z >= z0) & (z < z0 + SLICE_M)
        if m.sum() < 60:
            continue
        cx, cy, hx, hy, rms = _slice_shape(L[m, :2], rect)
        slices.append(dict(z0=z0, z1=z0 + SLICE_M, cx=cx, cy=cy, hx=hx, hy=hy, rms=rms, n=int(m.sum())))
    if not slices:
        raise AnalysisError("No se encontraron paredes verticales.")

    # agrupar cortes consecutivos de tamaño similar en tramos
    sections = []
    for s in slices:
        cur = sections[-1] if sections else None
        if cur and abs(s["z0"] - cur["z1"]) < 1e-6 + SLICE_M and \
                abs(s["hx"] - cur["hx"]) < SECTION_TOL_M and abs(s["hy"] - cur["hy"]) < SECTION_TOL_M:
            cur["slices"].append(s)
            cur["z1"] = s["z1"]
            k = len(cur["slices"])
            for key in ("cx", "cy", "hx", "hy"):
                cur[key] = float(np.median([q[key] for q in cur["slices"]]))
        else:
            sections.append(dict(z0=s["z0"], z1=s["z1"], cx=s["cx"], cy=s["cy"], hx=s["hx"], hy=s["hy"], slices=[s]))
    sections = [s for s in sections if len(s["slices"]) >= 2]
    if not sections:
        raise AnalysisError("No se encontró un tramo de pared continuo.")
    chamber = max(sections, key=lambda s: s["z1"] - s["z0"])

    # fondo: superficies horizontales dentro de la huella de la cámara, cerca de su parte baja
    inside = (np.abs(L[:, 0] - chamber["cx"]) < chamber["hx"] * 0.9) & \
             (np.abs(L[:, 1] - chamber["cy"]) < chamber["hy"] * 0.9)
    flat = np.abs(N_L[:, 2]) > FLOOR_NZ_MIN
    low = flat & inside & (z < chamber["z0"] + 0.3)
    floor_z = float(np.percentile(z[low], 50)) if low.sum() > 30 else float(chamber["z0"])
    floor_n = int(low.sum())

    # parte superior: último tramo de pared conectado con la cámara; terreno alrededor de la abertura
    stack = [s for s in sections if s["z0"] >= chamber["z0"] - 1e-6]
    top_wall = max(s["z1"] for s in stack)
    ext = (np.maximum(np.abs(L[:, 0] - chamber["cx"]) - chamber["hx"],
                      np.abs(L[:, 1] - chamber["cy"]) - chamber["hy"]) > 0.05)
    ground = flat & ext & (z > top_wall - 0.3)
    top_z = float(np.percentile(z[ground], 50)) if ground.sum() > 30 else float(top_wall)
    ground_found = ground.sum() > 30

    ch_rms = float(np.median([s["rms"] for s in chamber["slices"]]))
    ch_std = float(np.std([s["hx"] for s in chamber["slices"]]) + np.std([s["hy"] for s in chamber["slices"]]))
    n_sl = len(chamber["slices"])
    size_conf = _confidence([ch_rms < 0.01 and ch_std < 0.01 and n_sl >= 8,
                             ch_rms < 0.02 and ch_std < 0.02 and n_sl >= 4,
                             n_sl >= 2])
    depth_conf = _confidence([floor_n > 200 and ground_found,
                              floor_n > 50 and ground_found,
                              floor_n > 30 or ground_found])
    return dict(
        shape="rechthoekig" if rect else "rond",
        sections=sections, chamber=chamber,
        center_xy=np.array([chamber["cx"], chamber["cy"]]),
        width_x=2 * chamber["hx"], width_y=2 * chamber["hy"],
        diameter=2 * chamber["hx"] if not rect else None,
        floor_z=floor_z, top_z=top_z, top_wall_z=top_wall,
        depth=top_z - floor_z, ground_found=bool(ground_found),
        rms=ch_rms, size_conf=size_conf, depth_conf=depth_conf,
    )


# ---------------------------------------------------------------- conexiones

def _outside_distance(xy, sec, rect):
    dx, dy = xy[:, 0] - sec["cx"], xy[:, 1] - sec["cy"]
    if rect:
        return np.maximum(np.abs(dx) - sec["hx"], np.abs(dy) - sec["hy"])
    return np.hypot(dx, dy) - sec["hx"]


def _section_at(sections, z):
    for s in sections:
        if s["z0"] - SLICE_M <= z <= s["z1"] + SLICE_M:
            return s
    return None


def detect_connections(L, N_L, put, rect):
    ch = put["chamber"]
    z = L[:, 2]
    # las aansluitingen se buscan en la cámara (no en cuello/marco de la tapa)
    zmask = (z > put["floor_z"] - 0.3) & (z < ch["z1"] + SLICE_M)
    # detrás de la pared = fuera de TODOS los tramos que cubren esa altura (evita rellanos entre tramos)
    covered = np.zeros(len(L), bool)
    outside = np.ones(len(L), bool)
    for sec in put["sections"]:
        m = (z >= sec["z0"] - SLICE_M) & (z <= sec["z1"] + SLICE_M)
        covered |= m
        outside &= ~m | (_outside_distance(L[:, :2], sec, rect) > BEHIND_WALL_M)
    idx = np.flatnonzero(zmask & covered & outside)
    if len(idx) < PIPE_MIN_POINTS:
        return []
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L[idx])).cluster_dbscan(
        PIPE_CLUSTER_EPS_M, 10))

    conns = []
    for lab in range(labels.max() + 1 if len(labels) else 0):
        ids = idx[labels == lab]
        if len(ids) < PIPE_MIN_POINTS:
            continue
        Q, NQ = L[ids], N_L[ids]
        if np.mean(np.abs(NQ[:, 2]) > FLOOR_NZ_MIN) > 0.7:
            continue  # superficie casi toda horizontal: rellano o terreno, no un tubo
        sec =_section_at(put["sections"], float(np.median(Q[:, 2]))) or ch
        cxy = np.array([sec["cx"], sec["cy"]])
        m = Q[:, :2].mean(axis=0) - cxy
        # dirección: normal de la pared atravesada (supuesta perpendicular a la pared)
        if rect:
            k = int(np.argmax(np.abs(m) - np.array([sec["hx"], sec["hy"]])))
            d2 = np.zeros(2); d2[k] = np.sign(m[k])
        else:
            d2 = m / np.linalg.norm(m)
        d = np.array([d2[0], d2[1], 0.0])
        s_axis = np.array([-d2[1], d2[0], 0.0])  # horizontal, perpendicular a la tubería

        # sección transversal (s, z); quitar superficie horizontal inferior (agua/sedimento en el tubo)
        s = Q @ s_axis
        bottom = (np.abs(NQ[:, 2]) > FLOOR_NZ_MIN) & (Q[:, 2] < Q[:, 2].min() + 0.02)
        sz = np.column_stack([s, Q[:, 2]])[~bottom]
        wall_depth = (np.abs(m) - np.array([sec["hx"], sec["hy"]]))[k] if rect else np.linalg.norm(m) - sec["hx"]

        conn = dict(points=Q, direction=d, s_axis=s_axis, n=len(Q),
                    opening_w=float(np.ptp(np.percentile(s, [2, 98]))),
                    opening_h=float(np.ptp(np.percentile(Q[:, 2], [2, 98]))),
                    visible_len=float(np.ptp(Q @ d)),
                    wall_center=None, diameter=None, conf=ONZEKER, note="")
        wall_s, wall_z = float(np.median(s)), float(np.median(Q[:, 2]))
        radial_wall = (sec["hx"] if not rect else [sec["hx"], sec["hy"]][k])
        conn["wall_center"] = np.array([cxy[0], cxy[1], 0]) + d * radial_wall + s_axis * (wall_s - cxy @ s_axis[:2])
        conn["wall_center"][2] = wall_z

        if len(sz) >= 12:
            sc, zc, r, inl, rms = fit_circle_robust(sz)
            ang = np.arctan2(sz[inl, 1] - zc, sz[inl, 0] - sc)
            coverage = np.unique(((ang + np.pi) // (np.pi / 9)).astype(int)).size / 18
            sane = 0.03 < r < 0.8 and r < 1.5 * max(conn["opening_w"], 0.05)
            rel = rms / r if r > 0 else 1
            # bootstrap: variabilidad del radio si se repite el ajuste con otros subconjuntos
            rng = np.random.default_rng(0)
            arc = sz[inl]
            boot = [fit_circle_2d(arc[rng.integers(0, len(arc), len(arc))])[2] for _ in range(40)]
            r_std = float(np.std(boot))
            spread = r_std / r if r > 0 else 1
            conn.update(fit_r=r, fit_rms=rms, coverage=coverage, inliers=int(inl.sum()), d_std=2 * r_std)
            conn["conf"] = ONZEKER if not sane else _confidence([
                coverage >= 0.5 and rel < 0.03 and spread < 0.05 and inl.sum() >= 100,
                coverage >= 0.33 and rel < 0.06 and spread < 0.10 and inl.sum() >= 50,
                coverage >= 0.22 and rel < 0.10 and spread < 0.25 and inl.sum() >= 25])
            if conn["conf"] != ONZEKER:
                center = np.array([cxy[0], cxy[1], 0]) + d * radial_wall
                center += s_axis * (sc - cxy @ s_axis[:2])
                center[2] = zc
                conn.update(diameter=2 * r, radius=r, center=center, invert_z=zc - r)
            else:
                conn["note"] = f"arco visible ~{coverage * 360:.0f}°, Ø variaría ±{2 * r_std * 1000:.0f} mm"
        else:
            conn["note"] = "demasiado pocos puntos en la sección"
        ref = conn.get("center", conn["wall_center"])
        conn["angle_deg"] = float(np.degrees(np.arctan2(ref[1] - ch["cy"], ref[0] - ch["cx"])) % 360)
        conn["height"] = float(ref[2] - put["floor_z"])
        conn["depth"] = float(put["top_z"] - ref[2])
        conn["wall_depth"] = float(wall_depth)
        conns.append(conn)
    conns.sort(key=lambda c: c["angle_deg"])
    return conns


# ---------------------------------------------------------------- entrada principal

def analyze(pcd):
    """Analiza una nube de puntos Open3D. Devuelve dict con frame, put, conexiones y la nube en local."""
    P, N = preprocess(pcd)
    center, R, symmetry = estimate_frame(P, N)
    rect = symmetry >= RECT_SYMMETRY_MIN
    L, N_L = (P - center) @ R.T, N @ R.T
    put = detect_put(L, N_L, rect)

    # recentrar: origen en el centro de la cámara a la altura del fondo
    shift = np.array([put["center_xy"][0], put["center_xy"][1], put["floor_z"]])
    L -= shift
    for s in put["sections"]:
        s["cx"] -= shift[0]; s["cy"] -= shift[1]; s["z0"] -= shift[2]; s["z1"] -= shift[2]
        for q in s["slices"]:
            q["cx"] -= shift[0]; q["cy"] -= shift[1]
    put["top_z"] -= shift[2]; put["top_wall_z"] -= shift[2]; put["floor_z"] = 0.0
    put["center_xy"] = np.zeros(2)
    conns = detect_connections(L, N_L, put, rect)

    tilt = float(np.degrees(np.arccos(min(abs(R[2] @ np.array([0, 0, 1.0])), 1.0))))
    return dict(put=put, connections=conns, local_points=L, rect=rect, symmetry=symmetry,
                R=R, origin=center + shift @ R, axis_tilt_vs_ply_z=tilt, n_points=len(P))


def load_and_analyze(path):
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(path))
        pcd = o3d.geometry.PointCloud(mesh.vertices)
    if len(pcd.points) == 0:
        raise AnalysisError(f"No se pudieron leer puntos de {path}")
    return analyze(pcd)


def report(res):
    """Informe de texto (consola)."""
    put, lines = res["put"], []
    f = lambda v, unit="mm": "—" if v is None else (f"{v * 1000:.0f} mm" if unit == "mm" else f"{v:.2f} m")
    lines.append("=== PUT ===")
    lines.append(f"Vorm:      {put['shape']} (simetría 90°: {res['symmetry']:.2f})")
    lines.append(f"Eje vs Z del PLY: {res['axis_tilt_vs_ply_z']:.1f}°")
    if put["diameter"] is not None:
        lines.append(f"Diameter:  {f(put['diameter'])}  [{put['size_conf']}]")
        lines.append(f"Radius:    {f(put['diameter'] / 2)}  [{put['size_conf']}]")
    lines.append(f"Width X:   {f(put['width_x'])}  [{put['size_conf']}]")
    lines.append(f"Width Y:   {f(put['width_y'])}  [{put['size_conf']}]")
    lines.append(f"Depth:     {f(put['depth'], 'm')}  [{put['depth_conf']}]"
                 + ("" if put["ground_found"] else "  (sin terreno: hasta el borde de la pared)"))
    lines.append(f"Buitenmaat: niet meetbaar vanaf binnen  [{ONZEKER}]")
    for i, s in enumerate(put["sections"]):
        lines.append(f"  tramo {i + 1}: z {s['z0']:+.2f}..{s['z1']:+.2f} m  "
                     f"{2 * s['hx'] * 1000:.0f} x {2 * s['hy'] * 1000:.0f} mm")
    lines.append(f"\n=== AANSLUITINGEN ({len(res['connections'])}) ===")
    for i, c in enumerate(res["connections"]):
        lines.append(f"Aansluiting {i + 1}  [{c['conf']}]")
        if c["diameter"] is not None:
            lines.append(f"  Diameter: {f(c['diameter'])} ± {f(c['d_std'])}   Radius: {f(c['radius'])}"
                         f"   (arco ~{c['coverage'] * 360:.0f}°, rms {c['fit_rms'] * 1000:.1f} mm)")
            lines.append(f"  BOB t.o.v. bodem: {f(c['invert_z'])}")
        else:
            lines.append(f"  Diameter: onzeker ({c['note']})")
        lines.append(f"  Zichtbare opening: {f(c['opening_w'])} breed x {f(c['opening_h'])} hoog")
        lines.append(f"  Hoogte (as boven bodem): {f(c['height'])}   Diepte (onder maaiveld): {f(c['depth'], 'm')}")
        lines.append(f"  Hoek rond put: {c['angle_deg']:.0f}°   Richting: haaks op wand (aangenomen)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "scan.ply"
    print(report(load_and_analyze(path)))
