"""Análisis automático de una put (pozo de registro) a partir de un escaneo PLY.

Todo internamente en metros. Sistema local de la put:
  origen = centro de la cámara a la altura del fondo, Z = eje vertical (hacia arriba),
  X/Y = paredes (put rectangular) o proyección del X del PLY (put redonda).

Incertidumbres: "±" = incertidumbre expandida k=2 (~95 %):
  U = 2 * sqrt(σ_estadística² + σ_modelo² + σ_escala²)
  σ_estadística: bootstrap por bloques espaciales (no por puntos sueltos, que subestima).
  σ_modelo: discrepancia entre dos métodos independientes (cuando existe).
  σ_escala: SCALE_REL_SIGMA * valor (supuesto sobre la escala de Polycam/LiDAR).
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
PLANE_THRESH_M = 0.01       # RANSAC paredes
SURFACE_THRESH_M = 0.015    # RANSAC fondo / maaiveld
BLOCK_M = 0.10              # tamaño de bloque del bootstrap espacial
N_BOOT = 200
SCALE_REL_SIGMA = 0.005     # SUPUESTO: 0,5 % (1σ) de error de escala del escaneo; calibrar con cinta
K_EXPAND = 2.0
NOMINALS_MM = [110, 125, 160, 200, 250, 315, 400, 500, 600, 630, 700, 800, 1000]

HOOG, MIDDEL, LAAG, ONZEKER = "HOOG", "MIDDEL", "LAAG", "ONZEKER"
MEASURED, ESTIMATED, UNKNOWN = "MEASURED", "ESTIMATED", "UNKNOWN"
CONF_ORDER = [HOOG, MIDDEL, LAAG, ONZEKER]


class AnalysisError(RuntimeError):
    pass


# ---------------------------------------------------------------- utilidades

def expanded(value, *sigmas, scale=True):
    """Incertidumbre expandida (k=2) combinando sigmas y el término de escala."""
    s2 = sum(float(s) ** 2 for s in sigmas)
    if scale:
        s2 += (SCALE_REL_SIGMA * abs(value)) ** 2
    return K_EXPAND * np.sqrt(s2)


def rel_conf(rel, limits, ok=True):
    """Nivel según incertidumbre relativa y límites (HOOG, MIDDEL, LAAG)."""
    if not ok or not np.isfinite(rel):
        return ONZEKER
    for level, lim in zip([HOOG, MIDDEL, LAAG], limits):
        if rel < lim:
            return level
    return ONZEKER


def worst(*levels):
    return max(levels, key=CONF_ORDER.index)


def meas(value, U, conf, **extra):
    return dict(value=float(value), U=float(U), conf=conf, **extra)


def block_ids(coords, size=BLOCK_M):
    """Grupos de índices por bloque espacial (coords: Nx1 o Nx2)."""
    q = np.floor(np.asarray(coords) / size).astype(np.int64)
    q = q.reshape(len(q), -1)
    _, ids = np.unique(q, axis=0, return_inverse=True)
    ids = ids.ravel()
    order = np.argsort(ids, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(ids[order])) + 1)


def resample_blocks(groups, rng):
    """Bootstrap por bloques: se remuestrean bloques enteros (captura errores espacialmente correlados)."""
    return np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])


def fit_plane_lsq(P):
    c = P.mean(axis=0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[2], c


def fit_plane_ransac(P, thresh, rng, hint=None, iters=300):
    """Plano robusto: RANSAC + reajuste por mínimos cuadrados. hint = normal aproximada esperada."""
    best = None
    for _ in range(iters):
        s = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        if hint is not None and abs(n @ hint) < 0.9:
            continue
        inl = np.abs((P - s[0]) @ n) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 10:
        best = np.ones(len(P), bool)
    n, c = fit_plane_lsq(P[best])
    inl = np.abs((P - c) @ n) < thresh
    n, c = fit_plane_lsq(P[inl])
    if hint is not None and n @ hint < 0:
        n = -n
    rms = float(np.sqrt(np.mean(((P[inl] - c) @ n) ** 2)))
    return n, c, inl, rms


def plane_height_at(n, c, x, y):
    """Z del plano n·(p-c)=0 en (x, y)."""
    return c[2] - (n[0] * (x - c[0]) + n[1] * (y - c[1])) / n[2]


def fit_circle_geometric(xy, cx, cy, r, iters=30):
    """Ajuste geométrico (Gauss-Newton) de círculo: minimiza distancias reales al círculo."""
    for _ in range(iters):
        dx, dy = xy[:, 0] - cx, xy[:, 1] - cy
        d = np.maximum(np.hypot(dx, dy), 1e-12)
        J = np.column_stack([-dx / d, -dy / d, -np.ones(len(d))])
        delta, *_ = np.linalg.lstsq(J, -(d - r), rcond=None)
        cx, cy, r = cx + delta[0], cy + delta[1], r + delta[2]
        if np.linalg.norm(delta) < 1e-8:
            break
    return cx, cy, abs(r)


def nominal_candidates(d_mm, u_mm):
    """Maatvoeringen nominales compatibles: el diámetro interior de un tubo nominal N está entre ~0,9N y N."""
    lo, hi = d_mm - u_mm, d_mm + u_mm
    return [n for n in NOMINALS_MM if 0.9 * n <= hi and n >= lo]


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


# ---------------------------------------------------------------- put: tramos

def _slice_shape(xy, rect):
    """Tamaño de un corte horizontal de pared. Rect: (cx, cy, hx, hy, rms). Redonda: (cx, cy, r, r, rms)."""
    if rect:
        lo, hi = np.percentile(xy, 3, axis=0), np.percentile(xy, 97, axis=0)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        d = np.minimum(np.abs(np.abs(xy[:, 0] - c[0]) - h[0]), np.abs(np.abs(xy[:, 1] - c[1]) - h[1]))
        return c[0], c[1], h[0], h[1], float(np.sqrt(np.mean(np.minimum(d, 0.05) ** 2)))
    cx, cy, r, _, rms = fit_circle_robust(xy)
    return cx, cy, r, r, rms


def find_sections(L, N_L, rect):
    """Cortes cada SLICE_M agrupados en tramos de tamaño constante. Tamaños aproximados (percentiles)."""
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
    sections = []
    for s in slices:
        cur = sections[-1] if sections else None
        if cur and abs(s["z0"] - cur["z1"]) < 1e-6 + SLICE_M and \
                abs(s["hx"] - cur["hx"]) < SECTION_TOL_M and abs(s["hy"] - cur["hy"]) < SECTION_TOL_M:
            cur["slices"].append(s)
            cur["z1"] = s["z1"]
            for key in ("cx", "cy", "hx", "hy"):
                cur[key] = float(np.median([q[key] for q in cur["slices"]]))
        else:
            sections.append(dict(z0=s["z0"], z1=s["z1"], cx=s["cx"], cy=s["cy"], hx=s["hx"], hy=s["hy"], slices=[s]))
    sections = [s for s in sections if len(s["slices"]) >= 2]
    if not sections:
        raise AnalysisError("No se encontró un tramo de pared continuo.")
    return sections


# ---------------------------------------------------------------- put: paredes (precisión)

WALLS = [("+X", 0, 1), ("-X", 0, -1), ("+Y", 1, 1), ("-Y", 1, -1)]


def measure_rect_walls(L, N_L, ch, rng):
    """Ajusta los 4 planos interiores de la cámara (RANSAC) y mide anchos con bootstrap por bloques."""
    z = L[:, 2]
    zlo, zhi = ch["z0"] + 0.05, ch["z1"] - 0.05
    base = (np.abs(N_L[:, 2]) < WALL_NZ_MAX) & (z > zlo) & (z < zhi)
    walls = {}
    for key, ax, sign in WALLS:
        c0, h = (ch["cx"], ch["hx"]) if ax == 0 else (ch["cy"], ch["hy"])
        hint = np.zeros(3); hint[ax] = sign
        m = base & (np.abs(N_L[:, ax]) > 0.8) & (sign * (L[:, ax] - c0) > 0.5 * h)
        P = L[m]
        if len(P) < 100:
            raise AnalysisError(f"Pared {key}: muy pocos puntos ({len(P)}).")
        n, c, inl, rms = fit_plane_ransac(P, PLANE_THRESH_M, rng, hint)
        Pin = P[inl]
        other = 1 - ax
        ids = block_ids(np.column_stack([Pin[:, other], Pin[:, 2]]))
        # cobertura: celdas de 5 cm ocupadas sobre el área teórica de la pared
        span = 2 * (ch["hy"] if ax == 0 else ch["hx"])
        occ = np.unique(np.floor(np.column_stack([Pin[:, other] - (ch["cy"] if ax == 0 else ch["cx"]) + span / 2,
                                                  Pin[:, 2] - zlo]) / 0.05).astype(int), axis=0)
        total = max(1, int(np.ceil(span / 0.05)) * int(np.ceil((zhi - zlo) / 0.05)))
        walls[key] = dict(n=n, c=c, points=Pin, ids=ids, rms=rms, n_used=int(inl.sum()), n_total=len(P),
                          coverage=min(1.0, len(occ) / total))

    def widths(wp):
        out = []
        for a, b in [("+X", "-X"), ("+Y", "-Y")]:
            na, ca = wp[a]; nb, cb = wp[b]
            n = na - nb; n /= np.linalg.norm(n)
            out.append((ca - cb) @ n)
        return out

    wp = {k: (w["n"], w["c"]) for k, w in walls.items()}
    wx, wy = widths(wp)
    boot = []
    for _ in range(N_BOOT):
        bp = {}
        for k, w in walls.items():
            idx = resample_blocks(w["ids"], rng)
            n, c = fit_plane_lsq(w["points"][idx])
            if n @ w["n"] < 0:
                n = -n
            bp[k] = (n, c)
        boot.append(widths(bp))
    sx, sy = np.std(boot, axis=0)

    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))
    nx = walls["+X"]["n"] - walls["-X"]["n"]; nx /= np.linalg.norm(nx)
    ny = walls["+Y"]["n"] - walls["-Y"]["n"]; ny /= np.linalg.norm(ny)
    geo = dict(parallel_x=ang(walls["+X"]["n"], walls["-X"]["n"]),
               parallel_y=ang(walls["+Y"]["n"], walls["-Y"]["n"]),
               perpendicular=abs(90.0 - float(np.degrees(np.arccos(np.clip(nx @ ny, -1, 1))))))
    cx = float((walls["+X"]["c"][0] + walls["-X"]["c"][0]) / 2)
    cy = float((walls["+Y"]["c"][1] + walls["-Y"]["c"][1]) / 2)

    def width_meas(value, sigma, keys):
        U = expanded(value, sigma)
        ok = all(walls[k]["n_used"] >= 200 for k in keys)
        cov = min(walls[k]["coverage"] for k in keys)
        conf = rel_conf(U / value, (0.015, 0.03, 0.08), ok)
        if cov < 0.3:
            conf = worst(conf, LAAG)
        return meas(value, U, conf, sigma_stat=float(sigma))

    return dict(walls=walls, geometry=geo, cx=cx, cy=cy,
                width_x=width_meas(wx, sx, ("+X", "-X")), width_y=width_meas(wy, sy, ("+Y", "-Y")),
                coverage=float(np.mean([w["coverage"] for w in walls.values()])))


def measure_round_wall(ch, rng):
    """Put redonda: diámetro = mediana de los cortes; incertidumbre por bootstrap de cortes."""
    d = np.array([2 * s["hx"] for s in ch["slices"]])
    boot = [np.median(rng.choice(d, len(d))) for _ in range(N_BOOT)]
    D = float(np.median(d))
    U = expanded(D, np.std(boot))
    conf = rel_conf(U / D, (0.015, 0.03, 0.08), len(d) >= 4)
    m = meas(D, U, conf, sigma_stat=float(np.std(boot)))
    return dict(walls=None, geometry=None, cx=ch["cx"], cy=ch["cy"], width_x=m, width_y=m, diameter=m,
                coverage=None)


# ---------------------------------------------------------------- put: fondo y maaiveld

def measure_surface(P, cx, cy, rng):
    """Plano robusto de una superficie horizontal. Devuelve (altura en (cx,cy), σ bootstrap, info)."""
    n, c, inl, rms = fit_plane_ransac(P, SURFACE_THRESH_M, rng, np.array([0, 0, 1.0]))
    Pin = P[inl]
    z = plane_height_at(n, c, cx, cy)
    ids = block_ids(Pin[:, :2])
    boot = []
    for _ in range(N_BOOT):
        bn, bc = fit_plane_lsq(Pin[resample_blocks(ids, rng)])
        if abs(bn[2]) > 0.5:
            boot.append(plane_height_at(bn, bc, cx, cy))
    slope = float(np.degrees(np.arccos(min(abs(n[2]), 1.0))))
    return float(z), float(np.std(boot)), dict(n_used=int(inl.sum()), rms=rms, slope_deg=slope,
                                              blocks=len(ids))


def detect_put(L, N_L, rect, rng):
    sections = find_sections(L, N_L, rect)
    chamber = max(sections, key=lambda s: s["z1"] - s["z0"])
    walls = measure_rect_walls(L, N_L, chamber, rng) if rect else measure_round_wall(chamber, rng)
    # los anchos precisos sustituyen a los aproximados de los cortes
    chamber["cx"], chamber["cy"] = walls["cx"], walls["cy"]
    chamber["hx"], chamber["hy"] = walls["width_x"]["value"] / 2, walls["width_y"]["value"] / 2

    z = L[:, 2]
    flat = np.abs(N_L[:, 2]) > FLOOR_NZ_MIN
    inside = (np.abs(L[:, 0] - chamber["cx"]) < chamber["hx"] * 0.9) & \
             (np.abs(L[:, 1] - chamber["cy"]) < chamber["hy"] * 0.9)
    low = flat & inside & (z < chamber["z0"] + 0.3)
    if low.sum() < 30:
        raise AnalysisError("No se encontró el fondo de la put.")
    floor_z, floor_s, floor_info = measure_surface(L[low], chamber["cx"], chamber["cy"], rng)

    stack = [s for s in sections if s["z0"] >= chamber["z0"] - 1e-6]
    top_wall = max(s["z1"] for s in stack)
    ext = np.maximum(np.abs(L[:, 0] - chamber["cx"]) - chamber["hx"],
                     np.abs(L[:, 1] - chamber["cy"]) - chamber["hy"]) > 0.05
    ground = flat & ext & (z > top_wall - 0.3)
    ground_found = ground.sum() > 30
    if ground_found:
        top_z, top_s, ground_info = measure_surface(L[ground], chamber["cx"], chamber["cy"], rng)
    else:
        top_z, top_s, ground_info = float(top_wall), SLICE_M, None

    depth = top_z - floor_z
    U = expanded(depth, floor_s, top_s)
    depth_conf = rel_conf(U / depth, (0.015, 0.03, 0.08), floor_info["n_used"] >= 50)
    if not ground_found:
        depth_conf = worst(depth_conf, LAAG)
    return dict(
        shape="rechthoekig" if rect else "rond",
        sections=sections, chamber=chamber, walls=walls,
        width_x=walls["width_x"], width_y=walls["width_y"], diameter=walls.get("diameter"),
        depth=meas(depth, U, depth_conf, sigma_floor=floor_s, sigma_top=top_s),
        floor_z=floor_z, floor_sigma=floor_s, floor_info=floor_info,
        top_z=top_z, top_sigma=top_s, ground_info=ground_info, ground_found=bool(ground_found),
        top_wall_z=top_wall,
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


def _measure_arc(sz, rng, floor_sigma, top_z, top_sigma):
    """Mide un tubo a partir de su sección (s, z). Devuelve dict con medidas y estado."""
    sc, zc, r, inl, _ = fit_circle_robust(sz)
    arc = sz[inl]
    sc, zc, r = fit_circle_geometric(arc, sc, zc, r)
    rms = float(np.sqrt(np.mean((np.hypot(arc[:, 0] - sc, arc[:, 1] - zc) - r) ** 2)))
    ang = np.arctan2(arc[:, 1] - zc, arc[:, 0] - sc)
    coverage = np.unique(((ang + np.pi) // (np.pi / 18)).astype(int)).size / 36

    # método independiente: cuerda + flecha del arco visible (válido si el arco no es casi completo)
    zb, zt = np.percentile(arc[:, 1], [3, 97])
    h = zt - zb
    base = arc[arc[:, 1] < zb + 0.015, 0]
    chord = float(np.ptp(np.percentile(base, [2, 98]))) if len(base) >= 5 else 0.0
    r_cs = chord ** 2 / (8 * h) + h / 2 if (h > 0.01 and chord > 0.02 and coverage < 0.6) else np.nan

    # kruin (parte superior interior) observada directamente
    near_top = arc[(np.abs(arc[:, 0] - sc) < 0.25 * r) & (arc[:, 1] > zc)]
    crown_obs = len(near_top) >= 5

    # bootstrap por bloques angulares de 10°
    ids = block_ids((ang + np.pi)[:, None], np.pi / 18)
    boot = []
    for _ in range(N_BOOT):
        b = arc[resample_blocks(ids, rng)]
        bsc, bzc, br = fit_circle_geometric(b, sc, zc, r, iters=15)
        bt = b[(np.abs(b[:, 0] - bsc) < 0.25 * br) & (b[:, 1] > bzc)]
        crown_b = np.percentile(bt[:, 1], 90) if len(bt) >= 3 else bzc + br
        boot.append((2 * br, bzc, bzc - br, crown_b))
    boot = np.array(boot)
    s_d, s_zc, s_bob, s_crown = np.std(boot, axis=0)

    D = 2 * r
    crown = float(np.percentile(near_top[:, 1], 90)) if crown_obs else zc + r
    # σ_modelo: discrepancia entre métodos + sensibilidad del radio a un error sistemático δ
    # (medio voxel) en cuerda y flecha: en arcos planos el radio es muy sensible.
    geo_r = 0.0
    if np.isfinite(r_cs):
        delta = VOXEL_M / 2
        geo_r = float(np.hypot((0.5 - chord ** 2 / (8 * h ** 2)) * delta, chord / (4 * h) * delta))
    model_d = float(np.hypot(abs(D - 2 * r_cs) / 2 if np.isfinite(r_cs) else 0.0, 2 * geo_r))
    U_d = expanded(D, s_d, model_d)
    sane = 0.03 < r < 0.8 and len(arc) >= 12
    if coverage >= 0.5 and U_d / D < 0.05 and len(arc) >= 100:
        d_conf = HOOG
    elif coverage >= 0.33 and U_d / D < 0.10 and len(arc) >= 50:
        d_conf = MIDDEL
    elif sane and U_d / D < 0.30:
        d_conf = LAAG
    else:
        d_conf = ONZEKER
    status = {HOOG: MEASURED, MIDDEL: MEASURED, LAAG: ESTIMATED, ONZEKER: UNKNOWN}[d_conf]

    def height(value, s_stat, model, cap):
        U = expanded(value, s_stat, model, floor_sigma, scale=False)
        U = float(np.hypot(U, K_EXPAND * SCALE_REL_SIGMA * abs(value)))
        c = HOOG if U < 0.01 else MIDDEL if U < 0.025 else LAAG if U < 0.06 else ONZEKER
        return meas(value, U, worst(c, cap))

    cap = {MEASURED: HOOG, ESTIMATED: LAAG, UNKNOWN: ONZEKER}[status]
    alt_r = r_cs if np.isfinite(r_cs) else r
    out = dict(status=status, d_conf=d_conf, fit_r=r, fit_rms=rms, coverage=coverage, inliers=len(arc),
               r_chord=None if not np.isfinite(r_cs) else float(r_cs), center_s=sc, center_z=zc,
               diameter=meas(D, U_d, d_conf, sigma_stat=float(s_d), sigma_model=float(model_d)),
               axis_h=height(zc, s_zc, np.hypot(abs(zc - (crown - alt_r)) / 2, geo_r), cap),
               bob=height(zc - r, s_bob, np.hypot(abs((zc - r) - (crown - 2 * alt_r)) / 2, 2 * geo_r), cap),
               crown=height(crown, s_crown, 0.0, HOOG if crown_obs else worst(cap, LAAG)))
    out["bob_depth"] = meas(top_z - out["bob"]["value"],
                            float(np.hypot(out["bob"]["U"], K_EXPAND * top_sigma)), out["bob"]["conf"])
    U_mm = U_d * 1000
    out["nominal"] = nominal_candidates(D * 1000, U_mm) if status != UNKNOWN else []
    return out


def detect_connections(L, N_L, put, rect, rng):
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
        sec = _section_at(put["sections"], float(np.median(Q[:, 2]))) or ch
        if sec is not ch and _section_at([ch], float(np.median(Q[:, 2]))):
            sec = ch
        cxy = np.array([sec["cx"], sec["cy"]])
        m = Q[:, :2].mean(axis=0) - cxy
        # dirección: normal de la pared atravesada (supuesta perpendicular a la pared)
        if rect:
            k = int(np.argmax(np.abs(m) - np.array([sec["hx"], sec["hy"]])))
            d2 = np.zeros(2); d2[k] = np.sign(m[k])
            radial_wall = [sec["hx"], sec["hy"]][k]
        else:
            d2 = m / np.linalg.norm(m)
            radial_wall = sec["hx"]
        d = np.array([d2[0], d2[1], 0.0])
        s_axis = np.array([-d2[1], d2[0], 0.0])  # horizontal, perpendicular a la tubería

        # sección transversal (s, z); quitar superficie horizontal inferior (agua/sedimento en el tubo)
        s = Q @ s_axis
        bottom = (np.abs(NQ[:, 2]) > FLOOR_NZ_MIN) & (Q[:, 2] < Q[:, 2].min() + 0.02)
        sz = np.column_stack([s, Q[:, 2]])[~bottom]
        s_c0 = cxy @ s_axis[:2]
        wall_center = np.array([cxy[0], cxy[1], 0.0]) + d * radial_wall + s_axis * (float(np.median(s)) - s_c0)
        wall_center[2] = float(np.median(Q[:, 2]))

        conn = dict(points=Q, direction=d, s_axis=s_axis, n=len(Q), wall_center=wall_center,
                    opening_w=float(np.ptp(np.percentile(s, [2, 98]))),
                    opening_h=float(np.ptp(np.percentile(Q[:, 2], [2, 98]))),
                    visible_len=float(np.ptp(Q @ d)), status=UNKNOWN, note="")
        if len(sz) >= 12:
            arc = _measure_arc(sz, rng, put["floor_sigma"], put["top_z"], put["top_sigma"])
            conn.update(arc)
            center = np.array([cxy[0], cxy[1], 0.0]) + d * radial_wall + s_axis * (arc["center_s"] - s_c0)
            center[2] = arc["center_z"]
            conn["center"], conn["radius"] = center, arc["fit_r"]
            if arc["status"] == UNKNOWN:
                conn["note"] = f"boog ~{arc['coverage'] * 360:.0f}°, onvoldoende geometrie"
        else:
            conn["note"] = "te weinig punten in doorsnede"
        ref = conn.get("center", wall_center)
        conn["angle_deg"] = float(np.degrees(np.arctan2(ref[1] - ch["cy"], ref[0] - ch["cx"])) % 360)
        conns.append(conn)
    conns.sort(key=lambda c: c["angle_deg"])
    return conns


# ---------------------------------------------------------------- entrada principal

def analyze(pcd, seed=0):
    """Analiza una nube de puntos Open3D. Devuelve dict con frame, put, conexiones y la nube en local."""
    rng = np.random.default_rng(seed)
    n_raw = len(pcd.points)
    P, N = preprocess(pcd)
    center, R, symmetry = estimate_frame(P, N)
    rect = symmetry >= RECT_SYMMETRY_MIN
    L, N_L = (P - center) @ R.T, N @ R.T
    put = detect_put(L, N_L, rect, rng)

    # recentrar: origen en el centro de la cámara a la altura del fondo
    ch = put["chamber"]
    shift = np.array([ch["cx"], ch["cy"], put["floor_z"]])
    L -= shift
    for s in put["sections"]:
        s["cx"] -= shift[0]; s["cy"] -= shift[1]; s["z0"] -= shift[2]; s["z1"] -= shift[2]
        for q in s["slices"]:
            q["cx"] -= shift[0]; q["cy"] -= shift[1]
    if put["walls"]["walls"]:
        for w in put["walls"]["walls"].values():
            w["c"] = w["c"] - shift; w["points"] = w["points"] - shift
    put["top_z"] -= shift[2]; put["top_wall_z"] -= shift[2]; put["floor_z"] = 0.0
    conns = detect_connections(L, N_L, put, rect, rng)

    levels = [put["width_x"]["conf"], put["width_y"]["conf"], put["depth"]["conf"]]
    dims = np.ptp(L, axis=0)
    quality = dict(points_raw=n_raw, points_analysed=len(P), wall_coverage=put["walls"]["coverage"],
                   measurement=worst(*levels),
                   scale="metrisch (m)" if 0.3 < put["width_x"]["value"] < 5 and dims.max() < 50 else "controleer schaal")
    tilt = float(np.degrees(np.arccos(min(abs(R[2] @ np.array([0, 0, 1.0])), 1.0))))
    return dict(put=put, connections=conns, local_points=L, rect=rect, symmetry=symmetry, quality=quality,
                R=R, origin=center + shift @ R, axis_tilt_vs_ply_z=tilt, n_points=len(P))


def load_and_analyze(path):
    pcd = o3d.io.read_point_cloud(str(path))
    if len(pcd.points) == 0:
        mesh = o3d.io.read_triangle_mesh(str(path))
        pcd = o3d.geometry.PointCloud(mesh.vertices)
    if len(pcd.points) == 0:
        raise AnalysisError(f"No se pudieron leer puntos de {path}")
    return analyze(pcd)


# ---------------------------------------------------------------- informe

def fmt_len(m, detail=False):
    """< 1 m -> mm; >= 1 m -> m (con mm en detalle)."""
    v, U = m["value"], m["U"]
    if abs(v) < 1.0 or detail:
        return f"{v * 1000:.0f} ± {U * 1000:.0f} mm"
    return f"{v:.3f} ± {U:.3f} m"


def report(res):
    """Informe de texto (consola)."""
    put, q, lines = res["put"], res["quality"], []
    lines.append("=== PUT ===")
    lines.append(f"Vorm:            {put['shape']} (simetría 90°: {res['symmetry']:.2f}); eje vs Z PLY {res['axis_tilt_vs_ply_z']:.1f}°")
    if put["diameter"]:
        lines.append(f"Diameter:        {fmt_len(put['diameter'])}  [{put['diameter']['conf']}]")
    lines.append(f"Binnenmaat X:    {fmt_len(put['width_x'])}  [{put['width_x']['conf']}]")
    lines.append(f"Binnenmaat Y:    {fmt_len(put['width_y'])}  [{put['width_y']['conf']}]")
    lines.append(f"Diepte:          {fmt_len(put['depth'])} ({fmt_len(put['depth'], True)})  [{put['depth']['conf']}]"
                 + ("" if put["ground_found"] else "  (sin maaiveld: hasta el borde de la pared)"))
    lines.append(f"Buitenmaat:      niet meetbaar vanaf binnen  [{ONZEKER}]")
    w = put["walls"]
    if w["walls"]:
        for k, wl in w["walls"].items():
            lines.append(f"  wand {k}: {wl['n_used']:5d}/{wl['n_total']:5d} ptn  rms {wl['rms'] * 1000:.1f} mm  dekking {wl['coverage'] * 100:.0f}%")
        g = w["geometry"]
        lines.append(f"  parallel X {g['parallel_x']:.2f}°  parallel Y {g['parallel_y']:.2f}°  haaksheid X/Y {g['perpendicular']:.2f}°")
    fi, gi = put["floor_info"], put["ground_info"]
    lines.append(f"  bodem: {fi['n_used']} ptn, rms {fi['rms'] * 1000:.1f} mm, σ {put['floor_sigma'] * 1000:.1f} mm, helling {fi['slope_deg']:.1f}°")
    if gi:
        lines.append(f"  maaiveld: {gi['n_used']} ptn, rms {gi['rms'] * 1000:.1f} mm, σ {put['top_sigma'] * 1000:.1f} mm, helling {gi['slope_deg']:.1f}°")
    for i, s in enumerate(put["sections"]):
        lines.append(f"  tramo {i + 1}: z {s['z0']:+.2f}..{s['z1']:+.2f} m  ~{2 * s['hx'] * 1000:.0f} x {2 * s['hy'] * 1000:.0f} mm")
    lines.append(f"\n=== AANSLUITINGEN ({len(res['connections'])}) ===")
    for i, c in enumerate(res["connections"]):
        lines.append(f"A{i + 1}  richting {c['angle_deg']:.0f}°  [{c['status']}]")
        if "diameter" in c:
            d = c["diameter"]
            tag = "Diameter:" if c["status"] == MEASURED else "Diameter: ONZEKER   Geschat:"
            if c["status"] != UNKNOWN:
                lines.append(f"  {tag} {fmt_len(d)}  [{d['conf']}]  (boog ~{c['coverage'] * 360:.0f}°, "
                             f"rms {c['fit_rms'] * 1000:.1f} mm, koorde-methode "
                             f"{'—' if c['r_chord'] is None else f'{2 * c['r_chord'] * 1000:.0f} mm'})")
                if c["nominal"]:
                    lines.append(f"  Waarschijnlijk nominaal: " + " / ".join(f"Ø{n}" for n in c["nominal"]))
            else:
                lines.append(f"  Diameter: ONZEKER ({c['note']})")
            for key, name in [("crown", "Kruin (binnen-bovenkant)"), ("axis_h", "Hoogte as"), ("bob", "BOB")]:
                lines.append(f"  {name + ':':<26}{fmt_len(c[key])} boven bodem  [{c[key]['conf']}]")
            lines.append(f"  {'BOB onder maaiveld:':<26}{fmt_len(c['bob_depth'])}  [{c['bob_depth']['conf']}]")
        else:
            lines.append(f"  Diameter: ONZEKER ({c['note']})")
        lines.append(f"  Zichtbare opening: {c['opening_w'] * 1000:.0f} x {c['opening_h'] * 1000:.0f} mm; "
                     f"richting buis: haaks op wand (aangenomen)")
    lines.append("\n=== SCAN QUALITY ===")
    lines.append(f"Points analysed:     {q['points_analysed']:,} / {q['points_raw']:,}")
    if q["wall_coverage"] is not None:
        lines.append(f"Geometry coverage:   {q['wall_coverage'] * 100:.0f}% wanden")
    lines.append(f"Measurement quality: {q['measurement']}")
    lines.append(f"Scale:               {q['scale']} / Polycam")
    lines.append(f"(± = 2σ; incluye {SCALE_REL_SIGMA * 100:.1f}% σ de escala supuesta)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "scan.ply"
    t0 = time.time()
    res = load_and_analyze(path)
    print(report(res))
    print(f"\nTiempo de análisis: {time.time() - t0:.2f} s")
