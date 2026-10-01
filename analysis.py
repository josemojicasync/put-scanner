"""Análisis automático de una put (pozo de registro) a partir de un escaneo PLY.

Todo internamente en metros. Sistema local de la put:
  origen = centro de la cámara a la altura del fondo, Z = eje vertical (hacia arriba),
  X/Y = paredes (put rectangular) o proyección del X del PLY (put redonda).

Incertidumbres: "±" = incertidumbre expandida k=2 (~95 %):
  U = 2 * sqrt(σ_estadística² + σ_modelo² + σ_escala²)
  σ_estadística: bootstrap por bloques espaciales (no por puntos sueltos, que subestima).
  σ_modelo: discrepancia entre dos métodos independientes (cuando existe).
  σ_escala: SCALE_REL_SIGMA * valor (supuesto sobre la escala de Polycam/LiDAR).

Cada medida es un dict: value, U, conf (HOOG/MIDDEL/LAAG/ONZEKER), status (MEASURED/ESTIMATED/UNKNOWN)
y method. Una medida UNKNOWN tiene value=None: nunca se inventa un número.
"""
import numpy as np
import open3d as o3d

from measurement import fit_circle_2d, fit_circle_robust

# ---------------------------------------------------------------- parámetros
# Cada umbral tiene una justificación geométrica o física; no están ajustados a un scan concreto.
VOXEL_M = 0.01              # resolución de trabajo: 1 cm (≈ espaciado típico de Polycam LiDAR)
NORMAL_RADIUS_M = 0.05      # 5 vecinos de voxel por lado: normal estable sin mezclar superficies
SLICE_M = 0.05
WALL_NZ_MAX = 0.35          # |n·z| < 0.35 -> pared (inclinación > 70°)
FLOOR_NZ_MIN = 0.85         # |n·z| > 0.85 -> superficie horizontal (inclinación < 32°)
SIDE_NORMAL_MIN = 0.8       # normal alineada con una pared (±37°): excluye el interior de tubos
SECTION_TOL_M = 0.03        # cortes con tamaño dentro de esta tolerancia -> mismo tramo
RECT_SYMMETRY_MIN = 0.5     # |<e^{4iθ}>| de las normales: rectángulo ≈ 1, círculo ≈ 0; 0.5 = punto medio
ROUND_SYMMETRY_MAX = 0.25
CIRCLE_REL_RMS_ROUND = 0.02 # residuo mediano/R de un círculo ajustado: cuadrado ≈ 9 %, círculo con ruido 5 mm < 1 %
BEHIND_WALL_M = 0.03        # 3 cm detrás de la pared: > 5σ del ruido de pared típico (≤ 6 mm)
PIPE_CLUSTER_EPS_M = 0.05
PIPE_MIN_POINTS = 40
PLANE_THRESH_M = 0.01       # RANSAC paredes
SURFACE_THRESH_M = 0.015    # RANSAC fondo / maaiveld (superficies más rugosas)
MIN_WALL_POINTS = 100
WALL_MISSING_COVERAGE = 0.10  # < 10 % de la pared con puntos -> pared no escaneada
BLOCK_M = 0.10              # tamaño de bloque del bootstrap espacial
N_BOOT = 200
SCALE_REL_SIGMA = 0.005     # SUPUESTO: 0,5 % (1σ) de error de escala del escaneo; calibrar con validation/
K_EXPAND = 2.0
# Diámetros de tubería: la serie de tamaños de alcantarillado sigue números preferentes R10 (ISO 3),
# paso 10^(1/10) ≈ +26 %. Si el intervalo 95 % (2U) es más ancho que un paso, el diámetro no permite
# identificar el tamaño: se declara UNKNOWN.
R10_STEP = 10 ** 0.1 - 1
D_UNKNOWN_REL = R10_STEP / 2          # U/D > 12.9 % -> UNKNOWN
D_MEASURED_REL = 0.05                 # U/D ≤ 5 % (y arco ≥ 180°) -> MEASURED
ARC_MIN_DEG = 60                      # < 60°: el arco es casi recto (flecha < 13 % de r): sin información de curvatura
ARC_MEASURED_DEG = 180                # ≥ 180°: el diámetro se observa como cuerda máxima, no se extrapola
                                      # (en < 180° una ovalización del tubo cambia la curvatura observada)
DIR_AXIS_RATIO_MAX = 0.25   # λmin/λmed de las normales del tubo: eje bien definido
DIR_MIN_LENGTH_M = 0.15     # longitud visible mínima del tubo a lo largo de su eje
DIR_MEASURED_U_DEG = 2.0
SLOPE_MEASURED_U_DEG = 0.25 # pendientes de alcantarillado típicas 0,1–0,6°: hace falta U ≤ 0,25°
SLOPE_ESTIMATED_U_DEG = 1.0
NOMINALS_MM = [110, 125, 160, 200, 250, 300, 315, 400, 500, 600, 630, 800, 1000]

HOOG, MIDDEL, LAAG, ONZEKER = "HOOG", "MIDDEL", "LAAG", "ONZEKER"
MEASURED, ESTIMATED, UNKNOWN = "MEASURED", "ESTIMATED", "UNKNOWN"
CONF_ORDER = [HOOG, MIDDEL, LAAG, ONZEKER]
STATUS_OF_CONF = {HOOG: MEASURED, MIDDEL: MEASURED, LAAG: ESTIMATED, ONZEKER: UNKNOWN}

WALLS = [("+X", 0, 1), ("-X", 0, -1), ("+Y", 1, 1), ("-Y", 1, -1)]


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


def meas(value, U, conf, method="", status=None, **extra):
    """Medida: value/U en unidades SI (m, grados), conf, status, method."""
    return dict(value=None if value is None else float(value), U=None if U is None else float(U), conf=conf,
                status=status or STATUS_OF_CONF[conf], method=method, **extra)


def unknown(method, **extra):
    return meas(None, None, ONZEKER, method, UNKNOWN, **extra)


def cap_status(m, max_status):
    """Limita una medida a ESTIMATED (p. ej. si la forma de la put no está confirmada)."""
    if max_status == ESTIMATED and m["status"] == MEASURED:
        m = dict(m, status=ESTIMATED, conf=worst(m["conf"], LAAG))
    return m


def block_ids(coords, size=BLOCK_M):
    """Grupos de índices por bloque espacial (coords: Nx1, Nx2 o Nx3)."""
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


def nominal_candidates(lo_mm, hi_mm):
    """Tamaños nominales compatibles con un diámetro interior en [lo, hi]. El interior de un tubo
    nominal N está entre ~0,9N (PVC, pared incluida) y N (hormigón, nominal = interior).
    Es una SUGERENCIA, nunca una medida."""
    hi_mm = np.inf if hi_mm is None else hi_mm
    return [n for n in NOMINALS_MM if 0.9 * n <= hi_mm and n >= lo_mm]


def coverage_cells(uv, u_range, v_range, cell=0.05):
    """Fracción de celdas (cell x cell) ocupadas en un rectángulo u_range x v_range."""
    if len(uv) == 0:
        return 0.0
    m = (uv[:, 0] >= u_range[0]) & (uv[:, 0] < u_range[1]) & (uv[:, 1] >= v_range[0]) & (uv[:, 1] < v_range[1])
    if not m.any():
        return 0.0
    occ = np.unique(np.floor((uv[m] - [u_range[0], v_range[0]]) / cell).astype(int), axis=0)
    total = max(1, int(np.ceil((u_range[1] - u_range[0]) / cell)) * int(np.ceil((v_range[1] - v_range[0]) / cell)))
    return min(1.0, len(occ) / total)


# ---------------------------------------------------------------- preprocesado

def preprocess(pcd):
    """Limpia, reduce y calcula normales. No modifica el objeto original."""
    p = o3d.geometry.PointCloud(pcd)
    p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.5)
    n_clean = len(p.points)
    if len(p.points) > 30000:
        p = p.voxel_down_sample(VOXEL_M)
    p.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=NORMAL_RADIUS_M, max_nn=30))
    if len(p.points) < 1000:
        raise AnalysisError(f"Muy pocos puntos tras limpiar ({len(p.points)}).")
    return np.asarray(p.points).copy(), np.asarray(p.normals).copy(), n_clean


def point_spacing(pcd, rng, n=5000):
    """Mediana de la distancia al vecino más cercano (nube original, muestra)."""
    P = np.asarray(pcd.points)
    if len(P) < 10:
        return np.nan
    idx = rng.choice(len(P), min(n, len(P)), replace=False)
    tree = o3d.geometry.KDTreeFlann(pcd)
    d = []
    for i in idx:
        _, nn, d2 = tree.search_knn_vector_3d(P[i], 2)
        d.append(np.sqrt(d2[1]))
    return float(np.median(d))


# ---------------------------------------------------------------- orientación

def _least_normal_direction(normals):
    """Dirección con menor proyección de normales: eje de una estructura de paredes verticales."""
    w, v = np.linalg.eigh(normals.T @ normals / len(normals))
    return v[:, 0]


def _ground_score(P, N, e):
    """Fracción de puntos en superficies perpendiculares a `e` situadas FUERA de la huella de las paredes
    (el maaiveld). Solo el eje vertical real tiene un terreno perpendicular que se extiende fuera de la put;
    con otro eje candidato, las superficies 'horizontales' son paredes dentro de la huella."""
    wall = np.abs(N @ e) < WALL_NZ_MAX
    horiz = np.abs(N @ e) > FLOOR_NZ_MIN
    if wall.sum() < 100 or horiz.sum() < 30:
        return 0.0
    c = P[wall].mean(axis=0)
    rad = lambda Q: np.linalg.norm(np.cross(Q - c, e), axis=1)
    r_wall = np.percentile(rad(P[wall]), 90)
    return float(np.mean(rad(P[horiz]) > 1.3 * r_wall) * horiz.sum() / len(P))


def estimate_frame(P, N):
    """Devuelve (origen provisional, R 3x3 con filas = ejes locales X,Y,Z, simetría rectangular)."""
    # Candidatos: direcciones principales de las normales. Por defecto, la menos representada
    # (las paredes verticales tienen normales perpendiculares al eje). Pero en una caja la geometría es
    # simétrica y el área decide: si falta una pared o el terreno es grande, otra dirección puede quedar
    # menos representada. Si el maaiveld es visible, decide el candidato con terreno fuera de la huella.
    _, V = np.linalg.eigh(N.T @ N / len(N))
    scores = [_ground_score(P, N, V[:, i]) for i in range(3)]
    best = int(np.argmax(scores))
    others = sorted(scores)[-2]
    axis = V[:, best] if scores[best] > 0.02 and scores[best] > 2 * others else V[:, 0]
    for _ in range(3):  # refinar solo con puntos de pared
        wall = np.abs(N @ axis) < WALL_NZ_MAX
        axis = _least_normal_direction(N[wall])

    # sentido: el terreno (superficie horizontal que se extiende fuera de las paredes) está arriba;
    # el bodem (superficie horizontal dentro de la huella) está abajo
    t = P @ axis
    lo, hi = np.percentile(t, [2, 98])
    flat = np.abs(N @ axis) > FLOOR_NZ_MIN
    wall = np.abs(N @ axis) < WALL_NZ_MAX
    center = P[wall].mean(axis=0)
    radial = np.linalg.norm(np.cross(P - center, axis), axis=1)
    r_wall = np.percentile(radial[wall], 90)
    band = 0.15 * (hi - lo)
    end_hi, end_lo = flat & (t > hi - band), flat & (t < lo + band)
    spread_hi = np.percentile(radial[end_hi], 90) if end_hi.sum() > 20 else None
    spread_lo = np.percentile(radial[end_lo], 90) if end_lo.sum() > 20 else None
    if spread_hi is not None and spread_lo is not None:
        flip = spread_lo > spread_hi
    elif spread_lo is not None:   # solo superficie abajo: si está fuera de la huella es terreno -> girar
        flip = spread_lo > 1.2 * r_wall
    elif spread_hi is not None:   # solo superficie arriba: si está dentro de la huella es el bodem -> girar
        flip = spread_hi <= 1.2 * r_wall
    else:
        flip = False
    if flip:
        axis = -axis

    # X/Y: alinear con las paredes si la put es rectangular (X local = dirección de pared más cercana al X del PLY)
    px = np.eye(3)[0] - axis * axis[0]
    if np.linalg.norm(px) < 0.3:  # eje casi paralelo al X del PLY
        px = np.eye(3)[1] - axis * axis[1]
    e1 = px / np.linalg.norm(px)
    e2 = np.cross(axis, e1)
    wall = np.abs(N @ axis) < WALL_NZ_MAX
    theta = np.arctan2(N[wall] @ e2, N[wall] @ e1)
    m4 = np.mean(np.exp(4j * theta))
    symmetry = float(abs(m4))
    if symmetry >= (RECT_SYMMETRY_MIN + ROUND_SYMMETRY_MAX) / 2:
        rot = np.angle(m4) / 4
    else:  # redonda: X local = X del PLY proyectado
        rot = np.arctan2(np.eye(3)[0] @ e2, np.eye(3)[0] @ e1)
    x = np.cos(rot) * e1 + np.sin(rot) * e2
    y = np.cross(axis, x)
    return center, np.vstack([x, y, axis]), symmetry


def axis_uncertainty(L, N_L, rng, n_boot=100):
    """σ (grados) de la dirección del eje vertical: bootstrap por bloques de las normales de pared."""
    wall = np.abs(N_L[:, 2]) < WALL_NZ_MAX
    Nw, Pw = N_L[wall], L[wall]
    if len(Nw) < 100:
        return np.nan
    groups = block_ids(Pw, BLOCK_M)
    ang = []
    for _ in range(n_boot):
        a = _least_normal_direction(Nw[resample_blocks(groups, rng)])
        ang.append(np.degrees(np.arccos(min(abs(a[2]), 1.0))))
    return float(np.sqrt(np.mean(np.square(ang))))


# ---------------------------------------------------------------- put: tramos

def _slice_shape(xy, nxy, rect):
    """Tamaño de un corte horizontal de pared. Rect: (cx, cy, hx, hy, rms). Redonda: (cx, cy, r, r, rms).
    Rectangular: cada lado se mide solo con puntos cuya normal es perpendicular a ese lado (mediana);
    así el interior de un tubo (normales perpendiculares a su propio eje) no ensancha el corte."""
    if rect:
        c0 = np.median(xy, axis=0)
        lo, hi = np.percentile(xy, 3, axis=0), np.percentile(xy, 97, axis=0)
        for ax in (0, 1):
            side = np.abs(nxy[:, ax]) > SIDE_NORMAL_MIN
            p, n = xy[side & (xy[:, ax] > c0[ax]), ax], xy[side & (xy[:, ax] < c0[ax]), ax]
            if len(p) >= 10:
                hi[ax] = np.median(p)
            if len(n) >= 10:
                lo[ax] = np.median(n)
        c, h = (lo + hi) / 2, (hi - lo) / 2
        d = np.minimum(np.abs(np.abs(xy[:, 0] - c[0]) - h[0]), np.abs(np.abs(xy[:, 1] - c[1]) - h[1]))
        return c[0], c[1], h[0], h[1], float(np.sqrt(np.mean(np.minimum(d, 0.05) ** 2)))
    cx, cy, r, _, rms = fit_circle_robust(xy)
    return cx, cy, r, r, rms


def find_sections(L, N_L, rect):
    """Cortes cada SLICE_M agrupados en tramos de tamaño constante. Tamaños aproximados (indicativos)."""
    wall = np.abs(N_L[:, 2]) < WALL_NZ_MAX
    z = L[:, 2]
    slices = []
    for z0 in np.arange(np.percentile(z, 0.5), np.percentile(z, 99.5), SLICE_M):
        m = wall & (z >= z0) & (z < z0 + SLICE_M)
        if m.sum() < 60:
            continue
        cx, cy, hx, hy, rms = _slice_shape(L[m, :2], N_L[m, :2], rect)
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


def classify_shape(L, N_L, ch, symmetry):
    """Decide rectangular / rond / onbekend con dos evidencias independientes:
    simetría de 90° de las normales y residuo de un ajuste de círculo a los puntos de pared."""
    z = L[:, 2]
    m = (np.abs(N_L[:, 2]) < WALL_NZ_MAX) & (z > ch["z0"]) & (z < ch["z1"])
    xy = L[m, :2]
    circle_rel = np.nan
    if len(xy) >= 50:
        # círculo de mínimos cuadrados sobre TODOS los puntos de pared; residuo mediano (robusto a tubos):
        # cuadrado -> ≈ 9 % del radio, círculo con ruido de 5 mm -> < 1 %
        cx, cy, r = fit_circle_2d(xy)
        cx, cy, r = fit_circle_geometric(xy, cx, cy, r)
        d = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r)
        circle_rel = float(np.median(d) / r) if r > 0 else np.nan
    if symmetry >= RECT_SYMMETRY_MIN and not circle_rel <= CIRCLE_REL_RMS_ROUND:
        shape, status = "rechthoekig", MEASURED
    elif symmetry <= ROUND_SYMMETRY_MAX and circle_rel <= CIRCLE_REL_RMS_ROUND:
        shape, status = "rond", MEASURED
    else:
        shape, status = "onbekend", UNKNOWN
    conf = HOOG if status == MEASURED else ONZEKER
    return meas(None, None, conf, "normal_symmetry+circle_residual", status, label=shape,
                symmetry=float(symmetry), circle_rel_rms=circle_rel)


# ---------------------------------------------------------------- put: paredes (precisión)

def measure_rect_walls(L, N_L, ch, rng):
    """Ajusta los planos interiores de la cámara (RANSAC) y mide anchos con bootstrap por bloques.
    Una pared sin puntos suficientes se marca como ausente; la anchura correspondiente se estima a
    partir de la extensión de las paredes adyacentes (ESTIMATED) o queda UNKNOWN."""
    z = L[:, 2]
    zlo, zhi = ch["z0"] + 0.05, ch["z1"] - 0.05
    base = (np.abs(N_L[:, 2]) < WALL_NZ_MAX) & (z > zlo) & (z < zhi)
    walls, missing, debug_missing = {}, [], {}
    for key, ax, sign in WALLS:
        c0, h = (ch["cx"], ch["hx"]) if ax == 0 else (ch["cy"], ch["hy"])
        hint = np.zeros(3); hint[ax] = sign
        m = base & (np.abs(N_L[:, ax]) > SIDE_NORMAL_MIN) & (sign * (L[:, ax] - c0) > 0.5 * h)
        P = L[m]
        other = 1 - ax
        span = 2 * (ch["hy"] if ax == 0 else ch["hx"])
        oc = ch["cy"] if ax == 0 else ch["cx"]
        if len(P) < MIN_WALL_POINTS:
            missing.append(key)
            debug_missing[key] = dict(candidates=P, reason=f"{len(P)} < {MIN_WALL_POINTS} puntos candidatos")
            continue
        n, c, inl, rms = fit_plane_ransac(P, PLANE_THRESH_M, rng, hint)
        Pin = P[inl]
        cov = coverage_cells(np.column_stack([Pin[:, other], Pin[:, 2]]), (oc - span / 2, oc + span / 2), (zlo, zhi))
        if cov < WALL_MISSING_COVERAGE:
            missing.append(key)
            debug_missing[key] = dict(candidates=P, reason=f"cobertura {cov:.0%} < {WALL_MISSING_COVERAGE:.0%}")
            continue
        # (la detección de capa duplicada está en quality.py: necesita conocer las aberturas de tubos)
        # deriva: posición del plano en la mitad inferior vs superior de la pared
        zm = np.median(Pin[:, 2])
        lo_p, hi_p = Pin[Pin[:, 2] < zm], Pin[Pin[:, 2] >= zm]
        drift = np.nan
        if len(lo_p) > 50 and len(hi_p) > 50:
            n_lo, c_lo = fit_plane_lsq(lo_p); n_hi, c_hi = fit_plane_lsq(hi_p)
            drift = float(abs((c_hi - c_lo) @ n))
        walls[key] = dict(n=n, c=c, points=Pin, ids=block_ids(np.column_stack([Pin[:, other], Pin[:, 2]])),
                          rms=rms, n_used=int(inl.sum()), n_total=len(P), coverage=cov, drift_m=drift,
                          # diagnóstico: todos los candidatos y cuáles usó el ajuste
                          candidates=P, inlier_mask=inl, select=dict(z_range=(zlo, zhi), side_c0=c0, side_h=h))

    # alturas de evaluación del perfil de anchura (abajo / medio / arriba del tramo medido)
    z_prof = (zlo + 0.05, (zlo + zhi) / 2, zhi - 0.05)

    def width_at(na, ca, nb, cb, ax, z):
        """Distancia entre dos planos a lo largo del eje `ax`, en la línea central de la cámara a altura z."""
        other = 1 - ax
        oval = ch["cy"] if ax == 0 else ch["cx"]
        xs = []
        for n, c in ((na, ca), (nb, cb)):
            s = n[other] * (oval - c[other]) + n[2] * (z - c[2])
            xs.append(c[ax] - s / n[ax])
        return abs(xs[0] - xs[1])

    def widths(wp):
        out = {}
        for name, a, b, ax in [("x", "+X", "-X", 0), ("y", "+Y", "-Y", 1)]:
            if a in wp and b in wp:
                na, ca = wp[a]; nb, cb = wp[b]
                n = na - nb; n /= np.linalg.norm(n)
                out[name] = (ca - cb) @ n
                out[name + "_prof"] = [width_at(na, ca, nb, cb, ax, z) for z in z_prof]
        return out

    wp = {k: (w["n"], w["c"]) for k, w in walls.items()}
    wv = widths(wp)
    boot = []
    for _ in range(N_BOOT):
        bp = {}
        for k, w in walls.items():
            n, c = fit_plane_lsq(w["points"][resample_blocks(w["ids"], rng)])
            bp[k] = (n if n @ w["n"] > 0 else -n, c)
        boot.append(widths(bp))

    def width_meas(name, keys, ax):
        if name in wv:
            value = float(wv[name])
            sigma = float(np.std([b[name] for b in boot]))
            # paredes no paralelas: la anchura depende de la altura. Si no se sabe a qué altura se compara,
            # la variación medida se trata como distribución uniforme: σ_modelo = rango / √12
            prof = wv[name + "_prof"]
            var = prof[2] - prof[0]
            sigma_var = float(np.std([b[name + "_prof"][2] - b[name + "_prof"][0] for b in boot]))
            sigma_taper = abs(var) / np.sqrt(12)
            U = expanded(value, sigma, sigma_taper)
            ok = all(walls[k]["n_used"] >= 200 for k in keys)
            cov = min(walls[k]["coverage"] for k in keys)
            conf = rel_conf(U / value, (0.015, 0.03, 0.08), ok)
            if cov < 0.3:   # menos del 30 % de la pared: el plano se apoya en una zona pequeña
                conf = worst(conf, LAAG)
            profile = dict(z=list(z_prof), bottom=prof[0], mid=prof[1], top=prof[2], variation=var,
                           sigma_variation=sigma_var,
                           # significativa: > 3σ del ruido de los planos y > 3 mm (resolución útil)
                           significant=bool(abs(var) > 3 * sigma_var and abs(var) > 0.003))
            return meas(value, U, conf, "ransac_wall_planes_distance", sigma_stat=sigma, sigma_taper=sigma_taper,
                        profile=profile)
        # una pared ausente: extensión de las paredes perpendiculares (sus extremos llegan a las esquinas)
        perp = [k for k in walls if k[1] != keys[0][1]]
        if perp:
            ext = np.concatenate([walls[k]["points"][:, ax] for k in perp])
            lo, hi = np.percentile(ext, [1, 99])
            value = float(hi - lo)
            U = expanded(value, 0.01)  # σ 1 cm: esquinas redondeadas / borde de pared mal definido
            return meas(value, U, LAAG, "adjacent_wall_extent", ESTIMATED, missing_walls=[k for k in keys if k in missing])
        return unknown("no_walls", missing_walls=list(keys))

    wx = width_meas("x", ("+X", "-X"), 0)
    wy = width_meas("y", ("+Y", "-Y"), 1)

    def center(ax, a, b, width):
        if a in walls and b in walls:
            return float((walls[a]["c"][ax] + walls[b]["c"][ax]) / 2)
        if width["value"] is not None:
            if a in walls:
                return float(walls[a]["c"][ax] - width["value"] / 2)
            if b in walls:
                return float(walls[b]["c"][ax] + width["value"] / 2)
        return float(ch["cx"] if ax == 0 else ch["cy"])

    ang = lambda a, b: float(np.degrees(np.arccos(np.clip(abs(a @ b), -1, 1))))
    geo = {}
    if "+X" in walls and "-X" in walls:
        geo["parallel_x"] = ang(walls["+X"]["n"], walls["-X"]["n"])
    if "+Y" in walls and "-Y" in walls:
        geo["parallel_y"] = ang(walls["+Y"]["n"], walls["-Y"]["n"])
    if len(walls) == 4:
        nx = walls["+X"]["n"] - walls["-X"]["n"]; nx /= np.linalg.norm(nx)
        ny = walls["+Y"]["n"] - walls["-Y"]["n"]; ny /= np.linalg.norm(ny)
        geo["perpendicular"] = abs(90.0 - float(np.degrees(np.arccos(np.clip(nx @ ny, -1, 1)))))

    return dict(walls=walls, missing=missing, debug_missing=debug_missing, geometry=geo,
                cx=center(0, "+X", "-X", wx), cy=center(1, "+Y", "-Y", wy),
                width_x=wx, width_y=wy, diameter=None, circularity=None,
                coverage=float(np.mean([walls[k]["coverage"] if k in walls else 0.0 for k, _, _ in WALLS])))


def measure_round_wall(L, N_L, ch, rng):
    """Put redonda: círculo robusto sobre todos los puntos de pared de la cámara (geométrico),
    incertidumbre por bootstrap de bloques (ángulo x altura), circularidad = residuo."""
    z = L[:, 2]
    m = (np.abs(N_L[:, 2]) < WALL_NZ_MAX) & (z > ch["z0"] + 0.05) & (z < ch["z1"] - 0.05)
    P = L[m]
    if len(P) < MIN_WALL_POINTS:
        D = unknown("too_few_wall_points")
        return dict(walls=None, missing=["wand"], geometry={}, cx=ch["cx"], cy=ch["cy"], width_x=D, width_y=D,
                    diameter=D, circularity=unknown("too_few_wall_points"), coverage=0.0)
    cx, cy, r, inl, _ = fit_circle_robust(P[:, :2])
    near = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r) < max(0.03, 0.05 * r)  # excluye aberturas
    W = P[near]
    cx, cy, r = fit_circle_geometric(W[:, :2], cx, cy, r)
    res = np.hypot(W[:, 0] - cx, W[:, 1] - cy) - r
    rms, p95 = float(np.sqrt(np.mean(res ** 2))), float(np.percentile(np.abs(res), 95))
    ang = np.arctan2(W[:, 1] - cy, W[:, 0] - cx)
    sub = rng.choice(len(W), min(len(W), 8000), replace=False)  # el bootstrap por bloques no necesita todos los puntos
    groups = block_ids(np.column_stack([ang[sub] * r, W[sub, 2]]))
    boot = []
    for _ in range(N_BOOT):
        B = W[sub][resample_blocks(groups, rng)]
        boot.append(fit_circle_geometric(B[:, :2], cx, cy, r, iters=10))
    boot = np.array(boot)
    s_d = float(np.std(2 * boot[:, 2]))
    s_c = float(np.hypot(np.std(boot[:, 0]), np.std(boot[:, 1])))
    D = 2 * r
    cov = coverage_cells(np.column_stack([ang * r, W[:, 2]]), (-np.pi * r, np.pi * r), (ch["z0"] + 0.05, ch["z1"] - 0.05))
    U = expanded(D, s_d)
    conf = rel_conf(U / D, (0.015, 0.03, 0.08), len(W) >= 200)
    if cov < 0.3:
        conf = worst(conf, LAAG)
    dm = meas(D, U, conf, "geometric_circle_fit_wall", sigma_stat=s_d, radius=r)
    circ = meas(rms, None, HOOG if rms < 0.005 else MIDDEL if rms < 0.01 else LAAG, "circle_residual_rms",
                MEASURED, p95=p95)
    return dict(walls=None, missing=[], geometry={}, cx=float(cx), cy=float(cy), width_x=dm, width_y=dm,
                diameter=dm, circularity=circ, coverage=cov, center_sigma=s_c)


# ---------------------------------------------------------------- put: fondo y maaiveld

LEVEL_MIN_FRACTION = 0.15   # un nivel debe tener ≥ 15 % de los puntos (dentro de ±1 cm)
LEVEL_MIN_SEP_M = 0.025     # niveles separados > 2,5 cm (> umbral de RANSAC; si no, es la misma superficie)


def height_levels(z):
    """Niveles horizontales distintos en una superficie (p. ej. banket y canal, tapa y pavimento): picos del
    histograma de alturas con ≥ 15 % de los puntos, separados > 2,5 cm y con un valle entre ellos. Una pendiente
    continua da un histograma ancho sin valles y NO se considera multinivel."""
    if len(z) < 50:
        return []
    lo, hi = np.percentile(z, [1, 99])
    edges = np.arange(lo - 0.005, hi + 0.01, 0.005)
    h, _ = np.histogram(z, edges)
    hs = np.convolve(h, np.ones(3) / 3, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    peaks = [i for i in range(1, len(hs) - 1) if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] >= 0.25 * hs.max()]
    levels = []
    for i in sorted(peaks, key=lambda i: -hs[i]):
        zc = centers[i]
        sup = int(np.sum(np.abs(z - zc) < 0.01))
        if sup < LEVEL_MIN_FRACTION * len(z):
            continue
        ok = True
        for lv in levels:
            if abs(zc - lv["z"]) < LEVEL_MIN_SEP_M:
                ok = False
                break
            a, b = sorted([np.argmin(np.abs(centers - zc)), np.argmin(np.abs(centers - lv["z"]))])
            if hs[a:b + 1].min() > 0.5 * min(hs[a], hs[b]):   # sin valle: es una pendiente, no dos niveles
                ok = False
                break
        if ok:
            levels.append(dict(z=float(zc), support=sup))
    return levels if len(levels) >= 2 else []


def measure_surface(P, cx, cy, rng):
    """Plano robusto de una superficie horizontal. Devuelve (altura en (cx,cy), σ bootstrap, info).
    Si la superficie tiene varios niveles, el plano se ajusta SOLO al nivel dominante (un plano inclinado entre
    dos niveles puede tener más inliers y buen RMS, pero es una superficie que no existe)."""
    levels = height_levels(P[:, 2])
    if levels:
        main = max(levels, key=lambda lv: lv["support"])
        sel = np.abs(P[:, 2] - main["z"]) < SURFACE_THRESH_M
        n, c, inl_s, rms = fit_plane_ransac(P[sel], SURFACE_THRESH_M, rng, np.array([0, 0, 1.0]))
        inl = np.zeros(len(P), bool)
        inl[np.flatnonzero(sel)[inl_s]] = True
    else:
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
                                              blocks=len(ids), normal=n, point=c, inliers=Pin,
                                              candidates=P, inlier_mask=inl, eval_xy=(float(cx), float(cy)),
                                              levels=levels)


SURFACE_ALT_MIN_FRACTION = 0.15  # otra superficie con ≥ 15 % de los puntos del plano principal es plausible
SURFACE_ALT_MAX_ANGLE = 10.0     # casi paralela


def surface_candidates(info, cx, cy, rng, max_extra=2):
    """Superficies horizontales plausibles: la principal (RANSAC) y otras encontradas con RANSAC en los
    puntos rechazados. Una alternativa cuenta si tiene ≥ 15 % de los puntos del plano principal, es casi
    paralela y está a más de 2x el umbral de RANSAC (si no, es la misma superficie)."""
    if not info:
        return []
    n, c = info["normal"], info["point"]
    out = [dict(kind="primary", z=float(plane_height_at(n, c, cx, cy)), points=int(info["n_used"]),
                rms=float(info["rms"]), slope_deg=float(info["slope_deg"]))]
    P = info["candidates"]
    if info.get("levels"):
        # superficie multinivel: cada nivel distinto del principal es una alternativa plausible
        for lv in info["levels"]:
            sel = np.abs(P[:, 2] - lv["z"]) < SURFACE_THRESH_M
            if sel.sum() < 30:
                continue
            n2, c2 = fit_plane_lsq(P[sel])
            n2 = n2 if n2[2] > 0 else -n2
            z2 = float(plane_height_at(n2, c2, cx, cy))
            if abs(z2 - out[0]["z"]) > 2 * SURFACE_THRESH_M:
                res2 = (P[sel] - c2) @ n2
                out.append(dict(kind="alternative", z=z2, points=int(sel.sum()), rms=float(np.sqrt(np.mean(res2 ** 2))),
                                slope_deg=float(np.degrees(np.arccos(min(abs(n2[2]), 1.0)))), angle_deg=0.0,
                                inliers=P[sel], method="height_level"))
        return out
    rest = P[~info["inlier_mask"]]
    for _ in range(max_extra):
        if len(rest) < max(30, SURFACE_ALT_MIN_FRACTION * info["n_used"]):
            break
        n2, c2, inl2, rms2 = fit_plane_ransac(rest, SURFACE_THRESH_M, rng, np.array([0, 0, 1.0]))
        ang = float(np.degrees(np.arccos(np.clip(abs(n2 @ n), -1, 1))))
        z2 = float(plane_height_at(n2, c2, cx, cy))
        if inl2.sum() < SURFACE_ALT_MIN_FRACTION * info["n_used"] or ang > SURFACE_ALT_MAX_ANGLE:
            break
        if abs(z2 - out[0]["z"]) > 2 * SURFACE_THRESH_M:
            out.append(dict(kind="alternative", z=z2, points=int(inl2.sum()), rms=float(rms2),
                            slope_deg=float(np.degrees(np.arccos(min(abs(n2[2]), 1.0)))), angle_deg=ang,
                            inliers=rest[inl2]))
        rest = rest[~inl2]
    return out


def detect_put(L, N_L, rect, shape_known, rng):
    sections = find_sections(L, N_L, rect)
    chamber = max(sections, key=lambda s: s["z1"] - s["z0"])
    walls = measure_rect_walls(L, N_L, chamber, rng) if rect else measure_round_wall(L, N_L, chamber, rng)
    if not shape_known:  # forma no confirmada: ninguna dimensión puede ser MEASURED
        for k in ("width_x", "width_y", "diameter"):
            if walls.get(k):
                walls[k] = cap_status(walls[k], ESTIMATED)
    # los anchos precisos sustituyen a los aproximados de los cortes (si existen)
    chamber["cx"], chamber["cy"] = walls["cx"], walls["cy"]
    if walls["width_x"]["value"] is not None:
        chamber["hx"] = walls["width_x"]["value"] / 2
    if walls["width_y"]["value"] is not None:
        chamber["hy"] = walls["width_y"]["value"] / 2

    z = L[:, 2]
    flat = np.abs(N_L[:, 2]) > FLOOR_NZ_MIN
    if rect:
        inside = (np.abs(L[:, 0] - chamber["cx"]) < chamber["hx"] * 0.9) & \
                 (np.abs(L[:, 1] - chamber["cy"]) < chamber["hy"] * 0.9)
    else:
        inside = np.hypot(L[:, 0] - chamber["cx"], L[:, 1] - chamber["cy"]) < chamber["hx"] * 0.9
    low = flat & inside & (z < chamber["z0"] + 0.3)
    if low.sum() < 30:
        raise AnalysisError("No se encontró el fondo de la put (sin superficie horizontal dentro de la cámara).")
    floor_z, floor_s, floor_info = measure_surface(L[low], chamber["cx"], chamber["cy"], rng)
    fi = floor_info["inliers"]
    floor_info["coverage"] = coverage_cells(fi[:, :2], (chamber["cx"] - chamber["hx"] * 0.9, chamber["cx"] + chamber["hx"] * 0.9),
                                            (chamber["cy"] - chamber["hy"] * 0.9, chamber["cy"] + chamber["hy"] * 0.9))
    if not rect:
        floor_info["coverage"] = min(1.0, floor_info["coverage"] * 4 / np.pi)  # huella circular dentro del cuadrado

    stack = [s for s in sections if s["z0"] >= chamber["z0"] - 1e-6]
    top_wall = max(s["z1"] for s in stack)
    if rect:
        ext = np.maximum(np.abs(L[:, 0] - chamber["cx"]) - chamber["hx"],
                         np.abs(L[:, 1] - chamber["cy"]) - chamber["hy"]) > 0.05
    else:
        ext = np.hypot(L[:, 0] - chamber["cx"], L[:, 1] - chamber["cy"]) - chamber["hx"] > 0.05
    ground = flat & ext & (z > top_wall - 0.3)
    ground_found = ground.sum() > 30
    if ground_found:
        top_z, top_s, ground_info = measure_surface(L[ground], chamber["cx"], chamber["cy"], rng)
        gi = ground_info["inliers"]
        sectors = np.unique(((np.arctan2(gi[:, 1] - chamber["cy"], gi[:, 0] - chamber["cx"]) + np.pi) // (np.pi / 4)).astype(int))
        ground_info["sectors"] = int(len(sectors))  # de 8: rodea la abertura o solo un lado
    else:
        top_z, top_s, ground_info = float(top_wall), SLICE_M, None

    # superficies candidatas: ¿hay otra superficie plausible (banket vs canal/agua; tapa vs pavimento)?
    bottom_cands = surface_candidates(floor_info, chamber["cx"], chamber["cy"], rng)
    ref_cands = surface_candidates(ground_info, chamber["cx"], chamber["cy"], rng) if ground_found else []
    amb_b = max([abs(c["z"] - bottom_cands[0]["z"]) for c in bottom_cands[1:]], default=0.0)
    amb_t = max([abs(c["z"] - ref_cands[0]["z"]) for c in ref_cands[1:]], default=0.0)

    depth = top_z - floor_z
    # la referencia real puede ser cualquiera de las superficies plausibles: σ_modelo = separación / 2
    U = expanded(depth, floor_s, top_s, amb_b / 2, amb_t / 2)
    depth_conf = rel_conf(U / depth, (0.015, 0.03, 0.08), floor_info["n_used"] >= 50)
    extra = dict(sigma_floor=floor_s, sigma_top=top_s, ambiguity_bottom=amb_b, ambiguity_top=amb_t)
    if not ground_found:  # sin maaiveld: solo hasta el borde superior de la pared detectada
        depth_m = meas(depth, U, worst(depth_conf, LAAG), "wall_top_to_bodem", ESTIMATED, **extra)
    elif amb_b > 0 or amb_t > 0:  # referencia no inequívoca
        depth_m = meas(depth, U, worst(depth_conf, LAAG), "plane_to_plane(maaiveld,bodem)", ESTIMATED,
                       reason="varias superficies plausibles de " + " y ".join(
                           n for n, a in (("bodem", amb_b), ("maaiveld", amb_t)) if a > 0), **extra)
    else:
        depth_m = meas(depth, U, depth_conf, "plane_to_plane(maaiveld,bodem)", **extra)
    return dict(
        bottom_surface_candidates=bottom_cands, reference_surface_candidates=ref_cands,
        sections=sections, chamber=chamber, walls=walls,
        width_x=walls["width_x"], width_y=walls["width_y"], diameter=walls.get("diameter"),
        circularity=walls.get("circularity"),
        depth=depth_m,
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


def _is_flat(Q):
    """¿Explica una recta la sección del grupo tan bien como un círculo? (en el plano vertical que mejor
    contiene el grupo). Un arco que no se distingue de una recta no aporta información de tubo."""
    c = Q.mean(axis=0)
    _, _, vt = np.linalg.svd((Q - c)[:, :2], full_matrices=False)
    for direction in vt:  # plano solo si es recto en AMBAS direcciones horizontales (un tubo es recto a lo largo)
        s = (Q[:, :2] - c[:2]) @ direction
        sz = np.column_stack([s, Q[:, 2]])
        A = np.column_stack([s, np.ones(len(s))])
        coef, *_ = np.linalg.lstsq(A, sz[:, 1], rcond=None)
        line_rms = float(np.sqrt(np.mean((A @ coef - sz[:, 1]) ** 2)))
        cx, cy, r = fit_circle_2d(sz)
        cx, cy, r = fit_circle_geometric(sz, cx, cy, r)
        circ_rms = float(np.sqrt(np.mean((np.hypot(sz[:, 0] - cx, sz[:, 1] - cy) - r) ** 2)))
        if not (line_rms <= 1.5 * circ_rms + 0.002 or r > 1.0):
            return False
    return True


def _is_curved_invert(s, z):
    """¿Es la franja inferior el fondo curvo de un tubo? Cuadrático convexo significativamente mejor que lineal."""
    s = s - s.mean()
    A1 = np.column_stack([np.ones(len(s)), s])
    A2 = np.column_stack([np.ones(len(s)), s, s ** 2])
    c1, *_ = np.linalg.lstsq(A1, z, rcond=None)
    c2, *_ = np.linalg.lstsq(A2, z, rcond=None)
    rms1 = np.sqrt(np.mean((A1 @ c1 - z) ** 2))
    rms2 = np.sqrt(np.mean((A2 @ c2 - z) ** 2))
    # c2[2] = 1/(2r): para tubos de Ø110–Ø1000 entre 1 y 18 m⁻¹
    return c2[2] > 0.5 and rms1 > 1.5 * rms2 + 0.001


def measure_pipe_axis(Q, NQ, d0, rng):
    """Eje del tubo a partir de sus normales: en un cilindro todas las normales son perpendiculares
    al eje, así que el eje es la dirección de menor proyección de las normales (como el eje de la put).
    Solo es válido si se ve un tramo del tubo (no solo el borde del agujero) y el arco abre las normales."""
    m = np.abs(NQ @ d0) < 0.5           # superficie del tubo (excluye caras frontales / borde del hueco)
    out = dict(ok=False, length=0.0, ratio=np.nan, axis=d0)
    if m.sum() < 50:
        return out
    Qm, Nm = Q[m], NQ[m]
    axial = Qm @ d0
    out["length"] = float(np.ptp(np.percentile(axial, [2, 98])))
    w, v = np.linalg.eigh(Nm.T @ Nm / len(Nm))
    out["ratio"] = float(w[0] / max(w[1], 1e-12))
    a = v[:, 0] if v[:, 0] @ d0 > 0 else -v[:, 0]
    out["axis"] = a
    if out["length"] < DIR_MIN_LENGTH_M or out["ratio"] > DIR_AXIS_RATIO_MAX:
        return out
    groups = block_ids(Qm, 0.05)
    hz, sl = [], []
    for _ in range(N_BOOT // 2):
        idx = resample_blocks(groups, rng)
        _, vb = np.linalg.eigh(Nm[idx].T @ Nm[idx] / len(idx))
        b = vb[:, 0] if vb[:, 0] @ d0 > 0 else -vb[:, 0]
        hz.append(np.degrees(np.arctan2(b[1], b[0])))
        sl.append(np.degrees(np.arcsin(np.clip(b[2], -1, 1))))
    hz = np.unwrap(np.radians(hz))
    out.update(ok=True, sigma_azimuth_deg=float(np.degrees(np.std(hz))), sigma_slope_deg=float(np.std(sl)))
    return out


def _measure_arc(sz, rng, floor_sigma, top_z, top_sigma, extra_sigma_d=0.0, sz_extent=None):
    """sz_extent: puntos de la superficie del tubo para observar kruin/BOB directamente (por defecto = sz).
    El diámetro usa los inliers del círculo; la kruin/BOB no deben depender de que la corona caiga a < 5 mm
    del círculo (un tubo ovalado tiene la corona más baja que el círculo medio)."""
    ext = sz if sz_extent is None else sz_extent
    """Mide un tubo a partir de su sección (s, z). Devuelve medidas con estado explícito.
    Reglas (ver parámetros): arco < 60° o U/D > paso R10/2 -> diámetro UNKNOWN;
    arco ≥ 180° y U/D ≤ 5 % -> MEASURED; resto -> ESTIMATED."""
    sc, zc, r, inl, _ = fit_circle_robust(sz)
    arc = sz[inl]
    sc, zc, r = fit_circle_geometric(arc, sc, zc, r)
    rms = float(np.sqrt(np.mean((np.hypot(arc[:, 0] - sc, arc[:, 1] - zc) - r) ** 2)))
    ang = np.arctan2(arc[:, 1] - zc, arc[:, 0] - sc)
    bins = np.unique(((ang + np.pi) // (np.pi / 18)).astype(int))
    coverage = bins.size / 36
    arc_deg = coverage * 360

    # método independiente: cuerda + flecha del arco visible (válido si el arco no es casi completo)
    zb, zt = np.percentile(arc[:, 1], [3, 97])
    h = zt - zb
    base = arc[arc[:, 1] < zb + 0.015, 0]
    chord = float(np.ptp(np.percentile(base, [2, 98]))) if len(base) >= 5 else 0.0
    r_cs = chord ** 2 / (8 * h) + h / 2 if (h > 0.01 and chord > 0.02 and coverage < 0.6) else np.nan

    # bootstrap por bloques angulares de 10°
    ids = block_ids((ang + np.pi)[:, None], np.pi / 18)
    boot = []
    for _ in range(N_BOOT):
        b = arc[resample_blocks(ids, rng)]
        bsc, bzc, br = fit_circle_geometric(b, sc, zc, r, iters=15)
        boot.append((2 * br, bzc, bzc - br))
    boot = np.array(boot)
    s_d, s_zc, s_bob = np.std(boot, axis=0)

    D = 2 * r
    # σ_modelo: discrepancia entre métodos + sensibilidad del radio a un error sistemático δ
    # (medio voxel) en cuerda y flecha: en arcos planos el radio es muy sensible.
    geo_r = 0.0
    if np.isfinite(r_cs):
        delta = VOXEL_M / 2
        geo_r = float(np.hypot((0.5 - chord ** 2 / (8 * h ** 2)) * delta, chord / (4 * h) * delta))
    model_d = float(np.hypot(abs(D - 2 * r_cs) / 2 if np.isfinite(r_cs) else 0.0, 2 * geo_r))
    # término externo (medición por bandas): variación entre bandas + sensibilidad a la orientación del corte
    model_d = float(np.hypot(model_d, extra_sigma_d))
    U_d = expanded(D, s_d, model_d)
    sane = 0.03 < r < 0.8 and len(arc) >= 12
    rel = U_d / D if D > 0 else np.inf
    if not sane or arc_deg < ARC_MIN_DEG or rel > D_UNKNOWN_REL:
        why = ("arco < 60°: sin información de curvatura" if arc_deg < ARC_MIN_DEG else
               "ajuste no plausible" if not sane else
               f"incertidumbre ±{U_d * 1000:.0f} mm > medio paso de tamaños nominales")
        diameter = unknown("partial_arc_circle_fit", reason=why, attempt_value=float(D), attempt_U=float(U_d))
        status = UNKNOWN
    elif arc_deg >= ARC_MEASURED_DEG and rel <= D_MEASURED_REL:
        diameter = meas(D, U_d, HOOG if rel <= 0.03 else MIDDEL, "geometric_circle_fit",
                        sigma_stat=float(s_d), sigma_model=model_d)
        status = MEASURED
    else:
        diameter = meas(D, U_d, LAAG, "partial_arc_circle_fit", ESTIMATED, sigma_stat=float(s_d), sigma_model=model_d)
        status = ESTIMATED

    # kruin observada directamente, independiente del círculo: punto más alto en el centro de la abertura,
    # válido solo si el máximo está dentro del tramo visible (no en un extremo cortado)
    s_lo, s_hi = np.percentile(ext[:, 0], [2, 98])
    smid, swid = (s_lo + s_hi) / 2, s_hi - s_lo
    top_pts = ext[np.abs(ext[:, 0] - smid) < 0.1 * swid]   # ±10 % del ancho: z ≥ 98 % del radio en un círculo
    s_at_max = ext[np.argmax(ext[:, 1]), 0]
    crown_inside = s_lo + 0.2 * swid < s_at_max < s_hi - 0.2 * swid
    crown_v = float(np.percentile(top_pts[:, 1], 95)) if len(top_pts) >= 5 else np.nan
    # la corona observada debe estar en la MITAD SUPERIOR del círculo ajustado (≥ centro + 0,7 r, tolera
    # ovalidad); si solo se ve la mitad inferior, el punto más alto central es el fondo, no la kruin
    upper = np.isfinite(crown_v) and crown_v >= zc + 0.7 * r
    if len(top_pts) >= 5 and crown_inside and upper:
        cb = [np.percentile(top_pts[rng.integers(0, len(top_pts), len(top_pts)), 1], 95) for _ in range(N_BOOT)]
        U = float(np.hypot(expanded(crown_v, np.std(cb), floor_sigma, scale=False), K_EXPAND * SCALE_REL_SIGMA * abs(crown_v)))
        crown = meas(crown_v, U, HOOG if U < 0.01 else MIDDEL if U < 0.025 else LAAG, "highest_point_observed")
    else:
        crown = unknown("crown_not_visible", reason="la parte superior del tubo no se observa" if not upper else "")

    # BOB (binnen-onderkant): observada directamente solo si el arco incluye la parte inferior del tubo Y el
    # punto más bajo cae sobre el círculo ajustado (si no, es el banket/agua a nivel de bodem, no el tubo);
    # si no, crown - D (depende del diámetro); sin diámetro -> UNKNOWN
    bottom_seen = False
    if status != UNKNOWN and any(b in bins for b in (8, 9)):  # sin círculo válido no se sabe qué es "el fondo"
        low_pts = ext[(np.abs(ext[:, 0] - sc) < 0.1 * r) & (ext[:, 1] < zc)]
        if len(low_pts) >= 5:
            bob_obs = float(np.percentile(low_pts[:, 1], 5))
            # coherente con el círculo (tolerando ovalidad de hasta 5 % del radio)
            bottom_seen = abs(bob_obs - (zc - r)) <= max(3 * rms, VOXEL_M, 0.05 * r)
    if bottom_seen:
        bob_v = bob_obs
        U = float(np.hypot(expanded(bob_v, s_bob, floor_sigma, scale=False), K_EXPAND * SCALE_REL_SIGMA * abs(bob_v)))
        bob = meas(bob_v, U, HOOG if U < 0.01 else MIDDEL if U < 0.025 else LAAG, "lowest_point_observed")
    elif status != UNKNOWN and crown["value"] is not None:
        bob_v = crown["value"] - D
        U = float(np.hypot(crown["U"], diameter["U"]))
        bob = meas(bob_v, U, worst(diameter["conf"], LAAG), "crown_minus_diameter", ESTIMATED)
    else:
        bob = unknown("needs_diameter")

    if crown["value"] is not None and bob["value"] is not None:
        hv = (crown["value"] + bob["value"]) / 2
        hU = float(np.hypot(crown["U"], bob["U"]) / 2)
        st = MEASURED if crown["status"] == MEASURED and bob["status"] == MEASURED else ESTIMATED
        center_h = meas(hv, hU, worst(crown["conf"], bob["conf"]) if st == MEASURED else LAAG, "(crown+bob)/2", st)
    else:
        center_h = unknown("needs_crown_and_bob")

    bob_depth = unknown("needs_bob") if bob["value"] is None else meas(
        top_z - bob["value"], float(np.hypot(bob["U"], K_EXPAND * top_sigma)), bob["conf"], "maaiveld_minus_bob", bob["status"])

    # altura interior vertical observada (kruin - BOB, ambos vistos): medida independiente del círculo.
    # Si difiere del diámetro medio, el tubo puede estar ovalizado (no se mezcla con el diámetro).
    if crown["status"] == MEASURED and bottom_seen:
        hv = crown["value"] - bob["value"]
        inner_height = meas(hv, float(np.hypot(crown["U"], bob["U"])), worst(crown["conf"], bob["conf"]),
                            "crown_minus_bob_observed")
    else:
        inner_height = unknown("needs_crown_and_bottom_observed")

    # sugerencia de tamaño nominal (NUNCA una medida): solo dentro del intervalo de un ajuste plausible;
    # la cuerda observada es una cota inferior (D ≥ cuerda)
    lo = max(chord, float(np.ptp(np.percentile(sz[:, 0], [2, 98]))))
    nominal = nominal_candidates(max(lo, D - U_d) * 1000, (D + U_d) * 1000) if sane and arc_deg >= ARC_MIN_DEG else []
    if len(nominal) > 3:  # el intervalo abarca demasiados tamaños: la sugerencia no aporta nada
        nominal = []

    return dict(status=status, fit_r=r, fit_rms=rms, coverage=coverage, arc_deg=arc_deg, inliers=len(arc),
                arc_points=arc, arc_bins=bins.tolist(), chord=chord, sagitta=float(h),
                r_chord=None if not np.isfinite(r_cs) else float(r_cs), center_s=sc, center_z=zc,
                diameter=diameter, crown=crown, bob=bob, axis_h=center_h, bob_depth=bob_depth,
                inner_height=inner_height, bottom_seen=bool(bottom_seen), nominal=nominal)


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
    # por debajo de la cámara (z < z0 de la cámara) cuenta la cámara
    below = z < ch["z0"] - SLICE_M
    covered |= below
    outside &= ~below | (_outside_distance(L[:, :2], ch, rect) > BEHIND_WALL_M)
    idx = np.flatnonzero(zmask & covered & outside)
    # diagnóstico: región de búsqueda (todos los puntos "detrás de la pared") y grupos descartados
    put["connection_search"] = dict(indices=idx, z_range=(float(put["floor_z"] - 0.3), float(ch["z1"] + SLICE_M)),
                                    behind_wall_m=BEHIND_WALL_M, eps_m=PIPE_CLUSTER_EPS_M)
    put["rejected_candidates"] = []
    if len(idx) < PIPE_MIN_POINTS:
        return []
    labels = np.array(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(L[idx])).cluster_dbscan(
        PIPE_CLUSTER_EPS_M, 10))
    put["connection_search"]["labels"] = labels

    conns = []
    for lab in range(labels.max() + 1 if len(labels) else 0):
        ids = idx[labels == lab]
        if len(ids) < PIPE_MIN_POINTS:
            put["rejected_candidates"].append(dict(indices=ids, reason=f"{len(ids)} < {PIPE_MIN_POINTS} puntos"))
            continue
        Q, NQ = L[ids], N_L[ids]
        if np.mean(np.abs(NQ[:, 2]) > FLOOR_NZ_MIN) > 0.7 and _is_flat(Q):
            put["rejected_candidates"].append(dict(indices=ids, reason="superficie horizontal plana"))
            continue  # superficie horizontal plana (rellano/terreno): una recta la explica igual que un círculo
        sec =_section_at(put["sections"], float(np.median(Q[:, 2]))) or ch
        if sec is not ch and _section_at([ch], float(np.median(Q[:, 2]))):
            sec = ch
        cxy = np.array([sec["cx"], sec["cy"]])
        m = Q[:, :2].mean(axis=0) - cxy
        # dirección supuesta: normal de la pared atravesada
        if rect:
            k = int(np.argmax(np.abs(m) - np.array([sec["hx"], sec["hy"]])))
            d2 = np.zeros(2); d2[k] = np.sign(m[k])
            radial_wall = [sec["hx"], sec["hy"]][k]
            wall_id = ("+" if d2[k] > 0 else "-") + "XY"[k]
        else:
            d2 = m / np.linalg.norm(m)
            radial_wall = sec["hx"]
            wall_id = "wand"
        d = np.array([d2[0], d2[1], 0.0])

        # dirección medida (si el tubo es visible en longitud suficiente)
        ax = measure_pipe_axis(Q, NQ, d, rng)
        dev = float(np.degrees(np.arctan2(ax["axis"][1], ax["axis"][0]) - np.arctan2(d[1], d[0])))
        dev = (dev + 180) % 360 - 180
        if ax["ok"]:
            U_az = K_EXPAND * ax["sigma_azimuth_deg"]
            direction = meas(dev, U_az, HOOG if U_az <= DIR_MEASURED_U_DEG else LAAG, "pipe_normals_axis",
                             MEASURED if U_az <= DIR_MEASURED_U_DEG else ESTIMATED, vector=ax["axis"].tolist())
            U_sl = K_EXPAND * ax["sigma_slope_deg"]
            slope_v = float(np.degrees(np.arcsin(np.clip(ax["axis"][2], -1, 1))))
            if U_sl <= SLOPE_MEASURED_U_DEG:
                slope = meas(slope_v, U_sl, HOOG, "pipe_normals_axis")
            elif U_sl <= SLOPE_ESTIMATED_U_DEG:
                slope = meas(slope_v, U_sl, LAAG, "pipe_normals_axis", ESTIMATED)
            else:
                slope = unknown("pipe_normals_axis", reason=f"±{U_sl:.1f}° > {SLOPE_ESTIMATED_U_DEG}°")
            if direction["status"] == MEASURED:  # sección transversal perpendicular al eje medido (horizontal)
                h = np.array([ax["axis"][0], ax["axis"][1], 0.0]); d = h / np.linalg.norm(h)
        else:
            why = (f"tubo visible {ax['length'] * 100:.0f} cm < {DIR_MIN_LENGTH_M * 100:.0f} cm"
                   if ax["length"] < DIR_MIN_LENGTH_M else "normales sin eje definido")
            direction = meas(0.0, None, LAAG, "assumed_wall_normal", ESTIMATED, vector=d.tolist(), reason=why)
            slope = unknown("not_observable", reason=why)
        s_axis = np.array([-d[1], d[0], 0.0])  # horizontal, perpendicular a la tubería

        # sección transversal (s, z); quitar la superficie horizontal inferior (agua/sedimento/banket) salvo si es
        # el fondo CURVO de un tubo seco: el fondo de un tubo sigue z ≈ s²/2r (cuadrático, convexo hacia abajo),
        # mientras que agua o banket son planos (lineales en s, aunque estén inclinados o sean rugosos).
        s = Q @ s_axis
        bottom = (np.abs(NQ[:, 2]) > FLOOR_NZ_MIN) & (Q[:, 2] < Q[:, 2].min() + 0.02)
        if bottom.sum() >= 10 and _is_curved_invert(s[bottom], Q[bottom, 2]):
            bottom[:] = False
        sz = np.column_stack([s, Q[:, 2]])[~bottom]
        s_c0 = cxy @ s_axis[:2]
        wall_center = np.array([cxy[0], cxy[1], 0.0]) + d * radial_wall + s_axis * (float(np.median(s)) - s_c0)
        wall_center[2] = float(np.median(Q[:, 2]))

        conn = dict(points=Q, normals=NQ, indices=ids, bottom_mask=bottom, sz_all=sz, section_used=sec,
                    direction=d, s_axis=s_axis, n=len(Q), wall_center=wall_center, wall=wall_id,
                    direction_m=direction, slope=slope, axis_info=ax,
                    hidden_bottom=dict(n=int(bottom.sum()), z=float(np.median(Q[bottom, 2])) if bottom.any() else None),
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
                conn["note"] = arc["diameter"].get("reason", "")
        else:
            conn["note"] = "te weinig punten in doorsnede"
            for key in ("diameter", "crown", "bob", "axis_h", "bob_depth", "inner_height"):
                conn[key] = unknown("too_few_points")
            conn.update(arc_deg=0.0, coverage=0.0, nominal=[], bottom_seen=False)
        ref = conn.get("center", wall_center)
        conn["angle_deg"] = float(np.degrees(np.arctan2(ref[1] - ch["cy"], ref[0] - ch["cx"])) % 360)
        conns.append(conn)
    conns.sort(key=lambda c: c["angle_deg"])
    for i, c in enumerate(conns):
        c["id"] = f"A{i + 1}"
    return conns


# ---------------------------------------------------------------- entrada principal

def analyze(pcd, seed=0):
    """Analiza una nube de puntos Open3D. Devuelve dict con frame, put, conexiones, calidad y la nube local."""
    from quality import assess_quality  # import local: quality depende de las constantes de este módulo
    rng = np.random.default_rng(seed)
    n_raw = len(pcd.points)
    spacing = point_spacing(pcd, rng)
    P, N, n_clean = preprocess(pcd)
    center, R, symmetry = estimate_frame(P, N)
    rect = symmetry >= (RECT_SYMMETRY_MIN + ROUND_SYMMETRY_MAX) / 2
    L, N_L = (P - center) @ R.T, N @ R.T

    sections = find_sections(L, N_L, rect)
    chamber0 = max(sections, key=lambda s: s["z1"] - s["z0"])
    shape = classify_shape(L, N_L, chamber0, symmetry)
    if shape["status"] == MEASURED:
        rect = shape["label"] == "rechthoekig"
    put = detect_put(L, N_L, rect, shape["status"] == MEASURED, rng)
    put["shape"] = shape

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
            w["c"] = w["c"] - shift; w["points"] = w["points"] - shift; w["candidates"] = w["candidates"] - shift
            sel = w["select"]
            sel["z_range"] = (sel["z_range"][0] - shift[2], sel["z_range"][1] - shift[2])
        for dm in put["walls"].get("debug_missing", {}).values():
            dm["candidates"] = dm["candidates"] - shift
    for info in (put["floor_info"], put["ground_info"]):
        if info:
            info["point"] = info["point"] - shift; info["inliers"] = info["inliers"] - shift
            info["candidates"] = info["candidates"] - shift
            info["eval_xy"] = (info["eval_xy"][0] - shift[0], info["eval_xy"][1] - shift[1])
    for cands in (put["bottom_surface_candidates"], put["reference_surface_candidates"]):
        for cnd in cands:
            cnd["z"] -= shift[2]
            if "inliers" in cnd:
                cnd["inliers"] = cnd["inliers"] - shift
    put["top_z"] -= shift[2]; put["top_wall_z"] -= shift[2]; put["floor_z"] = 0.0
    # método anterior (todos los puntos detrás de la pared = tubo): se conserva para comparar y porque su
    # agrupación alimenta la lista de candidatos rechazados del método nuevo
    legacy = detect_connections(L, N_L, put, rect, rng)
    put["rejected_candidates_legacy"] = put.get("rejected_candidates", [])
    if rect and put["walls"]["walls"]:
        from connections import detect_connections_v2
        conns, openings, rejected = detect_connections_v2(L, N_L, put, rect, rng)
    else:  # put redonda: el método por aberturas solo está implementado para paredes planas
        conns, openings, rejected = legacy, [], []
    put["openings"] = openings
    put["rejected_candidates"] = rejected

    # orientación del eje vertical respecto al Z del PLY
    tilt = float(np.degrees(np.arccos(min(abs(R[2] @ np.array([0, 0, 1.0])), 1.0))))
    s_axis = axis_uncertainty(L, N_L, rng)
    put["orientation"] = meas(tilt, K_EXPAND * s_axis, HOOG if K_EXPAND * s_axis < 1 else MIDDEL,
                              "least_normal_direction(wall_normals)",
                              axis_ply=R[2].tolist(), x_axis_ply=R[0].tolist(),
                              azimuth_x_deg=float(np.degrees(np.arctan2(R[0][1], R[0][0]))))
    origin = center + shift @ R
    s_c = put["walls"].get("center_sigma")
    if s_c is None:
        s_c = float(np.hypot(put["width_x"].get("sigma_stat", 0.01), put["width_y"].get("sigma_stat", 0.01)) / 2)
    put["center"] = meas(0.0, K_EXPAND * s_c, HOOG, "midpoint_of_wall_planes" if rect else "circle_center",
                         xyz_ply=origin.tolist())

    res = dict(put=put, connections=conns, connections_legacy=legacy, local_points=L, local_normals=N_L, rect=rect,
               symmetry=symmetry,
               R=R, origin=origin, axis_tilt_vs_ply_z=tilt, n_points=len(P), n_raw=n_raw, n_clean=n_clean,
               spacing=spacing)
    res["quality"] = assess_quality(res)
    return res


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
    """< 1 m -> mm; >= 1 m -> m (con mm en detalle). UNKNOWN -> texto."""
    v, U = m["value"], m["U"]
    if v is None:
        return "UNKNOWN"
    u = "" if U is None else (f" ± {U * 1000:.0f}" if abs(v) < 1.0 or detail else f" ± {U:.3f}")
    if abs(v) < 1.0 or detail:
        return f"{v * 1000:.0f}{u} mm"
    return f"{v:.3f}{u} m"


def _row(name, m, unit="len"):
    if unit == "deg":
        val = "UNKNOWN" if m["value"] is None else f"{m['value']:.1f}°" + ("" if m["U"] is None else f" ± {m['U']:.1f}°")
    else:
        val = fmt_len(m)
    extra = f"  ({m['reason']})" if m.get("reason") else ""
    return f"  {name:<22}{val:<22}{m['status']:<10} {m['conf']:<8} {m['method']}{extra}"


def report(res):
    """Informe de texto (consola)."""
    put, q, lines = res["put"], res["quality"], []
    sh = put["shape"]
    lines.append("=== PUT ===")
    lines.append(f"  {'Vorm':<22}{sh['label']:<22}{sh['status']:<10} {sh['conf']:<8} "
                 f"(simetría 90° {sh['symmetry']:.2f}, residuo círculo {sh['circle_rel_rms'] * 100:.1f} %)")
    lines.append(_row("Oriëntatie (vs Z PLY)", put["orientation"], "deg"))
    if put["diameter"]:
        lines.append(_row("Binnen diameter", put["diameter"]))
        if put["circularity"]:
            lines.append(_row("Circulariteit (rms)", put["circularity"]))
    else:
        lines.append(_row("Binnenmaat X", put["width_x"]))
        lines.append(_row("Binnenmaat Y", put["width_y"]))
        for nm, k in (("X", "width_x"), ("Y", "width_y")):
            pr = put[k].get("profile")
            if pr:
                lines.append(f"    perfil {nm} (onder/midden/boven): {pr['bottom'] * 1000:.0f} / {pr['mid'] * 1000:.0f} / "
                             f"{pr['top'] * 1000:.0f} mm  variatie {pr['variation'] * 1000:+.0f} mm "
                             f"({'significant' if pr['significant'] else 'binnen ruis'}; σ_taper {put[k]['sigma_taper'] * 1000:.1f} mm)")
    lines.append(_row("Diepte", put["depth"]))
    for nm, cands in (("bodem", put.get("bottom_surface_candidates", [])), ("maaiveld", put.get("reference_surface_candidates", []))):
        if len(cands) > 1:
            lines.append(f"    {nm}-kandidaten: " + "; ".join(f"{c['kind']} z={c['z'] * 1000:.0f} mm ({c['points']} ptn)" for c in cands))
    lines.append(f"  {'Buitenmaat':<22}{'UNKNOWN':<22}{'UNKNOWN':<10} {ONZEKER:<8} not_visible_from_inside")
    w = put["walls"]
    if w["walls"] is not None:
        for k, _, _ in WALLS:
            if k in w["walls"]:
                wl = w["walls"][k]
                dup = q["duplicate_fraction"].get(k, 0.0)
                lines.append(f"    wand {k}: {wl['n_used']:5d} ptn  rms {wl['rms'] * 1000:.1f} mm  dekking {wl['coverage'] * 100:.0f}%"
                             f"  dubbel {dup * 100:.0f}%  drift {wl['drift_m'] * 1000:.0f} mm")
            else:
                lines.append(f"    wand {k}: NIET GESCAND")
        g = w["geometry"]
        lines.append("    " + "  ".join(f"{k} {v:.2f}°" for k, v in g.items()))
    fi, gi = put["floor_info"], put["ground_info"]
    lines.append(f"    bodem: {fi['n_used']} ptn, rms {fi['rms'] * 1000:.1f} mm, σ {put['floor_sigma'] * 1000:.1f} mm, "
                 f"helling {fi['slope_deg']:.1f}°, dekking {fi['coverage'] * 100:.0f}%")
    if gi:
        lines.append(f"    maaiveld: {gi['n_used']} ptn, rms {gi['rms'] * 1000:.1f} mm, σ {put['top_sigma'] * 1000:.1f} mm, "
                     f"helling {gi['slope_deg']:.1f}°, rondom {gi['sectors']}/8 sectoren")
    for i, s in enumerate(put["sections"]):
        lines.append(f"    tramo {i + 1}: z {s['z0']:+.2f}..{s['z1']:+.2f} m  ~{2 * s['hx'] * 1000:.0f} x {2 * s['hy'] * 1000:.0f} mm (indicatief)")
    lines.append(f"\n=== AANSLUITINGEN ({len(res['connections'])}) ===")
    for c in res["connections"]:
        lines.append(f"{c['id']}  wand {c['wall']}  hoek {c['angle_deg']:.0f}°  zichtbare boog {c.get('arc_deg', 0):.0f}°  "
                     f"opening {c['opening_w'] * 1000:.0f} x {c['opening_h'] * 1000:.0f} mm  zichtbare lengte {c['visible_len'] * 100:.0f} cm")
        if "opening" in c:
            op = c["opening"]
            od = c.get("opening_diameter") or {}
            lines.append(f"    opening: {op['status']} ({op['reason']}); {op['width'] * 1000:.0f} x {op['height'] * 1000:.0f} mm")
            lines.append("    banden: " + " ".join(f"{b['t0'] * 100:.0f}-{b['t1'] * 100:.0f}cm:{b['label'][:4]}"
                                                   + (f"/Ø{2 * b['r'] * 1000:.0f}" if "r" in b else "") for b in c["bands"]))
            lines.append(f"    stabiel buistraject: {len(c['stable_run'])} banden, {c['pipe_length_observed'] * 100:.0f} cm; "
                         f"onderkant verborgen: {c.get('lower_pipe_occluded')}; reconstructie-vulling: "
                         f"{c.get('possible_reconstruction_fill')}")
        lines.append(_row("Diameter", c["diameter"]))
        if c.get("opening_diameter"):
            lines.append(_row("Ø wandopening", c["opening_diameter"]))
        if c.get("opening_top"):
            lines.append(_row("Bovenkant opening", c["opening_top"]))
        if c["diameter"]["value"] is None and "attempt_value" in c["diameter"]:
            lines.append(f"    (intento de ajuste descartado: {c['diameter']['attempt_value'] * 1000:.0f} ± {c['diameter']['attempt_U'] * 1000:.0f} mm)")
        lines.append(_row("Kruin", c["crown"]))
        lines.append(_row("BOB", c["bob"]))
        lines.append(_row("Hart (as)", c["axis_h"]))
        lines.append(_row("BOB onder maaiveld", c["bob_depth"]))
        lines.append(_row("Richting (afw. normaal)", c["direction_m"], "deg"))
        lines.append(_row("Helling", c["slope"], "deg"))
        if c.get("nominal"):
            lines.append(f"    Suggestie nominaal (NIET gemeten): " + " / ".join(f"Ø{n}" for n in c["nominal"]))
    rej_ops = [o for o in put.get("openings", []) if o["status"] != "CONFIRMED"]
    rej_cl = put.get("rejected_candidates", [])
    if rej_ops or rej_cl:
        lines.append(f"\n=== KANDIDATEN ZONDER AANSLUITING ({len(rej_ops) + len(rej_cl)}) ===")
        for o in rej_ops:
            lines.append(f"  opening wand {o['wall']} {o['kind']} {o['status']}: u={o['centroid_uz'][0]:+.2f} "
                         f"z={o['centroid_uz'][1]:.2f} m {o['width'] * 1000:.0f}x{o['height'] * 1000:.0f} mm — {o['reason']}")
        for r_ in rej_cl:
            lines.append(f"  cluster achter wand {r_.get('wall')} {r_['kind']}: {len(r_['indices'])} ptn — {r_['reason']}")
    lines.append("\n=== SCAN QUALITY ===")
    lines.append(f"  Punten: {q['total_points']:,} totaal, {q['usable_points']:,} bruikbaar, "
                 f"puntafstand {q['point_spacing_mm']:.1f} mm, ruis {q['noise_mm']:.1f} mm")
    for name, st in q["items"].items():
        lines.append(f"  {name:<14}{st}")
    for wmsg in q["warnings"]:
        lines.append(f"  ⚠ {wmsg}")
    lines.append(f"(± = 2σ; incluye {SCALE_REL_SIGMA * 100:.1f}% σ de escala supuesta)")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys
    import time
    from pathlib import Path
    sys.stdout.reconfigure(encoding="utf-8")  # σ, Ø, ⚠ también al redirigir a archivo en Windows
    ap = argparse.ArgumentParser(description="PUT SCANNER - análisis")
    ap.add_argument("scan", nargs="?", default=str(Path(__file__).parent / "data" / "scan.ply"))
    ap.add_argument("--json", help="guardar ScanResult en este archivo JSON")
    args = ap.parse_args()
    t0 = time.time()
    res = load_and_analyze(args.scan)
    dt = time.time() - t0
    print(report(res))
    print(f"\nTiempo de análisis: {dt:.2f} s")
    if args.json:
        from scan_result import build_scan_result
        build_scan_result(res, args.scan, dt).save(args.json)
        print(f"ScanResult guardado en {args.json}")
