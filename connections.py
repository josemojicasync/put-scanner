"""Aansluitingen: detección por ABERTURA en la cara de la pared y medición del tubo por BANDAS.

Principio: detección y medición son etapas separadas.
  1. En la cara de cada pared se buscan interrupciones (regiones vacías rodeadas de pared) en una rejilla 2D.
     Clases de geometría de pared: WALL FACE (incluye collares que sobresalen hasta FACE_FRONT_M),
     WALL RELIEF (superficie paralela poco profunda detrás de la cara: ranuras, nichos) y vacío.
  2. Solo una abertura CONFIRMED crea una aansluiting. Detrás de ella se sigue el tubo por bandas de
     profundidad; cada banda se ajusta por separado (sección perpendicular al eje actual) y el eje se
     re-estima con la línea de centros de las bandas (iterativo: corrige tubos oblicuos).
  3. Clases de puntos dentro de una abertura: OPENING RING (bandas cerca de la cara con otro radio),
     PIPE SURFACE (tramo estable de radio constante), WATER/SEDIMENT/BANKET (plano horizontal que corta
     la parte inferior), FILL (superficies transversales al eje: tapas/relleno de reconstrucción, escalones).
  4. El diámetro del tubo usa SOLO las bandas PIPE del tramo estable. El tamaño de la abertura se guarda
     aparte y nunca se mezcla con el diámetro del tubo.
Solo puts rectangulares (paredes planas). Para puts redondas se usa el método anterior (legacy).
"""
from collections import deque

import numpy as np
import open3d as o3d

import analysis as A

CELL_M = 0.02                 # rejilla de la cara: 2 cm (≥ 3x el espaciado típico de Polycam)
FACE_FRONT_M = 0.035          # material hasta 3,5 cm por delante de la cara cuenta como pared (collares)
RECESS_MAX_M = 0.08           # relieve: superficie paralela a ≤ 8 cm detrás de la cara (ranuras, nichos)
PARALLEL_N = 0.9              # |n·n_pared| > 0.9 (≤ 26°): superficie paralela a la pared
TRANSVERSE_N = 0.8            # |n·eje| > 0.8: superficie transversal al tubo (tapa, escalón, cara): no es pared de tubo
MIN_OPENING_D_M = 0.08        # abertura mínima relevante ≈ Ø80 mm
CONTOUR_CONFIRMED = 0.6       # ≥ 60 % del contorno (sin contar el borde inferior en el bodem) apoyado en pared
CONTOUR_POSSIBLE = 0.4
MIN_BEHIND_POINTS = 20        # evidencia de que se ve "a través" de la abertura
BAND_M = 0.05
BAND_MIN_POINTS = 30
MAX_DEPTH_M = 1.0
SEARCH_FACTOR = 1.5           # radio de búsqueda alrededor del eje = 1,5 x radio de la abertura + 5 cm
BAND_MAX_RMS = 0.006          # 2x el ruido típico de pared (3 mm)
BAND_MIN_ARC = 90             # grados: con menos, el radio de UNA banda no está determinado
STABLE_TOL_REL = 0.02         # radios compatibles dentro de max(5 mm, 2 %): 5 mm ≈ 2x el ruido de pared; más = otra superficie
STABLE_TOL_ABS = 0.005
AXIS_MAX_DEV_DEG = 60
AXIS_MIN_ARC = 180            # centro de banda fiable para el eje solo si se ven lados opuestos del tubo
FILL_FRACTION = 0.4           # banda dominada (≥ 40 %) por superficie transversal en el interior -> relleno/tapa


# ---------------------------------------------------------------- utilidades de rejilla (sin scipy)

def _shift(m, dx, dy):
    out = np.zeros_like(m)
    xs = slice(max(dx, 0), m.shape[0] + min(dx, 0)); xd = slice(max(-dx, 0), m.shape[0] + min(-dx, 0))
    ys = slice(max(dy, 0), m.shape[1] + min(dy, 0)); yd = slice(max(-dy, 0), m.shape[1] + min(-dy, 0))
    out[xs, ys] = m[xd, yd]
    return out


def dilate(m):
    out = m.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            out |= _shift(m, dx, dy)
    return out


def close(m):
    """Cierre morfológico 3x3: rellena huecos de muestreo de 1 celda en la ocupación de la pared."""
    d = dilate(m)
    return ~dilate(~d) | m


def label(mask):
    """Componentes 8-conexas -> lista de arrays (N,2) de índices de celda."""
    seen = np.zeros_like(mask, bool)
    comps = []
    for i, j in zip(*np.nonzero(mask)):
        if seen[i, j]:
            continue
        q, cells = deque([(i, j)]), []
        seen[i, j] = True
        while q:
            a, b = q.popleft()
            cells.append((a, b))
            for da in (-1, 0, 1):
                for db in (-1, 0, 1):
                    x, y = a + da, b + db
                    if 0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and mask[x, y] and not seen[x, y]:
                        seen[x, y] = True
                        q.append((x, y))
        comps.append(np.array(cells))
    return comps


def fast_circle_robust(xy, seed=0, iters=500, thresh=0.005, min_points=8):
    """Mismo algoritmo que measurement.fit_circle_robust (RANSAC de 3 puntos, umbral 5 mm, reajuste de mínimos
    cuadrados sobre los inliers) pero vectorizado: círculo circunscrito en forma cerrada y conteo de inliers
    para todas las muestras a la vez. Solo cambia la velocidad."""
    n = len(xy)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, (iters, 3))
    ok = (idx[:, 0] != idx[:, 1]) & (idx[:, 0] != idx[:, 2]) & (idx[:, 1] != idx[:, 2])
    p1, p2, p3 = xy[idx[ok, 0]], xy[idx[ok, 1]], xy[idx[ok, 2]]
    d = 2 * (p1[:, 0] * (p2[:, 1] - p3[:, 1]) + p2[:, 0] * (p3[:, 1] - p1[:, 1]) + p3[:, 0] * (p1[:, 1] - p2[:, 1]))
    good = np.abs(d) > 1e-12
    p1, p2, p3, d = p1[good], p2[good], p3[good], d[good]
    s1, s2, s3 = (p1 ** 2).sum(1), (p2 ** 2).sum(1), (p3 ** 2).sum(1)
    ux = (s1 * (p2[:, 1] - p3[:, 1]) + s2 * (p3[:, 1] - p1[:, 1]) + s3 * (p1[:, 1] - p2[:, 1])) / d
    uy = (s1 * (p3[:, 0] - p2[:, 0]) + s2 * (p1[:, 0] - p3[:, 0]) + s3 * (p2[:, 0] - p1[:, 0])) / d
    r = np.hypot(p1[:, 0] - ux, p1[:, 1] - uy)
    best = None
    if len(r):
        counts = np.zeros(len(r), int)
        for k in range(0, len(r), 100):   # por bloques para limitar memoria
            dd = np.abs(np.hypot(xy[None, :, 0] - ux[k:k + 100, None], xy[None, :, 1] - uy[k:k + 100, None]) - r[k:k + 100, None])
            counts[k:k + 100] = (dd < thresh).sum(1)
        j = int(np.argmax(counts))
        best = np.abs(np.hypot(xy[:, 0] - ux[j], xy[:, 1] - uy[j]) - r[j]) < thresh
    if best is None or best.sum() < min_points:
        best = np.ones(n, bool)
    cx, cy, rr = A.fit_circle_2d(xy[best])
    res = np.hypot(xy[best, 0] - cx, xy[best, 1] - cy) - rr
    return cx, cy, rr, best, float(np.sqrt(np.mean(res ** 2)))


def _circle(xy):
    """Círculo geométrico sobre todos los puntos (sin RANSAC) + rms + cobertura angular (grados)."""
    if len(xy) < 6:
        return None
    cx, cy, r = A.fit_circle_2d(xy)
    cx, cy, r = A.fit_circle_geometric(xy, cx, cy, r)
    res = np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r
    ang = np.arctan2(xy[:, 1] - cy, xy[:, 0] - cx)
    cov = np.unique(((ang + np.pi) // (np.pi / 18)).astype(int)).size * 10
    return dict(c=np.array([cx, cy]), r=float(r), rms=float(np.sqrt(np.mean(res ** 2))), arc=float(cov))


# ---------------------------------------------------------------- cara de pared

class WallFrame:
    """Sistema de la cara de pared: u horizontal a lo largo de la pared, z vertical, d = distancia firmada
    (positiva hacia FUERA de la cámara, es decir detrás de la cara)."""

    def __init__(self, key, w, ch, floor_z):
        self.key = key
        self.n = w["n"] / np.linalg.norm(w["n"])
        self.o = w["c"].copy()
        self.u = np.cross([0, 0, 1.0], self.n); self.u /= np.linalg.norm(self.u)
        self.tol = max(0.01, 2.5 * w["rms"])
        Pin = w["points"]
        uu = (Pin - self.o) @ self.u
        self.u_range = (float(np.percentile(uu, 0.5)), float(np.percentile(uu, 99.5)))
        self.z_range = (floor_z - 0.05, ch["z1"])

    def coords(self, P):
        rel = P - self.o
        return rel @ self.u, P[:, 2], rel @ self.n

    def point(self, u, z):
        p = self.o + self.u * u + np.array([0, 0, z - self.o[2]])
        return p - self.n * ((p - self.o) @ self.n)


def face_grid(L, N, fr):
    u, z, d = fr.coords(L)
    inside = (u >= fr.u_range[0]) & (u < fr.u_range[1]) & (z >= fr.z_range[0]) & (z < fr.z_range[1])
    nu = max(1, int(np.ceil((fr.u_range[1] - fr.u_range[0]) / CELL_M)))
    nz = max(1, int(np.ceil((fr.z_range[1] - fr.z_range[0]) / CELL_M)))
    iu = np.clip(((u - fr.u_range[0]) / CELL_M).astype(int), 0, nu - 1)
    iz = np.clip(((z - fr.z_range[0]) / CELL_M).astype(int), 0, nz - 1)
    par = np.abs(N @ fr.n) > PARALLEL_N
    # cara de pared = puntos con orientación de pared (agua, banket o canal horizontales delante del hueco
    # no son pared y no deben "cerrar" una abertura)
    face = inside & (d >= -FACE_FRONT_M) & (d <= fr.tol) & (np.abs(N @ fr.n) > 0.5)
    recess = inside & (d > fr.tol) & (d <= RECESS_MAX_M) & par
    behind = inside & (d > fr.tol) & (d < MAX_DEPTH_M)
    deep = behind & ~recess
    g = dict(nu=nu, nz=nz, iu=iu, iz=iz, u=u, z=z, d=d, face_mask=face, behind_mask=behind, recess_mask=recess)
    for name, m in (("face", face), ("recess", recess), ("deep", deep)):
        cnt = np.zeros((nu, nz), int)
        np.add.at(cnt, (iu[m], iz[m]), 1)
        g[name] = cnt
    occ = close(g["face"] > 0)
    g["occ"] = occ
    # Pie de la pared: en la base, donde la pared se encuentra con el banket/canal, no hay superficie con
    # orientación de pared en todo el ancho. La línea base es el percentil 25 del pie de cada columna (robusto
    # frente a las muescas); una abertura a nivel del bodem aparece como una muesca HACIA ARRIBA del pie.
    foot = np.array([np.flatnonzero(occ[i]).min() if occ[i].any() else np.nan for i in range(nu)], float)
    base = int(np.nanpercentile(foot, 25)) if np.isfinite(foot).any() else 0
    below = np.zeros((nu, nz), bool)
    below[:, :base + 1] = True
    g["foot_cells"], g["base_cell"], g["below"] = foot, base, below
    g["base_z"] = fr.z_range[0] + (base + 1) * CELL_M
    relief = ~occ & ~below & (g["recess"] >= 2) & (g["recess"] >= g["deep"])
    g["relief_cells"] = relief
    g["open_cells"] = ~occ & ~relief & ~below
    return g


def _region_features(comp, g, fr, L, N, floor_z):
    m = np.zeros((g["nu"], g["nz"]), bool)
    m[comp[:, 0], comp[:, 1]] = True
    area = len(comp) * CELL_M ** 2
    width = (np.ptp(comp[:, 0]) + 1) * CELL_M
    height = (np.ptp(comp[:, 1]) + 1) * CELL_M
    border = dilate(m) & ~m
    # el borde inferior que cae en la base de la pared (banket/canal/agua) no cuenta: allí no hay pared
    valid = ~g["below"][border]
    support = g["occ"][border][valid]
    touches_edge = bool((comp[:, 0] == 0).any() or (comp[:, 0] == g["nu"] - 1).any())
    contour = float(support.mean()) if len(support) else 0.0
    if touches_edge:
        contour *= 0.5  # en la esquina de la pared el contorno no está cerrado
    touches_floor = bool(comp[:, 1].min() <= g["base_cell"] + 2)
    # puntos de la cara en el borde (edge support) y ajuste de círculo/elipse del contorno
    in_border = border[g["iu"], g["iz"]] & g["face_mask"]
    edge_uv = np.column_stack([g["u"][in_border], g["z"][in_border]])
    circ = _circle(edge_uv) if len(edge_uv) >= 6 else None
    if circ is not None:
        # borde real = el punto de pared MÁS INTERIOR en cada sector de 5° (las celdas de borde se extienden
        # hasta 2 cm fuera del hueco y sesgarían el radio hacia fuera)
        rel = edge_uv - circ["c"]
        ang = ((np.arctan2(rel[:, 1], rel[:, 0]) + np.pi) // (np.pi / 36)).astype(int)
        rad = np.hypot(rel[:, 0], rel[:, 1])
        inner = [np.flatnonzero(ang == k)[np.argmin(rad[ang == k])] for k in np.unique(ang)]
        if len(inner) >= 6:
            edge_uv = edge_uv[inner]
            circ = _circle(edge_uv)
    # evidencia detrás de la abertura
    mb = dilate(m)
    sel_b = g["behind_mask"] & mb[g["iu"], g["iz"]]
    sel_r = g["recess_mask"] & mb[g["iu"], g["iz"]]
    behind_cells = float(np.mean((g["deep"][m] + g["recess"][m]) > 0))
    cu = fr.u_range[0] + (comp[:, 0].mean() + 0.5) * CELL_M
    cz = fr.z_range[0] + (comp[:, 1].mean() + 0.5) * CELL_M
    return dict(mask=m, area=area, width=width, height=height, contour=contour, touches_floor=touches_floor,
                touches_edge=touches_edge, edge_points=int(in_border.sum()), edge_uv=edge_uv, circle=circ,
                behind_idx=np.flatnonzero(sel_b), recess_idx=np.flatnonzero(sel_r), behind_cells=behind_cells,
                centroid_uz=(cu, cz))


def _is_relief(P, N, n_wall, d):
    """Superficie detrás de la cara predominantemente plana, paralela a la pared y poco profunda."""
    if len(P) < 15:
        return False, {}
    par = np.mean(np.abs(N @ n_wall) > PARALLEL_N)
    spread = float(np.percentile(d, 90) - np.percentile(d, 10))
    depth = float(np.median(d))
    return bool(par > 0.7 and spread < 0.03 and depth < RECESS_MAX_M), dict(parallel=float(par), spread=spread,
                                                                            depth=depth)


def detect_openings(L, N, put):
    """Aberturas en cada pared: lista de dicts con clasificación CONFIRMED / POSSIBLE / REJECTED."""
    walls = put["walls"]["walls"] or {}
    ch = put["chamber"]
    out = []
    for key, w in walls.items():
        fr = WallFrame(key, w, ch, put["floor_z"])
        g = face_grid(L, N, fr)
        for kind, cells in (("open", g["open_cells"]), ("relief", g["relief_cells"])):
            for comp in label(cells):
                if len(comp) * CELL_M ** 2 < np.pi * (MIN_OPENING_D_M / 2) ** 2:
                    continue
                top_z = fr.z_range[0] + (comp[:, 1].max() + 1) * CELL_M
                if top_z <= put["floor_z"] + 0.03:
                    continue  # franja bajo el nivel del bodem: allí no hay pared que interrumpir
                f = _region_features(comp, g, fr, L, N, put["floor_z"])
                f.update(wall=key, frame=fr, kind=kind)
                if kind == "relief":
                    ridx = f["recess_idx"]
                    rel, info = _is_relief(L[ridx], N[ridx], fr.n, g["d"][ridx])
                    f.update(status="REJECTED", reason="WALL_RELIEF: superficie paralela a "
                             f"{info.get('depth', 0) * 100:.0f} cm detrás de la cara, sin abertura", relief=info)
                    out.append(f)
                    continue
                bidx = f["behind_idx"]
                rel, info = _is_relief(L[bidx], N[bidx], fr.n, g["d"][bidx])
                circ = f["circle"]
                ring_ok = circ is not None and circ["rms"] <= 0.012 and circ["arc"] >= 270
                if rel:
                    f.update(status="REJECTED", reason="WALL_RELIEF: lo que hay detrás es un plano paralelo poco profundo",
                             relief=info)
                elif f["contour"] >= CONTOUR_CONFIRMED and (len(bidx) >= MIN_BEHIND_POINTS or ring_ok):
                    f.update(status="CONFIRMED", reason=f"contorno {f['contour']:.0%} apoyado en pared, "
                             f"{len(bidx)} ptn detrás")
                elif f["contour"] >= CONTOUR_POSSIBLE:
                    f.update(status="POSSIBLE", reason=f"contorno {f['contour']:.0%}, {len(bidx)} ptn detrás: "
                             "evidencia insuficiente")
                else:
                    f.update(status="REJECTED", reason=f"contorno {f['contour']:.0%} < {CONTOUR_POSSIBLE:.0%}: "
                             "hueco sin pared alrededor (zona no escaneada / esquina)")
                out.append(f)
    return out


# ---------------------------------------------------------------- tubo por bandas

def _frame_for_axis(a):
    u = np.cross([0, 0, 1.0], a)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross([0, 1.0, 0], a)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    if v[2] < 0:
        u, v = -u, -v
    return u, v


def _fit_line(C, wts):
    c = np.average(C, axis=0, weights=wts)
    _, _, vt = np.linalg.svd((C - c) * np.sqrt(wts)[:, None], full_matrices=False)
    return c, vt[0]


def _water_plane(Q, NQ, c0, a, r0):
    """Plano horizontal coherente en la parte inferior de la abertura (agua/sedimento/banket)."""
    u, v = _frame_for_axis(a)
    rel = Q - c0
    w = rel @ v
    hor = (np.abs(NQ[:, 2]) > 0.9) & (w < 0.2 * r0)
    if hor.sum() < 30:
        return None
    zs = Q[hor, 2]
    # nivel dominante: moda de alturas en bins de 1 cm
    hist, edges = np.histogram(zs, bins=np.arange(zs.min(), zs.max() + 0.02, 0.01))
    if not len(hist):
        return None
    k = int(np.argmax(hist))
    zw = float(np.median(zs[(zs >= edges[k] - 0.01) & (zs < edges[k + 1] + 0.01)]))
    on = hor & (np.abs(Q[:, 2] - zw) < 0.015)
    if on.sum() < 30:
        return None
    s_span = float(np.ptp(np.percentile(rel[on] @ u, [5, 95])))
    if s_span < 0.5 * r0 or A._is_curved_invert(rel[on] @ u, Q[on, 2]):
        return None  # demasiado estrecho o es el fondo curvo de un tubo seco
    return dict(z=zw, n=int(on.sum()), s_span=s_span)


def _bands(Q, NQ, c0, a, excl):
    """Ajuste independiente por bandas de profundidad (sección perpendicular a `a`)."""
    u, v = _frame_for_axis(a)
    rel = Q - c0
    t = rel @ a
    trans = np.abs(NQ @ a) > TRANSVERSE_N
    rows, t0 = [], 0.0
    tmax = min(MAX_DEPTH_M, float(t.max()) if len(t) else 0.0)
    while t0 < tmax:
        t1 = t0 + BAND_M
        # adaptar: ampliar la banda si tiene pocos puntos útiles
        while t1 < tmax + BAND_M and ((t >= t0) & (t < t1) & ~excl & ~trans).sum() < BAND_MIN_POINTS and t1 - t0 < 3 * BAND_M:
            t1 += BAND_M
        mb = (t >= t0) & (t < t1)
        use = mb & ~excl & ~trans
        row = dict(t0=float(t0), t1=float(t1), n=int(mb.sum()), n_use=int(use.sum()), n_trans=int((mb & trans).sum()),
                   valid=False, label="REJECTED", reason="", idx=np.flatnonzero(use))
        if use.sum() >= 12:
            tm = (t0 + t1) / 2
            sw = np.column_stack([rel[use] @ u, rel[use] @ v])
            sc, wc, r, inl, _ = fast_circle_robust(sw)
            if inl.sum() >= 6:
                sc, wc, r = A.fit_circle_geometric(sw[inl], sc, wc, r)
            res = np.hypot(sw[:, 0] - sc, sw[:, 1] - wc) - r
            near = np.abs(res) < 0.005
            ang = np.arctan2(sw[near, 1] - wc, sw[near, 0] - sc)
            arc = np.unique(((ang + np.pi) // (np.pi / 18)).astype(int)).size * 10
            rms = float(np.sqrt(np.mean(res[near] ** 2))) if near.any() else np.inf
            row.update(center=c0 + a * tm + u * sc + v * wc, r=float(r), rms=rms, arc=float(arc), n_inl=int(near.sum()),
                       inside_frac=float(np.mean(res < -0.02)), contamination=float(1 - near.mean()), inl_idx=row["idx"][near])
            # tapa / relleno: banda dominada por superficie transversal en el interior del círculo
            mt = mb & trans
            if mt.sum() >= BAND_MIN_POINTS and mt.sum() / max(mb.sum(), 1) >= FILL_FRACTION:
                rad = np.hypot(rel[mt] @ u - sc, rel[mt] @ v - wc)
                if np.median(rad) < 0.7 * max(r, 1e-3):
                    row.update(label="FILL", reason="superficie transversal que cierra el tubo")
            ok = (near.sum() >= 15 and arc >= BAND_MIN_ARC and rms <= BAND_MAX_RMS and 0.03 < r < 0.8)
            if row["label"] != "FILL":
                row["valid"] = bool(ok)
                if not ok:
                    row["reason"] = (f"arco {arc}° < {BAND_MIN_ARC}°" if arc < BAND_MIN_ARC else
                                     f"rms {rms * 1000:.1f} mm" if rms > BAND_MAX_RMS else
                                     f"{near.sum()} ptn sobre el círculo" if near.sum() < 15 else f"r={r * 1000:.0f} mm")
        else:
            row["reason"] = f"{int(use.sum())} ptn útiles"
        rows.append(row)
        t0 = t1
    return rows


def _axis_from_bands(rows, a_prev, n_wall, rng=None, n_boot=0):
    """Eje robusto por los centros de las bandas válidas (≥ 3, quitando la peor si el residuo es alto).
    Solo bandas con arco ≥ 180°: con menos, el centro se extrapola y se desplaza sistemáticamente hacia el lado
    no observado (la misma razón por la que un diámetro con < 180° no es MEASURED)."""
    vb = [b for b in rows if b["valid"] and b.get("arc", 0) >= AXIS_MIN_ARC]
    if len(vb) < 3:
        return None
    C = np.array([b["center"] for b in vb]); W = np.array([b["n_inl"] for b in vb], float)
    keep = np.ones(len(vb), bool)
    for _ in range(len(vb) - 3):
        c, d = _fit_line(C[keep], W[keep])
        res = np.linalg.norm(np.cross(C - c, d), axis=1)
        if res[keep].max() <= 0.008:
            break
        worst = np.argmax(np.where(keep, res, -1))
        keep[worst] = False
    c, d = _fit_line(C[keep], W[keep])
    if d @ a_prev < 0:
        d = -d
    res = np.linalg.norm(np.cross(C[keep] - c, d), axis=1)
    span = float(np.ptp((C[keep] - c) @ d))
    if res.max() > 0.008:
        return None   # los centros no forman una línea coherente: el eje no está determinado
    if np.degrees(np.arccos(np.clip(d @ n_wall, -1, 1))) > AXIS_MAX_DEV_DEG:
        return None
    out = dict(point=c, dir=d, res_max=float(res.max()), span=span, n=int(keep.sum()))
    if rng is not None and n_boot and keep.sum() >= 3:
        az, sl = [], []
        Ck, Wk = C[keep], W[keep]
        for _ in range(n_boot):
            i = rng.integers(0, len(Ck), len(Ck))
            if len(np.unique(i)) < 2:
                continue
            _, db = _fit_line(Ck[i], Wk[i])
            db = db if db @ d > 0 else -db
            az.append(np.degrees(np.arctan2(db[1], db[0]))); sl.append(np.degrees(np.arcsin(np.clip(db[2], -1, 1))))
        out["sigma_az"] = float(np.degrees(np.std(np.unwrap(np.radians(az))))) if az else np.inf
        out["sigma_slope"] = float(np.std(sl)) if sl else np.inf
    return out


def _stable_run(rows):
    """Tramo más largo de bandas válidas consecutivas con radio compatible (constante)."""
    vb = [i for i, b in enumerate(rows) if b["valid"]]
    best = []
    for s in range(len(vb)):
        run = [vb[s]]
        for k in range(s + 1, len(vb)):
            i = vb[k]
            if rows[i]["t0"] - rows[run[-1]]["t1"] > BAND_M + 1e-9:  # hueco de más de una banda: se corta
                break
            rs = [rows[j]["r"] for j in run + [i]]
            med = np.median(rs)
            if max(abs(x - med) for x in rs) > max(STABLE_TOL_ABS, STABLE_TOL_REL * med):
                break
            run.append(i)
        if len(run) > len(best):
            best = run
    return best


def measure_pipe(L, N, put, op, rng):
    """Sigue el tubo detrás de una abertura confirmada y lo mide por bandas."""
    fr = op["frame"]
    cu, cz = op["centroid_uz"]
    r0 = max(op["width"], op["height"]) / 2
    if op["circle"] is not None and op["circle"]["arc"] >= 180 and op["circle"]["rms"] <= 0.012:
        cu, cz = op["circle"]["c"]
        r0 = op["circle"]["r"]
    c0 = fr.point(cu, cz)
    a = fr.n.copy()
    u_all, z_all, d_all = fr.coords(L)
    at_face = d_all >= -0.01
    history = []

    def evaluate(c0, a, r0):
        rel = L - c0
        t = rel @ a
        dist = np.linalg.norm(np.cross(rel, a), axis=1)
        pool = at_face & (t >= -0.01) & (t <= MAX_DEPTH_M) & (dist <= SEARCH_FACTOR * r0 + 0.05)
        idx = np.flatnonzero(pool)
        Q, NQ = L[idx], N[idx]
        water = _water_plane(Q, NQ, c0, a, r0) if len(Q) else None
        excl = np.zeros(len(Q), bool)
        if water:
            excl |= (np.abs(NQ[:, 2]) > 0.9) & (np.abs(Q[:, 2] - water["z"]) < 0.015)
            excl |= Q[:, 2] < water["z"] - 0.015  # bajo el agua no hay superficie de tubo visible
        rows = _bands(Q, NQ, c0, a, excl) if len(Q) else []
        return idx, water, excl, rows

    def score(rows):
        return (len(_stable_run(rows)), sum(b["valid"] for b in rows))

    # iterar: cortar perpendicular al eje actual -> centros de bandas -> nuevo eje (corrige tubos oblicuos).
    # Un eje nuevo solo se acepta si NO empeora la evidencia (tramo estable y bandas válidas): unos pocos
    # centros contaminados pueden alinearse en cualquier dirección.
    a0_wall, c0_wall = a.copy(), c0.copy()
    idx, water, excl, rows = evaluate(c0, a, r0)
    cur = score(rows)
    for it in range(6):
        ax = _axis_from_bands(rows, a, fr.n)
        history.append(dict(a=a.copy(), c0=c0.copy(), score=cur))
        if ax is None:
            break
        new_a = ax["dir"]
        c_new = ax["point"] - new_a * (((ax["point"] - fr.o) @ fr.n) / (new_a @ fr.n))  # entrada en el plano de pared
        vb = [b for b in rows if b["valid"]]
        r_new = float(np.median([b["r"] for b in vb])) if vb else r0
        cand = evaluate(c_new, new_a, r_new)
        new_score = score(cand[3])
        if new_score < cur:
            history.append(dict(a=new_a.copy(), rejected=True, score=new_score))
            break
        change = np.degrees(np.arccos(np.clip(new_a @ a, -1, 1)))
        a, c0, r0 = new_a, c_new, r_new
        idx, water, excl, rows = cand
        cur = new_score
        if change < 0.3:
            break
    if len(_stable_run(rows)) < 3 and np.degrees(np.arccos(np.clip(a @ a0_wall, -1, 1))) > 0.5:
        # eje no confirmado por un tramo estable: volver a la dirección supuesta (normal de la pared)
        a, c0 = a0_wall, c0_wall
        idx, water, excl, rows = evaluate(c0, a, r0)
    axis = _axis_from_bands(rows, a, fr.n, rng, n_boot=100)
    run = _stable_run(rows)
    for i, b in enumerate(rows):
        if b["label"] == "FILL":
            continue
        if i in run:
            b["label"] = "PIPE"
        elif b["valid"]:
            b["label"] = "OPENING_RING" if run and i < run[0] else "REJECTED"
            b["reason"] = b["reason"] or (f"radio {b['r'] * 1000:.0f} mm distinto del tramo estable" if run else
                                          "sin tramo estable")
    return dict(idx=idx, c0=c0, axis=a, axis_fit=axis, rows=rows, run=run, water=water, r0=r0, history=history,
                excl_water=excl if len(idx) else np.zeros(0, bool))


def crown_profile(rows, rel, u, v, c0):
    """Altura de la corona (punto más alto en el centro de la sección) por banda."""
    out = []
    for b in rows:
        if b["label"] == "FILL" or b["n_use"] < 12:
            out.append(None)
            continue
        idx = b["idx"]
        s, w = rel[idx] @ u, rel[idx] @ v
        lo, hi = np.percentile(s, [2, 98])
        mid, wid = (lo + hi) / 2, hi - lo
        top = np.abs(s - mid) < 0.1 * wid
        s_at_max = s[np.argmax(w)]
        if top.sum() < 5 or not (lo + 0.2 * wid < s_at_max < hi - 0.2 * wid):
            out.append(None)  # el máximo está en un extremo cortado: la corona no se ve en esta banda
            continue
        out.append(float(c0[2] + np.percentile(w[top], 95)))
    return out


CROWN_TOL_M = 0.005           # corona a la misma altura (±5 mm ≈ 2x ruido) en bandas consecutivas
CROWN_MAX_TREND = np.tan(np.radians(5.0))  # pendientes de alcantarillado reales << 5°


def crown_from_profile(rows, rel, u, v, c0, floor_sigma):
    """Kruin sin tramo de tubo estable: solo si la corona se mantiene a la misma altura en ≥ 2 bandas
    consecutivas Y el perfil de alturas no muestra una tendencia > 5°. Una 'corona' que baja de forma continua
    con la profundidad es el límite de visibilidad (oclusión al escanear desde arriba), no la kruin."""
    prof = crown_profile(rows, rel, u, v, c0)
    for b, cz in zip(rows, prof):
        b["crown_z"] = cz
    known = [(0.5 * (b["t0"] + b["t1"]), cz) for b, cz in zip(rows, prof) if cz is not None]
    if len(known) >= 3:
        t_, z_ = np.array(known).T
        trend = float(np.polyfit(t_, z_, 1)[0])
        if abs(trend) > CROWN_MAX_TREND:
            return A.unknown("crown_band_profile", reason=f"la altura del punto más alto cambia {np.degrees(np.arctan(trend)):+.0f}° "
                             "con la profundidad: es el límite de visibilidad, no la kruin")
    best = []
    for s in range(len(prof)):
        run = []
        for k in range(s, len(prof)):
            if prof[k] is None:
                break
            vals = [prof[j] for j in run + [k]]
            if max(vals) - min(vals) > CROWN_TOL_M:
                break
            run.append(k)
        if len(run) > len(best):
            best = run
    if len(best) < 2:
        return A.unknown("crown_band_profile", reason="la corona no se mantiene en ≥ 2 bandas consecutivas "
                         "(el punto más alto puede ser el borde del agujero en la pared)")
    vals = np.array([prof[k] for k in best])
    v0 = float(np.median(vals))
    U = float(np.hypot(A.expanded(v0, np.std(vals), floor_sigma, scale=False), A.K_EXPAND * A.SCALE_REL_SIGMA * abs(v0)))
    conf = A.HOOG if U < 0.01 else A.MIDDEL if U < 0.025 else A.LAAG
    return A.meas(v0, U, conf, "crown_band_profile", bands=len(best))


# ---------------------------------------------------------------- fallback: radio común de bandas parciales
# Con arcos de 100-150° por banda, el círculo de CADA banda está mal condicionado: un pequeño desplazamiento del
# centro cambia mucho el radio aunque los puntos sean del mismo cilindro. Si no hay tramo estable, se prueba UN
# solo círculo común para todas las bandas detrás de la cara. Solo se acepta si los datos lo determinan.
JOINT_MIN_BANDS = 3        # con 2 bandas, quitar una deja una sola banda mal condicionada: no hay comprobación
JOINT_MIN_BAND_PTS = 10    # cada banda debe aportar puntos sobre el círculo común (si no, no es independiente)
JOINT_MIN_POINTS = 60
JOINT_MIN_ARC_DEG = 120    # el arco combinado debe superar claramente el mínimo por banda (90°)
JOINT_MAX_TILT_DEG = 30    # el modelo de eje libre más allá de esto no está determinado


def _free_axis_cylinder(s, w, t, s0, w0, r, iters=40):
    """Cilindro con radio común y centro que deriva linealmente con la profundidad t (eje libre).
    Devuelve dict(r, ds, dw, rms, res, s_corr, w_corr, sigma_r) o None si diverge. sigma_r sale de la covarianza
    de los parámetros: si radio y deriva del eje son casi degenerados (arco parcial), sigma_r es grande."""
    tc = t - t.mean()
    p = np.array([s0, w0, 0.0, 0.0, r])
    J = None
    for _ in range(iters):
        ds_, dw_ = s - p[0] - p[2] * tc, w - p[1] - p[3] * tc
        d = np.maximum(np.hypot(ds_, dw_), 1e-12)
        J = np.column_stack([-ds_ / d, -dw_ / d, -ds_ * tc / d, -dw_ * tc / d, -np.ones(len(d))])
        step, *_ = np.linalg.lstsq(J, -(d - p[4]), rcond=None)
        p = p + step
        if not np.all(np.isfinite(p)):
            return None
        if np.linalg.norm(step) < 1e-9:
            break
    s_c, w_c = s - p[2] * tc, w - p[3] * tc
    res = np.hypot(s_c - p[0], w_c - p[1]) - p[4]
    rms = float(np.sqrt(np.mean(res ** 2)))
    try:
        cov = rms ** 2 * np.linalg.inv(J.T @ J)
        sigma_r = float(np.sqrt(max(cov[4, 4], 0.0)))
    except np.linalg.LinAlgError:
        sigma_r = np.inf
    return dict(r=float(abs(p[4])), s0=float(p[0]), w0=float(p[1]), ds=float(p[2]), dw=float(p[3]), rms=rms,
                res=res, s_corr=s_c, w_corr=w_c, sigma_r=sigma_r)


def _joint_partial_pipe_fit(rows, Q, c0, a, u, v, rng, floor_s, top_z, top_s, sigma_axis):
    """Ajuste conjunto de un círculo común a las bandas parciales del tubo detrás de la cara de la pared.

    Usa SOLO: bandas válidas (no FILL) con t ≥ 5 cm (la banda de la cara puede ser abertura/collar), y dentro de
    cada banda los puntos que ya estaban sobre su arco local (sin agua/banket ni superficies transversales: esos
    se excluyeron al formar las bandas). El tamaño de la abertura NO interviene.

    Comprobaciones de identificabilidad (cualquiera que falle -> no determinado, el diámetro sigue UNKNOWN):
      - ≥ 3 bandas que aporten cada una ≥ 10 puntos sobre el círculo común, ≥ 60 puntos en total;
      - arco combinado ≥ 120°; RMS ≤ BAND_MAX_RMS;
      - consistencia longitudinal: residuo medio de cada banda y tendencia del radio con la profundidad dentro de
        la misma tolerancia que define un tramo estable (max(5 mm, 2 % r)) -> rechaza conos y radios incompatibles;
      - leave-one-band-out: cada ajuste sin una banda dentro de la misma tolerancia;
      - modelo de eje libre convergente y con inclinación ≤ 30°.
    Incertidumbre adicional (entra en _measure_arc, que aplica además bootstrap, cuerda-flecha, escala y la regla
    R10): jackknife leave-one-band-out + sensibilidad al eje (eje rígido vs eje libre + incertidumbre del eje)."""
    out = dict(ok=False, method="joint_partial_pipe_bands", reason="", bands=[])
    cand = [i for i, b in enumerate(rows) if b["valid"] and b["label"] != "FILL" and b["t0"] >= BAND_M - 1e-9
            and len(b.get("inl_idx", [])) >= 6]
    out["bands"] = cand
    if len(cand) < JOINT_MIN_BANDS:
        out["reason"] = f"{len(cand)} bandas válidas detrás de la cara (< {JOINT_MIN_BANDS}): sin comprobación independiente"
        return out
    idx = np.concatenate([rows[i]["inl_idx"] for i in cand])
    bid = np.concatenate([np.full(len(rows[i]["inl_idx"]), k) for k, i in enumerate(cand)])
    rel = Q[idx] - c0
    s, w, t = rel @ u, rel @ v, rel @ a
    sz = np.column_stack([s, w])
    # Círculo común por mínimos cuadrados geométricos sobre TODOS los puntos de arco local (ya limpios: estaban a
    # < 5 mm del arco de su banda), con un recorte robusto. No se usa RANSAC: maximizar inliers puede elegir un
    # subconjunto de bandas; aquí son los residuos por banda los que deciden si existe un radio común.
    sc, wc, r = A.fit_circle_2d(sz)
    sc, wc, r = A.fit_circle_geometric(sz, sc, wc, r)
    for _ in range(2):
        dr = np.hypot(s - sc, w - wc) - r
        mad = 1.4826 * np.median(np.abs(dr - np.median(dr)))
        keep = np.abs(dr) <= max(3 * mad, 0.005)
        if keep.sum() < 12:
            break
        sc, wc, r = A.fit_circle_geometric(sz[keep], sc, wc, r)

    def stats(se, we, sc_, wc_, r_):
        dr_ = np.hypot(se - sc_, we - wc_) - r_
        on_ = np.abs(dr_) < 0.005
        ang_ = np.arctan2(we[on_] - wc_, se[on_] - sc_)
        band_res_ = np.array([np.median(dr_[bid == k]) for k in range(len(cand))])
        t_b = np.array([np.median(t[bid == k]) for k in range(len(cand))])
        trend_ = float(np.polyfit(t_b, band_res_, 1)[0] * np.ptp(t_b))
        return dict(dr=dr_, on=on_, per_band=np.bincount(bid[on_], minlength=len(cand)),
                    arc=np.unique(((ang_ + np.pi) // (np.pi / 18)).astype(int)).size * 10,
                    rms=float(np.sqrt(np.mean(dr_[on_] ** 2))) if on_.any() else np.inf,
                    band_res=band_res_, trend=trend_)

    def consistent(st_):
        return np.max(np.abs(st_["band_res"])) <= tol and abs(st_["trend"]) <= tol

    tol = max(STABLE_TOL_ABS, STABLE_TOL_REL * r)
    st = stats(s, w, sc, wc, r)
    out.update(D_rigid=2 * r, tolerance_mm=tol * 1000, band_depth_cm=[round(rows[i]["t0"] * 100) for i in cand],
               band_offsets_rigid_mm=(st["band_res"] * 1000).round(1).tolist(),
               radius_trend_rigid_mm=st["trend"] * 1000)
    model, se, we = "rigid_axis", s, w
    sigma_model_r = 0.0
    if not consistent(st):
        # Desvíos por banda con tendencia: puede ser un eje inclinado (los centros derivan con la profundidad).
        # Se prueba el cilindro de eje libre y se exige que el radio quede DETERMINADO (no degenerado con la deriva).
        free = _free_axis_cylinder(s, w, t, sc, wc, r)
        if free is None:
            out["reason"] = "bandas incompatibles con un radio común y el modelo de eje libre no converge"
            return out
        tilt = float(np.degrees(np.arctan(np.hypot(free["ds"], free["dw"]))))
        sf = stats(free["s_corr"], free["w_corr"], free["s0"], free["w0"], free["r"])
        out.update(D_free_axis=2 * free["r"], free_axis_tilt_deg=tilt, sigma_r_free_mm=free["sigma_r"] * 1000,
                   band_offsets_free_mm=(sf["band_res"] * 1000).round(1).tolist())
        if tilt > JOINT_MAX_TILT_DEG:
            out["reason"] = (f"bandas incompatibles con un radio común (desvíos {out['band_offsets_rigid_mm']} mm); "
                             f"el eje libre necesitaría {tilt:.0f}° de inclinación")
            return out
        if not consistent(sf):
            out["reason"] = (f"bandas incompatibles con un radio común, ni con eje rígido (desvíos "
                             f"{out['band_offsets_rigid_mm']} mm) ni con eje libre ({out['band_offsets_free_mm']} mm); "
                             f"tolerancia {tol * 1000:.1f} mm")
            return out
        # Con arcos parciales, un cono y un cilindro inclinado son casi indistinguibles para el modelo de eje libre.
        # Lo que los separa: en un cilindro inclinado, el radio PROPIO de cada banda no cambia con la profundidad
        # (solo se desplaza el centro); en un cono crece/decrece de forma sistemática. Tendencia significativa
        # (> tolerancia de radio y t > 3 frente a la dispersión entre bandas) -> no es un cilindro.
        r_own = np.array([rows[i]["r"] for i in cand])
        t_own = np.array([0.5 * (rows[i]["t0"] + rows[i]["t1"]) for i in cand])
        coef = np.polyfit(t_own, r_own, 1)
        resid = r_own - np.polyval(coef, t_own)
        dof = max(len(cand) - 2, 1)
        se = np.sqrt(np.sum(resid ** 2) / dof / max(np.sum((t_own - t_own.mean()) ** 2), 1e-12))
        change = float(coef[0] * np.ptp(t_own))
        t_stat = abs(coef[0]) / max(se, 1e-12)
        out.update(own_radius_trend_mm=change * 1000, own_radius_trend_t=float(t_stat))
        if abs(change) > tol and t_stat > 3:
            out["reason"] = (f"el radio propio de las bandas cambia {change * 1000:+.0f} mm con la profundidad "
                             f"(t={t_stat:.1f}): no es un cilindro (cono/transición), el eje libre lo enmascararía")
            return out
        if 2 * A.K_EXPAND * free["sigma_r"] / (2 * free["r"]) > A.D_UNKNOWN_REL:
            out["reason"] = (f"con este arco el radio y la deriva del eje no son separables "
                             f"(σ_r {free['sigma_r'] * 1000:.0f} mm)")
            return out
        model, se, we = "free_axis", free["s_corr"], free["w_corr"]
        sc, wc, r, st = free["s0"], free["w0"], free["r"], sf
        sigma_model_r = free["sigma_r"]
    on = st["on"]
    out.update(model=model, D=2 * r, arc_deg=float(st["arc"]), rms=st["rms"], points=int(on.sum()),
               per_band_points=st["per_band"].tolist(), band_offsets_mm=(st["band_res"] * 1000).round(1).tolist())
    if (st["per_band"] >= JOINT_MIN_BAND_PTS).sum() < JOINT_MIN_BANDS or on.sum() < JOINT_MIN_POINTS:
        out["reason"] = (f"solo {(st['per_band'] >= JOINT_MIN_BAND_PTS).sum()} bandas aportan puntos al círculo común "
                         f"({on.sum()} ptn)")
        return out
    if st["arc"] < JOINT_MIN_ARC_DEG:
        out["reason"] = f"arco combinado {st['arc']}° < {JOINT_MIN_ARC_DEG}°"
        return out
    if st["rms"] > BAND_MAX_RMS:
        out["reason"] = f"RMS {st['rms'] * 1000:.1f} mm > {BAND_MAX_RMS * 1000:.0f} mm"
        return out
    # leave-one-band-out con el MISMO modelo
    lobo = []
    for k in range(len(cand)):
        m = on & (bid != k)
        if m.sum() < 12:
            continue
        if model == "rigid_axis":
            lobo.append(2 * A.fit_circle_geometric(np.column_stack([se[m], we[m]]), sc, wc, r)[2])
        else:
            fk = _free_axis_cylinder(s[m], w[m], t[m], sc, wc, r)
            lobo.append(np.inf if fk is None else 2 * fk["r"])
    lobo = np.array(lobo)
    out["lobo_D_mm"] = [round(float(x) * 1000, 1) if np.isfinite(x) else None for x in lobo]
    if len(lobo) < JOINT_MIN_BANDS - 1 or not np.all(np.isfinite(lobo)) or np.max(np.abs(lobo - 2 * r)) > 2 * tol:
        out["reason"] = f"inestable al retirar una banda (Ø {out['lobo_D_mm']} mm frente a {2 * r * 1000:.0f} mm)"
        return out
    n = len(lobo)
    sigma_jack = float(np.sqrt((n - 1) / n * np.sum((lobo - lobo.mean()) ** 2)))
    # sensibilidad a la orientación: el otro modelo de eje como alternativa + incertidumbre del eje de la conexión
    if model == "rigid_axis":
        free = _free_axis_cylinder(s[on], w[on], t[on], sc, wc, r)
        d_alt = abs(2 * free["r"] - 2 * r) / 2 if free is not None else 0.0
        out["D_free_axis"] = None if free is None else 2 * free["r"]
    else:
        d_alt = abs(out["D_rigid"] - 2 * r) / 2
    sigma_orient = float(np.hypot(d_alt, 2 * r * (1 / np.cos(min(sigma_axis, np.radians(30))) - 1) / 2))
    out.update(sigma_jack=sigma_jack, sigma_orient=sigma_orient, sigma_model_r=sigma_model_r)
    # medida final con las reglas comunes (bootstrap, cuerda-flecha, escala, arco, regla R10)
    meas_arc = A._measure_arc(np.column_stack([se[on], c0[2] + we[on]]), rng, floor_s, top_z, top_s,
                              extra_sigma_d=float(np.hypot(np.hypot(sigma_jack, sigma_orient), 2 * sigma_model_r)))
    d = meas_arc["diameter"]
    if d["value"] is None:
        out["reason"] = "ajuste conjunto no determinado: " + d.get("reason", "")
        out["attempt"] = (d.get("attempt_value"), d.get("attempt_U"))
        return out
    out.update(ok=True, arc=meas_arc, used_idx=idx[on])
    return out


# ---------------------------------------------------------------- ajuste robusto de cilindro 3D
# Ajuste DIRECTO de un cilindro a los puntos 3D del tubo (sin reducir antes cada banda a un círculo):
#   parámetros = punto del eje (2), dirección del eje (2), radio común (1)
#   residual_i = distancia(punto_i, eje) - radio,  pérdida robusta soft-L1 (escala = ruido de superficie).
# La identificabilidad del radio se cuantifica con un PERFIL del radio (para cada radio fijo se reoptimiza el eje),
# bootstrap por bloques, leave-one-band-out, sensibilidad al eje y tendencia del residual con la profundidad.
CYL_F0_M = 0.003           # escala de la pérdida soft-L1 ≈ ruido de superficie típico (RMS de pared ≤ 3 mm)
CYL_MIN_POINTS = 60
CYL_BLOCK_DEG = 10         # bloques del bootstrap: 10° de arco x 5 cm de profundidad (errores correlados localmente)
CYL_PROFILE_STEPS = np.linspace(-0.4, 0.4, 41)   # radios de prueba: ±40 % alrededor del óptimo
CYL_CHI2_95 = 3.84         # perfil de 1 parámetro, 95 %
CYL_MAX_TILT_DEG = 30
CYL_N_BOOT = 60
T_CRIT_95 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}   # t de Student bilateral 95% (sin scipy)


def _cyl_frame(c0, a0):
    u0, v0 = _frame_for_axis(a0)
    return u0, v0


def _cyl_residuals(P, x, c0, a0, u0, v0):
    ps, pw, al, be, r = x
    p = c0 + ps * u0 + pw * v0
    a = a0 + al * u0 + be * v0
    a = a / np.linalg.norm(a)
    d = P - p
    w = d - np.outer(d @ a, a)
    return np.linalg.norm(w, axis=1) - r


def _soft_l1(res, f0=CYL_F0_M):
    return 2 * f0 ** 2 * (np.sqrt(1 + (res / f0) ** 2) - 1)


def _fit_cylinder(P, x0, c0, a0, u0, v0, free=(True, True, True, True, True), iters=40):
    """Gauss-Newton con pesos IRLS soft-L1 (Jacobiano numérico). Devuelve (x, residuos, coste robusto)."""
    x = np.array(x0, float)
    free = np.array(free)
    eps = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    for _ in range(iters):
        res = _cyl_residuals(P, x, c0, a0, u0, v0)
        wts = 1 / np.sqrt(1 + (res / CYL_F0_M) ** 2)
        J = []
        for k in np.flatnonzero(free):
            xk = x.copy(); xk[k] += eps[k]
            J.append((_cyl_residuals(P, xk, c0, a0, u0, v0) - res) / eps[k])
        J = np.array(J).T
        A_ = J.T @ (J * wts[:, None])
        g = J.T @ (wts * res)
        A_ += 1e-9 * np.trace(A_) * np.eye(len(A_))
        try:
            step = -np.linalg.solve(A_, g)
        except np.linalg.LinAlgError:
            break
        x[free] += step
        if not np.all(np.isfinite(x)):
            return None
        if np.linalg.norm(step) < 1e-9:
            break
    res = _cyl_residuals(P, x, c0, a0, u0, v0)
    return x, res, float(np.sum(_soft_l1(res)))


def _cyl_axis(x, c0, a0, u0, v0):
    a = a0 + x[2] * u0 + x[3] * v0
    return c0 + x[0] * u0 + x[1] * v0, a / np.linalg.norm(a)


def pipe_cylinder_3d(Q, NQ, c0, a0, excl, rng, sigma_axis, t_min=BAND_M):
    """Ajuste robusto de cilindro 3D a TODOS los puntos válidos del tubo detrás de la cara (t ≥ 5 cm).
    excl: puntos a excluir (agua/banket/sedimento, superficies transversales = tapa/escalón/cara).
    No usa la abertura de la pared. Devuelve un dict con el ajuste, su identificabilidad y la decisión de medida."""
    out = dict(ok_fit=False, measured=False, method="robust_3d_cylinder_fit", reason="")
    u0, v0 = _cyl_frame(c0, a0)
    rel = Q - c0
    t = rel @ a0
    sel = ~excl & (t >= t_min - 1e-9)
    P = Q[sel]
    if len(P) < CYL_MIN_POINTS:
        out["reason"] = f"{len(P)} puntos de tubo detrás de la cara (< {CYL_MIN_POINTS})"
        return out
    # radio inicial: mediana de la distancia al eje actual (no se usa la abertura)
    rp = P - c0
    rad0 = np.linalg.norm(rp - np.outer(rp @ a0, a0), axis=1)
    r_init = float(np.median(rad0))
    # 1) ajuste inicial con eje fijo y centro libre (estable), 2) ajuste completo
    f = _fit_cylinder(P, [0, 0, 0, 0, r_init], c0, a0, u0, v0, free=(True, True, False, False, True))
    if f is None:
        out["reason"] = "el ajuste inicial no converge"
        return out
    # recorte: solo la superficie cercana al cilindro (ventana amplia; la pérdida robusta decide dentro)
    near = np.abs(f[1]) < max(0.03, 0.2 * f[0][4])
    P = P[near]
    if len(P) < CYL_MIN_POINTS:
        out["reason"] = f"solo {len(P)} puntos cerca de una superficie cilíndrica"
        return out
    # Contaminación UNILATERAL: el escáner no ve a través de la pared del tubo, así que fuera de la superficie interior
    # solo hay ruido; DENTRO puede haber agua, relleno, restos... (residuos negativos). El ruido se estima con el lado
    # EXTERIOR del residual (limpio) y se excluyen los puntos > 3σ hacia dentro; se itera hasta estabilizar.
    keep = np.ones(len(P), bool)
    fx = _fit_cylinder(P, f[0], c0, a0, u0, v0)
    n_interior = 0
    for _ in range(4):
        if fx is None:
            break
        res_all = _cyl_residuals(P, fx[0], c0, a0, u0, v0)
        pos = res_all[res_all > 0]
        s_noise = max(np.median(pos) / 0.674 if len(pos) >= 10 else CYL_F0_M, 0.001)
        new_keep = res_all > -3 * s_noise
        if new_keep.sum() < CYL_MIN_POINTS or np.array_equal(new_keep, keep):
            break
        keep = new_keep
        fx = _fit_cylinder(P[keep], fx[0], c0, a0, u0, v0)
    n_interior = int((~keep).sum())
    P = P[keep]
    x_fixed = np.array(fx[0] if fx is not None else f[0], float)
    x_fixed[2:4] = 0.0   # eje fijo = eje supuesto de la conexión (sin inclinación adicional)
    ff = _fit_cylinder(P, x_fixed, c0, a0, u0, v0, free=(True, True, False, False, True))
    if fx is None or ff is None:
        out["reason"] = "el ajuste de cilindro no converge"
        return out
    out["n_interior_excluded"] = n_interior
    x, res, cost = fx
    p_ax, a_ax = _cyl_axis(x, c0, a0, u0, v0)
    tilt = float(np.degrees(np.arccos(np.clip(abs(a_ax @ a0), -1, 1))))
    r = abs(x[4])
    D = 2 * r
    # estadísticos de la superficie
    mad = 1.4826 * np.median(np.abs(res - np.median(res)))
    inl = np.abs(res) <= max(3 * mad, 0.005)
    d = P - p_ax
    tt = d @ a_ax
    uu, vv = _frame_for_axis(a_ax)
    ang = np.arctan2(d @ vv, d @ uu)
    arc = np.unique(((ang[inl] + np.pi) // np.radians(CYL_BLOCK_DEG)).astype(int)).size * CYL_BLOCK_DEG
    band_of = np.floor((tt - tt.min()) / BAND_M).astype(int)
    bands = [k for k in np.unique(band_of[inl]) if (inl & (band_of == k)).sum() >= 10]
    rms = float(np.sqrt(np.mean(res[inl] ** 2)))
    out.update(ok_fit=True, D=D, axis=a_ax.tolist(), axis_point=p_ax, tilt_deg=tilt, rms=rms,
               res_pct_mm=(np.percentile(res, [5, 25, 50, 75, 95]) * 1000).round(1).tolist(),
               n_points=int(len(P)), n_inliers=int(inl.sum()), arc_deg=float(arc), n_bands=len(bands),
               visible_length=float(np.ptp(np.percentile(tt[inl], [2, 98]))) if inl.sum() > 5 else 0.0,
               t_range=(float(np.percentile(tt[inl], 2)), float(np.percentile(tt[inl], 98))) if inl.sum() > 5 else (0, 0),
               D_fixed_axis=2 * abs(ff[0][4]))
    # tendencia del residual medio por banda con la profundidad (cono / transición -> no es un cilindro)
    tol = max(STABLE_TOL_ABS, STABLE_TOL_REL * r)
    if len(bands) >= 2:
        bm = np.array([np.median(res[inl & (band_of == k)]) for k in bands])
        bt = np.array([np.median(tt[inl & (band_of == k)]) for k in bands])
        trend = float(np.polyfit(bt, bm, 1)[0] * np.ptp(bt)) if len(bands) >= 2 else 0.0
        out.update(band_offsets_mm=(bm * 1000).round(1).tolist(), residual_trend_mm=trend * 1000)
    else:
        trend = 0.0
    # radio PROPIO de cada banda (dirección del eje ajustado fija, centro y radio libres): con arco parcial un eje
    # libre puede absorber un cono (la corona que sube parece un eje inclinado); el radio propio que crece lo delata.
    own_r, own_t = [], []
    for k in bands:
        mk = inl & (band_of == k)
        if mk.sum() < 20:
            continue
        fk = _fit_cylinder(P[mk], x, c0, a0, u0, v0, free=(True, True, False, False, True), iters=25)
        if fk is not None and 0.5 * r < abs(fk[0][4]) < 2 * r:   # fuera de ese rango: banda mal condicionada
            own_r.append(abs(fk[0][4]))
            own_t.append(float(np.median(tt[mk])))
    own_change, own_tstat, t_crit = 0.0, 0.0, np.inf
    if len(own_r) >= 3:
        own_t, own_r = np.array(own_t), np.array(own_r)
        A_ = np.column_stack([own_t - own_t.mean(), np.ones(len(own_t))])
        coef, *_ = np.linalg.lstsq(A_, own_r, rcond=None)
        dof = len(own_r) - 2
        s2 = float(np.sum((own_r - A_ @ coef) ** 2)) / dof if dof > 0 else np.nan
        se = np.sqrt(s2 / np.sum((own_t - own_t.mean()) ** 2)) if dof > 0 else np.nan
        own_change = float(coef[0] * np.ptp(own_t))
        own_tstat = float(abs(coef[0]) / max(se, 1e-9)) if np.isfinite(se) else 0.0
        t_crit = max(3.0, T_CRIT_95.get(dof, 3.0))   # t de Student bilateral 95% para los g.l. reales
    out.update(own_band_D_mm=[round(2 * float(z) * 1000, 1) for z in own_r], own_radius_change_mm=own_change * 1000,
               own_radius_tstat=own_tstat, own_radius_tcrit=t_crit)
    cone = abs(own_change) > tol and own_tstat > t_crit
    # perfil del radio: para cada radio fijo, reoptimizar eje y centro; Δχ² con N efectivo = nº de bloques
    Pi = P[inl]
    blocks = np.column_stack([((ang[inl] + np.pi) // np.radians(CYL_BLOCK_DEG)).astype(int), band_of[inl]])
    n_eff = len(np.unique(blocks, axis=0))
    sigma2 = max(float(np.sum(_soft_l1(res[inl]))) / max(len(Pi) - 5, 1), 1e-12)
    prof = []
    xb = x.copy()
    for k in CYL_PROFILE_STEPS:
        rk = r * (1 + k)
        xr = xb.copy(); xr[4] = rk
        fr_ = _fit_cylinder(Pi, xr, c0, a0, u0, v0, free=(True, True, True, True, False), iters=25)
        if fr_ is None:
            prof.append((2 * rk, np.inf, np.inf))
            continue
        ck = float(np.sum(_soft_l1(fr_[1])))
        prof.append((2 * rk, ck, float(np.sqrt(np.mean(fr_[1] ** 2)))))
    cmin = min(p[1] for p in prof)
    dchi = [((p[1] - cmin) / sigma2) * n_eff / len(Pi) for p in prof]
    bounded = dchi[0] > CYL_CHI2_95 and dchi[-1] > CYL_CHI2_95
    # límites del intervalo: cruce de Δχ² = 3.84 interpolado entre puntos de la rejilla en √Δχ² (≈ lineal en |r - r̂|),
    # para que un intervalo más estrecho que el paso de la rejilla no colapse a un punto
    i0 = int(np.argmin(dchi))
    sq, rr_ = np.sqrt(np.maximum(dchi, 0)), [p[0] for p in prof]
    thr = np.sqrt(CYL_CHI2_95)

    def _cross(step):
        i = i0
        while 0 <= i + step < len(prof):
            if sq[i + step] > thr:
                f = (thr - sq[i]) / max(sq[i + step] - sq[i], 1e-12)
                return rr_[i] + f * (rr_[i + step] - rr_[i])
            i += step
        return rr_[i]
    ok_r = [_cross(-1), _cross(+1)] if np.isfinite(dchi[i0]) else []
    out.update(profile=[(round(p[0] * 1000, 1), round(dc, 2), round(p[2] * 1000, 2)) for p, dc in zip(prof, dchi)],
               profile_interval_mm=(round(float(min(ok_r)) * 1000, 1), round(float(max(ok_r)) * 1000, 1)) if ok_r else None,
               profile_bounded=bounded, n_eff_blocks=int(n_eff))
    # ¿la inclinación del eje está identificada? eje fijo (supuesto) vs eje libre con los mismos puntos, Δχ² efectivo
    # con 2 g.l. (5.99 al 95%). Si no es significativa, la inclinación no se distingue de un límite de visibilidad.
    x_fix = x.copy(); x_fix[2:4] = 0.0
    fa = _fit_cylinder(Pi, x_fix, c0, a0, u0, v0, free=(True, True, False, False, True), iters=25)
    if fa is not None:
        c_fix = float(np.sum(_soft_l1(fa[1])))
        c_free = float(np.sum(_soft_l1(res[inl])))
        d_ax = (c_fix - c_free) / sigma2 * n_eff / len(Pi)
        out.update(rms_fixed_axis=float(np.sqrt(np.mean(fa[1] ** 2))), axis_dchi2_eff=round(float(d_ax), 2),
                   axis_tilt_significant=bool(d_ax > 5.99))
    # leave-one-band-out (bandas de 5 cm a lo largo del eje ajustado)
    lobo = []
    for k in bands:
        m = ~(band_of[inl] == k)
        if m.sum() < CYL_MIN_POINTS // 2:
            continue
        fk = _fit_cylinder(Pi[m], x, c0, a0, u0, v0, iters=25)
        lobo.append(2 * abs(fk[0][4]) if fk is not None else np.inf)
    lobo = np.array(lobo)
    out["lobo_D_mm"] = [round(float(z) * 1000, 1) if np.isfinite(z) else None for z in lobo]
    # jackknife ANGULAR: quitar cada cuarto del arco observado. LOBO y bootstrap solo prueban la variación a lo largo
    # del tubo; el error por sección no circular (ovalidad, deformación) depende de QUÉ parte de la circunferencia
    # se ve, y solo aparece al cambiar la parte del arco usada.
    ai = ang[inl]
    srt = np.sort(ai)
    gaps = np.diff(np.r_[srt, srt[0] + 2 * np.pi])
    start = srt[(np.argmax(gaps) + 1) % len(srt)]          # inicio del arco contiguo (después del mayor hueco)
    rel_ang = (ai - start) % (2 * np.pi)
    q = np.quantile(rel_ang, [0.25, 0.5, 0.75])
    sector = np.searchsorted(q, rel_ang)
    angD = []
    for k in range(4):
        mk = sector != k
        fk = _fit_cylinder(Pi[mk], x, c0, a0, u0, v0, iters=25) if mk.sum() >= CYL_MIN_POINTS // 2 else None
        angD.append(2 * abs(fk[0][4]) if fk is not None else np.inf)
    angD = np.array(angD)
    sigma_ang = (float(np.sqrt(3 / 4 * np.sum((angD - angD.mean()) ** 2))) if np.all(np.isfinite(angD)) else np.inf)
    out.update(angular_jack_D_mm=[round(float(z) * 1000, 1) if np.isfinite(z) else None for z in angD],
               sigma_ang_mm=sigma_ang * 1000)
    # bootstrap por bloques (10° x 5 cm), no por puntos
    uniq, inv = np.unique(blocks, axis=0, return_inverse=True)
    inv = inv.ravel()
    groups = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    bootD = []
    for _ in range(CYL_N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        fb = _fit_cylinder(Pi[idx], x, c0, a0, u0, v0, iters=15)
        if fb is not None:
            bootD.append(2 * abs(fb[0][4]))
    sigma_boot = float(np.std(bootD)) if len(bootD) > 5 else np.inf
    n = len(lobo)
    sigma_jack = (float(np.sqrt((n - 1) / n * np.sum((lobo - lobo.mean()) ** 2)))
                  if n >= 2 and np.all(np.isfinite(lobo)) else np.inf)
    sigma_prof = (max(ok_r) - min(ok_r)) / (2 * 1.96) if bounded else np.inf
    sigma_axis_term = float(np.hypot(abs(D - out["D_fixed_axis"]) / 2,
                                     D * (1 / np.cos(min(sigma_axis, np.radians(30))) - 1) / 2))
    # bootstrap por bloques, jackknife por bandas, perfil y jackknife angular son estimaciones de la MISMA
    # incertidumbre (variabilidad del radio con estos datos): se usa la mayor (conservador), no su suma.
    # El jackknife angular es el único que ve el error de forma de la sección (validado en Put 15 A4: arcos
    # parciales de un tubo bien medido daban errores de 2-3 U sin él).
    # A ella se añaden los términos de MODELO: sensibilidad al eje y escala (dentro de expanded()).
    sigma_stat = max(sigma_boot, sigma_jack, sigma_prof, sigma_ang)
    U = A.expanded(D, sigma_stat, sigma_axis_term)
    out.update(sigma_boot_mm=sigma_boot * 1000, sigma_jack_mm=sigma_jack * 1000, sigma_profile_mm=sigma_prof * 1000,
               sigma_stat_mm=sigma_stat * 1000, U_if_summed_mm=A.expanded(D, sigma_boot, sigma_jack, sigma_prof,
                                                                          sigma_axis_term) * 1000,
               sigma_axis_mm=sigma_axis_term * 1000, U=U, boot_D_range_mm=(
                   (round(float(min(bootD)) * 1000, 1), round(float(max(bootD)) * 1000, 1)) if bootD else None))
    # decisión (como máximo ESTIMATED: el tubo solo se ve parcialmente)
    why = None
    if tilt > CYL_MAX_TILT_DEG:
        why = f"el eje ajustado se inclina {tilt:.0f}° respecto al eje del tubo: orientación no determinada"
    elif arc < A.ARC_MIN_DEG:
        why = f"arco cubierto {arc}° < {A.ARC_MIN_DEG}°: sin información de curvatura"
    elif rms > BAND_MAX_RMS:
        why = (f"RMS {rms * 1000:.1f} mm > {BAND_MAX_RMS * 1000:.0f} mm (límite de banda): la superficie no se describe "
               "como un cilindro (ovalidad, deformación o superficies ajenas)")
    elif not bounded:
        why = "el perfil del radio no está acotado: radios muy distintos explican los puntos casi igual de bien"
    elif abs(trend) > tol and len(bands) >= 3:
        why = f"el residual cambia {trend * 1000:+.1f} mm con la profundidad: no es un cilindro (cono/transición)"
    elif cone:
        why = (f"el diámetro propio de las bandas cambia {2 * own_change * 1000:+.0f} mm con la profundidad "
               f"(t = {own_tstat:.1f}): no es un cilindro (cono/transición)")
    elif not np.isfinite(U) or n < 2:
        why = "sin estimación de variabilidad entre bandas (una sola banda)"
    elif len(own_r) < 3:
        why = f"solo {len(own_r)} bandas con radio propio fiable: no se puede distinguir un cilindro de un cono"
    elif U / D > A.D_UNKNOWN_REL:
        why = f"incertidumbre ±{U * 1000:.0f} mm > medio paso de tamaños nominales"
    # forma de cilindro aceptable (eje, RMS, perfil, tendencia): solo entonces el radio sirve como radio de DIBUJO
    out["shape_ok"] = bool(tilt <= CYL_MAX_TILT_DEG and arc >= A.ARC_MIN_DEG and rms <= BAND_MAX_RMS and bounded
                           and not (abs(trend) > tol and len(bands) >= 3) and not cone)
    if why:
        out["reason"] = why
        return out
    out["measured"] = True
    return out


def _apply_cylinder_fit(conn, cf):
    """Diámetro ESTIMATED por ajuste de cilindro 3D. Kruin/BOB no se derivan de él."""
    D, U = cf["D"], cf["U"]
    conn["diameter"] = A.meas(D, U, A.LAAG, "robust_3d_cylinder_fit", A.ESTIMATED,
                              arc_deg=cf["arc_deg"], bands=cf["n_bands"], profile_interval_mm=cf["profile_interval_mm"],
                              reason="tubo parcialmente visible: cilindro 3D robusto con perfil del radio acotado")
    conn["status"] = A.ESTIMATED
    conn["radius"] = D / 2
    conn["nominal"] = A.nominal_candidates((D - U) * 1000, (D + U) * 1000)
    if len(conn["nominal"]) > 3:
        conn["nominal"] = []
    reason = "no observada directamente: no se deriva del cilindro estimado (tubo parcial)"
    for k in ("bob", "axis_h", "bob_depth", "inner_height"):
        conn[k] = A.unknown("not_observed_partial_fit", reason=reason)
    conn["note"] = (f"diámetro estimado por cilindro 3D robusto (arco {cf['arc_deg']:.0f}°, "
                    f"{cf['n_bands']} bandas, {cf['visible_length'] * 100:.0f} cm)")


def visual_geometry(conn, fr, cf=None, band_radii=None):
    """Geometría SOLO para dibujar el tubo. Separada de la medida: si el diámetro es UNKNOWN, el radio de dibujo
    es provisional (visual_only = True) y nunca se exporta como diámetro.
    Prioridad del radio de dibujo: diámetro medido/estimado > cilindro 3D con forma aceptable (rechazado solo por
    incertidumbre) > mediana de los radios de bandas del tramo estable > tamaño de la abertura."""
    st = conn["diameter"]["status"]
    axis = np.asarray(conn["axis"], float)
    entry = np.asarray(conn["c0"], float)
    length = conn.get("tube_length_observed") or 0.0
    shape_ok = bool(cf and cf.get("ok_fit") and cf.get("shape_ok"))
    if cf and cf.get("ok_fit"):
        length = max(length, cf["t_range"][1])        # profundidad realmente observada de los puntos del tubo
    if shape_ok:
        axis = np.asarray(cf["axis"], float)
        if axis @ fr.n < 0:
            axis = -axis
        p = cf["axis_point"]
        entry = p - axis * (((p - fr.o) @ fr.n) / (axis @ fr.n))     # entrada en el plano de la pared
    if st in (A.MEASURED, A.ESTIMATED) and conn["diameter"]["value"] is not None:
        r, src, vo = conn["diameter"]["value"] / 2, ("measured_diameter" if st == A.MEASURED else "estimated_diameter"), False
    elif shape_ok and 0.02 < cf["D"] / 2 < 0.8:
        r, src, vo = cf["D"] / 2, "visual_only_3d_fit", True
    elif band_radii is not None and len(band_radii) >= 2:
        r, src, vo = float(np.median(band_radii)), "visual_only_band_radii", True
    else:
        op = conn["opening"]
        r, src, vo = max(op["width"], op["height"]) / 2, "visual_only_opening_size", True
    if length <= 0:
        length = conn.get("visible_len") or 0.05
    return dict(confirmed_pipe=conn["opening"]["status"] == "CONFIRMED", axis=axis.tolist(), entry=entry.tolist(),
                visible_length=float(length), display_radius=float(r), display_radius_source=src, visual_only=vo,
                note="SOLO VISUAL: no es una medida del diámetro" if vo else "")


def _apply_joint_fit(conn, jf, Q, c0, u, v):
    """Diámetro ESTIMATED (como máximo) del ajuste conjunto. Kruin y BOB NO se derivan del círculo estimado:
    la kruin queda como la determinó el perfil de bandas y la BOB queda UNKNOWN (no observada directamente)."""
    arc = jf["arc"]
    d = dict(arc["diameter"])
    d.update(status=A.ESTIMATED, conf=A.LAAG, method="joint_partial_pipe_bands", bands=len(jf["bands"]),
             sigma_jack=jf["sigma_jack"], sigma_orient=jf["sigma_orient"], combined_arc_deg=jf["arc_deg"],
             reason="radio común de bandas parciales; centro/radio de cada banda mal condicionados")
    conn["diameter"] = d
    conn["status"] = A.ESTIMATED
    conn["radius"] = arc["fit_r"]
    conn["center"] = c0 + u * arc["center_s"] + v * (arc["center_z"] - c0[2])
    conn.update({k: arc[k] for k in ("arc_deg", "coverage", "fit_r", "fit_rms", "center_s", "center_z",
                                     "arc_points", "arc_bins", "chord", "sagitta", "inliers", "r_chord")})
    conn["nominal"] = arc["nominal"]
    used = np.zeros(len(Q), bool)
    used[jf["used_idx"]] = True
    conn["used_mask"] = used
    conn["sz_all"] = np.column_stack([(Q[used] - c0) @ u, c0[2] + (Q[used] - c0) @ v])
    reason = "no observada directamente: no se deriva del círculo estimado (ajuste parcial)"
    conn["bob"] = A.unknown("not_observed_partial_fit", reason=reason)
    conn["axis_h"] = A.unknown("not_observed_partial_fit", reason=reason)
    conn["bob_depth"] = A.unknown("not_observed_partial_fit", reason=reason)
    conn["inner_height"] = A.unknown("not_observed_partial_fit", reason=reason)
    conn["note"] = (f"diámetro estimado por radio común de {len(jf['bands'])} bandas parciales "
                    f"(arco combinado {jf['arc_deg']:.0f}°)")


def build_connection(L, N, put, op, pm, rng):
    """Convierte apertura + perfil de bandas en un dict de aansluiting compatible con el resto del código."""
    fr, rows, run = op["frame"], pm["rows"], pm["run"]
    idx, c0, a = pm["idx"], pm["c0"], pm["axis"]
    Q, NQ = L[idx], N[idx]
    u, v = _frame_for_axis(a)
    rel = Q - c0
    t = rel @ a
    trans = np.abs(NQ @ a) > TRANSVERSE_N
    used = np.zeros(len(Q), bool)
    for i in run:
        used[rows[i]["inl_idx"]] = True   # solo puntos SOBRE el círculo de cada banda PIPE (índices locales)
    fill_any = any(b["label"] == "FILL" for b in rows)
    run_D = [2 * rows[i]["r"] for i in run]
    ax = pm["axis_fit"]
    # proyección final: perpendicular al eje; altura equivalente referida a la cara de la pared (t = 0)
    sz = np.column_stack([rel @ u, c0[2] + rel @ v])
    floor_s, top_z, top_s = put["floor_sigma"], put["top_z"], put["top_sigma"]
    conn = dict(points=Q, normals=NQ, indices=idx, wall=fr.key, wall_center=fr.point(*op["centroid_uz"]),
                opening=op, bands=rows, stable_run=run, axis=a, c0=c0, s_axis=u, v_axis=v,
                transverse_mask=trans, water_mask=pm["excl_water"], used_mask=used,
                bottom_mask=pm["excl_water"], possible_reconstruction_fill=bool(fill_any),
                opening_w=float(op["width"]), opening_h=float(op["height"]),
                visible_len=float(np.ptp(t)) if len(t) else 0.0,
                pipe_length_observed=float(rows[run[-1]]["t1"] - rows[run[0]]["t0"]) if run else 0.0,
                tube_length_observed=float(max([b["t1"] for b in rows if b["valid"]], default=0.0)
                                           - min([b["t0"] for b in rows if b["valid"]], default=0.0)),
                hidden_bottom=dict(n=int(pm["excl_water"].sum()), z=pm["water"]["z"] if pm["water"] else None),
                status=A.UNKNOWN, note="", direction=a.copy())
    # abertura: tamaño propio (nunca mezclado con el diámetro del tubo)
    circ = op["circle"]
    if circ is not None and circ["arc"] >= 180 and circ["rms"] <= 0.012:
        conn["opening_diameter"] = A.meas(2 * circ["r"], A.expanded(2 * circ["r"], 2 * circ["rms"]), A.MIDDEL,
                                          "wall_face_contour_circle", A.MEASURED if circ["arc"] >= 270 else A.ESTIMATED)
    else:
        # el círculo del contorno, si existe con curvatura suficiente (arco ≥ 60°), se conserva como intento
        att = (dict(attempt_value=2 * circ["r"], attempt_U=A.expanded(2 * circ["r"], 2 * circ["rms"]))
               if circ is not None and circ["arc"] >= A.ARC_MIN_DEG else {})
        conn["opening_diameter"] = A.unknown("wall_face_contour_circle",
                                             reason="contorno insuficiente para un círculo", **att)
    # borde superior de la abertura en la cara (geometría de la pared, NO la kruin del tubo)
    if len(op["edge_uv"]):
        cu, cz = op["centroid_uz"]
        near = (np.abs(op["edge_uv"][:, 0] - cu) < max(op["width"] * 0.15, 0.02)) & (op["edge_uv"][:, 1] > cz)
        if near.any():   # el punto de pared más bajo POR ENCIMA del hueco = borde superior de la abertura
            conn["opening_top"] = A.meas(float(op["edge_uv"][near, 1].min()), A.K_EXPAND * CELL_M / 2, A.MIDDEL,
                                         "wall_face_edge", A.MEASURED)

    # dirección/pendiente por centros de bandas
    if ax is not None and ax["span"] >= A.DIR_MIN_LENGTH_M and ax["res_max"] <= 0.008:
        dev = float(np.degrees(np.arctan2(np.cross(fr.n, ax["dir"])[2], fr.n @ ax["dir"])))
        U_az = A.K_EXPAND * ax["sigma_az"]
        conn["direction_m"] = A.meas(dev, U_az, A.HOOG if U_az <= A.DIR_MEASURED_U_DEG else A.LAAG, "band_centers_axis",
                                     A.MEASURED if U_az <= A.DIR_MEASURED_U_DEG else A.ESTIMATED, vector=ax["dir"].tolist())
        U_sl = A.K_EXPAND * ax["sigma_slope"]
        sl = float(np.degrees(np.arcsin(np.clip(ax["dir"][2], -1, 1))))
        if U_sl <= A.SLOPE_MEASURED_U_DEG:
            conn["slope"] = A.meas(sl, U_sl, A.HOOG, "band_centers_axis")
        elif U_sl <= A.SLOPE_ESTIMATED_U_DEG:
            conn["slope"] = A.meas(sl, U_sl, A.LAAG, "band_centers_axis", A.ESTIMATED)
        else:
            conn["slope"] = A.unknown("band_centers_axis", reason=f"±{U_sl:.1f}° > {A.SLOPE_ESTIMATED_U_DEG}°",
                                      attempt_value=sl, attempt_U=float(U_sl))
        sigma_axis = np.radians(max(ax["sigma_az"], ax["sigma_slope"]))
    else:
        why = ("menos de 3 bandas válidas" if ax is None else
               f"tramo con eje {ax['span'] * 100:.0f} cm < {A.DIR_MIN_LENGTH_M * 100:.0f} cm" if ax["span"] < A.DIR_MIN_LENGTH_M
               else f"centros no alineados ({ax['res_max'] * 1000:.0f} mm)")
        dev = float(np.degrees(np.arctan2(np.cross(fr.n, a)[2], fr.n @ a)))
        meth = "assumed_wall_normal" if abs(dev) < 0.5 else "band_centers_axis(unconfirmed)"
        conn["direction_m"] = A.meas(dev, None, A.LAAG, meth, A.ESTIMATED, vector=a.tolist(), reason=why)
        conn["slope"] = A.unknown("not_observable", reason=why)
        sigma_axis = np.radians(10.0)  # eje no confirmado: sensibilidad de orientación conservadora
    conn["axis_info"] = dict(axis=a, ratio=np.nan, length=float(ax["span"]) if ax else 0.0)

    # diámetro: SOLO bandas PIPE del tramo estable, y el tramo debe continuar DETRÁS de la banda de la cara
    # (si solo la cara es estable, el radio puede ser el del agujero/collar y no el del tubo)
    behind = [i for i in run if rows[i]["t0"] >= BAND_M - 1e-9]
    if len(run) >= 2 and len(behind) < 2:
        conn["note"] = "solo la zona de la cara de la pared es estable: puede ser la abertura, no el tubo"
        run = []
        conn["stable_run"] = run
        for b in rows:
            if b["label"] == "PIPE":
                b["label"], b["reason"] = "OPENING_RING", "tramo estable limitado a la cara de la pared"
        used[:] = False
    if len(run) >= 2 and used.sum() >= 12:
        sigma_bands = float(np.std(run_D, ddof=1))
        D0 = float(np.median(run_D))
        sigma_orient = D0 * (1 / np.cos(min(sigma_axis, np.radians(30))) - 1) / 2
        # superficie del tubo para kruin/BOB: puntos de las bandas PIPE dentro de una ventana radial alrededor
        # del círculo de su banda (admite ovalidad; excluye superficies ajenas como otro tubo cercano)
        ext_mask = np.zeros(len(Q), bool)
        for i in run:
            b = rows[i]
            ii = b["idx"]
            cb = b["center"]
            relb = Q[ii] - cb
            rad = np.linalg.norm(relb - np.outer(relb @ a, a), axis=1)
            ext_mask[ii[np.abs(rad - b["r"]) < max(0.03, 0.15 * b["r"])]] = True
        conn["extent_mask"] = ext_mask
        arc = A._measure_arc(sz[used], rng, floor_s, top_z, top_s,
                             extra_sigma_d=float(np.hypot(sigma_bands, sigma_orient)), sz_extent=sz[ext_mask])
        conn.update(arc)
        if len(run) < 3 and arc["diameter"]["status"] == A.MEASURED:
            conn["diameter"] = A.cap_status(arc["diameter"], A.ESTIMATED)
            conn["diameter"]["reason"] = "solo 2 bandas estables"
            conn["status"] = A.ESTIMATED
        conn["diameter"]["bands"] = len(run)
        conn["diameter"]["sigma_bands"] = sigma_bands
        conn["center"] = c0 + u * arc["center_s"] + v * (arc["center_z"] - c0[2])
        conn["radius"] = arc["fit_r"]
        conn["sz_all"] = sz[used]
        if arc["status"] == A.UNKNOWN:
            conn["note"] = arc["diameter"].get("reason", "")
        # (fase 7) el antiguo fallback "stable_partial_pipe_bands" estaba aquí: solo se ejecutaba con un tramo
        # estable ya existente (inútil para tubos sin tramo estable) y promediaba diámetros de bandas
        # individuales. Se ha sustituido por _joint_partial_pipe_fit en la rama sin tramo estable.
    else:
        why = conn["note"] or ("bandas incompatibles: ningún tramo de radio constante" if sum(b["valid"] for b in rows) >= 2
                               else "menos de 2 bandas válidas detrás de la abertura")
        conn["note"] = why
        # sin tramo de tubo estable: la kruin solo se acepta si la corona se mantiene a la misma altura en
        # ≥ 2 bandas consecutivas (superficie de tubo); el borde superior del agujero en la cara NO es la kruin
        usable = ~pm["excl_water"] & ~trans
        if usable.sum() >= 12:
            arc = A._measure_arc(sz[usable], rng, floor_s, top_z, top_s)
            conn.update({k: arc[k] for k in ("arc_deg", "coverage", "fit_r", "fit_rms", "center_s", "center_z",
                                             "arc_points", "arc_bins", "chord", "sagitta", "inliers", "r_chord")})
        else:
            conn.update(arc_deg=0.0, coverage=0.0)
        conn["crown"] = crown_from_profile(rows, rel, u, v, c0, floor_s)
        conn["diameter"] = A.unknown("pipe_bands", reason=why)
        for k in ("bob", "axis_h", "bob_depth", "inner_height"):
            conn[k] = A.unknown("needs_diameter")
        conn.update(nominal=[], bottom_seen=False, status=A.UNKNOWN)
        conn["sz_all"] = sz[usable] if usable.any() else np.zeros((0, 2))
        # fallback: ajuste CONJUNTO de un único círculo a las bandas parciales detrás de la cara (ver función)
        jf = _joint_partial_pipe_fit(rows, Q, c0, a, u, v, rng, floor_s, top_z, top_s, sigma_axis)
        conn["joint_fit"] = jf
        if jf["ok"]:
            _apply_joint_fit(conn, jf, Q, c0, u, v)
    # ajuste de cilindro 3D robusto: siempre como diagnóstico (comparación de métodos); fija el diámetro SOLO si los
    # métodos anteriores lo dejaron UNKNOWN y el cilindro es identificable (perfil acotado, LOBO, bootstrap, R10)
    excl3d = pm["excl_water"] | trans
    sub = np.ones(len(Q), bool)
    if len(Q) > 4000:   # rendimiento: submuestra determinista
        sub[:] = False
        sub[np.random.default_rng(1).choice(len(Q), 4000, replace=False)] = True
    cf = pipe_cylinder_3d(Q[sub], NQ[sub], c0, a, excl3d[sub], rng, sigma_axis)
    conn["cylinder_fit"] = cf
    if conn["diameter"]["value"] is None and cf.get("measured"):
        _apply_cylinder_fit(conn, cf)
        p_ax, a_ax = cf["axis_point"], np.asarray(cf["axis"])
        conn["center"] = p_ax - a_ax * (((p_ax - fr.o) @ fr.n) / (a_ax @ fr.n))
    elif conn["diameter"]["value"] is None and cf.get("ok_fit"):
        conn["note"] = (conn["note"] + "; " if conn["note"] else "") + f"cilindro 3D: {cf['reason']}"
    conn["visual_geometry"] = visual_geometry(conn, fr, cf, [rows[i]["r"] for i in run if "r" in rows[i]])
    if conn["diameter"]["value"] is None:
        # sin diámetro no hay sugerencia nominal: un intervalo de un ajuste rechazado no es base para sugerir tamaños
        conn["nominal"] = []
    # agua: ¿la parte inferior del tubo está oculta?
    wz = pm["water"]["z"] if pm["water"] else None
    if wz is not None and conn.get("radius") is not None and conn["diameter"]["value"] is not None:
        bottom_z = conn["center"][2] - conn["radius"]
        conn["lower_pipe_occluded"] = bool(wz > bottom_z + 0.01)
    else:
        conn["lower_pipe_occluded"] = wz is not None
    if conn["lower_pipe_occluded"] and conn["bob"]["status"] == A.MEASURED and conn["bob"]["method"] == "lowest_point_observed":
        conn["bob"] = A.unknown("lower_pipe_occluded")  # bajo el agua no hay geometría directa del fondo
    return conn


def rejected_behind_clusters(L, N, put, rect, conns, openings):
    """Grupos detrás de la pared (método anterior) que NO se explican por una abertura confirmada."""
    from analysis import _outside_distance  # noqa
    search = put.get("connection_search", {})
    idx, labels = search.get("indices"), search.get("labels")
    out = []
    if idx is None or labels is None or not len(labels):
        return out
    claimed = np.zeros(len(L), bool)
    for c in conns:
        claimed[c["indices"]] = True
    walls = {k: WallFrame(k, w, put["chamber"], put["floor_z"]) for k, w in (put["walls"]["walls"] or {}).items()}
    for lab in range(labels.max() + 1):
        ids = idx[labels == lab]
        if len(ids) < A.PIPE_MIN_POINTS or claimed[ids].mean() > 0.5:
            continue
        Q, NQ = L[ids], N[ids]
        if np.mean(np.abs(NQ[:, 2]) > A.FLOOR_NZ_MIN) > 0.7 and A._is_flat(Q):
            continue  # rellano / terreno horizontal: nunca fue candidato a conexión
        m = Q[:, :2].mean(axis=0)
        key = max(walls, key=lambda k: m @ walls[k].n[:2]) if walls else None
        rel_info = {}
        kind = "UNASSOCIATED"
        reason = "puntos detrás de la pared sin abertura confirmada en la cara"
        if key:
            fr = walls[key]
            _, _, d = fr.coords(Q)
            is_rel, rel_info = _is_relief(Q, NQ, fr.n, d)
            if is_rel:
                kind = "WALL_RELIEF"
                reason = (f"plano paralelo a la pared ({rel_info['parallel']:.0%} de normales paralelas), "
                          f"espesor {rel_info['spread'] * 100:.1f} cm, a {rel_info['depth'] * 100:.0f} cm detrás: "
                          "relieve, no tubo")
        out.append(dict(indices=ids, kind=kind, wall=key, reason=reason, relief=rel_info))
    return out


def detect_connections_v2(L, N, put, rect, rng):
    """Pipeline nuevo. Devuelve (conexiones, candidatos de abertura, rechazados)."""
    openings = detect_openings(L, N, put)
    conns = []
    for op in openings:
        if op["status"] != "CONFIRMED":
            continue
        pm = measure_pipe(L, N, put, op, rng)
        if not len(pm["idx"]):
            op["status"], op["reason"] = "POSSIBLE", "abertura sin puntos detrás tras el seguimiento"
            continue
        c = build_connection(L, N, put, op, pm, rng)
        ref = c.get("center", c["wall_center"])
        ch = put["chamber"]
        c["angle_deg"] = float(np.degrees(np.arctan2(ref[1] - ch["cy"], ref[0] - ch["cx"])) % 360)
        conns.append(c)
    conns.sort(key=lambda c: c["angle_deg"])
    for i, c in enumerate(conns):
        c["id"] = f"A{i + 1}"
    rejected = rejected_behind_clusters(L, N, put, rect, conns, openings)
    return conns, openings, rejected
