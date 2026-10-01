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
        conn["opening_diameter"] = A.unknown("wall_face_contour_circle",
                                             reason="contorno insuficiente para un círculo")
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
            conn["slope"] = A.unknown("band_centers_axis", reason=f"±{U_sl:.1f}° > {A.SLOPE_ESTIMATED_U_DEG}°")
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

            # Fallback conservador: varias bandas parciales coherentes pueden
            # estimar el tubo sin usar el tamano de la abertura de pared.
            partial = [rows[i] for i in run
                       if rows[i].get("valid")
                       and rows[i].get("arc", 0) >= BAND_MIN_ARC
                       and rows[i].get("n_inl", 0) >= 15
                       and rows[i].get("rms", np.inf) <= BAND_MAX_RMS]
            if len(partial) >= 3:
                Ds = np.array([2.0 * b["r"] for b in partial], dtype=float)
                Dp = float(np.median(Ds))
                spread = float(np.std(Ds, ddof=1))
                med_arc = float(np.median([b["arc"] for b in partial]))
                max_dev = float(np.max(np.abs(Ds - Dp)))
                tol_d = 2.0 * max(STABLE_TOL_ABS, STABLE_TOL_REL * (Dp / 2.0))
                if max_dev <= tol_d + 1e-12:
                    partial_penalty = Dp * 0.05 * max(0.0, min(1.0, (180.0 - med_arc) / 90.0))
                    rms_d = 2.0 * float(np.median([b["rms"] for b in partial]))
                    sigma = float(np.hypot(np.hypot(spread, rms_d), np.hypot(partial_penalty, sigma_orient)))
                    U = A.expanded(Dp, sigma)
                    if np.isfinite(U) and Dp > 0 and U / Dp <= 0.15:
                        conn["diameter"] = A.meas(Dp, U, A.LAAG, "stable_partial_pipe_bands", A.ESTIMATED,
                                                  bands=len(partial), median_arc_deg=med_arc,
                                                  sigma_bands=spread,
                                                  reason="diameter estimated from coherent partial pipe bands")
                        conn["status"] = A.ESTIMATED
                        conn["radius"] = Dp / 2.0
                        conn["note"] = (f"geschat uit {len(partial)} consistente gedeeltelijke buisbanden; "
                                        f"mediaan boog {med_arc:.0f} graden")
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
