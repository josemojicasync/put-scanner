"""DIAGNÓSTICO de las mediciones: qué geometría usa el algoritmo para cada medida y por qué da ese valor.

No cambia ningún resultado del análisis: solo lee lo que analysis.py registró y lo explica.

Uso:
    python diagnose.py data/scan.ply            -> informe en consola y en results/debug/<scan>_diagnose.txt
    python debug_geometry.py data/scan.ply      -> visor 3D DEBUG GEOMETRY (usa este módulo)

Pruebas principales:
  * residuos firmados de TODOS los candidatos respecto al modelo (no solo el RMS de los inliers);
  * "segunda superficie": RANSAC sobre los puntos rechazados. Si existe otra superficie paralela con
    muchos puntos, el RMS bajo del ajuste no garantiza que se eligió la superficie correcta;
  * perfil por alturas: el plano se reajusta por bandas de 20 cm (escalones, cuello, inclinación);
  * contaminación: fracción de puntos de una región que pertenecen a otra (tubos, banket, marco...);
  * candidatos a conexión: características geométricas y propuesta CANDIDATE/CONFIRMED (solo diagnóstico).
"""
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

import analysis as A

ALT_MIN_FRACTION = 0.15   # una segunda superficie con ≥ 15 % de los puntos del ajuste principal es relevante
BAND_M = 0.20


# ---------------------------------------------------------------- utilidades

def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def residual_stats(P, n, c, thresh):
    d = (P - c) @ n
    return dict(d=d, p01=pct(d, 1), p05=pct(d, 5), p50=pct(d, 50), p95=pct(d, 95), p99=pct(d, 99),
                out_neg=float(np.mean(d < -thresh)) if len(d) else 0.0,
                out_pos=float(np.mean(d > thresh)) if len(d) else 0.0)


def second_surface(P, inl, n, c, thresh, rng, parallel_deg=10.0):
    """RANSAC sobre los candidatos rechazados: ¿hay otra superficie (casi) paralela con muchos puntos?"""
    rej = P[~inl]
    if len(rej) < max(50, ALT_MIN_FRACTION * inl.sum()):
        return None
    n2, c2, inl2, rms2 = A.fit_plane_ransac(rej, thresh, rng, n)
    ang = float(np.degrees(np.arccos(np.clip(abs(n2 @ n), -1, 1))))
    if inl2.sum() < ALT_MIN_FRACTION * inl.sum() or ang > parallel_deg:
        return None
    return dict(n=n2, c=c2, points=rej[inl2], n_points=int(inl2.sum()), rms=rms2, angle_deg=ang,
                offset_m=float((c2 - c) @ n), fraction=float(inl2.sum() / inl.sum()))


def band_profile(P, n, c, z0, z1, axis_coord=None):
    """Posición del plano por bandas de altura (offset firmado respecto al plano global)."""
    rows = []
    for zb in np.arange(z0, z1, BAND_M):
        m = (P[:, 2] >= zb) & (P[:, 2] < zb + BAND_M)
        if m.sum() < 30:
            rows.append((zb, None, int(m.sum())))
            continue
        rows.append((zb, float(np.median((P[m] - c) @ n)), int(m.sum())))
    return rows


def near_connections(P, res, wall_key=None, margin=0.15):
    """Máscara de puntos cerca de alguna abertura de tubo (en el plano de la pared)."""
    m = np.zeros(len(P), bool)
    for cn in res["connections"]:
        if wall_key is not None and cn["wall"] != wall_key:
            continue
        wc = cn["wall_center"]
        rad = max(cn["opening_w"], cn["opening_h"]) / 2 + margin
        m |= np.linalg.norm(P - wc, axis=1) < rad + 0.6  # incluye el tubo detrás de la pared
    return m


def plane_x_at(n, c, ax, other_val, z, other_ax):
    """Coordenada `ax` del plano en el punto (other_ax=other_val, z)."""
    rest = [i for i in range(3) if i != ax]
    val = {other_ax: other_val, 2: z}
    s = sum(n[i] * (val[i] - c[i]) for i in rest)
    return c[ax] - s / n[ax]


# ---------------------------------------------------------------- diagnóstico por región

def diag_walls(res, rng):
    put = res["put"]
    ch = put["chamber"]
    out, geo = {}, {}
    walls = put["walls"]["walls"] or {}
    for key, ax, sign in A.WALLS:
        lines = [f"WALL {key}"]
        if key not in walls:
            dm = put["walls"].get("debug_missing", {}).get(key)
            lines.append(f"  NO usada: {dm['reason'] if dm else 'sin candidatos'}")
            out[f"WALL {key}"] = lines
            geo[f"WALL {key}"] = dict(candidates=dm["candidates"] if dm else np.zeros((0, 3)))
            continue
        w = walls[key]
        P, inl, n, c = w["candidates"], w["inlier_mask"], w["n"], w["c"]
        sel = w["select"]
        rs = residual_stats(P, n, c, A.PLANE_THRESH_M)
        alt = second_surface(P, inl, n, c, A.PLANE_THRESH_M, rng)
        prof = band_profile(P[inl], n, c, sel["z_range"][0], sel["z_range"][1])
        contam_pipe = float(np.mean(near_connections(P[inl], res, key))) if inl.any() else 0.0
        z_top_frac = float(np.mean(P[inl][:, 2] > ch["z1"] - 0.1)) if inl.any() else 0.0
        pos = float(c @ n)
        lines += [
            f"  selección: normal ∥ ±{'XY'[ax]} (|n|>{A.SIDE_NORMAL_MIN}), lado {'+' if sign > 0 else '-'}, "
            f"z {sel['z_range'][0]:.2f}..{sel['z_range'][1]:.2f} m (tramo cámara ± 5 cm)",
            f"  candidatos {len(P)}  usados {int(inl.sum())}  rechazados {int((~inl).sum())}",
            f"  RMS {w['rms'] * 1000:.1f} mm   normal {np.round(n, 4).tolist()}   posición n·c = {pos * 1000:.1f} mm",
            f"  inclinación respecto al eje vertical: {np.degrees(np.arcsin(abs(n[2]))):.2f}°",
            f"  residuos candidatos (mm): p1 {rs['p01'] * 1000:+.0f}  p5 {rs['p05'] * 1000:+.0f}  p50 {rs['p50'] * 1000:+.1f}"
            f"  p95 {rs['p95'] * 1000:+.0f}  p99 {rs['p99'] * 1000:+.0f}   fuera: hacia dentro {rs['out_neg']:.0%} / hacia fuera {rs['out_pos']:.0%}",
            f"  contaminación: {contam_pipe:.0%} de los inliers cerca de una abertura de tubo; "
            f"{z_top_frac:.0%} en los 10 cm superiores del tramo (posible cuello/marco)",
        ]
        if alt:
            lines.append(f"  ¡SEGUNDA SUPERFICIE! {alt['n_points']} ptn ({alt['fraction']:.0%} del ajuste) paralela "
                         f"({alt['angle_deg']:.1f}°) a {alt['offset_m'] * 1000:+.0f} mm "
                         f"({'hacia fuera' if alt['offset_m'] > 0 else 'hacia dentro'}), RMS {alt['rms'] * 1000:.1f} mm")
        else:
            lines.append("  segunda superficie: no (los rechazados no forman otro plano paralelo relevante)")
        lines.append("  perfil por altura (offset mediano del plano local respecto al global):")
        for zb, off, cnt in prof:
            lines.append(f"    z {zb:5.2f}-{zb + BAND_M:4.2f}: " + ("—" if off is None else f"{off * 1000:+6.1f} mm") + f"  ({cnt} ptn)")
        out[f"WALL {key}"] = lines
        geo[f"WALL {key}"] = dict(candidates=P, inlier_mask=inl, n=n, c=c, residuals=rs["d"], alt=alt,
                                  profile=prof)
    return out, geo


def diag_width(res, axis_name):
    put = res["put"]
    ch = put["chamber"]
    walls = put["walls"]["walls"] or {}
    a, b = f"+{axis_name}", f"-{axis_name}"
    ax = 0 if axis_name == "X" else 1
    other = 1 - ax
    m = put["width_x" if ax == 0 else "width_y"]
    lines = [f"Binnenmaat {axis_name} = " + ("UNKNOWN" if m["value"] is None else f"{m['value'] * 1000:.1f} ± {m['U'] * 1000:.1f} mm")
             + f"   [{m['status']}, {m['method']}]"]
    seg = None
    if a in walls and b in walls:
        wa, wb = walls[a], walls[b]
        zmid = (ch["z0"] + ch["z1"]) / 2
        xa = plane_x_at(wa["n"], wa["c"], ax, ch["cy"] if ax == 0 else ch["cx"], zmid, other)
        xb = plane_x_at(wb["n"], wb["c"], ax, ch["cy"] if ax == 0 else ch["cx"], zmid, other)
        nn = wa["n"] - wb["n"]; nn /= np.linalg.norm(nn)
        for k, w in ((a, wa), (b, wb)):
            lines.append(f"  Plano {k}: {w['n_used']} ptn, RMS {w['rms'] * 1000:.1f} mm, normal {np.round(w['n'], 4).tolist()}, "
                         f"centroide {np.round(w['c'] * 1000, 0).tolist()} mm")
        lines.append(f"  método usado: (centroide {a} − centroide {b}) · normal media = {((wa['c'] - wb['c']) @ nn) * 1000:.1f} mm")
        lines.append(f"  distancia en el centro de la cámara (z={zmid:.2f} m): {abs(xa - xb) * 1000:.1f} mm")
        zs = np.arange(ch["z0"] + 0.1, ch["z1"] - 0.05, 0.2)
        ws = [abs(plane_x_at(wa["n"], wa["c"], ax, 0.0, z, other) - plane_x_at(wb["n"], wb["c"], ax, 0.0, z, other)) for z in zs]
        lines.append(f"  distancia entre planos según altura: " + ", ".join(f"z{z:.1f}:{v * 1000:.0f}" for z, v in zip(zs, ws)))
        lines.append(f"  paralelismo {np.degrees(np.arccos(np.clip(abs(wa['n'] @ wb['n']), -1, 1))):.2f}° "
                     "(planos no paralelos -> la 'anchura' depende de dónde se mide)")
        p0 = np.zeros(3); p1 = np.zeros(3)
        p0[ax], p1[ax] = xb, xa
        p0[other] = p1[other] = ch["cy"] if ax == 0 else ch["cx"]
        p0[2] = p1[2] = zmid
        seg = (p0, p1)
    else:
        lines.append(f"  falta {a if a not in walls else b}: valor estimado por extensión de las paredes adyacentes")
    return lines, seg


def diag_surface(res, which, rng):
    put = res["put"]
    info = put["floor_info"] if which == "BOTTOM" else put["ground_info"]
    if info is None:
        return [which, "  no encontrado"], {}
    P, inl, n, c = info["candidates"], info["inlier_mask"], info["normal"], info["point"]
    rs = residual_stats(P, n, c, A.SURFACE_THRESH_M)
    alt = second_surface(P, inl, n, c, A.SURFACE_THRESH_M, rng)
    ch = put["chamber"]
    ex, ey = info["eval_xy"]
    zc = A.plane_height_at(n, c, ex, ey)
    z_hist = np.histogram(P[:, 2], bins=np.arange(P[:, 2].min(), P[:, 2].max() + 0.02, 0.02))
    lines = [which]
    if which == "BOTTOM":
        lines.append("  selección: normal vertical (|n_z|>0.85), dentro del 90 % de la huella, z < z0 cámara + 0,3 m")
        # franja del canal: alineada con un tubo que llega al bodem (dentro de su anchura de abertura)
        chan_in, chan_cand = np.zeros(int(inl.sum()), bool), np.zeros(len(P), bool)
        for cn in res["connections"]:
            if cn["points"][:, 2].min() < put["floor_z"] + 0.10:
                for M, Pp in ((chan_in, P[inl]), (chan_cand, P)):
                    M |= np.abs((Pp - cn["wall_center"]) @ cn["s_axis"]) < cn["opening_w"] / 2
        lines.append(f"  contaminación: {np.mean(chan_in):.0%} de los inliers (y {np.mean(chan_cand):.0%} de los candidatos) "
                     "en la franja de un tubo que llega al bodem (canal/agua)")
        for lvl_name, sel in (("inliers en franja de canal", chan_in), ("inliers fuera de la franja", ~chan_in)):
            if sel.any():
                lines.append(f"    altura mediana {lvl_name}: {np.median(P[inl][sel][:, 2]) * 1000:+.0f} mm")
    else:
        lines.append("  selección: normal vertical, > 5 cm fuera de la huella de la cámara, z > borde superior de pared − 0,3 m")
        rad = np.hypot(P[inl][:, 0], P[inl][:, 1]) if inl.any() else np.zeros(0)
        top = max(put["sections"], key=lambda s: s["z1"])
        r_open = max(top["hx"], top["hy"]) * 1.42
        lines.append(f"  distancia de los inliers al eje: p5 {pct(rad, 5):.2f} m  p50 {pct(rad, 50):.2f} m  p95 {pct(rad, 95):.2f} m"
                     f"  (abertura superior ≈ {2 * top['hx'] * 1000:.0f} x {2 * top['hy'] * 1000:.0f} mm, cámara {2 * ch['hx'] * 1000:.0f} mm)")
        near_rim = float(np.mean(rad < r_open + 0.15)) if len(rad) else 0
        lines.append(f"  {near_rim:.0%} de los inliers a < 15 cm del marco superior (posible marco/tapa en vez de terreno)")
        lines.append(f"  NOTA: la región empieza en la huella de la CÁMARA ({ch['hx'] * 1000:.0f} mm del eje), no en la abertura "
                     f"superior; la losa/marco entre ambos {'puede' if r_open < ch['hx'] else 'no'} quedar fuera")
    lines += [
        f"  candidatos {len(P)}  usados {int(inl.sum())}  RMS {info['rms'] * 1000:.1f} mm  inclinación {info['slope_deg']:.2f}°",
        f"  normal {np.round(n, 4).tolist()}",
        f"  altura evaluada en (x,y)=({ex * 1000:.0f}, {ey * 1000:.0f}) mm: z = {zc * 1000:.1f} mm",
        f"  residuos candidatos (mm): p1 {rs['p01'] * 1000:+.0f}  p50 {rs['p50'] * 1000:+.1f}  p99 {rs['p99'] * 1000:+.0f}"
        f"   fuera: debajo {rs['out_neg']:.0%} / encima {rs['out_pos']:.0%}",
        "  histograma de alturas de candidatos (bins 2 cm): " + " ".join(
            f"{e * 1000:.0f}:{h}" for e, h in zip(z_hist[1][:-1], z_hist[0]) if h > 0.02 * len(P)),
    ]
    if alt:
        lines.append(f"  ¡SEGUNDA SUPERFICIE! {alt['n_points']} ptn ({alt['fraction']:.0%}) a {alt['offset_m'] * 1000:+.0f} mm "
                     f"({'encima' if alt['offset_m'] * n[2] > 0 else 'debajo'}), RMS {alt['rms'] * 1000:.1f} mm")
    else:
        lines.append("  segunda superficie: no")
    return lines, dict(candidates=P, inlier_mask=inl, n=n, c=c, residuals=rs["d"], alt=alt, eval=(ex, ey, zc))


def diag_depth(res, bottom_geo, ground_geo):
    put = res["put"]
    m = put["depth"]
    lines = [f"Diepte = {m['value'] * 1000:.1f} ± {m['U'] * 1000:.1f} mm   [{m['status']}, {m['method']}]"]
    seg = None
    if bottom_geo and ground_geo:
        ex, ey, zb = bottom_geo["eval"]
        _, _, zt = ground_geo["eval"]
        lines += [f"  plano bodem:    z = {zb * 1000:.1f} mm en ({ex * 1000:.0f}, {ey * 1000:.0f}) mm",
                  f"  plano maaiveld: z = {zt * 1000:.1f} mm en el mismo (x,y) — EXTRAPOLADO sobre la abertura",
                  f"  diepte = {(zt - zb) * 1000:.1f} mm (vertical, a lo largo del eje estimado de la put)"]
        gi = put["ground_info"]
        lines.append(f"  el maaiveld tiene inclinación {gi['slope_deg']:.2f}°: en la anchura de la put (≈{2 * put['chamber']['hx']:.2f} m) "
                     f"su altura varía ±{np.tan(np.radians(gi['slope_deg'])) * put['chamber']['hx'] * 1000:.0f} mm")
        seg = (np.array([ex, ey, zb]), np.array([ex, ey, zt]))
    else:
        lines.append("  sin maaiveld: diepte medida hasta el borde superior de la pared")
    return lines, seg


# ---------------------------------------------------------------- candidatos a conexión

def candidate_features(res, Q, NQ, conn=None):
    put = res["put"]
    ch = put["chamber"]
    rect = res["rect"]
    out_d = A._outside_distance(Q[:, :2], ch, rect)
    m = Q[:, :2].mean(axis=0) - [ch["cx"], ch["cy"]]
    if rect:
        k = int(np.argmax(np.abs(m) - np.array([ch["hx"], ch["hy"]])))
        d = np.zeros(3); d[k] = np.sign(m[k])
    else:
        d = np.array([m[0], m[1], 0.0]) / np.linalg.norm(m)
    s_axis = np.array([-d[1], d[0], 0.0])
    s, z, t = Q @ s_axis, Q[:, 2], Q @ d
    vox = np.unique(np.floor(Q / 0.01).astype(int), axis=0)
    sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Q)).cluster_dbscan(0.02, 5)
    sub = np.array(sub)
    comps = np.bincount(sub[sub >= 0]) if (sub >= 0).any() else np.array([0])
    # curvatura en la sección (s, z)
    sz = np.column_stack([s, z])
    Al = np.column_stack([s, np.ones(len(s))])
    coef, *_ = np.linalg.lstsq(Al, z, rcond=None)
    line_rms = float(np.sqrt(np.mean((Al @ coef - z) ** 2)))
    cx, cy, r = A.fit_circle_2d(sz)
    cx, cy, r = A.fit_circle_geometric(sz, cx, cy, r)
    circ_rms = float(np.sqrt(np.mean((np.hypot(sz[:, 0] - cx, sz[:, 1] - cy) - r) ** 2)))
    flat_low = (np.abs(NQ[:, 2]) > A.FLOOR_NZ_MIN) & (z < put["floor_z"] + 0.05)
    wall_like = (np.abs(NQ @ d) > 0.9) & (out_d < 0.06)
    corner = min(abs(abs(Q[:, 1 - k].mean() - (ch["cy"] if k == 0 else ch["cx"])) - (ch["hy"] if k == 0 else ch["hx"])), 9) if rect else np.nan
    f = dict(
        n=len(Q), width=float(np.ptp(np.percentile(s, [2, 98]))), height=float(np.ptp(np.percentile(z, [2, 98]))),
        depth=float(np.ptp(np.percentile(t, [2, 98]))), area_m2=len(vox) * 1e-4,
        bbox_vol_m3=float(np.prod(np.ptp(Q, axis=0))),
        components=int(len(comps)), largest_frac=float(comps.max() / max(len(Q), 1)),
        line_rms=line_rms, circle_rms=circ_rms, circle_r=float(r),
        curvature_gain=float(line_rms / max(circ_rms, 1e-4)),
        penetration_p50=pct(out_d, 50), penetration_p95=pct(out_d, 95),
        z_min=float(z.min()), z_max=float(z.max()), dist_floor=float(z.min() - put["floor_z"]),
        dist_chamber_top=float(ch["z1"] - z.max()), dist_corner=float(corner),
        frac_flat_low=float(np.mean(flat_low)), frac_wall_like=float(np.mean(wall_like)),
        wall=("-" if d[k] < 0 else "+") + "XY"[k] if rect else "wand", direction=d, s_axis=s_axis,
        visible_arc=None if conn is None else conn.get("arc_deg"),
    )
    # PROPUESTA de clasificación (solo diagnóstico; no cambia el análisis):
    reasons = []
    if f["penetration_p95"] < 0.08:
        reasons.append(f"penetra solo {f['penetration_p95'] * 100:.0f} cm detrás de la pared (< 8 cm)")
    if f["curvature_gain"] < 1.5 or not (0.04 < f["circle_r"] < 0.8):
        reasons.append(f"sin curvatura de tubo (recta/círculo {f['curvature_gain']:.1f}, r={f['circle_r'] * 1000:.0f} mm)")
    if f["largest_frac"] < 0.8:
        reasons.append(f"discontinuo ({f['components']} fragmentos, mayor {f['largest_frac']:.0%})")
    if f["frac_wall_like"] > 0.3:
        reasons.append(f"{f['frac_wall_like']:.0%} de puntos con orientación de pared (relieve de la pared)")
    if f["frac_flat_low"] > 0.5:
        reasons.append(f"{f['frac_flat_low']:.0%} horizontal a nivel de bodem (banket/agua)")
    if conn is not None and conn.get("arc_deg", 0) < A.ARC_MIN_DEG:
        reasons.append(f"arco visible {conn.get('arc_deg', 0):.0f}° < {A.ARC_MIN_DEG}°")
    f["proposal"] = "CONFIRMED" if not reasons else "CANDIDATE"
    f["proposal_reasons"] = reasons
    return f


def _m(m, unit="mm"):
    if m is None or m.get("value") is None:
        return "UNKNOWN" + (f" ({m.get('reason')})" if m and m.get("reason") else "")
    if unit == "deg":
        return f"{m['value']:.1f}°" + ("" if m["U"] is None else f" ± {m['U']:.1f}°") + f" [{m['status']}, {m['method']}]"
    return f"{m['value'] * 1000:.0f}" + ("" if m["U"] is None else f" ± {m['U'] * 1000:.0f}") + f" mm [{m['status']}, {m['method']}]"


def opening_lines(op):
    circ = op.get("circle")
    return [f"  ABERTURA en la cara de la pared {op['wall']}: {op['status']} — {op['reason']}",
            f"    centro (u,z)=({op['centroid_uz'][0] * 1000:.0f}, {op['centroid_uz'][1] * 1000:.0f}) mm, "
            f"{op['width'] * 1000:.0f} x {op['height'] * 1000:.0f} mm, área {op['area'] * 1e4:.0f} cm²",
            f"    contorno apoyado en pared {op['contour']:.0%} (sin contar el borde en la base), {op['edge_points']} ptn de borde, "
            f"{len(op['behind_idx'])} ptn detrás, toca bodem {op['touches_floor']}, toca esquina {op['touches_edge']}",
            "    ajuste del contorno: " + ("—" if circ is None else
                                          f"círculo r={circ['r'] * 1000:.0f} mm, rms {circ['rms'] * 1000:.1f} mm, arco {circ['arc']:.0f}°")]


def diag_connections(res):
    """Pipeline nuevo: abertura -> bandas -> tramo estable -> medidas. Explica qué puntos usa cada medida."""
    L, N = res["local_points"], res["local_normals"]
    put = res["put"]
    out, feats = {}, {}
    for cn in res["connections"]:
        if "bands" not in cn:   # put redonda: método anterior
            out[cn["id"]] = [f"{cn['id']} (método anterior, put redonda)"]
            continue
        lines = [f"{cn['id']} — pared {cn['wall']}, ángulo {cn['angle_deg']:.0f}°"]
        lines += opening_lines(cn["opening"])
        lines.append(f"    Ø abertura: {_m(cn.get('opening_diameter'))}   borde superior: {_m(cn.get('opening_top'))}")
        lines.append(f"  EJE: {np.round(cn['axis'], 3).tolist()}  entrada {np.round(cn['c0'] * 1000, 0).tolist()} mm")
        lines.append("  BANDAS (profundidad desde la cara | etiqueta | ptn útiles | Ø | arco | rms | contaminación | razón):")
        for b in cn["bands"]:
            lines.append(f"    {b['t0'] * 100:4.0f}-{b['t1'] * 100:3.0f} cm  {b['label']:12s} {b['n_use']:5d}  "
                         + (f"Ø{2 * b['r'] * 1000:5.0f}  {b['arc']:4.0f}°  {b['rms'] * 1000:4.1f} mm  {b['contamination']:.0%}"
                            if "r" in b else " " * 34) + f"  {b['reason']}")
        Q = cn["points"]
        lines.append(f"  PUNTOS: {len(Q)} en la región del tubo; usados para el diámetro {int(cn['used_mask'].sum())}; "
                     f"agua/banket {int(cn['water_mask'].sum())}; transversales (tapa/escalón/cara) {int(cn['transverse_mask'].sum())}")
        lines.append(f"  tramo estable: {len(cn['stable_run'])} bandas, {cn['pipe_length_observed'] * 100:.0f} cm de tubo; "
                     f"bandas válidas en {cn['tube_length_observed'] * 100:.0f} cm")
        lines.append(f"  parte inferior oculta: {cn['lower_pipe_occluded']}   relleno/tapa: {cn['possible_reconstruction_fill']}")
        if cn["note"]:
            lines.append(f"  nota: {cn['note']}")
        lines += [f"  Diameter (tubo):  {_m(cn['diameter'])}",
                  f"  Kruin:            {_m(cn['crown'])}",
                  f"  BOB:              {_m(cn['bob'])}",
                  f"  Hart:             {_m(cn['axis_h'])}",
                  f"  Richting:         {_m(cn['direction_m'], 'deg')}",
                  f"  Helling:          {_m(cn['slope'], 'deg')}"]
        out[cn["id"]] = lines
    for i, o in enumerate([o for o in put.get("openings", []) if o["status"] != "CONFIRMED"]):
        out[f"O{i + 1}"] = [f"O{i + 1} — candidato de abertura NO confirmado ({o['kind']})"] + opening_lines(o)
    for i, rc in enumerate(put.get("rejected_candidates", [])):
        out[f"R{i + 1}"] = [f"R{i + 1} — grupo detrás de la pared {rc.get('wall')}: {rc['kind']}", f"  {rc['reason']}",
                            f"  {len(rc['indices'])} puntos"]
    return out, feats


def _diag_connections_legacy(res):
    """Diagnóstico del método anterior (fase 5), conservado para comparar."""
    L, N = res["local_points"], res["local_normals"]
    put = res["put"]
    out, feats = {}, {}
    for cn in res.get("connections_legacy", []):
        f = candidate_features(res, cn["points"], cn["normals"], cn)
        feats[cn["id"]] = f
        lines = [f"{cn['id']} (aceptado como conexión por analysis.py: ≥ {A.PIPE_MIN_POINTS} ptn y no es superficie horizontal plana)"]
        lines += _feature_lines(f)
        d = cn["diameter"]
        lines.append(f"  diámetro: " + ("UNKNOWN" if d["value"] is None else f"{d['value'] * 1000:.0f} ± {d['U'] * 1000:.0f} mm")
                     + f" [{d['status']}]" + (f" — {d.get('reason')}" if d.get("reason") else ""))
        if "arc_points" in cn:
            arc, sz = cn["arc_points"], cn["sz_all"]
            res_c = np.hypot(arc[:, 0] - cn["center_s"], arc[:, 1] - cn["center_z"]) - cn["fit_r"]
            lines += [f"  sección: {len(sz)} ptn tras quitar {int(cn['bottom_mask'].sum())} de la superficie inferior plana; "
                      f"inliers del círculo {len(arc)} ({len(arc) / max(len(sz), 1):.0%})",
                      f"  círculo: centro (s,z)=({cn['center_s'] * 1000:.0f}, {cn['center_z'] * 1000:.0f}) mm, r={cn['fit_r'] * 1000:.0f} mm, "
                      f"RMS {cn['fit_rms'] * 1000:.1f} mm, residuos p5/p95 {pct(res_c, 5) * 1000:+.1f}/{pct(res_c, 95) * 1000:+.1f} mm",
                      f"  arco observado: {cn['arc_deg']:.0f}° (bins de 10° con inliers: {cn['arc_bins']})",
                      f"  cuerda {cn['chord'] * 1000:.0f} mm, flecha {cn['sagitta'] * 1000:.0f} mm, método cuerda-flecha: "
                      + ("—" if cn.get("r_chord") is None else f"Ø{2 * cn['r_chord'] * 1000:.0f} mm")]
            pts_not_inl = len(sz) - len(arc)
            lines.append(f"  puntos de la sección NO usados por el círculo: {pts_not_inl} ({pts_not_inl / max(len(sz), 1):.0%})")
            lines += section_breakdown(cn)
        out[cn["id"]] = lines
    for i, rc in enumerate(put.get("rejected_candidates", [])):
        Q, NQ = L[rc["indices"]], N[rc["indices"]]
        key = f"R{i + 1}"
        lines = [f"{key} (descartado por analysis.py: {rc['reason']})"]
        if len(Q) >= 10:
            f = candidate_features(res, Q, NQ)
            feats[key] = f
            lines += _feature_lines(f)
        out[key] = lines
    return out, feats


def section_breakdown(cn):
    """¿Qué son los puntos de la sección que el círculo no usa? Residuos de TODOS los puntos, inliers por
    profundidad a lo largo del tubo y prueba 'qué pasaría si' proyectando con el eje medido (no cambia nada)."""
    Q = cn["points"][~cn["bottom_mask"]]
    sz = cn["sz_all"]
    dr = np.hypot(sz[:, 0] - cn["center_s"], sz[:, 1] - cn["center_z"]) - cn["fit_r"]
    bins = [0, 0.005, 0.02, 0.05, 0.10, np.inf]
    h = np.histogram(np.abs(dr), bins=bins)[0] / max(len(dr), 1)
    lines = ["  residuos de TODOS los puntos de la sección |d−r|: "
             + "  ".join(f"{lab}:{v:.0%}" for lab, v in zip(["<5mm", "5-20", "20-50", "50-100", ">100mm"], h)),
             f"    hacia dentro del círculo (d<−2 cm) {np.mean(dr < -0.02):.0%}, hacia fuera (d>+2 cm) {np.mean(dr > 0.02):.0%}"]
    t = Q @ cn["direction"]
    t = t - t.min()
    inl = np.abs(dr) < 0.005
    rows = []
    for t0 in np.arange(0, t.max() + 1e-9, 0.05):
        m = (t >= t0) & (t < t0 + 0.05)
        if m.sum() >= 10:
            rows.append(f"{t0 * 100:.0f}-{(t0 + 0.05) * 100:.0f}cm:{inl[m].mean():.0%}({m.sum()})")
    lines.append("  inliers por profundidad a lo largo del tubo (desde la pared): " + " ".join(rows))
    # ajuste INDEPENDIENTE por bandas de profundidad: ¿el agujero en la pared y el tubo tienen el mismo círculo?
    band_rows = []
    for t0 in np.arange(0, t.max() + 1e-9, 0.10):
        m = (t >= t0) & (t < t0 + 0.10)
        if m.sum() < 40:
            continue
        sc, zc, r, inl_b, _ = A.fit_circle_robust(sz[m])
        if inl_b.sum() < 15:
            continue
        sc, zc, r = A.fit_circle_geometric(sz[m][inl_b], sc, zc, r)
        ang = np.arctan2(sz[m][inl_b][:, 1] - zc, sz[m][inl_b][:, 0] - sc)
        arc = np.unique(((ang + np.pi) // (np.pi / 18)).astype(int)).size * 10
        band_rows.append(f"{t0 * 100:.0f}-{(t0 + 0.1) * 100:.0f}cm: D{2 * r * 1000:.0f} c({sc * 1000:.0f},{zc * 1000:.0f}) "
                         f"arco{arc}° inl{inl_b.mean():.0%}")
    if band_rows:
        lines.append("  círculo independiente por banda de profundidad (desde la pared):")
        lines += ["    " + r for r in band_rows]
    ax = cn.get("axis_info", {})
    if ax.get("axis") is not None:
        a = np.asarray(ax["axis"], float)
        u = np.cross(a, [0, 0, 1.0]); u /= max(np.linalg.norm(u), 1e-9)
        v = np.cross(u, a)
        sz2 = np.column_stack([Q @ u, Q @ v])
        sc, zc, r, inl2, _ = A.fit_circle_robust(sz2)
        sc, zc, r = A.fit_circle_geometric(sz2[inl2], sc, zc, r)
        dr2 = np.hypot(sz2[:, 0] - sc, sz2[:, 1] - zc) - r
        slope = np.degrees(np.arcsin(np.clip(a[2], -1, 1)))
        lines.append(f"  QUÉ PASARÍA SI se proyecta según el eje medido por normales (pendiente {slope:+.1f}°, "
                     f"ratio λ {ax.get('ratio', np.nan):.2f}, longitud {ax.get('length', 0) * 100:.0f} cm): "
                     f"Ø {2 * r * 1000:.0f} mm, inliers <5 mm {np.mean(np.abs(dr2) < 0.005):.0%} (ahora {np.mean(inl):.0%})")
    return lines


def _feature_lines(f):
    return [
        f"  pared {f['wall']}  puntos {f['n']}  ancho {f['width'] * 1000:.0f}  alto {f['height'] * 1000:.0f}  "
        f"profundidad visible {f['depth'] * 1000:.0f} mm",
        f"  superficie ≈ {f['area_m2']:.3f} m²  volumen caja {f['bbox_vol_m3'] * 1000:.1f} dm³",
        f"  continuidad: {f['components']} fragmento(s), mayor {f['largest_frac']:.0%}",
        f"  curvatura (sección): RMS recta {f['line_rms'] * 1000:.1f} mm / RMS círculo {f['circle_rms'] * 1000:.1f} mm "
        f"(x{f['curvature_gain']:.1f}), r {f['circle_r'] * 1000:.0f} mm",
        f"  penetración detrás de la pared: p50 {f['penetration_p50'] * 100:.1f} cm  p95 {f['penetration_p95'] * 100:.1f} cm",
        f"  posición: z {f['z_min']:.2f}..{f['z_max']:.2f} m, {f['dist_floor'] * 100:.0f} cm sobre bodem, "
        f"{f['dist_chamber_top'] * 100:.0f} cm bajo el techo de la cámara, {f['dist_corner'] * 100:.0f} cm de la esquina",
        f"  contaminación: {f['frac_wall_like']:.0%} orientado como pared, {f['frac_flat_low']:.0%} horizontal a nivel de bodem",
        f"  visible arc: " + ("—" if f["visible_arc"] is None else f"{f['visible_arc']:.0f}°"),
        f"  PROPUESTA (diagnóstico): {f['proposal']}" + ("" if not f["proposal_reasons"] else
                                                       " — " + "; ".join(f["proposal_reasons"])),
    ]


def diag_merge(res):
    """¿Pertenecen varios grupos al mismo tubo? Compara y prueba el ajuste conjunto (no fusiona)."""
    lines = ["MERGE (pares en la misma pared)"]
    conns = res["connections"]
    found = False
    for i in range(len(conns)):
        for j in range(i + 1, len(conns)):
            a, b = conns[i], conns[j]
            if a["wall"] != b["wall"]:
                continue
            found = True
            gap = float(np.min(np.linalg.norm(a["points"][::5, None, :] - b["points"][None, ::5, :], axis=2)))
            lines.append(f"  {a['id']} + {b['id']} (pared {a['wall']}): separación mínima {gap * 100:.1f} cm, "
                         f"ángulos {a['angle_deg']:.0f}°/{b['angle_deg']:.0f}°")
            for c in (a, b):
                d = c["diameter"]
                attempt = d["value"] if d["value"] is not None else d.get("attempt_value")
                lines.append(f"     {c['id']}: centro (s,z)=({c.get('center_s', np.nan) * 1000:.0f}, {c.get('center_z', np.nan) * 1000:.0f}) mm, "
                             f"Ø ajuste {('—' if attempt is None else f'{attempt * 1000:.0f}')} mm, arco {c.get('arc_deg', 0):.0f}°, "
                             f"dirección {np.round(c['direction_m']['vector'], 3).tolist()}")
            s_axis = a["s_axis"]
            sz = np.vstack([np.column_stack([a["points"] @ s_axis, a["points"][:, 2]])[~a["bottom_mask"]],
                            np.column_stack([b["points"] @ s_axis, b["points"][:, 2]])[~b["bottom_mask"]]])
            if len(sz) >= 12:
                joint = A._measure_arc(sz, np.random.default_rng(0), res["put"]["floor_sigma"], res["put"]["top_z"],
                                       res["put"]["top_sigma"])
                jd = joint["diameter"]
                jv = jd["value"] if jd["value"] is not None else jd.get("attempt_value")
                jU = jd["U"] if jd["value"] is not None else jd.get("attempt_U")
                lines.append(f"     ajuste CONJUNTO: Ø {('—' if jv is None else f'{jv * 1000:.0f} ± {jU * 1000:.0f}')} mm, "
                             f"arco {joint['arc_deg']:.0f}°, RMS {joint['fit_rms'] * 1000:.1f} mm, inliers {joint['inliers']}/{len(sz)} "
                             f"[{jd['status']}]")
                best_single = max(a.get("fit_rms", 0), b.get("fit_rms", 0))
                compatible = (joint["fit_rms"] <= 1.5 * best_single + 0.001 and joint["inliers"] >= 0.6 * len(sz)
                              and gap < 0.10)
                lines.append(f"     -> {'COMPATIBLE: probablemente el mismo tubo' if compatible else 'no compatible (o insuficiente)'}"
                             f" (criterio: RMS conjunto ≤ 1,5×, ≥ 60 % inliers, separación < 10 cm)")
    if not found:
        lines.append("  ningún par en la misma pared")
    return lines


def diag_orientation(res):
    put = res["put"]
    o = put["orientation"]
    lines = ["ORIENTACIÓN",
             f"  eje vertical estimado (en PLY): {np.round(o['axis_ply'], 4).tolist()}, {o['value']:.2f}° respecto a Z del PLY (±{o['U']:.2f}°)",
             f"  simetría 90° de normales {res['symmetry']:.2f}"]
    for key, w in (put["walls"]["walls"] or {}).items():
        lines.append(f"  pared {key}: inclinación respecto al eje {np.degrees(np.arcsin(abs(w['n'][2]))):.2f}° "
                     "(paredes verticales -> ≈0°; > 1° indica eje mal estimado o pared inclinada)")
    fi = put["floor_info"]
    lines.append(f"  bodem: inclinación {fi['slope_deg']:.2f}° respecto al plano perpendicular al eje")
    if put["ground_info"]:
        lines.append(f"  maaiveld: inclinación {put['ground_info']['slope_deg']:.2f}°")
    return lines


# ---------------------------------------------------------------- imágenes 2D (sin dependencias extra)

def _canvas(w=900, h=700, bg=(18, 20, 24)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bg
    return img


def _plot(img, xy, color, extent, size=2):
    (x0, x1), (y0, y1) = extent
    h, w = img.shape[:2]
    px = ((xy[:, 0] - x0) / (x1 - x0) * (w - 40) + 20).astype(int)
    py = (h - 20 - (xy[:, 1] - y0) / (y1 - y0) * (h - 40)).astype(int)
    for dx in range(-size // 2, size // 2 + 1):
        for dy in range(-size // 2, size // 2 + 1):
            ok = (px + dx >= 0) & (px + dx < w) & (py + dy >= 0) & (py + dy < h)
            img[py[ok] + dy, px[ok] + dx] = color


def _extent(xy, margin=0.05, aspect=900 / 700):
    lo, hi = xy.min(axis=0) - margin, xy.max(axis=0) + margin
    c, span = (lo + hi) / 2, hi - lo
    span = np.maximum(span, [span[1] * aspect, span[0] / aspect])
    return (c[0] - span[0] / 2, c[0] + span[0] / 2), (c[1] - span[1] / 2, c[1] + span[1] / 2)


def _save(img, path):
    o3d.io.write_image(str(path), o3d.geometry.Image(np.ascontiguousarray(img)))


def save_section_png(cn, path):
    """Sección perpendicular al eje del tubo (pipeline nuevo): verde = usados para el diámetro, naranja = bandas
    de abertura/collar, rojo = resto no usado, gris = agua/banket, magenta = superficies transversales (tapa,
    escalón), amarillo = círculo final, blanco = centro."""
    if "bands" not in cn:
        return _save_section_png_legacy(cn, path)
    Q = cn["points"]
    rel = Q - cn["c0"]
    sz = np.column_stack([rel @ cn["s_axis"], cn["c0"][2] + rel @ cn["v_axis"]])
    ring = np.zeros(len(Q), bool)
    for b in cn["bands"]:
        if b["label"] == "OPENING_RING":
            ring[b["idx"]] = True
    img = _canvas()
    ext = _extent(sz)
    other = ~(cn["used_mask"] | cn["water_mask"] | cn["transverse_mask"] | ring)
    _plot(img, sz[other], (255, 80, 80), ext)
    _plot(img, sz[ring & ~cn["used_mask"]], (255, 160, 40), ext)
    _plot(img, sz[cn["water_mask"]], (110, 110, 110), ext)
    _plot(img, sz[cn["transverse_mask"]], (230, 60, 230), ext)
    _plot(img, sz[cn["used_mask"]], (80, 230, 110), ext)
    if cn["diameter"]["value"] is not None:
        t = np.linspace(0, 2 * np.pi, 2000)
        rr = np.column_stack([cn["center_s"] + cn["fit_r"] * np.cos(t), cn["center_z"] + cn["fit_r"] * np.sin(t)])
        _plot(img, rr, (255, 220, 60), ext, size=1)
        _plot(img, np.array([[cn["center_s"], cn["center_z"]]]), (255, 255, 255), ext, size=8)
    _save(img, path)


def _save_section_png_legacy(cn, path):
    Q = cn["points"]
    s_axis = cn["s_axis"]
    all_sz = np.column_stack([Q @ s_axis, Q[:, 2]])
    img = _canvas()
    ext = _extent(all_sz)
    _plot(img, all_sz[cn["bottom_mask"]], (110, 110, 110), ext)
    if "center_s" in cn:
        sz = cn["sz_all"]
        dr = np.hypot(sz[:, 0] - cn["center_s"], sz[:, 1] - cn["center_z"]) - cn["fit_r"]
        _plot(img, sz[dr < -0.005], (70, 130, 255), ext)
        _plot(img, sz[dr > 0.005], (255, 80, 80), ext)
        _plot(img, sz[np.abs(dr) <= 0.005], (80, 230, 110), ext)
        t = np.linspace(0, 2 * np.pi, 2000)
        ring = np.column_stack([cn["center_s"] + cn["fit_r"] * np.cos(t), cn["center_z"] + cn["fit_r"] * np.sin(t)])
        _plot(img, ring, (255, 220, 60), ext, size=1)
        _plot(img, np.array([[cn["center_s"], cn["center_z"]]]), (255, 255, 255), ext, size=8)
    else:
        _plot(img, all_sz, (200, 200, 200), ext)
    _save(img, path)


def save_section_depth_png(cn, path):
    """Misma sección coloreada por profundidad a lo largo del tubo (rojo = junto a la pared, azul = lejos)."""
    if "bands" in cn:
        Q = cn["points"]
        rel = Q - cn["c0"]
        sz = np.column_stack([rel @ cn["s_axis"], cn["c0"][2] + rel @ cn["v_axis"]])
        t = rel @ cn["axis"]
    else:
        Q = cn["points"][~cn["bottom_mask"]]
        sz = cn["sz_all"]
        t = Q @ cn["direction"]
    t = (t - t.min()) / max(np.ptp(t), 1e-6)
    img = _canvas()
    ext = _extent(sz)
    order = np.argsort(t)
    for k in np.array_split(order, 10):
        if len(k):
            f = float(t[k].mean())
            _plot(img, sz[k], (int(255 * (1 - f)), 80, int(255 * f)), ext)
    _save(img, path)


def save_residual_map_png(geo, path, thresh, horizontal=False):
    """Mapa de residuos firmados de los candidatos sobre el plano ajustado (vista de frente).
    La normal de pared apunta HACIA FUERA de la cámara: rojo = detrás de la pared / por encima del bodem
    o maaiveld; azul = hacia la cámara / por debajo; blanco = sobre el plano (±2 cm a saturación);
    gris oscuro = > 3x el umbral (no usados por el ajuste)."""
    P, n, c, d = geo["candidates"], geo["n"], geo["c"], geo["residuals"]
    if horizontal:
        uv = P[:, :2]
    else:
        u = np.cross([0, 0, 1.0], n); u /= np.linalg.norm(u)
        uv = np.column_stack([P @ u, P[:, 2]])
    img = _canvas()
    ext = _extent(uv)
    f = np.clip(d / 0.02, -1, 1)
    col = np.zeros((len(d), 3))
    col[:, 0] = np.where(f > 0, 255, 255 * (1 + f))
    col[:, 1] = 255 * (1 - np.abs(f))
    col[:, 2] = np.where(f < 0, 255, 255 * (1 - f))
    for lo in np.linspace(-1, 1, 21)[:-1]:
        m = (f >= lo) & (f < lo + 0.1 + 1e-9)
        if m.any():
            _plot(img, uv[m], tuple(int(x) for x in col[m][0]), ext)
    _plot(img, uv[np.abs(d) > thresh * 3], (60, 60, 60), ext, size=1)
    _save(img, path)


# ---------------------------------------------------------------- informe completo

def diagnose(res, seed=1):
    rng = np.random.default_rng(seed)
    sections, geom = {}, {}
    w_lines, w_geo = diag_walls(res, rng)
    sections.update(w_lines); geom.update(w_geo)
    for ax in ("X", "Y"):
        lines, seg = diag_width(res, ax)
        sections[f"MEASURE {ax}"] = lines
        geom[f"MEASURE {ax}"] = dict(segment=seg, walls=[f"WALL +{ax}", f"WALL -{ax}"])
    b_lines, b_geo = diag_surface(res, "BOTTOM", rng)
    g_lines, g_geo = diag_surface(res, "GROUND", rng)
    sections["BOTTOM"], sections["GROUND"] = b_lines, g_lines
    geom["BOTTOM"], geom["GROUND"] = b_geo, g_geo
    d_lines, seg = diag_depth(res, b_geo, g_geo)
    sections["MEASURE DEPTH"] = d_lines
    geom["MEASURE DEPTH"] = dict(segment=seg)
    c_lines, feats = diag_connections(res)
    sections.update(c_lines)
    geom["features"] = feats
    sections["MERGE"] = diag_merge(res)
    sections["ORIENTATION"] = diag_orientation(res)
    return sections, geom


ORDER = ["MEASURE X", "WALL +X", "WALL -X", "MEASURE Y", "WALL +Y", "WALL -Y", "MEASURE DEPTH", "BOTTOM",
         "GROUND", "ORIENTATION"]


def report_text(path, sections):
    lines = [f"=== DIAGNÓSTICO: {Path(path).name} ===", ""]
    keys = ORDER + [k for k in sections if k not in ORDER and k != "MERGE"] + ["MERGE"]
    for k in keys:
        if k in sections:
            lines += sections[k] + [""]
    return "\n".join(lines)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    path = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent / "data" / "scan.ply")
    res = A.load_and_analyze(path)
    sections, geom = diagnose(res)
    text = report_text(path, sections)
    print(text)
    stem = Path(path).stem.replace(" ", "")
    outdir = Path(__file__).parent / "results" / "debug"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{stem}_diagnose.txt").write_text(text, encoding="utf-8")
    for cn in res["connections"]:
        save_section_png(cn, outdir / f"{stem}_{cn['id']}_section.png")
        save_section_depth_png(cn, outdir / f"{stem}_{cn['id']}_depth.png")
    for key in ("WALL +X", "WALL -X", "WALL +Y", "WALL -Y"):
        if "n" in geom.get(key, {}):
            save_residual_map_png(geom[key], outdir / f"{stem}_{key.replace(' ', '_').replace('+', 'p').replace('-', 'm')}_residuals.png",
                                  A.PLANE_THRESH_M)
    for key in ("BOTTOM", "GROUND"):
        if "n" in geom.get(key, {}):
            save_residual_map_png(geom[key], outdir / f"{stem}_{key}_residuals.png", A.SURFACE_THRESH_M, horizontal=True)
    print(f"\n(informe e imágenes en {outdir.relative_to(Path(__file__).parent)})")
