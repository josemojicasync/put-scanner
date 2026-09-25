"""Medición de una boca de tubería a partir de puntos del borde (unidades: metros)."""
import numpy as np

MIN_POINTS = 8
RANSAC_ITERS = 500
RANSAC_THRESHOLD_M = 0.005  # 5 mm de tolerancia al borde
UP_AXIS = 2  # eje vertical del escaneo (0=X, 1=Y, 2=Z)
AXIS_NAMES = "XYZ"


class MeasurementError(ValueError):
    pass


def fit_plane_pca(points):
    """Plano local por PCA: devuelve centroide, eje u, eje v y normal."""
    centroid = points.mean(axis=0)
    _, s, vt = np.linalg.svd(points - centroid, full_matrices=False)
    if s[1] < 1e-6 * max(s[0], 1e-12):
        raise MeasurementError("Los puntos están casi en línea recta; selecciona puntos a lo largo de todo el borde.")
    return centroid, vt[0], vt[1], vt[2]


def fit_circle_2d(xy):
    """Ajuste algebraico (Kasa) de círculo por mínimos cuadrados: (cx, cy, r)."""
    A = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    (cx, cy, c), *_ = np.linalg.lstsq(A, b, rcond=None)
    return cx, cy, np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0))


def fit_circle_robust(xy, seed=0):
    """RANSAC de 3 puntos + reajuste por mínimos cuadrados sobre los inliers."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(RANSAC_ITERS):
        sample = xy[rng.choice(len(xy), 3, replace=False)]
        cx, cy, r = fit_circle_2d(sample)
        if not np.isfinite(r) or r == 0:
            continue
        inliers = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) < RANSAC_THRESHOLD_M
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best is None or best.sum() < MIN_POINTS:
        best = np.ones(len(xy), bool)  # sin consenso: usar todos los puntos
    cx, cy, r = fit_circle_2d(xy[best])
    residuals = np.hypot(xy[best, 0] - cx, xy[best, 1] - cy) - r
    return cx, cy, r, best, float(np.sqrt(np.mean(residuals ** 2)))


def describe_orientation(normal):
    """Texto aproximado de la orientación del eje de la tubería (normal del plano de la boca)."""
    n = normal / np.linalg.norm(normal)
    tilt = np.degrees(np.arccos(min(abs(n[UP_AXIS]), 1.0)))  # 0° = eje vertical
    if tilt < 20:
        kind = "vertical (boca mirando hacia arriba/abajo)"
    elif tilt > 70:
        kind = "horizontal (entra por la pared de la put)"
    else:
        kind = "inclinada"
    h = [i for i in range(3) if i != UP_AXIS]
    azimuth = np.degrees(np.arctan2(n[h[1]], n[h[0]])) % 180
    return (f"{kind}; eje {np.round(n, 3)}, {tilt:.1f}° respecto a la vertical ({AXIS_NAMES[UP_AXIS]}), "
            f"rumbo horizontal {azimuth:.0f}° desde +{AXIS_NAMES[h[0]]}")


def measure_pipe(points):
    """Mide la boca a partir de puntos 3D del borde. Devuelve un dict con resultados en metros."""
    points = np.unique(np.asarray(points, float), axis=0)
    if len(points) < MIN_POINTS:
        raise MeasurementError(
            f"Solo hay {len(points)} puntos distintos seleccionados; se necesitan al menos {MIN_POINTS} "
            "repartidos alrededor del borde de la boca.")
    centroid, u, v, normal = fit_plane_pca(points)
    rel = points - centroid
    xy = np.column_stack([rel @ u, rel @ v])
    cx, cy, r, inliers, rms = fit_circle_robust(xy)
    return {
        "points": points,
        "inliers": inliers,
        "center": centroid + cx * u + cy * v,
        "radius": r,
        "normal": normal,
        "u": u,
        "v": v,
        "rms": rms,
        "orientation": describe_orientation(normal),
    }
