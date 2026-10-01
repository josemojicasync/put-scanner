"""ScanResult: resultado estructurado e independiente del visor, guardable como JSON.

Unidades: longitudes en mm, ángulos en grados. Coordenadas en el sistema local de la put
(origen = centro de la cámara a la altura del bodem, Z hacia arriba); `transform_local_to_ply`
(4x4, metros) permite volver al PLY original. Pensado para que módulos futuros (Modellen/Prefabs,
Rapporten, Projecten) usen las medidas sin volver a analizar el PLY.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

SCHEMA_VERSION = "1.0"
CONFIDENCE_EN = {"HOOG": "HIGH", "MIDDEL": "MEDIUM", "LAAG": "LOW", "ONZEKER": "NONE"}


@dataclass
class Measurement:
    value: Optional[object]          # número, texto o None (UNKNOWN: nunca un número inventado)
    uncertainty: Optional[float]     # incertidumbre expandida k=2 (~95 %), mismas unidades
    unit: str
    confidence: str                  # HIGH / MEDIUM / LOW / NONE
    status: str                      # MEASURED / ESTIMATED / UNKNOWN
    method: str
    note: str = ""

    @property
    def is_measured(self):
        return self.status == "MEASURED"


@dataclass
class Plane:
    level_mm: Measurement
    slope_deg: float
    rms_mm: float
    points: int
    coverage: Optional[float] = None


@dataclass
class Wall:
    id: str
    scanned: bool
    points: int = 0
    rms_mm: Optional[float] = None
    coverage: Optional[float] = None
    normal: Optional[list] = None
    drift_mm: Optional[float] = None


@dataclass
class PutResult:
    shape: Measurement
    width_x_mm: Measurement
    width_y_mm: Measurement
    diameter_mm: Measurement
    radius_mm: Measurement
    circularity_mm: Measurement
    depth_mm: Measurement
    bottom_level_mm: Measurement
    reference_level_mm: Measurement
    center: Measurement               # value = [x, y, z] en el PLY (mm)
    orientation: Measurement          # value = inclinación del eje vs Z del PLY (grados)
    bottom_plane: Optional[Plane]
    reference_plane: Optional[Plane]
    walls: list = field(default_factory=list)
    wall_geometry: dict = field(default_factory=dict)
    sections: list = field(default_factory=list)   # tramos indicativos (ESTIMATED)
    width_x_profile: dict = field(default_factory=dict)  # anchura abajo/medio/arriba (mm) y variación
    width_y_profile: dict = field(default_factory=dict)
    bottom_surface_candidates: list = field(default_factory=list)     # superficies plausibles de bodem (z mm)
    reference_surface_candidates: list = field(default_factory=list)  # superficies plausibles de maaiveld


@dataclass
class Connection:
    id: str
    wall: str
    position_mm: list                 # centro del tubo (o de la abertura) en coordenadas locales
    angle_deg: float                  # alrededor de la put, 0° = +X local
    direction: Measurement            # desviación respecto a la normal de pared (grados)
    slope_deg: Measurement
    diameter_mm: Measurement
    crown_mm: Measurement             # kruin, sobre el bodem
    invert_bob_mm: Measurement        # BOB, sobre el bodem
    center_height_mm: Measurement     # hart, sobre el bodem
    bob_depth_mm: Measurement         # BOB bajo maaiveld
    inner_height_mm: Measurement      # kruin - BOB observados (ovalidad)
    visible_arc_deg: float
    observed_opening_mm: dict
    visible_length_mm: float
    nominal_suggestions: list = field(default_factory=list)   # SUGERENCIA, no medida
    notes: list = field(default_factory=list)
    opening_diameter_mm: Optional[Measurement] = None  # abertura en la cara de la pared (≠ diámetro del tubo)
    opening_top_mm: Optional[Measurement] = None        # borde superior de la abertura (no es la kruin)
    opening: dict = field(default_factory=dict)         # evidencia de la abertura en la cara
    pipe_profile: list = field(default_factory=list)    # por banda: profundidad, centro, radio, arco, etiqueta
    pipe_length_observed_mm: float = 0.0                # longitud del tramo de tubo estable usado para el diámetro
    tube_length_observed_mm: float = 0.0                # longitud con bandas válidas (cualquier radio)
    lower_pipe_occluded: Optional[bool] = None
    possible_reconstruction_fill: bool = False


@dataclass
class ScanQuality:
    total_points: int
    usable_points: int
    analysed_points: int
    point_spacing_mm: float
    wall_density_pts_m2: Optional[float]
    noise_mm: Optional[float]
    chamber_coverage: float
    wall_coverage: dict
    duplicate_fraction: dict
    bottom_coverage: float
    connection_coverage_deg: dict
    scale: dict
    missing_geometry: list
    items: dict
    warnings: list


@dataclass
class ScanResult:
    schema_version: str
    source: dict
    analysed_at: str
    analysis_time_s: float
    coordinate_system: str
    transform_local_to_ply: list
    put: PutResult
    connections: list
    quality: ScanQuality
    parameters: dict
    rejected_candidates: list = field(default_factory=list)  # candidatos que NO son conexión, con su razón

    # ---- JSON
    def to_dict(self):
        return _clean(asdict(self))

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return from_dict(cls, json.loads(Path(path).read_text(encoding="utf-8")))


def _clean(o):
    """Convierte numpy y NaN a tipos JSON."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def from_dict(cls, data):
    """Reconstruye dataclasses anidadas desde un dict (JSON)."""
    if data is None or not is_dataclass(cls):
        return data
    kwargs = {}
    hints = {f.name: f.type for f in fields(cls)}
    for name, typ in hints.items():
        if name not in data:
            continue
        v = data[name]
        target = _resolve(typ)
        if target is not None and isinstance(v, dict):
            v = from_dict(target, v)
        elif name in _LIST_TYPES.get(cls.__name__, {}) and isinstance(v, list):
            v = [from_dict(_LIST_TYPES[cls.__name__][name], x) for x in v]
        kwargs[name] = v
    return cls(**kwargs)


def _resolve(typ):
    """Dataclass destino de un campo (también dentro de Optional[...])."""
    if is_dataclass(typ):
        return typ
    for arg in getattr(typ, "__args__", None) or ():
        if is_dataclass(arg):
            return arg
    return None


_LIST_TYPES = {"ScanResult": {"connections": Connection}, "PutResult": {"walls": Wall}}


# ---------------------------------------------------------------- construcción desde el análisis

def _m(d, unit="mm", scale=1000.0, note=""):
    """dict de medida de analysis.py (SI) -> Measurement."""
    if d is None:
        return Measurement(None, None, unit, "NONE", "UNKNOWN", "not_applicable", note)
    v, U = d["value"], d["U"]
    if v is not None and unit in ("mm",):
        v = round(v * scale, 1)
    elif v is not None:
        v = round(v, 2)
    if U is not None:
        U = round(U * scale, 1) if unit == "mm" else round(U, 2)
    note = note or d.get("reason", "")
    return Measurement(v, U, unit, CONFIDENCE_EN[d["conf"]], d["status"], d["method"], note)


def _plane(info, level_mm, sigma_m):
    if not info:
        return None
    return Plane(level_mm=Measurement(round(level_mm, 1), round(2 * sigma_m * 1000, 1), "mm", "HIGH", "MEASURED",
                                      "ransac_plane"),
                 slope_deg=round(info["slope_deg"], 2), rms_mm=round(info["rms"] * 1000, 1), points=info["n_used"],
                 coverage=info.get("coverage"))


def file_info(path):
    p = Path(path)
    h = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
    return dict(name=p.name, size_bytes=p.stat().st_size if p.is_file() else None, sha256=h)


def _profile(m):
    p = (m or {}).get("profile")
    if not p:
        return {}
    return dict(z_mm=[round(z * 1000) for z in p["z"]], bottom_mm=round(p["bottom"] * 1000, 1),
                mid_mm=round(p["mid"] * 1000, 1), top_mm=round(p["top"] * 1000, 1),
                variation_mm=round(p["variation"] * 1000, 1), significant=p["significant"],
                sigma_taper_mm=round(m.get("sigma_taper", 0) * 1000, 1))


def _cands(cands):
    return [dict(kind=c["kind"], z_mm=round(c["z"] * 1000, 1), points=c["points"], rms_mm=round(c["rms"] * 1000, 1),
                 slope_deg=round(c["slope_deg"], 2)) for c in cands]


def _opening(op):
    if not op:
        return {}
    circ = op.get("circle")
    return dict(status=op["status"], reason=op["reason"], center_u_mm=round(op["centroid_uz"][0] * 1000),
                center_z_mm=round(op["centroid_uz"][1] * 1000), width_mm=round(op["width"] * 1000),
                height_mm=round(op["height"] * 1000), contour_support=round(op["contour"], 2),
                edge_points=op["edge_points"], behind_points=len(op["behind_idx"]),
                touches_floor=op["touches_floor"],
                contour_circle=None if circ is None else dict(r_mm=round(circ["r"] * 1000, 1),
                                                             rms_mm=round(circ["rms"] * 1000, 1), arc_deg=circ["arc"]))


def _bands(c):
    out = []
    for b in c.get("bands", []):
        out.append(dict(depth_mm=[round(b["t0"] * 1000), round(b["t1"] * 1000)], label=b["label"], points=b["n_use"],
                        diameter_mm=round(2 * b["r"] * 1000, 1) if "r" in b else None,
                        center_mm=[round(x * 1000, 1) for x in b["center"]] if "center" in b else None,
                        rms_mm=round(b["rms"] * 1000, 2) if np.isfinite(b.get("rms", np.nan)) else None,
                        arc_deg=b.get("arc"), contamination=round(b.get("contamination", 0), 2),
                        valid=b["valid"], reason=b["reason"]))
    return out


def _rejected(put):
    out = []
    for o in put.get("openings", []):
        if o["status"] != "CONFIRMED":
            out.append(dict(stage="OPENING", wall=o["wall"], kind=o["kind"], status=o["status"], reason=o["reason"],
                            center_u_mm=round(o["centroid_uz"][0] * 1000), center_z_mm=round(o["centroid_uz"][1] * 1000),
                            width_mm=round(o["width"] * 1000), height_mm=round(o["height"] * 1000)))
    for r in put.get("rejected_candidates", []):
        out.append(dict(stage="BEHIND_WALL_CLUSTER", wall=r.get("wall"), kind=r["kind"], status="REJECTED",
                        reason=r["reason"], points=len(r["indices"])))
    return out


def build_scan_result(res, path, elapsed_s):
    import analysis as A
    put, q = res["put"], res["quality"]
    sh = put["shape"]
    shape = Measurement(sh["label"], None, "", CONFIDENCE_EN[sh["conf"]], sh["status"], sh["method"],
                        f"symmetry90={sh['symmetry']:.2f}, circle_residual={sh['circle_rel_rms'] * 100:.1f}%")
    diam = put.get("diameter")
    radius = None if diam is None else dict(diam, value=None if diam["value"] is None else diam["value"] / 2,
                                            U=None if diam["U"] is None else diam["U"] / 2)
    walls = []
    if put["walls"]["walls"] is not None:
        for key, _, _ in A.WALLS:
            w = put["walls"]["walls"].get(key)
            walls.append(Wall(key, False) if w is None else Wall(
                key, True, w["n_used"], round(w["rms"] * 1000, 1), round(w["coverage"], 3), w["n"].tolist(),
                None if not np.isfinite(w["drift_m"]) else round(w["drift_m"] * 1000, 1)))
    T = np.eye(4)
    T[:3, :3] = res["R"].T
    T[:3, 3] = res["origin"]
    depth = put["depth"]
    pr = PutResult(
        shape=shape,
        width_x_mm=_m(put["width_x"]) if put["diameter"] is None else _m(None, note="rond"),
        width_y_mm=_m(put["width_y"]) if put["diameter"] is None else _m(None, note="rond"),
        diameter_mm=_m(diam, note="" if diam else "rechthoekig"),
        radius_mm=_m(radius, note="" if diam else "rechthoekig"),
        circularity_mm=_m(put.get("circularity"), note="" if diam else "rechthoekig"),
        depth_mm=_m(depth),
        bottom_level_mm=Measurement(0.0, round(2 * put["floor_sigma"] * 1000, 1), "mm", "HIGH", "MEASURED",
                                    "ransac_plane(bodem) = referentie 0"),
        reference_level_mm=_m(dict(depth, method="ransac_plane(maaiveld)")),
        center=Measurement([round(x * 1000, 1) for x in put["center"]["xyz_ply"]], round(put["center"]["U"] * 1000, 1),
                           "mm (PLY)", CONFIDENCE_EN[put["center"]["conf"]], put["center"]["status"], put["center"]["method"]),
        orientation=_m(put["orientation"], unit="deg",
                       note=f"axis_ply={np.round(put['orientation']['axis_ply'], 4).tolist()}"),
        bottom_plane=_plane(put["floor_info"], 0.0, put["floor_sigma"]),
        reference_plane=_plane(put["ground_info"], put["top_z"] * 1000, put["top_sigma"]),
        walls=walls,
        wall_geometry={k: round(v, 2) for k, v in put["walls"]["geometry"].items()},
        sections=[dict(z0_mm=round(s["z0"] * 1000), z1_mm=round(s["z1"] * 1000),
                       size_x_mm=round(2 * s["hx"] * 1000), size_y_mm=round(2 * s["hy"] * 1000), status="ESTIMATED",
                       method="slice_side_medians") for s in put["sections"]],
        width_x_profile=_profile(put["width_x"]), width_y_profile=_profile(put["width_y"]),
        bottom_surface_candidates=_cands(put.get("bottom_surface_candidates", [])),
        reference_surface_candidates=_cands(put.get("reference_surface_candidates", [])),
    )
    conns = []
    for c in res["connections"]:
        pos = c.get("center", c["wall_center"])
        d = c["direction_m"]
        notes = [w.split(": ", 1)[1] for w in q["warnings"] if w.startswith(c["id"] + ":")]
        conns.append(Connection(
            id=c["id"], wall=c["wall"], position_mm=[round(x * 1000, 1) for x in pos], angle_deg=round(c["angle_deg"], 1),
            direction=_m(d, unit="deg", note=d.get("reason", "")), slope_deg=_m(c["slope"], unit="deg"),
            diameter_mm=_m(c["diameter"]), crown_mm=_m(c["crown"]), invert_bob_mm=_m(c["bob"]),
            center_height_mm=_m(c["axis_h"]), bob_depth_mm=_m(c["bob_depth"]), inner_height_mm=_m(c.get("inner_height")),
            visible_arc_deg=round(c.get("arc_deg", 0.0), 0),
            observed_opening_mm=dict(width=round(c["opening_w"] * 1000), height=round(c["opening_h"] * 1000)),
            visible_length_mm=round(c["visible_len"] * 1000),
            nominal_suggestions=list(c.get("nominal", [])), notes=notes,
            opening_diameter_mm=_m(c.get("opening_diameter")) if c.get("opening_diameter") else None,
            opening_top_mm=_m(c.get("opening_top")) if c.get("opening_top") else None,
            opening=_opening(c.get("opening")), pipe_profile=_bands(c),
            pipe_length_observed_mm=round(c.get("pipe_length_observed", 0.0) * 1000),
            tube_length_observed_mm=round(c.get("tube_length_observed", 0.0) * 1000),
            lower_pipe_occluded=c.get("lower_pipe_occluded"),
            possible_reconstruction_fill=bool(c.get("possible_reconstruction_fill", False))))
    quality = ScanQuality(**{k: q[k] for k in (
        "total_points", "usable_points", "analysed_points", "point_spacing_mm", "wall_density_pts_m2", "noise_mm",
        "chamber_coverage", "wall_coverage", "duplicate_fraction", "bottom_coverage", "connection_coverage_deg",
        "scale", "missing_geometry", "items", "warnings")})
    params = dict(scale_rel_sigma=A.SCALE_REL_SIGMA, k_expand=A.K_EXPAND, voxel_m=A.VOXEL_M,
                  d_unknown_rel=A.D_UNKNOWN_REL, d_measured_rel=A.D_MEASURED_REL, arc_min_deg=A.ARC_MIN_DEG,
                  arc_measured_deg=A.ARC_MEASURED_DEG)
    return ScanResult(
        schema_version=SCHEMA_VERSION, source=file_info(path), analysed_at=datetime.now().isoformat(timespec="seconds"),
        analysis_time_s=round(elapsed_s, 2),
        coordinate_system="local put frame: origin = chamber centre at bodem level, Z up, X/Y along walls; mm",
        transform_local_to_ply=T.tolist(), put=pr, connections=conns, quality=quality, parameters=params,
        rejected_candidates=_rejected(put))
