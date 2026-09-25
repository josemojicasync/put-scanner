"""Put Scanner - prototipo: carga un escaneo PLY local y mide bocas de tubería seleccionadas a mano."""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

from measurement import MIN_POINTS, MeasurementError, measure_pipe
from viewer import scanner_cloud, show_overview, show_result, style_legacy

SCAN_PATH = Path(__file__).parent / "data" / "scan.ply"
EXPAND_RADIUS_M = 0.008  # añade vecinos a cada clic para densificar el borde (0 = desactivado)


def load_scan(path):
    """Carga el PLY como malla si tiene triángulos; si no, como nube de puntos."""
    mesh = o3d.io.read_triangle_mesh(str(path))
    if mesh.has_triangles():
        mesh.compute_vertex_normals()
        return mesh, np.asarray(mesh.vertices)
    pcd = o3d.io.read_point_cloud(str(path))
    return pcd, np.asarray(pcd.points)


def check_scale(dims):
    """Heurística: Polycam exporta en metros; un escaneo de putter/green ronda 0.05-20 m."""
    largest = dims.max()
    if largest < 0.01:
        return "SOSPECHOSA: < 1 cm. Posible escala incorrecta."
    if largest <= 50:
        return "OK: coherente con metros."
    if largest <= 5000:
        return "SOSPECHOSA: muy grande para metros. ¿Unidades en cm o mm?"
    return "SOSPECHOSA: dimensiones enormes. Revisa la exportación."


def print_summary(path, geometry, points):
    min_b, max_b = points.min(axis=0), points.max(axis=0)
    dims = max_b - min_b
    kind = "malla" if isinstance(geometry, o3d.geometry.TriangleMesh) else "nube de puntos"
    print(f"Archivo:  {path}")
    print(f"Tipo:     {kind}")
    print(f"Puntos:   {len(points):,}")
    print(f"Dimensiones: X={dims[0]:.4f}  Y={dims[1]:.4f}  Z={dims[2]:.4f}")
    print(f"Mín:      {np.round(min_b, 4)}")
    print(f"Máx:      {np.round(max_b, 4)}")
    print(f"Escala:   {check_scale(dims)}")


def pick_points(pcd):
    """Abre el visor de edición y devuelve los índices de los puntos elegidos con el ratón."""
    print(f"""
==============================================================
  PUT SCANNER - MEDICIÓN DE TUBERÍA
==============================================================
  PASO 1 - Navega hasta una boca de tubería
             arrastrar = rotar | rueda = zoom | Ctrl+arrastrar = mover
  PASO 2 - Shift + clic izquierdo para marcar puntos alrededor del BORDE
             (Shift + clic derecho deshace el último punto)
  PASO 3 - Selecciona mínimo {MIN_POINTS} puntos (mejor 15-30) repartidos
             por todo el contorno de la boca
  PASO 4 - Pulsa Q para analizar
--------------------------------------------------------------
  Extra: + / - cambia el tamaño de los puntos
==============================================================""")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="PUT SCANNER - Selección | Shift+clic marcar | Q analizar",
                      width=1280, height=800)
    vis.add_geometry(scanner_cloud(np.asarray(pcd.points)))
    style_legacy(vis)
    vis.run()
    vis.destroy_window()
    return vis.get_picked_points()


def expand_selection(pcd, indices):
    """Añade los vecinos cercanos a cada punto clicado."""
    if EXPAND_RADIUS_M <= 0:
        return list(indices)
    tree = o3d.geometry.KDTreeFlann(pcd)
    selected = set(indices)
    for i in indices:
        _, nbrs, _ = tree.search_radius_vector_3d(pcd.points[i], EXPAND_RADIUS_M)
        selected.update(nbrs)
    return sorted(selected)


def main():
    parser = argparse.ArgumentParser(description="Put Scanner")
    parser.add_argument("scan", nargs="?", default=SCAN_PATH, type=Path)
    parser.add_argument("--ver", action="store_true", help="solo visualizar (Fase 1), sin medir")
    args = parser.parse_args()

    path = args.scan
    if not path.is_file() or path.stat().st_size == 0:
        sys.exit(f"ERROR: no existe o está vacío: {path}")

    geometry, points = load_scan(path)
    if len(points) == 0:
        sys.exit(f"ERROR: no se pudieron leer puntos de {path}")
    print_summary(path, geometry, points)

    if args.ver:
        print("\nVisor: arrastrar=rotar | rueda=zoom | Ctrl+arrastrar=mover | cerrar ventana=salir")
        print("Ejes: X=rojo  Y=verde  Z=azul")
        show_overview(points)
        return

    # La selección trabaja sobre los puntos (en memoria; el archivo no se modifica).
    pcd = geometry if isinstance(geometry, o3d.geometry.PointCloud) else \
        o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    picked = pick_points(pcd)
    print(f"\nPuntos clicados: {len(picked)}")
    indices = expand_selection(pcd, picked) if picked else []
    try:
        result = measure_pipe(points[indices])
    except MeasurementError as e:
        sys.exit(f"ERROR: {e}")

    c = result["center"]
    print("\n=== TUBERÍA SELECCIONADA ===")
    print(f"Puntos utilizados: {int(result['inliers'].sum())} de {len(result['points'])}"
          f" ({len(picked)} clics + vecinos a {EXPAND_RADIUS_M * 1000:.0f} mm)")
    print(f"Diámetro: {result['radius'] * 2000:.1f} mm")
    print(f"Radio: {result['radius'] * 1000:.1f} mm")
    print(f"Centro: X={c[0]:.4f}, Y={c[1]:.4f}, Z={c[2]:.4f} m")
    print(f"Orientación: {result['orientation']}")
    print(f"Error de ajuste (RMS): {result['rms'] * 1000:.1f} mm")

    print("\nAbriendo vista de resultado (cerrar la ventana para salir)...")
    show_result(points, result, len(picked))


if __name__ == "__main__":
    main()
