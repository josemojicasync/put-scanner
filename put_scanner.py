import open3d as o3d
import numpy as np
from pathlib import Path

ARCHIVO = Path("scan.ply")

print("\n=== PUT SCANNER - PROTOTYPE ===\n")

if not ARCHIVO.exists():
    print(f"ERROR: No encuentro {ARCHIVO.resolve()}")
    raise SystemExit

print("Cargando escaneo...")
cloud = o3d.io.read_point_cloud(str(ARCHIVO))

points = np.asarray(cloud.points)

if len(points) == 0:
    print("ERROR: El PLY no contiene una nube de puntos.")
    raise SystemExit

print(f"Puntos encontrados: {len(points):,}")

minimo = cloud.get_min_bound()
maximo = cloud.get_max_bound()
tamano = maximo - minimo

print("\nDimensiones del escaneo:")
print(f"X = {tamano[0]:.3f}")
print(f"Y = {tamano[1]:.3f}")
print(f"Z = {tamano[2]:.3f}")

print("\nAbriendo visor 3D...")
print("Ratón izquierdo = rotar")
print("Rueda = zoom")
print("Shift + ratón = mover")

ejes = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=max(np.max(tamano) * 0.15, 0.1)
)

o3d.visualization.draw_geometries(
    [cloud, ejes],
    window_name="PUT SCANNER - Polycam Scan",
    width=1400,
    height=850
)

print("\nVisor cerrado.")
