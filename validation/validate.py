"""Validación del scanner contra medidas reales (cinta métrica, planos, fabricante).

Uso:
    python validation/validate.py data/scan.ply validation/ground_truth.json

ground_truth.json (en mm; omite o deja en null lo que no hayas medido):
{
  "scan": "scan.ply",
  "measured_by": "nombre", "date": "2026-10-01", "method": "rolmaat",
  "put": {"width_x_mm": 800, "width_y_mm": 800, "diameter_mm": null, "depth_mm": 2700},
  "connections": [
    {"id": "A1", "angle_deg": 88, "diameter_mm": 315, "crown_mm": null, "invert_bob_mm": null}
  ]
}
Las aansluitingen se emparejan por "angle_deg" (la más cercana, ±20°) o, si falta, por "id".

Cada ejecución se guarda en validation/results/ (JSON) y se añade a validation/results/history.csv
para comparar muchos scans en el futuro.
"""
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import load_and_analyze  # noqa: E402
from scan_result import build_scan_result  # noqa: E402

RESULTS = Path(__file__).parent / "results"
PUT_FIELDS = [("width_x_mm", "Binnenmaat X"), ("width_y_mm", "Binnenmaat Y"),
              ("diameter_mm", "Diameter put"), ("depth_mm", "Diepte")]
CONN_FIELDS = [("diameter_mm", "Diameter"), ("crown_mm", "Kruin"), ("invert_bob_mm", "BOB"),
               ("center_height_mm", "Hart")]


def compare(name, m, real):
    """Fila de comparación. Un valor UNKNOWN no se compara (no es un error del scanner: no midió)."""
    row = dict(name=name, real=real, scanner=m.value, U=m.uncertainty, status=m.status, confidence=m.confidence,
               error=None, rel_error=None, within_U=None)
    if m.value is not None and real is not None:
        err = m.value - real
        row.update(error=round(err, 1), rel_error=round(err / real * 100, 2) if real else None,
                   within_U=None if m.uncertainty is None else abs(err) <= m.uncertainty)
    return row


def match_connection(sr, gt):
    if gt.get("angle_deg") is not None and sr.connections:
        best = min(sr.connections, key=lambda c: abs((c.angle_deg - gt["angle_deg"] + 180) % 360 - 180))
        if abs((best.angle_deg - gt["angle_deg"] + 180) % 360 - 180) <= 20:
            return best
        return None
    return next((c for c in sr.connections if c.id == gt.get("id")), None)


def fmt(v, digits=0):
    return "—" if v is None else f"{v:.{digits}f}"


def main(scan, truth_path):
    sys.stdout.reconfigure(encoding="utf-8")
    truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    t0 = time.time()
    res = load_and_analyze(scan)
    sr = build_scan_result(res, scan, time.time() - t0)

    rows = []
    for key, label in PUT_FIELDS:
        real = truth.get("put", {}).get(key)
        if real is not None:
            rows.append(compare(label, getattr(sr.put, key), real))
    for gt in truth.get("connections", []):
        c = match_connection(sr, gt)
        for key, label in CONN_FIELDS:
            real = gt.get(key)
            if real is None:
                continue
            name = f"{gt.get('id', '?')} {label}"
            if c is None:
                rows.append(dict(name=name, real=real, scanner=None, U=None, status="NOT_DETECTED", confidence="NONE",
                                 error=None, rel_error=None, within_U=None))
            else:
                rows.append(compare(f"{name} (= {c.id})", getattr(c, key), real))

    print("MEASUREMENT VALIDATION")
    print(f"scan: {sr.source['name']}   ground truth: {Path(truth_path).name}\n")
    print(f"{'':<26}{'Scanner':>10}{'± U':>8}{'Real':>9}{'Error':>9}{'Rel.':>8}  {'In ±U':<6} Status")
    for r in rows:
        inside = "—" if r["within_U"] is None else ("ja" if r["within_U"] else "NEE")
        err = "—" if r["error"] is None else f"{r['error']:+.0f} mm"
        rel = "—" if r["rel_error"] is None else f"{r['rel_error']:+.1f}%"
        print(f"{r['name']:<26}{fmt(r['scanner']):>10}{fmt(r['U']):>8}{fmt(r['real']):>9}{err:>9}{rel:>8}  {inside:<6} {r['status']}")

    compared = [r for r in rows if r["error"] is not None]
    if compared:
        mae = sum(abs(r["error"]) for r in compared) / len(compared)
        cover = [r["within_U"] for r in compared if r["within_U"] is not None]
        print(f"\nGemiddelde absolute fout: {mae:.1f} mm over {len(compared)} maten")
        if cover:
            print(f"Werkelijke waarde binnen ±U: {sum(cover)}/{len(cover)} "
                  f"(verwacht ~95% als de onzekerheid klopt; veel minder = onzekerheid te optimistisch)")
    not_measured = [r for r in rows if r["error"] is None]
    if not_measured:
        print(f"Niet vergeleken (UNKNOWN / niet gedetecteerd): {', '.join(r['name'] for r in not_measured)}")

    # guardar para comparar scans en el futuro
    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS / f"{stamp}_{Path(scan).stem}.json"
    out.write_text(json.dumps(dict(scan=sr.source, ground_truth=truth, rows=rows, scan_result=sr.to_dict()),
                              indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    hist = RESULTS / "history.csv"
    new = not hist.exists()
    with hist.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "scan", "sha256", "measure", "scanner", "U", "real", "error", "rel_error_pct",
                        "within_U", "status", "confidence", "scale_rel_sigma"])
        for r in rows:
            w.writerow([stamp, sr.source["name"], sr.source["sha256"], r["name"], r["scanner"], r["U"], r["real"],
                        r["error"], r["rel_error"], r["within_U"], r["status"], r["confidence"],
                        sr.parameters["scale_rel_sigma"]])
    print(f"\nOpgeslagen: {out.relative_to(ROOT)} en {hist.relative_to(ROOT)}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
