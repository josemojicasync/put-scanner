"""ScanQuality: evaluación de la calidad del scan y mensajes concretos de qué falta.

Estados por elemento: GOOD / FAIR / POOR / RESCAN (RESCAN = la medida no es posible con este scan).
Los umbrales son geométricos/físicos (ver comentarios), no ajustados a un scan concreto.
"""
import numpy as np

from analysis import ESTIMATED, MEASURED, SCALE_REL_SIGMA, UNKNOWN, WALLS

GOOD, FAIR, POOR, RESCAN = "GOOD", "FAIR", "POOR", "RESCAN"
ORDER = [GOOD, FAIR, POOR, RESCAN]

COVERAGE_GOOD, COVERAGE_FAIR = 0.8, 0.5   # < 50 % de la pared: el plano se apoya en menos de la mitad
NOISE_GOOD_MM, NOISE_FAIR_MM = 3.0, 6.0   # ruido de profundidad LiDAR/fotogrametría típico: 1–3 mm
SPACING_GOOD_MM, SPACING_FAIR_MM = 10.0, 20.0  # ≤ 1/5 del tubo más pequeño (Ø110) para ver su curvatura
DUPLICATE_MAX = 0.15     # > 15 % de las celdas de pared con una segunda capa a 2–6 cm: superficie duplicada
DRIFT_MAX_M = 0.01       # la misma pared desplazada > 1 cm entre mitad inferior y superior
WATER_LEVEL_TOL_M = 0.03 # superficie horizontal dentro del tubo a ±3 cm del plano de bodem


def _worse(a, b):
    return max(a, b, key=ORDER.index)


def dm_value(c):
    return c["diameter"]["value"]


def duplicate_fractions(res):
    """Fracción de celdas de 10 cm de cada pared con una segunda capa (puntos con la misma orientación
    a 2–6 cm del plano, > 5σ del ruido). Se excluye el entorno de cada abertura de tubo: mortero,
    manguitos y rebordes alrededor de una penetración no forman parte del plano de la pared.
    Un error de registro, en cambio, duplica zonas extensas de pared."""
    walls = res["put"]["walls"]["walls"] or {}
    L, N = res["local_points"], res["local_normals"]
    ch = res["put"]["chamber"]
    out = {}
    for key, ax, _ in WALLS:
        if key not in walls:
            continue
        w = walls[key]
        other = 1 - ax
        m = (np.abs(N[:, ax]) > 0.9) & (np.abs(N[:, 2]) < 0.35) & (np.abs((L - w["c"]) @ w["n"]) < 0.06) & \
            (L[:, 2] > ch["z0"]) & (L[:, 2] < ch["z1"])
        for c in res["connections"]:
            if c["wall"] == key:
                rad = max(c["opening_w"], c["opening_h"]) / 2 + 0.15
                wc = c["wall_center"]
                m &= np.hypot(L[:, other] - wc[other], L[:, 2] - wc[2]) > rad
        Ln = L[m]
        if not len(Ln):
            out[key] = 0.0
            continue
        far = np.abs((Ln - w["c"]) @ w["n"]) > 0.02
        cells = np.floor(np.column_stack([Ln[:, other], Ln[:, 2]]) / 0.10).astype(int)
        _, inv, cnt = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        frac = np.bincount(inv.ravel(), weights=far.astype(float)) / cnt
        out[key] = float(np.mean(frac[cnt >= 5] > 0.3)) if (cnt >= 5).any() else 0.0
    return out


def assess_quality(res):
    put, conns = res["put"], res["connections"]
    items, warnings, missing = {}, [], []

    # ---- densidad, ruido, escala
    spacing_mm = res["spacing"] * 1000
    walls = put["walls"]["walls"]
    if walls:
        noise_mm = float(np.median([w["rms"] for w in walls.values()]) * 1000) if walls else np.nan
    else:
        circ = put.get("circularity")
        noise_mm = circ["value"] * 1000 if circ and circ["value"] is not None else np.nan
    density = GOOD if spacing_mm <= SPACING_GOOD_MM else FAIR if spacing_mm <= SPACING_FAIR_MM else POOR
    if density != GOOD:
        warnings.append(f"Puntdichtheid laag (puntafstand {spacing_mm:.0f} mm): kleine aansluitingen zijn mogelijk "
                        "niet herkenbaar. Scan langzamer en dichterbij.")

    # ---- geometría de la put
    put_state = GOOD
    shape = put["shape"]
    if shape["status"] != MEASURED:
        put_state = POOR
        warnings.append("Vorm van de put niet eenduidig (rechthoekig/rond): maten zijn hooguit geschat.")
    wall_cov = {}
    dups = duplicate_fractions(res)
    if walls is not None:
        for key, _, _ in WALLS:
            if key in walls:
                w = walls[key]
                wall_cov[key] = w["coverage"]
                if w["coverage"] < COVERAGE_FAIR:
                    put_state = _worse(put_state, POOR)
                    warnings.append(f"Wand {key} slechts {w['coverage'] * 100:.0f}% gescand. Scan de hele wand van boven tot onder.")
                elif w["coverage"] < COVERAGE_GOOD:
                    put_state = _worse(put_state, FAIR)
                if dups.get(key, 0.0) > DUPLICATE_MAX:
                    put_state = _worse(put_state, POOR)
                    warnings.append(f"Wand {key}: dubbele geometrie ({dups[key] * 100:.0f}% van de wand in een tweede laag, "
                                    "buiten de aansluitingen). Mogelijke registratiefout van de scan; opnieuw scannen.")
                if np.isfinite(w["drift_m"]) and w["drift_m"] > DRIFT_MAX_M:
                    put_state = _worse(put_state, FAIR)
                    warnings.append(f"Wand {key}: boven- en onderkant liggen {w['drift_m'] * 1000:.0f} mm verschoven. "
                                    "Mogelijke drift van de scan (of werkelijk scheve wand).")
            else:
                wall_cov[key] = 0.0
                missing.append(f"wand {key}")
                put_state = _worse(put_state, POOR)
                warnings.append(f"Wand {key} niet gescand: binnenmaat {'X' if 'X' in key else 'Y'} kan niet direct gemeten worden.")
        if len(put["walls"]["missing"]) >= 2:
            put_state = RESCAN
        for ax_name, key in (("X", "width_x"), ("Y", "width_y")):
            prof = put[key].get("profile")
            if prof and prof["significant"]:   # variación > 3σ del ruido de los planos y > 3 mm
                warnings.append(f"Wanden niet parallel / mogelijke scan drift: binnenmaat {ax_name} varieert "
                                f"{prof['bottom'] * 1000:.0f} -> {prof['top'] * 1000:.0f} mm over de gemeten hoogte "
                                f"(onder -> boven). Opgenomen in de onzekerheid; controleer met een handmaat.")
    else:
        cov = put["walls"]["coverage"]
        wall_cov["wand"] = cov
        if cov < COVERAGE_FAIR:
            put_state = _worse(put_state, POOR)
            warnings.append(f"Putwand slechts {cov * 100:.0f}% gescand. Scan rondom de hele wand.")
        elif cov < COVERAGE_GOOD:
            put_state = _worse(put_state, FAIR)
    if np.isfinite(noise_mm) and noise_mm > NOISE_GOOD_MM:
        put_state = _worse(put_state, FAIR if noise_mm <= NOISE_FAIR_MM else POOR)
        warnings.append(f"Ruis op de wanden {noise_mm:.1f} mm (normaal max. {NOISE_GOOD_MM:.0f} mm): "
                        "bewegingsonscherpte, reflecterend of nat oppervlak.")
    items["Put"] = put_state

    # ---- bodem
    fi = put["floor_info"]
    bottom = GOOD if fi["coverage"] >= 0.6 else FAIR if fi["coverage"] >= 0.3 else POOR
    if bottom != GOOD:
        warnings.append(f"Bodem slechts {fi['coverage'] * 100:.0f}% gescand: het bodemvlak is minder zeker. "
                        "Scan de bodem van bovenaf met de telefoon recht naar beneden.")
    items["Bodem"] = bottom

    # ---- maaiveld
    gi = put["ground_info"]
    if gi is None:
        items["Maaiveld"] = RESCAN
        missing.append("maaiveld")
        warnings.append("Maaiveld niet gescand: diepte alleen tot bovenkant van de wand (geschat). "
                        "Scan ook het terrein rondom de put.")
    else:
        items["Maaiveld"] = GOOD if gi["sectors"] >= 4 else FAIR
        if gi["sectors"] < 4:
            warnings.append(f"Maaiveld maar aan {gi['sectors']}/8 zijden rond de put gescand: helling van het terrein onzeker.")

    # ---- referentievlakken niet eenduidig (meerdere plausibele bodem-/maaiveldvlakken)
    d = put["depth"]
    for name, item, amb, cands in (("bodem", "Bodem", d.get("ambiguity_bottom", 0), put.get("bottom_surface_candidates", [])),
                                   ("maaiveld", "Maaiveld", d.get("ambiguity_top", 0), put.get("reference_surface_candidates", []))):
        if amb > 0:
            items[item] = _worse(items[item], FAIR)
            warnings.append(f"Meerdere mogelijke {name}vlakken ({len(cands)}; tot {amb * 1000:.0f} mm verschil): "
                            "referentie niet eenduidig, diepte geschat en onzekerheid vergroot.")

    # ---- aansluitingen
    conn_cov = {}
    for c in conns:
        cid, st = c["id"], c["diameter"]["status"]
        conn_cov[cid] = c.get("arc_deg", 0.0)
        items[cid] = GOOD if st == MEASURED else FAIR if st == ESTIMATED else RESCAN
        if st == UNKNOWN:
            why = c.get("note") or ""
            warnings.append(f"{cid}: diameter niet betrouwbaar meetbaar (zichtbare boog {c.get('arc_deg', 0):.0f}°, "
                            f"opening {c['opening_w'] * 1000:.0f} mm{'; ' + why if why else ''}). "
                            "Scan meer van de buis, recht ervoor en vanuit een lagere hoek.")
        elif st == ESTIMATED:
            warnings.append(f"{cid}: diameter alleen geschat (zichtbare boog {c.get('arc_deg', 0):.0f}°, "
                            f"{c['diameter'].get('bands', '?')} stabiele banden).")
        occluded = c.get("lower_pipe_occluded")
        if occluded is None:   # método anterior (put redonda)
            hb = c["hidden_bottom"]
            occluded = hb["n"] >= 10 and hb["z"] is not None and not c.get("bottom_seen")
        if occluded:
            hb = c["hidden_bottom"]
            where = "op bodemniveau " if hb.get("z") is not None and abs(hb["z"] - put["floor_z"]) < WATER_LEVEL_TOL_M else ""
            warnings.append(f"{cid}: onderkant van de buis verborgen door een horizontaal vlak {where}"
                            "(water, slib of banket). BOB niet direct zichtbaar.")
        if c["crown"]["status"] == UNKNOWN:
            warnings.append(f"{cid}: kruin niet zichtbaar (bovenkant van de buis niet in de scan).")
        if c.get("possible_reconstruction_fill"):
            warnings.append(f"{cid}: afsluitend vlak dwars in de buis (mogelijke reconstructie-vulling van Polycam); "
                            "uitgesloten van de meting.")
        od = c.get("opening_diameter")
        if od and od["value"] is not None and dm_value(c) is not None and abs(od["value"] - dm_value(c)) > np.hypot(
                od["U"], c["diameter"]["U"]):
            warnings.append(f"{cid}: wandopening Ø{od['value'] * 1000:.0f} mm verschilt van de buis Ø{dm_value(c) * 1000:.0f} mm "
                            "(sparing/kraag/manchet): voor de buis is alleen het buisoppervlak gebruikt.")
        ih, dm = c.get("inner_height"), c["diameter"]
        if ih and ih["value"] is not None and dm["value"] is not None:
            diff = dm["value"] - ih["value"]
            if abs(diff) > np.hypot(ih["U"], dm["U"]):  # verschil groter dan de gecombineerde onzekerheid
                warnings.append(f"{cid}: verticale binnenhoogte {ih['value'] * 1000:.0f} mm wijkt af van de gemiddelde "
                                f"diameter {dm['value'] * 1000:.0f} mm: mogelijke ovaliteit/vervorming van de buis.")

    # ---- kandidaten die GEEN aansluiting werden (geen bevestigde opening in het wandvlak)
    rejected = put.get("rejected_candidates", [])
    reliefs = [x for x in rejected if x["kind"] == "WALL_RELIEF"]
    reliefs += [o for o in put.get("openings", []) if o.get("kind") == "relief"]
    if reliefs:
        walls_r = sorted({x.get("wall") or "?" for x in reliefs})
        warnings.append(f"Wandreliëf (sleuf/nis, geen aansluiting) op wand {', '.join(walls_r)}: niet als aansluiting gemeten.")
    possible = [o for o in put.get("openings", []) if o.get("status") == "POSSIBLE"]
    if possible:
        warnings.append(f"{len(possible)} mogelijke opening(en) met onvoldoende bewijs (geen gesloten contour in het wandvlak "
                        "of niets erachter): niet als aansluiting opgenomen. Controleer ter plaatse.")

    # ---- escala
    wx = put["width_x"]["value"] or put["width_y"]["value"]
    plausible = wx is not None and 0.3 < wx < 5.0
    scale = dict(units="m (Polycam PLY)", verified=False, assumed_rel_sigma=SCALE_REL_SIGMA, plausible=plausible)
    items["Schaal"] = FAIR if plausible else POOR
    if plausible:
        warnings.append(f"Schaal niet geverifieerd met een referentiemaat (aanname ±{SCALE_REL_SIGMA * 100:.1f}%). "
                        "Valideer met handmaten via validation/validate.py.")
    else:
        warnings.append("Afmetingen niet plausibel voor een put in meters: controleer de eenheden van de export.")

    walls_area = None
    if walls:
        ch = put["chamber"]
        h = ch["z1"] - ch["z0"]
        walls_area = sum((2 * ch["hy"] if k[1] == "X" else 2 * ch["hx"]) * h * w["coverage"] for k, w in walls.items())
    n_wall = sum(w["n_used"] for w in walls.values()) if walls else None
    return dict(
        total_points=int(res["n_raw"]), usable_points=int(res["n_clean"]), analysed_points=int(res["n_points"]),
        point_spacing_mm=float(spacing_mm),
        wall_density_pts_m2=float(n_wall / walls_area) if walls_area else None,
        noise_mm=float(noise_mm) if np.isfinite(noise_mm) else None,
        chamber_coverage=float(put["walls"]["coverage"]), wall_coverage=wall_cov, duplicate_fraction=dups,
        bottom_coverage=float(fi["coverage"]), connection_coverage_deg=conn_cov,
        scale=scale, missing_geometry=missing, items=items, warnings=warnings,
    )
