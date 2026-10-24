"""Parameter tables of the manuscript as LaTeX tabular snippets (Copernicus rules) in tables/.

File names are those of v3 and do not follow the table numbers of manuscript v4.0:
tab01_scenario = Table 9, tab02_texture = Table 2, tab03_diagenesis = Table 3,
tab04_rockphysics = Table 4, tab05_e3_ranges = Table 11 (tabA1_verification = Table 7 is written
by figA_verification.py).  Table 1 (user inputs) and Table 5 (earth-model units) are in the
manuscript itself."""
from common import TAB, TEXTURE_LABELS, scenario
from e3_run import PARAMS

sc = scenario()
ck = sc["carbokitten"]


def write(name, header, rows, cols):
    with open(TAB / f"{name}.tex", "w") as f:
        f.write(f"\\begin{{tabular}}{{{cols}}}\n\\tophline\n{header} \\\\\n\\middlehline\n")
        for r in rows:
            f.write(" & ".join(str(c) for c in r) + " \\\\\n")
        f.write("\\bottomhline\n\\end{tabular}\n")
    print(f"wrote tables/{name}.tex")


# Table 9: forward-model and survey parameters of the base case
rows = [("Grid", f"{ck['grid_size'][0]} $\\times$ {ck['grid_size'][1]} cells of {ck['phys_scale_m']:.0f} m, periodic"),
        ("Time", f"{ck['t0_Myr']} Myr start, {ck['dt_yr']:.0f} yr steps, {ck['steps']} steps"),
        ("Sea level", "CarboKitten validation curve, 8.0--15.5 Ma window"),
        ("Subsidence", f"{ck['subsidence_rate_m_per_Myr']:.0f} m Myr$^{{-1}}$"),
        ("Disintegration / lithification", f"{ck['disintegration_rate_m_per_Myr']:.0f} m Myr$^{{-1}}$ / {ck['lithification_time_yr']:.0f} yr"),
        ("Insolation", f"{ck['insolation_W_m2']:.0f} W m$^{{-2}}$"),
        ("Section output", f"row {ck['profile_row']}, every {ck['profile_write_interval']} steps"),
        ("Water depth / overburden", f"{sc['earthmodel']['water_depth_m']:.0f} m / {sc['earthmodel']['overburden_m']:.0f} m"),
        ("Seismic", f"{sc['seismic']['dominant_frequency_Hz']:.0f} Hz Ricker, {sc['seismic']['dt_ms']:.0f} ms, {sc['earthmodel']['trace_spacing_m']:.0f} m traces")]
for f in ck["facies"]:
    if f["type"] == "benthic":
        rows.append((f"Facies: {f['name']}", f"$g_m$ {f['maximum_growth_rate_m_per_Myr']:.0f} m Myr$^{{-1}}$, $k$ {f['extinction_coefficient_per_m']} m$^{{-1}}$"))
    elif f["type"] == "pelagic":
        rows.append((f"Facies: {f['name']}", f"{f['maximum_growth_rate_per_Myr']} Myr$^{{-1}}$ to {f['maximum_production_depth_m']:.0f} m"))
    else:
        rows.append((f"Facies: {f['name']}", "no production; receives disintegrated oligophotic sediment"))
write("tab01_scenario", "Parameter & Value", rows, "ll")

# Table 2: texture membership
tx = sc["texture"]
rows = []
for cls in ("shallow", "mid", "deep"):
    for f, row in zip(ck["facies"], tx[cls]):
        rows.append([cls if f is ck["facies"][0] else "", f["name"]] + [f"{v:.2f}" for v in row])
write("tab02_texture", "Depth class & Facies & " + " & ".join(t[:5] + "." for t in TEXTURE_LABELS), rows, "ll" + "c" * 6)

# Table 3: diagenesis
d = sc["diagenesis"]
rows = [("Meteoric exposure", f"$\\tau_{{crit}}$ = {d['tau_crit_Myr']} Myr, $\\Delta\\phi_{{mold}}$ = {d['dphi_mold']}, $\\Delta\\phi_{{cem}}$ = {d['dphi_cem']}"),
        ("Burial cementation", f"$z_{{ch}}$ = {d['z_ch_m']:.0f} m, $c_{{ch}}$ = {d['c_ch_per_km']} km$^{{-1}}$"),
        ("Dolomitization (optional)", f"top {d['dolomite_zone_m']:.0f} m, fraction {d['dolomite_fraction']}, $\\Delta\\phi$ = {d['dphi_dolomite']}"),
        ("Deep-burial leaching (optional)", f"top {d['leaching_zone_m']:.0f} m, $\\Delta\\phi_{{vug}}$ = {d['dphi_vug']}"),
        ("Meteoric zone thickness", f"{sc['grid']['meteoric_zone_m']:.0f} m")]
for t, pt in zip(TEXTURE_LABELS, d["pore_types"]):
    rows.append((f"Pore types: {t.lower()}", " / ".join(f"{v:.2f}" for v in pt) + " (inter. / stiff / crack)"))
write("tab03_diagenesis", "Rule & Parameters", rows, "ll")

# Table 4: rock physics and compaction
r = sc["rockphysics"]
rows = [(n, f"{v[0]}", f"{v[1]}", f"{v[2]}") for n, v in
        (("Calcite", r["calcite"]), ("Dolomite", r["dolomite"]), ("Clay", r["clay"]),
         ("Brine", r["brine"]), ("Gas", r["gas"]))]
rows.append(("Pore aspect ratios", "inter. " + str(r["aspect_ratios"][0]), "stiff " + str(r["aspect_ratios"][1]),
             "crack " + str(r["aspect_ratios"][2])))
for t, p0, c in zip(TEXTURE_LABELS, sc["compaction"]["phi0"], sc["compaction"]["c_per_km"]):
    rows.append((f"Compaction: {t.lower()}", f"$\\phi_0$ {p0}", f"$c$ {c} km$^{{-1}}$", ""))
write("tab04_rockphysics", "Material & $K$ (GPa) & $\\mu$ (GPa) & $\\rho$ (g cm$^{-3}$)", rows, "llll")

# Table 11: E3 parameters
rows = [(p[0].replace("_", "\\_"), f"{p[2]:g}", f"{p[3]:g}", "yes" if p[4] else "no") for p in PARAMS]
write("tab05_e3_ranges", "Parameter & Low & High & New forward run", rows, "lccc")
