# CarboSeis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23608277.svg)](https://doi.org/10.5281/zenodo.23608277)

A workflow for constructing seismic models of carbonate platforms simulated with
[CarboKitten.jl](https://github.com/MindTheGap-ERC/CarboKitten.jl).

Given a carbonate formation simulated with CarboKitten, CarboSeis builds its seismic model once the
user supplies the petrophysical parameters of the formation. The user constructs the forward model
with CarboKitten (or supplies an existing run), describes the formation in one scenario file, and
obtains an acoustic earth model and synthetic seismic sections. Each step uses established methods;
CarboSeis integrates them into a verified and reproducible chain in which rock properties follow
the simulated depositional, exposure and burial history of every layer:

1. **Stratigraphic grid** — time-lines, water depth at deposition and exposure time from CarboKitten's HDF5 output
2. **Texture** — producer fractions and depositional water depth mapped to Dunham-type textures
3. **Compaction** — solid-volume-conserving, texture-dependent Athy trends
4. **Diagenesis** — rule-based meteoric, burial and optional dolomitization and leaching overprints
5. **Rock physics** — Xu–Payne-type differential effective medium with three pore types, then Gassmann
6. **Earth model** — water, clastic overburden, onlapping fill and basement; exact Backus averaging onto a depth grid
7. **Seismic** — normal-incidence 1-D convolution in two-way time and point-spread-function (PSF) imaging, optional band-limited noise

Status: version 0.1.0, research code accompanying a model description paper in preparation for
*Geoscientific Model Development* (the manuscript is a separate folder beside this one, see
[Manuscript](#manuscript)). CarboSeis was called SeisKitten during development.

## Install

```
git clone https://github.com/yupenglee2006-ui/CarboSeis.git carboseis && cd carboseis
python -m pip install -e ".[dev]"
python scripts/fetch_inputs.py        # downloads the validation sea-level curve
python -m pytest -q                   # verification, reader, command-line and safeguard tests
```

Python 3.11 or newer; depends on NumPy, SciPy, h5py and Matplotlib.

## Build a seismic model

```
carboseis run scenarios/luconia_base.toml --out build/luconia_base_carboseis.h5
```

reads the CarboKitten output named in the scenario (`[carbokitten] output`) and writes one HDF5
file with every intermediate quantity and both seismic sections. Use another CarboKitten run with
`--input path/to/run.h5`. Single parameters can be changed without editing the scenario file:

```
carboseis run scenarios/luconia_base.toml --set rockphysics.gas_column_m=100 \
    --set seismic.dominant_frequency_Hz=45 --out build/gas_45Hz.h5
```

Other options: `--levels 1` (1-D convolution only, faster), `--frequency HZ`,
`--constant-impedance` (one impedance per facies, the conventional reference of example E1) and
`--standin` (test data, see below). The same chain is available from Python:

```python
import carboseis
sc = carboseis.load_scenario("scenarios/luconia_base.toml", {"diagenesis.dphi_mold": 0.10})
res = carboseis.run_chain(sc)                  # steps 1-7; res.layers, res.em, res.l1, res.l2
carboseis.write_result("build/result.h5", res, sc)
```

and each step can be called on its own (`grid.build_grid`, `properties.texture_fractions`,
`properties.compact`, `properties.diagenesis`, `rockphysics.elastic_properties`,
`earthmodel.assemble`, `seismic.level1`, `seismic.level2`).

### What the user supplies

The scenario file holds two kinds of input (manuscript Sect. 2, Table 1). The `[carbokitten]`
table describes the forward model and is read by the CarboSeis-CarboKitten bridge; changing it
needs a new CarboKitten run. All other tables are the petrophysical description of the formation;
changing them reuses the stored run, so a new seismic model takes seconds to minutes.

| Step | Scenario table | Parameters |
| --- | --- | --- |
| Forward model | `[carbokitten]` | grid, time step, sea level, subsidence, insolation, producers, disintegration, transport |
| 1 Stratigraphic grid | `[grid]` | section group, meteoric zone thickness and profile |
| 2 Texture | `[texture]` | membership matrix per depth class, wave bases, transition width |
| 3 Compaction | `[compaction]` | depositional porosity and Athy coefficient per texture (overburden in `[earthmodel]`) |
| 4 Diagenesis | `[diagenesis]` | pore-type partition per texture; meteoric, burial, dolomitization and leaching rules |
| 5 Rock physics | `[rockphysics]` | mineral and fluid moduli and densities, pore aspect ratios, gas column |
| 6 Earth model | `[earthmodel]` | water depth, overburden, clastic velocity trend, basement, grid spacing |
| 7 Seismic | `[seismic]` | wavelet, dominant frequency, sample interval, PSF dip limit and window, noise |

### What CarboSeis writes

| HDF5 group | Content |
| --- | --- |
| `/` (attributes) | CarboSeis version, creation time, scenario name, forward-model file, stand-in flag |
| `/scenario` | the scenario file as read and all parameters after `--set` overrides (JSON) |
| `/layers` | per CarboKitten layer and column: time-lines, water depth, exposure time, facies and texture fractions, compacted thickness, burial depth, porosity by pore type, mineral fractions, Vp, Vs, density, impedance |
| `/earth_model` | depth grid: Vp, Vs, density, impedance, reflectivity, unit code, layer index, age, porosity, dominant texture and facies |
| `/seismic/level1`, `/seismic/level2` | synthetic sections in two-way time (`t`, `x`, `data`); level 2 also holds the PSF image in depth |

Every dataset has a `units` attribute. The layout is documented in `src/carboseis/output.py`.

## Run CarboKitten (Julia)

CarboKitten is run through the **CarboSeis-CarboKitten bridge** (`ck-bridge/`, GPL-3.0-or-later),
which holds the Julia driver, the pinned Julia environment and notes on CarboKitten v0.6.1. It
ships inside this folder; a checkout without it can point to another copy with `CARBOSEIS_CK_BRIDGE`
(the lookup order is: environment variable, `ck-bridge/`, then `../carboseis-ck-bridge`).

```
make carbokitten          # = julia --project=ck-bridge ck-bridge/ck_run.jl scenarios/luconia_base.toml
```

### Julia version

All CarboKitten runs of one study must come from the same Julia minor version: Julia does not
guarantee identical random-number streams across minor versions, so a run with the same seed can
differ between, for example, Julia 1.12 and 1.13. The bridge is resolved for **Julia 1.12.7**
(`julia_version` in its `Manifest.toml`). With [juliaup](https://github.com/JuliaLang/juliaup), install
it once next to your default Julia:

```
juliaup add 1.12.7
```

`make` and `scripts/e3_run.py` then select it through the `JULIAUP_CHANNEL` environment variable, so
a newer default Julia (e.g. 1.13) can stay installed. Three safeguards keep runs consistent:

- the bridge driver stops before loading CarboKitten if Julia has another minor version, and
  records the Julia, CarboKitten and bridge versions in every output file;
- `scripts/check_output.py` reports these versions and warns if they are missing or differ;
- `scripts/e3_run.py` starts Julia once before any run, refuses cached runs from another Julia
  version or without a recorded version, and stops (instead of recording a model failure) when
  Julia or its environment fails to start, e.g. a juliaup launcher error.

For exploratory work with another Julia only, set `CARBOSEIS_ALLOW_ANY_JULIA=1`.

### Base case

The driver reproduces the CarboKitten validation case (dome pre-run, then a 4 Myr run of a
Central Luconia-type platform) and writes `data/carbokitten/luconia_base.h5`. CarboKitten itself
is used unmodified. Overrides: `--set subsidence_rate_m_per_Myr=160`.

**Check every run before using it.**

```
python scripts/check_output.py data/carbokitten/luconia_base.h5 --json build/check_output.json
```

prints the platform relief (centre above flanks), the final water depth, the V1 consistency of the
time-lines, the depositional water depth and exposure time in the platform interior, the texture
budget and the largest single deposit, and lists any warnings. Notes on reading CarboKitten v0.6.1 output are in
[COMPATIBILITY.md](COMPATIBILITY.md); notes on running CarboKitten are in the bridge's `NOTES.md`.

## Start from scratch

To regenerate every CarboKitten run with the current versions (recommended after installing a new
version of CarboSeis or the bridge):

```
make clean-runs          # removes data/carbokitten (base case and pre-run) and build/ (all E3 runs, caches)
make reproduce           # inputs, base case, check, tests, figures, tables, E1 diagnostics, E3, paper
```

`make reproduce` runs, in order: `inputs carbokitten check test figures tables diagnostics e3
figures-e3 paper`; `make check` stops the sequence if the base case has any warning, before the
hours of E3. The Makefile works with a POSIX shell and with Windows cmd.exe (GNU make for Windows).
On Windows PowerShell without `make`, the same steps are:

```
Remove-Item -Recurse -Force data\carbokitten, build
python scripts/fetch_inputs.py
$env:JULIAUP_CHANNEL = "1.12.7"
julia --project=ck-bridge -e "using Pkg; Pkg.instantiate()"
julia --project=ck-bridge ck-bridge/ck_run.jl scenarios/luconia_base.toml
python scripts/check_output.py --json build/check_output.json --strict
cd scripts
foreach ($f in "fig01_workflow","fig02_scenario","fig03_base_geology","fig04_base_elastic","fig05_base_seismic","fig06_e1_sections","fig07_e1_metrics","fig08_e2_timelines","fig09_e2_agespan","figA_verification") { python "$f.py" }
python make_tables.py; python base_stats.py; python e1_diagnostics.py; python make_numbers.py
python e3_run.py --r 10 --seeds 10
python fig10_e3_morris.py; python fig11_e3_detectability.py; python make_numbers.py
```

## Reproduce the figures and tables of the paper

```
make figures tables      # Figs. 1-9 and A1-A3, parameter tables; minutes
make diagnostics         # E1 layer-scale diagnostics (about 15 minutes, no new CarboKitten runs)
make e3 figures-e3       # E3 Morris screening (66 CarboKitten runs, hours) and Figs. 10-11
make paper               # syncs figures/ and tables/ into the manuscript folder, then compiles it
```

`make tables` also writes `tables/base_case.csv` (base-case statistics by zone, `base_stats.py`) and
`tables/numbers.tex` (`make_numbers.py`), which defines a LaTeX macro for every result quoted in the
manuscript, so the prose always matches the latest run. The chain cache in `build/cache` can grow
to several gigabytes; `make clean-cache` empties it.

Every figure has one script in `scripts/` (`fig01_workflow.py` ... `fig11_e3_detectability.py`,
`figA_verification.py`). If the CarboKitten output exists the scripts use it; otherwise they fall
back to a **development stand-in** (`carboseis.standin`): a simplified Python emulation of
CarboKitten that writes the same HDF5 layout as CarboKitten v0.6.1. Stand-in figures carry a red
"STAND-IN DATA" watermark and must not be used for publication. Force the stand-in with
`CARBOSEIS_STANDIN=1`. Intermediate results are cached in `build/cache`; the cache key includes
the scenario, the input file and a hash of the `carboseis` sources.

Example E3 (`scripts/e3_run.py`) needs 140 chain evaluations and 66 forward runs. Forward runs are
cached in `build/e3` under names that depend only on their CarboKitten parameters, so an
interrupted run resumes where it stopped (Ctrl+C is safe at any time). Progress, with an estimate of
the remaining time, is printed and appended to `build/e3/status.txt`; run `make figures-e3` after
the final "E3 finished" message. Runs that CarboKitten cannot complete, or that deposit more than
25 m in one output interval, are recorded in `build/e3/failures.json` and excluded (bridge
NOTES.md B4). Runs that fail before CarboKitten starts are not model failures: E3 stops and asks
for the environment to be fixed, and `make_numbers.py` leaves the E3 numbers pending if any such
failure is still recorded.

## Pack a CarboKitten run

```
python scripts/pack_output.py data/carbokitten/luconia_base.h5
```

writes `luconia_base_profile.h5` with only the `input` group and the section that CarboSeis reads
(re-compressed, values bit-identical, checked on writing); it is usually less than half the size of
the full output.

## Repository layout

| Path | Content |
| --- | --- |
| `src/carboseis/` | `io` (CarboKitten reader), `grid`, `properties` (texture, compaction, diagenesis), `rockphysics`, `earthmodel`, `seismic`, `pipeline`, `output` (HDF5 writer), `cli`, `analysis`, `standin` |
| `scenarios/luconia_base.toml` | scenario of the example application, read by both the bridge and CarboSeis |
| `scripts/` | one script per figure, `make_tables.py`, `base_stats.py`, `e1_diagnostics.py`, `make_numbers.py`, `e3_run.py`, `check_output.py`, `pack_output.py`, `fetch_inputs.py`, `sync_paper.py` |
| `tests/` | verification tests V1–V6, reader tests for the CarboKitten v0.6.1 layout, command-line tests |
| `ck-bridge/` | the CarboSeis-CarboKitten bridge: a Julia project that runs CarboKitten v0.6.1 from a CarboSeis scenario and writes the HDF5 file this package reads. Separate package, GPL-3.0-or-later, pinned to Julia 1.12.7 |
| `figures/`, `tables/` | generated by `scripts/` (Figs. 1–11 and A1–A3, parameter tables, `numbers.tex`) |

## Manuscript

This repository is the code of the paper; the manuscript is a **separate folder beside it**,
`../CarboSeis_manuscript_v5.0`, holding `manuscript.tex`, the bibliography, the Copernicus class and
its own copies of `figures/*.pdf` and `tables/*.tex`. The copies let the manuscript compile without
this folder, which is what the journal asks for; they are refreshed from the code:

```
make paper                       # syncs figures/ and tables/, then compiles the manuscript
python scripts/sync_paper.py     # sync only; --paper PATH or CARBOSEIS_PAPER=... to point elsewhere
```

## Data

The validation sea-level curve (file Morley_2021.txt) is not redistributed here; `scripts/fetch_inputs.py`
downloads it from the CarboKitten research-paper repository (pinned commit, checksum verified).

## Licence

MIT (see `LICENSE`). CarboSeis reads CarboKitten's output files and contains no CarboKitten code. The Julia driver, which loads CarboKitten.jl (GPL-3.0) as a
library, lives in `ck-bridge/` under GPL-3.0-or-later; it is a separate package with its own
manifest and licence file.

## Citation

See `CITATION.cff` (version 0.1.0, https://doi.org/10.5281/zenodo.23608277). Please also cite CarboKitten.jl: Hidding et al. (2026),
Geosci. Model Dev., 19, 6207–6229, https://doi.org/10.5194/gmd-19-6207-2026.
