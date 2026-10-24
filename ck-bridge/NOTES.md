# Notes on running CarboKitten v0.6.1 for CarboSeis

CarboKitten.jl (Hidding et al., 2026) is an open, well-documented carbonate forward model whose
HDF5 output and validation scripts made the coupling with CarboSeis straightforward. This file
collects the points we met while driving CarboKitten v0.6.1 from CarboSeis scenario files, so
that others coupling the two codes, and the CarboKitten community, can benefit from them. None of
them requires changes to CarboKitten; the driver `ck_run.jl` handles each of them.

## B1. Two names exported by two packages

`CarboKitten.BoundaryTrait` and `Interpolations` (used here for the sea-level curve) both export
`Coast` and `Periodic`. Loading both with plain `using` makes the names ambiguous. The driver imports
the boundary traits explicitly:

```julia
using CarboKitten.BoundaryTrait: Coast, Periodic
```

## B2. Sea-floor elevation is reconstructed from the sediment thickness

CarboKitten v0.6.1 records, for every output group, `production`, `deposition`, `disintegration`
and the cumulative `sediment_thickness`, together with the `input` group (time, sea level, initial
topography and the subsidence rate as an attribute). The sea-floor elevation (m, positive up) is
therefore rebuilt in the same way as in the research-paper repository
(`add_final_topography` in `runs/ValidationConstants.jl`):

    elevation(x, y, t) = initial_topography(x, y) + sediment_thickness(x, y, t)
                         - subsidence_rate * (t - t0)

and the water depth is `sea_level - elevation`. The driver uses the elevation at the end of the
dome pre-run as the initial topography of the main run. For a slice group, such as the dip section
`slice = (:, j)`, the initial topography must be taken from the same row `j`; CarboSeis's reader
does this.

## B3. File names are passed as `String`

`H5Output(f, input::Input, filename::String)` expects a `String`. Values parsed from command-line
overrides with `split` are `SubString`s, so the driver converts every path with `String(...)`
before handing it to CarboKitten.

## B4. Stability range of the transport scheme in parameter screening

The validation case uses a 400-year time step, forward-Euler transport with adaptive sub-steps,
a transport coefficient of 2 m/yr and a euphotic maximum growth rate of 1800 m/Myr. In
CarboSeis's Morris screening (example E3 of the manuscript) these parameters were varied well beyond the
validated values, up to 4 m/yr and 2400 m/Myr. Runs that combined a transport coefficient of at least
3 m/yr with a euphotic growth rate of at least 2000 m/Myr did not complete: 14 of 18 such
combinations stopped, and none of the other 38 combinations did. Before stopping, CarboKitten
reported a parcel of several hundred times the depositional resolution in a single step, which is
far more than carbonate production can supply and indicates that the explicit transport scheme
had left its stable range for that time step. As expected for explicit schemes, a shorter time
step or an implicit solver should extend the usable range.

When such a parcel exceeds the sediment buffer, `push_sediment!` in `src/SedimentStack.jl`
reaches the branch that is meant to fill the buffer, and the assignment `col .= frac` there
broadcasts the facies vector along the buffer axis, which raises a `DimensionMismatch`. Writing
the facies fractions as a row, for example `col .= frac'`, would let the run continue (and the
large parcel would still be reported by the existing warning). We offer this observation for the
CarboKitten developers to consider; CarboSeis itself does not depend on it, because its E3 driver
records such runs, excludes them and reports how many were excluded.

How CarboSeis treats these runs: `scripts/e3_run.py` writes each one to `build/e3/failures.json`
with its parameters, does not repeat it, and also rejects any completed run that deposits more than
25 m in one output interval (the buffer capacity of 50 x 0.5 m in this set-up).

## B5. One Julia minor version for all runs; launcher failures are not model failures

Julia does not guarantee the same random-number streams across minor versions, so CarboKitten runs
with the same `ca_random_seed` may differ between, for example, Julia 1.12 and 1.13. All runs of one
study (base case, pre-run and every E3 run) must therefore come from one minor version. Since
0.1.0.dev3 the driver stops before loading any package unless Julia has the minor version recorded
in `Manifest.toml` (1.12.7; a different patch version only gives a warning), and stamps the Julia,
CarboKitten and bridge versions on the `input` group of every output file.

In a CarboSeis E3 rerun on a workstation with juliaup, many runs
stopped with `Error: The Julia launcher failed to figure out which juliaup channel to use` before
Julia started, and were recorded as failed CarboKitten runs. Setting `JULIAUP_CHANNEL=1.12.7` selects
the channel explicitly. The driver now prints `carboseis-ck-bridge: starting CarboKitten run` just
before the main run; CarboSeis treats a failure without this line as an environment failure and
stops instead of recording it.

## Output contract used by CarboSeis

| HDF5 item | Shape seen from Python (h5py) | Use in CarboSeis |
| --- | --- | --- |
| `input/t`, `input/sea_level` | (steps + 1,) | time and sea level of every output frame |
| `input/initial_topography` | (ny, nx) | initial elevation, row of the section |
| `input` attribute `subsidence_rate` | scalar, m/Myr | elevation, Eq. above |
| `<group>/deposition`, `<group>/disintegration` | (frames, Y, X, facies) | layer thickness by facies |
| `<group>/sediment_thickness` | (frames, Y, X) | elevation, Eq. above |
| `<group>` attributes `write_interval`, `slice` | scalar, byte string | frame times, section row |
| `input` attributes `carboseis_julia_version`, `carboseis_carbokitten_version`, `carboseis_bridge_version` | strings (written by the driver) | provenance checks |

Frame 0 is the initial state, and frame j >= 1 holds the `write_interval` model steps that end at
`input/t[j * write_interval]`. Julia arrays are column-major, so h5py sees the axes in reverse order.
