# Compatibility notes

CarboSeis reads the HDF5 files written by CarboKitten.jl v0.6.1. Running CarboKitten from a CarboSeis
scenario is the job of the **CarboSeis-CarboKitten bridge** (`ck-bridge/`, GPL-3.0),
whose `NOTES.md` documents the points that concern running CarboKitten: the `Coast`/`Periodic` names
(formerly #1 here, now B1), the sea-floor elevation handed from the pre-run to the main run (#4, now B2),
`String` file names (#5, now B3) and the stability range met in the E3 screening (#7, now B4).

This file keeps the notes on the CarboSeis side: reading the v0.6.1 output layout (#2, #3), the row of
the initial topography used for a section (#6) and the naming of cached E3 runs (#8). The numbering of
earlier versions is kept so that references in old logs remain valid.

CarboSeis was called SeisKitten before v4.0; paths below use the current names
(`src/carboseis/`), and stand-in files written before v4.0 carry the attribute
`seiskitten_standin`, which the reader still accepts.


---

## 2. Output layout of CarboKitten 0.6.1: sediment thickness instead of bathymetry

- **Symptom**: reading the pre-run HDF5 output raised
  `KeyError: key "bathymetry" not found` (from the bridge `ck_run.jl` /
  `final_bathymetry`, and from `src/carboseis/io.py` when reading output
  groups).  The CarboKitten 0.6.1 `topography` / `profile` groups contain
  `deposition`, `disintegration`, `production` and `sediment_thickness` but no
  `bathymetry`.
- **Cause**: the v0.6 HDF5 writer does not emit `bathymetry`; the previous code
  assumed the older layout and read `fid["topography"]["bathymetry"]` directly.
- **Fix**: recompute the sea floor elevation (m, positive up) from the
  stratigraphic state instead of reading a dataset (see issue #4 for the
  sign convention — CarboKitten's official `water_depth` is *water depth*,
  not elevation, and must not be returned here):

  ```
  bathymetry(t) = initial_topography + sediment_thickness(t)
                  - subsidence_rate * (t - t0)
  ```

  (sea floor elevation, positive up; water depth =
  `sea_level - bathymetry`.)

  - **Julia side** (`final_bathymetry` in the bridge `ck_run.jl`): reads `input/t`,
    `input/initial_topography`, the `subsidence_rate` attribute on `input`,
    the `write_interval` attribute on the `topography` group, and the last
    frame of `topography/sediment_thickness`, then applies the formula for
    the final pre-run frame.  See the docstring above the function.
  - **Python side** (`read_section` in `src/carboseis/io.py`): when an output
    group has no `bathymetry` dataset, the array is recovered per-frame from
    `initial_topography` + `sediment_thickness` - `subsidence_rate*(t-t0)`,
    keeping the same `(frame, y, x)` layout so downstream code is unchanged.
- **Files**: the bridge `ck_run.jl` (`final_bathymetry`), `src/carboseis/io.py`
  (`read_section`).

---

## 3. Byte-string `slice` attribute

- **Symptom**: `read_section` stopped in `_parse_slice` with a
  `ValueError: invalid literal for int()` (or similar) while parsing the
  `slice` attribute of an output group, e.g. the `profile` group.
- **Cause**: HDF5 stores the `slice` attribute as a `bytes` object
  (`b':,50'`).  The previous code passed the raw value through `str(...)`
  first, producing the literal string `"b':,50'"` (with quotes and `b`
  prefix), which `_parse_slice` could not split into `:` / integer parts.
- **Fix**: every string attribute goes through `_decode`, which handles
  `bytes`, `str` and NumPy scalars before parsing.

- **Files**: `src/carboseis/io.py` (`_decode`, `read_section`).

---

## 6. Initial-topography row used for a section (fixed in v3.2)

- **Symptom**: no error.  With real CarboKitten output, the platform interior
  of Fig. 3b was dominated by wackestone and mudstone although the facies were
  euphotic and oligophotic (Fig. 3a); the meteoric rule had almost no effect
  (D1 = D0 in Fig. 6) and the platform showed hardly any internal reflectivity.
- **Cause**: the fallback of fix #2 added the *full* initial topography,
  shape (ny, nx), to the profile's `sediment_thickness`, shape (frames, 1, nx).
  NumPy broadcast this to (frames, ny, nx), and the section selection then took
  row 0, which lies in the basin, instead of the profile row.  Under the
  platform the elevation came out about 115 m too low, so the depositional water
  depth was about 115 m too large and exposure was almost never detected.  The
  time-lines (`z_top`), which use the correct row, were not affected, which is
  why Fig. 1b looked right.  The stand-in wrote a `bathymetry` dataset, so the
  tests never exercised the fallback.
- **Fix**: `read_section` restricts the initial topography to the group's own
  rows and columns before adding `sediment_thickness`.  If a file carries both
  `bathymetry` and `sediment_thickness`, they must agree or reading stops.  The
  stand-in now writes the v0.6.1 layout (no `bathymetry`, byte-string `slice`),
  so tests and development use the same code path as real output.
  `tests/test_io.py` fails on the v3.1 reader and passes on the fix.
- **Consequences**: Figs. 3-9, `tables/e1_metrics.csv`, `tables/e2_agespan.csv`
  and the E3 statistics computed with v3.1 are invalid; Figs. 1, 2 and A1-A3
  are not affected.  The CarboKitten HDF5 files themselves are fine, including
  the cached E3 runs in `build/e3`; only the CarboSeis evaluation has to be
  repeated.  The chain cache (`build/cache`) did not include the code version,
  so it would have served stale results; its key now includes a hash of
  `src/carboseis`.  Delete `build/cache` once anyway.
- **Files**: `src/carboseis/io.py`, `src/carboseis/standin.py`,
  `scripts/common.py`, `tests/test_io.py`.

---

## 8. E3 re-ran cached CarboKitten runs after the folder changed (fixed in v3.5)

- **Symptom**: in a fresh v3.2 folder, `make e3` repeated all seed replicates (about 6 min each)
  although the v3.1 runs had been copied into `build/e3`.  `make figures-e3` started while E3 was
  still in its seed phase failed with `KeyError: 'trajs is not a file in the archive'`.
- **Cause**: run files were named by a hash of the `[carbokitten]` table *including* absolute
  paths (`output`, `sea_level_file`, `prerun.output`), so the same parameters got a different name
  in another folder.  `results.npz` holds no trajectories until the first one is finished.
- **Fix**: names now hash only the parameters (file names reduced to their base name); runs cached
  under the old names in the same folder are renamed automatically, and the plan line printed at
  start-up reports how many runs are already available.  CarboKitten now writes to
  `<run>.partial.h5`, renamed only after success, and a cached file whose last frame was never
  written (interrupted run) is detected and run again, so E3 can be stopped with Ctrl+C at any
  time.  `make figures-e3` explains what is missing instead of failing, and marks figures made
  from an unfinished E3 as PARTIAL.  Progress lines with an estimate of the remaining time are
  printed and appended to `build/e3/status.txt`.
- **Files**: `scripts/e3_run.py`, `scripts/e3_common.py`, `scripts/fig10_e3_morris.py`,
  `scripts/fig11_e3_detectability.py`, `scripts/make_numbers.py`.

---

## 9. Julia launcher failures counted as model failures; Julia minor version (fixed in v4.0.1)

- **Symptom**: after moving to v4.0, the E3 numbers changed (25 -> 44 excluded evaluations, one seed
  replicate lost, nine failures outside the unstable corner).  `build/e3/failures.json` listed only
  `Error: The Julia launcher failed to figure out which juliaup channel to use`, and
  `Pkg.instantiate()` warned that the Manifest was resolved for Julia 1.12.7.
- **Cause**: juliaup could not start Julia for these runs, and `e3_run.py` treated every non-zero
  exit as a CarboKitten failure.  The default Julia was also not the 1.12.7 of the bridge Manifest;
  another minor version need not reproduce the random-number streams of runs made with 1.12.7.
- **Fix**: see bridge NOTES.md B5: one Julia minor version for all runs, recorded
  in every output file and checked by `e3_run.py` and `check_output.py`; environment failures stop E3.
- **Files**: `scripts/common.py`, `scripts/e3_run.py`, `scripts/make_numbers.py`,
  `scripts/check_output.py`, `src/carboseis/output.py`, `Makefile`, bridge `ck_run.jl`.

---

## Verification

`python -m pytest -q` passes (22 tests). Before using a CarboKitten run for figures, run

```
python scripts/check_output.py data/carbokitten/luconia_base.h5 --json build/check_output.json
```

and confirm: `centre_above_flanks` is true, `V1_max_abs_m` is below 1e-6, the median depositional water
depth in the platform interior is a few metres, `max_single_interval_deposit_m` is far below 25 m, and
`warnings` is empty (`--strict` exits with status 1 otherwise), and `provenance` shows
Julia 1.12.x.
