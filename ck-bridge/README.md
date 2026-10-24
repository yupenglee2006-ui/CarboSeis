# CarboSeis-CarboKitten bridge

A small Julia project that runs [CarboKitten.jl](https://github.com/MindTheGap-ERC/CarboKitten.jl)
v0.6.1 from a [CarboSeis](https://github.com/yupenglee2006-ui/CarboSeis) scenario file and writes the HDF5 output that CarboSeis reads.
It ships as the folder `ck-bridge/` of the CarboSeis repository but remains a separate package, so
that CarboSeis (Python) stays independent of a particular CarboKitten release, and so that the code
that loads CarboKitten as a library carries CarboKitten's licence (GPL-3.0-or-later).

## Julia version

The environment is resolved for **Julia 1.12.7** (`julia_version` in `Manifest.toml`). Julia does
not guarantee identical random-number streams across minor versions, so runs made with another
minor version (e.g. 1.13) cannot be mixed with runs made with 1.12. The driver therefore stops,
before loading any package, if Julia has another minor version (patch versions only give a warning),
and records `carboseis_julia_version`, `carboseis_carbokitten_version` and `carboseis_bridge_version`
as attributes of the `input` group of every output file. With
[juliaup](https://github.com/JuliaLang/juliaup), keep your default Julia and select 1.12.7 per
session:

```
juliaup add 1.12.7
$env:JULIAUP_CHANNEL = "1.12.7"        # PowerShell; bash: export JULIAUP_CHANNEL=1.12.7
```

CarboSeis' Makefile and its E3 driver set `JULIAUP_CHANNEL` themselves. For exploratory runs with
another Julia only, set `CARBOSEIS_ALLOW_ANY_JULIA=1`.

## Use

```
cd carboseis                                   # the folder that holds this one
julia --project=ck-bridge -e 'using Pkg; Pkg.instantiate()'
julia --project=ck-bridge ck-bridge/ck_run.jl scenarios/luconia_base.toml
```

The first lines of the output name the bridge, Julia and CarboKitten versions; the line
`carboseis-ck-bridge: starting CarboKitten run` marks the start of the main run, so that CarboSeis
can tell a model failure (after this line) from an environment failure (before it).

The run reproduces the CarboKitten validation case of Hidding et al. (2026): a dome pre-run and a
4 Myr run of a Central Luconia-type isolated platform. Parameters come from the `[carbokitten]`
table of the scenario file; single values can be overridden, e.g.
`--set subsidence_rate_m_per_Myr=160` or `--set facies.1.maximum_growth_rate_m_per_Myr=2000`.
Relative paths in the scenario are resolved against the CarboSeis folder that holds the scenario
(the parent of `scenarios/`), or against `CARBOSEIS_ROOT` if that environment variable is set.

CarboSeis finds the bridge through the environment variable `CARBOSEIS_CK_BRIDGE`, a folder
`ck-bridge` inside the CarboSeis repository (this folder), or a folder
`carboseis-ck-bridge` next to it (the layout used before CarboSeis v5.0).

`Manifest.toml` pins the exact package versions used for the paper.

Before 0.1.0.dev2 this package was called the SeisKitten-CarboKitten bridge (`seiskitten-ck-bridge`, environment variable `SEISKITTEN_ROOT`).

## Notes

`NOTES.md` lists the points specific to CarboKitten v0.6.1 that the driver handles, the stability
range met in CarboSeis's parameter screening, and the output contract that CarboSeis relies on.

## Licence and citation

GPL-3.0-or-later (see `LICENSE`), following CarboKitten.jl. Please cite CarboKitten.jl
(Hidding et al., 2026, Geosci. Model Dev., 19, 6207-6229, https://doi.org/10.5194/gmd-19-6207-2026)
and CarboSeis (see `CITATION.cff`).
