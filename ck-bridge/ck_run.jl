# SPDX-License-Identifier: GPL-3.0-or-later
#
# CarboSeis-CarboKitten bridge: run CarboKitten.jl (ALCAP model) from a CarboSeis scenario file.
#
#   julia --project=<bridge> <bridge>/ck_run.jl <carboseis>/scenarios/luconia_base.toml [--set key=value ...]
#
# Mirrors runs/validation_prerun.jl, runs/validation_case.jl and runs/ValidationConstants.jl of the
# CarboKitten research-paper repository (Hidding et al., 2026). The dome pre-run is executed first
# when its output does not exist yet; its final sea-floor elevation becomes the initial topography.
# Overrides use dotted keys inside [carbokitten], e.g. --set subsidence_rate_m_per_Myr=160
# or --set facies.1.maximum_growth_rate_m_per_Myr=2000 (facies index is 1-based).
# Relative paths in the scenario are resolved against the CarboSeis repository, i.e. the parent of
# the folder that holds the scenario file (override with the environment variable CARBOSEIS_ROOT).
# Written for CarboKitten v0.6.1; see NOTES.md for the points that are specific to this version.
#
# Safeguards (0.1.0.dev3, NOTES.md B5): the driver stops before loading any package unless Julia has
# the minor version this environment was resolved with (julia_version in Manifest.toml), prints a
# start marker just before each CarboKitten run, and records the Julia, CarboKitten and bridge
# versions as attributes of the `input` group of every output file.

const BRIDGE_VERSION = "0.1.0.dev3"
const START_MARKER = "carboseis-ck-bridge: starting CarboKitten run"   # CarboSeis looks for this line

using TOML
let want = VersionNumber(TOML.parsefile(joinpath(@__DIR__, "Manifest.toml"))["julia_version"])
    if (VERSION.major, VERSION.minor) != (want.major, want.minor)
        msg = "Julia $(VERSION) is running, but the CarboSeis-CarboKitten bridge was resolved for Julia " *
              "$(want). Julia does not guarantee the same random-number streams across minor versions, " *
              "so runs from different minor versions must not be mixed. With juliaup: " *
              "`juliaup add $(want)` and set the environment variable JULIAUP_CHANNEL=$(want)."
        if get(ENV, "CARBOSEIS_ALLOW_ANY_JULIA", "") == "1"
            @warn msg
        else
            println(stderr, "CARBOSEIS-ENVIRONMENT-ERROR: ", msg)
            exit(3)
        end
    elseif VERSION != want
        @warn "Julia $(VERSION) differs from $(want) only in the patch version; continuing."
    end
end

using CarboKitten
# NOTES.md B1: `Coast` and `Periodic` are exported both by `CarboKitten.BoundaryTrait` and by
# `Interpolations` (loaded below).  Importing them through the explicit
# `using CarboKitten.BoundaryTrait: Coast, Periodic` line keeps the names unambiguous.
using CarboKitten.BoundaryTrait: Coast, Periodic
using CarboKitten.Boxes: Box
using Unitful
using TOML
using DelimitedFiles
using Interpolations
using HDF5
using Statistics: mean

const CK_VERSION = string(pkgversion(CarboKitten))
println("carboseis-ck-bridge $(BRIDGE_VERSION): Julia $(VERSION), CarboKitten $(CK_VERSION)")
flush(stdout)

# Record the versions in the output file (attributes of the `input` group).  A failure here never
# turns a finished run into a failed one; CarboSeis then reports the run as unversioned.
function stamp!(path)
    try
        h5open(path, "r+") do fid
            g = fid["input"]
            write_attribute(g, "carboseis_julia_version", string(VERSION))
            write_attribute(g, "carboseis_carbokitten_version", CK_VERSION)
            write_attribute(g, "carboseis_bridge_version", BRIDGE_VERSION)
        end
    catch err
        println(stderr, "warning: could not record the versions in ", path, ": ", err)
    end
end

# Repository root of the CarboSeis checkout that owns the scenario file; set in load_scenario.
const ROOT = Ref{String}(pwd())
# NOTES.md B3: `rootpath` returns a `String`, never a `SubString`, because CarboKitten's
# `H5Output(f, input::Input, filename::String)` method expects a `String` file name.
rootpath(p) = isabspath(p) ? String(p) : joinpath(ROOT[], p)

function parse_value(s)
    for T in (Int, Float64)
        v = tryparse(T, s)
        v === nothing || return v
    end
    s == "true" && return true
    s == "false" && return false
    return s
end

function setdotted!(d, key, value)
    parts = split(key, ".")
    for p in parts[1:end-1]
        d = d isa AbstractVector ? d[parse(Int, p)] : d[p]
    end
    last = parts[end]
    if d isa AbstractVector
        d[parse(Int, last)] = value
    else
        d[last] = value
    end
end

function load_scenario(args)
    path = String(args[1])
    ROOT[] = get(ENV, "CARBOSEIS_ROOT", normpath(joinpath(dirname(abspath(path)), "..")))
    ck = TOML.parsefile(path)["carbokitten"]
    i = 2
    while i <= length(args)
        if args[i] == "--set"
            k, v = split(args[i+1], "="; limit=2)
            setdotted!(ck, k, parse_value(v))
            i += 2
        else
            error("unknown argument $(args[i])")
        end
    end
    return ck
end

function make_box(ck)
    gs = Tuple(ck["grid_size"])
    scale = ck["phys_scale_m"] * u"m"
    ck["boundary"] == "periodic" ? Box{Periodic{2}}(grid_size=gs, phys_scale=scale) :
                                   Box{Coast}(grid_size=gs, phys_scale=scale)
end

function sea_level_function(ck)
    data, _ = readdlm(rootpath(ck["sea_level_file"]), header=true)
    lo, hi = ck["sea_level_window_Ma"]
    keep = (data[:, 1] .>= lo) .& (data[:, 1] .<= hi)
    d = data[keep, :]
    d = d[sortperm(d[:, 1], rev=true), :]
    times = -Float64.(d[:, 1])                    # Myr, increasing (negative = before present)
    itp = linear_interpolation(times, Float64.(d[:, 2]))
    t0 = ck["t0_Myr"]
    scale = get(ck, "sea_level_scale", 1.0)
    stretch = get(ck, "sea_level_time_stretch", 1.0)
    tend = t0 + ck["steps"] * ck["dt_yr"] * 1e-6
    tgrid = range(t0, tend, length=2001)
    m = mean(itp.(t0 .+ (tgrid .- t0) ./ stretch))
    return t -> (m + scale * (itp(t0 + (ustrip(u"Myr", t) - t0) / stretch) - m)) * u"m"
end

function facies_list(ck)
    out = []
    for f in ck["facies"]
        tc = f["transport_coefficient_m_per_yr"] * u"m/yr"
        if f["type"] == "benthic"
            push!(out, ALCAP.Facies(viability_range=(4, 10), activation_range=(6, 10), active=f["active"],
                production=BenthicProduction(
                    maximum_growth_rate=f["maximum_growth_rate_m_per_Myr"] * u"m/Myr",
                    extinction_coefficient=f["extinction_coefficient_per_m"] * u"m^-1",
                    saturation_intensity=f["saturation_intensity_W_m2"] * u"W/m^2"),
                transport_coefficient=tc, name=f["name"]))
        elseif f["type"] == "pelagic"
            push!(out, ALCAP.Facies(viability_range=(4, 10), activation_range=(6, 10), active=f["active"],
                production=PelagicProduction(
                    maximum_growth_rate=f["maximum_growth_rate_per_Myr"] * u"1/Myr",
                    extinction_coefficient=f["extinction_coefficient_per_m"] * u"m^-1",
                    saturation_intensity=f["saturation_intensity_W_m2"] * u"W/m^2",
                    maximum_production_depth=f["maximum_production_depth_m"] * u"m"),
                transport_coefficient=tc, name=f["name"]))
        else
            push!(out, ALCAP.Facies(active=false, transport_coefficient=tc, name=f["name"]))
        end
    end
    return out
end

function transfer_function(T)
    nf = length(T)
    return f -> stack([sum(T[i][j] .* f[j, :, :] for j in 1:nf) for i in 1:nf], dims=1)
end

function run_prerun(ck, box, facies)
    pr = ck["prerun"]
    gx, gy = ck["grid_size"]
    L = ck["phys_scale_m"] * u"m"
    X, Y = gx * L, gy * L
    radius = pr["dome_radius_fraction"] * min(X, Y)
    top, floor_, slope = pr["dome_top_m"] * u"m", pr["dome_floor_m"] * u"m", pr["dome_flank_slope"]
    dome(x, y) = let dist = sqrt((x - X / 2)^2 + (y - Y / 2)^2)
        dist <= radius ? top : max(floor_, top - slope * (dist - radius))
    end
    period = pr["sea_level_period_Myr"] * u"Myr"
    amp, base = pr["sea_level_amplitude_m"] * u"m", pr["sea_level_base_m"] * u"m"
    input = ALCAP.Input(
        tag=ck["tag"] * "_prerun", box=box,
        time=TimeProperties(Δt=pr["dt_yr"] * u"yr", steps=pr["steps"]),
        output=Dict(:topography => OutputSpec(write_interval=pr["steps"])),
        ca_interval=pr["ca_interval"], ca_random_seed=pr["ca_random_seed"],
        initial_topography=dome,
        sea_level=t -> sin(t * 2π / period) * amp + base,
        subsidence_rate=pr["subsidence_rate_m_per_Myr"] * u"m/Myr",
        disintegration_rate=pr["disintegration_rate_m_per_Myr"] * u"m/Myr",
        insolation=ck["insolation_W_m2"] * u"W/m^2",
        sediment_buffer_size=ck["sediment_buffer_size"],
        depositional_resolution=ck["depositional_resolution_m"] * u"m",
        lithification_time=pr["lithification_time_yr"] * u"yr",
        disintegration_transfer=transfer_function(pr["disintegration_transfer"]),
        facies=facies)
    path = rootpath(pr["output"])
    mkpath(dirname(path))
    println("carboseis-ck-bridge: starting CarboKitten pre-run")
    flush(stdout)
    run_model(Model{ALCAP}, input, path)
    stamp!(path)
    return path
end

# Initial sea-floor elevation for the main run (NOTES.md B2): taken from the final state of the
# pre-run output (dome), as m positive up.  CarboKitten v0.6.1 records the cumulative sediment
# thickness rather than the elevation, so the elevation is rebuilt as in the research-paper
# repository (`add_final_topography` in runs/ValidationConstants.jl):
#
#   elevation = initial_topography + sediment_thickness(t) - subsidence_rate * (t - t0)
#
# (water depth = sea_level - elevation; the water depth is not the quantity needed here.)
# Inputs read from the pre-run HDF5:
#   - `input/t`                 : model time for every output step (Myr)
#   - `input/sea_level`         : sea level at every output step
#   - `input/initial_topography`: pre-run initial dome (ny, nx)
#   - attribute `subsidence_rate` on `input`
#   - attribute `write_interval` on the `topography` group
#   - `topography/sediment_thickness` : its last frame is the final sediment thickness
function final_bathymetry(path)
    """Sea floor elevation (m, positive up) at the end of the pre-run.

    CarboKitten v0.6 does not write a bathymetry data set.  Its official
    ``water_depth`` is *water depth*, not elevation:

        water_depth = sea_level - initial_topography - sediment_thickness
                      + subsidence_rate * (t - t0)

    so the sea floor elevation (positive up, as used for ``initial_topography``
    of the main run and by CarboSeis's io.read_section) is

        bathymetry = initial_topography + sediment_thickness
                     - subsidence_rate * (t - t0).

    Reads input/t, input/sea_level, input/initial_topography, attribute
    subsidence_rate, and the last frame of topography/sediment_thickness.
    """
    h5open(path, "r") do fid
        inp = fid["input"]
        t = read(inp["t"])
        h0 = read(inp["initial_topography"])            # (ny, nx)
        sub = read_attribute(inp, "subsidence_rate")
        grp = fid["topography"]
        wi = read_attribute(grp, "write_interval")
        th = read(grp["sediment_thickness"])            # (ny, nx, nw+1)
        n = size(th, 3)
        idx = 1 + (n - 1) * wi
        Σ = sub * (t[idx] - t[1])
        return (h0 .+ th[:, :, n] .- Σ) .* u"m"
    end
end

function main(args)
    ck = load_scenario(args)
    box = make_box(ck)
    facies = facies_list(ck)
    prerun_path = rootpath(ck["prerun"]["output"])
    isfile(prerun_path) || run_prerun(ck, box, facies)
    input = ALCAP.Input(
        tag=ck["tag"], box=box,
        time=TimeProperties(t0=ck["t0_Myr"] * u"Myr", Δt=ck["dt_yr"] * u"yr", steps=ck["steps"]),
        output=Dict(
            :topography => OutputSpec(write_interval=ck["volume_write_interval"]),
            :profile => OutputSpec(slice=(:, ck["profile_row"]), write_interval=ck["profile_write_interval"])),
        ca_interval=ck["ca_interval"], ca_random_seed=ck["ca_random_seed"],
        initial_topography=final_bathymetry(prerun_path),
        sea_level=sea_level_function(ck),
        subsidence_rate=ck["subsidence_rate_m_per_Myr"] * u"m/Myr",
        transport_solver=Val{:forward_euler},
        disintegration_rate=ck["disintegration_rate_m_per_Myr"] * u"m/Myr",
        insolation=ck["insolation_W_m2"] * u"W/m^2",
        sediment_buffer_size=ck["sediment_buffer_size"],
        depositional_resolution=ck["depositional_resolution_m"] * u"m",
        lithification_time=ck["lithification_time_yr"] * u"yr",
        disintegration_transfer=transfer_function(ck["disintegration_transfer"]),
        facies=facies)
    out = rootpath(ck["output"])
    mkpath(dirname(out))
    println(START_MARKER)
    flush(stdout)
    run_model(Model{ALCAP}, input, out)
    stamp!(out)
    println("wrote ", out)
end

main(ARGS)
