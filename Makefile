# CarboSeis: build seismic models and reproduce every figure and table of the manuscript.
# With CarboKitten output present (make carbokitten), figures use it; otherwise the
# watermarked Python stand-in is used.
#
# Portable: every recipe works both with a POSIX shell and with Windows cmd.exe (GNU make for
# Windows without sh.exe).  Recipes therefore use only commands, `cd` and `&&`; file removal and
# loops are done in Python.
PY ?= python
JULIA ?= julia
SCEN = scenarios/luconia_base.toml

# The manuscript is a separate folder beside this one (see README, "Manuscript").
PAPER ?= ../CarboSeis_manuscript_v5.0
LATEX ?= pdflatex
BIBTEX ?= bibtex

.PHONY: all inputs carbokitten standin run check test figures figures-e3 tables e3 diagnostics \
        paper clean clean-cache clean-runs reproduce

all: inputs test figures tables

inputs:
	$(PY) scripts/fetch_inputs.py

# CarboSeis-CarboKitten bridge (GPL-3.0, ck-bridge/ inside this folder); override with
# make CK_BRIDGE=/path/to/bridge
CK_BRIDGE ?= $(firstword $(CARBOSEIS_CK_BRIDGE) $(wildcard ck-bridge ../carboseis-ck-bridge))

# Julia for CarboKitten runs: the version the bridge environment was resolved with (julia_version in
# the bridge's Manifest.toml).  With juliaup this selects it even if another Julia is the default;
# install it once with `juliaup add 1.12.7`.  Keep equal to the bridge Manifest.
JULIAUP_CHANNEL ?= 1.12.7
export JULIAUP_CHANNEL

carbokitten: inputs
	$(if $(CK_BRIDGE),,$(error CarboSeis-CarboKitten bridge not found: ck-bridge/ is missing from this folder, or run make CK_BRIDGE=path))
	$(JULIA) --project=$(CK_BRIDGE) -e "using Pkg; Pkg.instantiate()"
	$(JULIA) --project=$(CK_BRIDGE) $(CK_BRIDGE)/ck_run.jl $(SCEN)

standin: inputs
	$(PY) -m carboseis.cli standin $(SCEN)

# Seismic model of the scenario: every intermediate quantity and both imaging levels in one HDF5 file
run:
	$(PY) -m carboseis.cli run $(SCEN) --out build/luconia_base_carboseis.h5

test:
	$(PY) -m pytest -q

check:
	$(PY) scripts/check_output.py --json build/check_output.json --strict

figures:
	cd scripts && $(PY) fig01_workflow.py && $(PY) fig02_scenario.py && $(PY) fig03_base_geology.py && $(PY) fig04_base_elastic.py && $(PY) fig05_base_seismic.py && $(PY) fig06_e1_sections.py && $(PY) fig07_e1_metrics.py && $(PY) fig08_e2_timelines.py && $(PY) fig09_e2_agespan.py && $(PY) figA_verification.py

tables:
	cd scripts && $(PY) make_tables.py && $(PY) base_stats.py && $(PY) make_numbers.py

diagnostics:
	cd scripts && $(PY) e1_diagnostics.py && $(PY) make_numbers.py

e3:
	cd scripts && $(PY) e3_run.py --r 10 --seeds 10

figures-e3:
	cd scripts && $(PY) fig10_e3_morris.py && $(PY) fig11_e3_detectability.py && $(PY) make_numbers.py

paper:
	$(if $(wildcard $(PAPER)),,$(error manuscript folder not found at '$(PAPER)': it is a separate folder, not part of this repository; set PAPER=path))
	$(PY) scripts/sync_paper.py --paper $(PAPER)
	cd $(PAPER) && $(LATEX) -interaction=nonstopmode manuscript.tex && $(BIBTEX) manuscript && $(LATEX) -interaction=nonstopmode manuscript.tex && $(LATEX) -interaction=nonstopmode manuscript.tex

clean-cache:
	$(PY) scripts/clean.py cache

clean:
	$(PY) scripts/clean.py outputs

# Remove every CarboKitten run (base case, pre-run and all E3 runs) to regenerate them from scratch
clean-runs:
	$(PY) scripts/clean.py runs

# Everything from scratch, in order: forward runs, checks, figures, tables, E1 diagnostics, E3 (hours)
reproduce: inputs carbokitten check test figures tables diagnostics e3 figures-e3 paper
