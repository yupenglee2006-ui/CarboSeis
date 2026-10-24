"""CarboSeis: seismic model construction for carbonate platforms simulated with CarboKitten.jl.

A scenario file describes the forward model and the petrophysical parameters of the
formation; :func:`run_chain` turns CarboKitten output into texture, porosity, elastic
properties, an earth model and synthetic seismic, and :func:`write_result` saves every
intermediate quantity to one HDF5 file.  The command ``carboseis run`` does both.
"""
__version__ = "0.1.0"

from .config import load_scenario  # noqa: E402,F401
from .output import write_result  # noqa: E402,F401
from .pipeline import ChainResult, run_chain  # noqa: E402,F401
