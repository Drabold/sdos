"""sdos: order-N spectral density of states for large sparse Hermitian problems.

Stochastic trace estimation (Drabold-Sankey) with the kernel polynomial method or maximum entropy,
plus shift-invert windows.  Works for real symmetric and complex Hermitian matrices, explicit or
matrix-free.  See ``sdos.estimate_dos`` and the ``sdos`` command.
"""

__version__ = "0.1.0"

from .api import dos
from .bounds import distance_coloring, spectral_bounds
from .dos import DOSResult, estimate_dos, fermi_and_band_energy, fermi_interval
from .results import load, save_result
from .io import read_matrix, write_dos, write_triplets
from .maxent import band_quantities_grid, maxent, maxent_bootstrap, maxent_density, maxent_fit
from .moments import chebyshev_moments, chebyshev_moments_plain, impartial_vector, make_vectors, moment_errors, stochastic_moments
from .operators import HermitianError, ScaledOperator, check_hermitian
from .reconstruct import band_quantities, jackson, lorentz, reconstruct
from .eigwin import WindowResult, count_states_below, homo_lumo, ipr, mulliken_weights, window_eigs
from .exact import exact_chebyshev_moments
from .generalized import GeneralizedOperator, generalized_bounds, pcg
from .phonon import vdos
from .projections import ProjectedResult, ldos, pdos, site_groups
from .thermo import HBAR_OMEGA_METAL, ThermoResult, harmonic_thermo
from .windows import shift_invert_dos, si_nodes

__all__ = [
    "__version__", "dos", "load", "save_result", "vdos", "DOSResult", "estimate_dos", "fermi_and_band_energy", "fermi_interval", "read_matrix", "write_dos",
    "write_triplets", "spectral_bounds", "distance_coloring", "chebyshev_moments", "make_vectors",
    "stochastic_moments", "moment_errors", "impartial_vector", "jackson", "lorentz", "reconstruct", "band_quantities",
    "maxent", "maxent_bootstrap", "maxent_fit", "maxent_density", "band_quantities_grid", "shift_invert_dos", "si_nodes",
    "check_hermitian", "HermitianError", "ScaledOperator", "exact_chebyshev_moments", "GeneralizedOperator",
    "generalized_bounds", "pcg", "window_eigs", "WindowResult", "count_states_below", "homo_lumo", "harmonic_thermo", "ThermoResult", "HBAR_OMEGA_METAL", "pdos", "ldos", "site_groups", "ProjectedResult", "ipr", "mulliken_weights", "chebyshev_moments_plain",
]
