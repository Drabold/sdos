"""Optional interfaces.  None of these is a hard dependency; each imports its third-party package lazily.

* :mod:`sdos.interfaces.lmp`  matrix-free dynamical matrix from forces (any force callable; LAMMPS through ``LammpsForces``)
* :mod:`sdos.interfaces.sisl_if`  H(k), S(k) from a ``sisl.Hamiltonian`` (e.g. SIESTA)
"""
