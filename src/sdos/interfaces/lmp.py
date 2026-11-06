"""Matrix-free vibrational problem from forces: D v = M^-1/2 Phi M^-1/2 v, Phi v from central differences of the forces.

``HessianOperator(forces, mass)`` is a SciPy LinearOperator built on any callable ``forces(x) -> F`` (x, F of shape (natoms, 3), forces = -dE/dx),
so it works with LAMMPS (``LammpsForces``; needs the ``lammps`` Python module, ``atom_modify map array`` and ``units metal``), with the Keating
model, or with any other force provider.  Rigid translations are projected out (the three zero modes are removed from the spectrum).  The step is
chosen so that the rms displacement per atom is ``delta`` (default 1e-3 A), independent of the system size.  Two force calls per product.

Finite-difference operators are Hermitian only to O(delta^2): pass ``hermitian_tol=1e-3`` (or larger) to ``sdos.dos`` / ``sdos.vdos``; the
recommended value is in ``HessianOperator.hermitian_tol``.  This is the route of the original ``lmpdos.py`` (which keeps working: it imports the same
``sdos`` functions).
"""

import numpy as np
import scipy.sparse.linalg as spla


class HessianOperator(spla.LinearOperator):
    """Mass-weighted dynamical matrix as a LinearOperator, translations projected out."""

    hermitian_tol = 1e-3

    def __init__(self, forces, x0, mass, delta=1e-3):
        x0 = np.asarray(x0, float)
        n = x0.shape[0]
        super().__init__(dtype=np.float64, shape=(3 * n, 3 * n))
        self.forces, self.delta, self.n = forces, delta, n
        self.x0 = x0.ravel().copy()
        self.mass = np.asarray(mass, float)
        self.w = 1.0 / np.sqrt(np.repeat(self.mass, 3))
        T = np.zeros((3 * n, 3))
        for a in range(3):
            T[a::3, a] = np.sqrt(self.mass)
        self.Tr, _ = np.linalg.qr(T)
        self.nprod = 0

    def project(self, v):
        return v - self.Tr @ (self.Tr.T @ v)

    def _matvec(self, v):
        v = self.project(np.asarray(v, dtype=np.float64).ravel())
        u = self.w * v
        nu = np.linalg.norm(u)
        if nu == 0.0:
            return np.zeros_like(v)
        s = self.delta * np.sqrt(u.size) / nu
        Fp = np.asarray(self.forces((self.x0 + s * u).reshape(-1, 3))).ravel()
        Fm = np.asarray(self.forces((self.x0 - s * u).reshape(-1, 3))).ravel()
        self.nprod += 1
        return self.project(self.w * (-(Fp - Fm) / (2 * s)))

    def _matmat(self, X):
        return np.column_stack([self._matvec(X[:, k]) for k in range(X.shape[1])])

    def _adjoint(self):
        return self


class LammpsForces:
    """Forces from a LAMMPS instance set up by an input file (``units metal``, ``atom_modify map array``; no run/minimize commands)."""

    def __init__(self, infile, comm=None, log=None):
        try:
            from lammps import lammps
        except ImportError as exc:
            raise ImportError("LammpsForces needs the LAMMPS Python module (build LAMMPS with PYTHON support, or pip install lammps)") from exc
        args = ["-screen", "none", "-log", log or "none"]
        self.comm = comm
        self.lmp = lammps(cmdargs=args, comm=comm) if comm is not None else lammps(cmdargs=args)
        self.lmp.file(infile)
        if self.lmp.extract_global("units") != "metal":
            raise ValueError("LAMMPS units must be 'metal'")
        self.n = int(self.lmp.get_natoms())
        self.lmp.command("run 0 post no")
        self.x0 = self._gather("x")
        types = np.ctypeslib.as_array(self.lmp.gather_atoms("type", 0, 1)).copy()
        ntypes = int(self.lmp.extract_global("ntypes"))
        mptr = self.lmp.extract_atom("mass")
        if mptr is None:
            raise ValueError("no per-type masses found (use the 'mass' command in the input)")
        self.mass = np.array([mptr[t] for t in range(ntypes + 1)])[types]
        self.calls = 0

    def _gather(self, name):
        a = self.lmp.gather_atoms(name, 1, 3)
        return np.ctypeslib.as_array(a).reshape(self.n, 3).copy()

    def __call__(self, x):
        x = np.ascontiguousarray(x, dtype=np.float64).reshape(-1)
        if self.comm is not None:
            self.comm.Bcast(x, root=0)
        self.lmp.scatter_atoms("x", 1, 3, np.ctypeslib.as_ctypes(x))
        self.lmp.command("run 0 post no")
        self.calls += 1
        return self._gather("f")

    def operator(self, delta=1e-3):
        return HessianOperator(self, self.x0, self.mass, delta)
