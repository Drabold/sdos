"""Reader and builder for DFTB Slater-Koster (``.skf``) files: non-orthogonal tight-binding (H, S) from atomic coordinates.

The parameter files are NOT distributed with sdos (their licences differ, e.g. the pbc set is CC BY-SA 4.0); point ``SKSet`` at a directory that holds the
``A-B.skf`` files of the user's set (``Type2FileNames``: ``<Prefix><A>-<B>.skf``).  Supported: the "simple" (non-extended) format with s and p orbitals;
H is the non-self-consistent (zeroth-order) DFTB Hamiltonian H0, with the atomic on-site energies of the files; S is the overlap matrix (unit diagonal).
The repulsive potential is not used (band-structure part only), and there is no SCC charge shift.

Conventions (as in DFTB+): atomic units in the files (Bohr, Hartree; converted to Angstrom, eV); 20 columns per table row in the order
Hdd0 Hdd1 Hdd2 Hpd0 Hpd1 Hpp0 Hpp1 Hsd0 Hsp0 Hss0 Sdd0 Sdd1 Sdd2 Spd0 Spd1 Spp0 Spp1 Ssd0 Ssp0 Sss0; Fortran repeat counts ``n*x`` are expanded;
for the pair file ``A-B`` the first orbital of each integral sits on A (``s`` on A, ``p`` on B for ``Hsp0``).  Orbital order here is (s, px, py, pz).
"""
import os
import re

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree

HARTREE = 27.211386245988          # eV
BOHR = 0.529177210903              # Angstrom
COL = dict(Hpp0=5, Hpp1=6, Hsp0=8, Hss0=9, Spp0=15, Spp1=16, Ssp0=18, Sss0=19)
NORB = {"s": 1, "p": 4}


def _tokens(text):
    out = []
    for t in re.split(r"[,\s]+", text.strip()):
        if not t:
            continue
        if "*" in t:
            n, v = t.split("*")
            out.extend([float(v.replace("d", "e").replace("D", "e"))] * int(n))
        else:
            out.append(float(t.replace("d", "e").replace("D", "e")))
    return out


class SKFile:
    """One ``A-B.skf`` file: grid spacing ``dx`` and the number of rows (Bohr), the table (ngrid, 20) in Hartree/dimensionless, the on-site data of a homonuclear file."""

    def __init__(self, path, homonuclear):
        with open(path) as f:
            lines = f.read().splitlines()
        if lines[0].lstrip().startswith("@"):
            raise NotImplementedError("extended-format .skf files (with f orbitals) are not supported")
        head = _tokens(lines[0])
        self.dx, self.ngrid = head[0], int(head[1])
        k = 1
        self.onsite = None
        if homonuclear:
            v = _tokens(lines[k])
            self.onsite = dict(Ed=v[0], Ep=v[1], Es=v[2], SPE=v[3])          # Hartree
            k += 1
        self.mass = _tokens(lines[k])[0]
        k += 1
        vals = []
        while k < len(lines):
            try:
                vals.extend(_tokens(lines[k]))
            except ValueError:                               # the "Spline" (repulsive) section starts
                break
            k += 1
        nrows = min(self.ngrid - 1, len(vals) // 20)          # the header counts the omitted r = 0 row: DFTB+ reads ngrid - 1 rows, row 1 at r = dx
        if nrows < self.ngrid - 1:
            raise ValueError(f"{path}: table shorter than announced ({len(vals) // 20} < {self.ngrid - 1} rows)")
        self.ngrid = nrows
        self.table = np.array(vals[:20 * nrows]).reshape(nrows, 20)
        flat = (np.abs(self.table - 1.0).max(axis=1) == 0) | (np.abs(self.table).max(axis=1) == 0)
        nz = np.where(~flat)[0]
        self.first = int(nz[0]) if len(nz) else self.ngrid              # leading placeholder rows ("20*1.0" or zeros) carry no data
        self.last_row = int(nz[-1]) if len(nz) else -1


TAIL = 1.0                 # Bohr: DFTB+ continues each integral past the end of its table by a quintic that goes to zero (with zero 1st and 2nd derivatives) TAIL later


class _Radial:
    """One tabulated integral: cubic spline through the table, then the DFTB+ quintic tail on (r_end, r_end + 1 Bohr), zero beyond.

    The tail matches the value, slope and curvature of the spline at the last table row and has a triple zero at r_end + TAIL (checked against DFTB+: whole
    H and S matrices of a 64-atom Si cell agree to ~1e-9).  Lengths in Angstrom."""

    def __init__(self, r, y, tail):
        self.r0, self.r1 = float(r[0]), float(r[-1])
        self.cs = CubicSpline(r, y)
        self.rend = self.r1 + tail
        y0, y1, y2 = float(self.cs(self.r1)), float(self.cs(self.r1, 1)), float(self.cs(self.r1, 2))
        L = tail
        A = np.array([[L ** 3, L ** 4, L ** 5], [3 * L ** 2, 4 * L ** 3, 5 * L ** 4], [6 * L, 12 * L ** 2, 20 * L ** 3]])
        rhs = -np.array([y0 + y1 * L + y2 * L ** 2 / 2, y1 + y2 * L, y2])
        self.c = np.linalg.solve(A, rhs)
        self.y = (y0, y1, y2)

    def __call__(self, r):
        r = np.asarray(r, float)
        t = r - self.r1
        y0, y1, y2 = self.y
        tail = y0 + y1 * t + y2 * t ** 2 / 2 + self.c[0] * t ** 3 + self.c[1] * t ** 4 + self.c[2] * t ** 5
        return np.where(r <= self.r1, self.cs(np.minimum(r, self.r1)), np.where(r < self.rend, tail, 0.0))


class SKSet:
    """A directory of .skf files for the given species; builds the radial functions (Angstrom, eV)."""

    def __init__(self, directory, species, max_l=None, grid_offset=1):
        self.species = list(species)
        self.max_l = {s: (max_l or {}).get(s, "s" if s == "H" else "p") for s in self.species}
        self.grid_offset = grid_offset                    # row j (0-based) sits at r = (j + grid_offset) dx  (DFTB+: 1)
        self.files = {}
        self.splines = {}
        self.onsite = {}
        rc = 0.0
        for a in self.species:
            for b in self.species:
                f = SKFile(os.path.join(directory, f"{a}-{b}.skf"), a == b)
                self.files[(a, b)] = f
                tail = TAIL * BOHR
                rows_ = slice(f.first, f.ngrid)                     # the table end is the last row (DFTB+), not the last nonzero one
                r = (np.arange(f.first, f.ngrid) + grid_offset) * f.dx * BOHR
                tab = f.table[rows_]
                self.splines[(a, b)] = {name: _Radial(r, tab[:, c] * (HARTREE if name[0] == "H" else 1.0), tail) for name, c in COL.items()}
                self.splines[(a, b)]["_rmin"] = r[0]
                self.splines[(a, b)]["_rmax"] = r[-1] + tail
                rc = max(rc, self.splines[(a, b)]["_rmax"])
                if a == b:
                    o = f.onsite
                    self.onsite[a] = (o["Es"] * HARTREE, o["Ep"] * HARTREE)
        self.rcut = rc

    def norb(self, species):
        return NORB[self.max_l[species]]

    def blocks(self, a, b, d):
        """(n, norb_a, norb_b) blocks of H and S for atoms of species a (at 0) and b (at d (n, 3), Angstrom)."""
        d = np.asarray(d, float)
        r = np.linalg.norm(d, axis=1)
        u = d / r[:, None]
        na, nb = self.norb(a), self.norb(b)
        ab, ba = self.splines[(a, b)], self.splines[(b, a)]
        if np.any(r < ab["_rmin"]):
            raise ValueError(f"pair distance below the first tabulated point of {a}-{b} ({ab['_rmin']:.3f} A)")
        live = r < ab["_rmax"]
        out = []
        for X in ("H", "S"):
            B = np.zeros((len(r), na, nb))
            f = lambda sp_, name: np.where(live, sp_[X + name](r), 0.0)
            B[:, 0, 0] = f(ab, "ss0")
            if nb == 4:
                s_p = f(ab, "sp0")
                for k in range(3):
                    B[:, 0, 1 + k] = u[:, k] * s_p
            if na == 4:
                p_s = f(ba, "sp0")                          # s on b, p on a: the integral of the b-a file along the reversed direction
                for k in range(3):
                    B[:, 1 + k, 0] = -u[:, k] * p_s
            if na == 4 and nb == 4:
                pps, ppp = f(ab, "pp0"), f(ab, "pp1")
                for k in range(3):
                    for m in range(3):
                        B[:, 1 + k, 1 + m] = u[:, k] * u[:, m] * (pps - ppp) + (ppp if k == m else 0.0)
            out.append(B)
        return out[0], out[1]


def _pairs(pos, lattice, rcut):
    """All (i, j, d) with |d| < rcut (d = x_j + T - x_i, T a lattice translation), i != j or T != 0, each unordered pair once per direction i<j or (i == j, T > 0)."""
    lattice = np.asarray(lattice, float)
    N = len(pos)
    orth = lattice.ndim == 1 or np.allclose(lattice - np.diag(np.diag(lattice)), 0)
    if lattice.ndim == 1:
        box = lattice
    elif orth:
        box = np.diag(lattice)
    if orth and np.all(box >= 2 * rcut):
        X = np.mod(pos, box)
        pr = cKDTree(X, boxsize=box).query_pairs(rcut, output_type="ndarray")
        I, J = pr[:, 0], pr[:, 1]
        d = X[J] - X[I]
        d -= box * np.rint(d / box)
        return I, J, d
    A = lattice if lattice.ndim == 2 else np.diag(lattice)
    heights = 1.0 / np.linalg.norm(np.linalg.inv(A), axis=0)
    m = np.ceil(rcut / heights).astype(int)
    tree = cKDTree(pos)
    Is, Js, Ds = [], [], []
    for i in range(-m[0], m[0] + 1):
        for j in range(-m[1], m[1] + 1):
            for l in range(-m[2], m[2] + 1):
                T = i * A[0] + j * A[1] + l * A[2]
                sd = tree.sparse_distance_matrix(cKDTree(pos + T), rcut, output_type="coo_matrix")
                a, b = sd.row, sd.col                       # a: original atom, b: image atom (index in the shifted copy) -> j = b
                sel = (a != b) | (np.abs(T).sum() > 0)
                a, b = a[sel], b[sel]
                d = pos[b] + T - pos[a]
                # keep each undirected bond once: (a, b, T) ~ (b, a, -T)
                keep = (a < b) | ((a == b) & ((i, j, l) > (0, 0, 0)))
                Is.append(a[keep]); Js.append(b[keep]); Ds.append(d[keep])
    return np.concatenate(Is), np.concatenate(Js), np.vstack(Ds)


def skf_hamiltonian(pos, symbols, lattice, skset, kpoint=None, chunk=None):
    """(H, S) as sparse CSR (complex when ``kpoint`` (1/Angstrom) is given: H_ij = sum_T h(d) exp(i k.d), the atomic gauge).

    ``lattice``: three box lengths (orthogonal cell) or a (3, 3) matrix of lattice vectors (rows).  Orbital order: atom by atom, (s, px, py, pz) for p-species.
    ``chunk``: build in blocks of that many atoms (single species, orthorhombic box larger than twice the cutoff): peak memory = the final matrices plus one block.
    """
    pos = np.asarray(pos, float)
    symbols = np.asarray(symbols)
    if chunk:                                   # single-species orthorhombic systems: chunked assembly (peak memory = final H, S + one block)
        if len(set(symbols.tolist())) != 1 or np.asarray(lattice).ndim != 1:
            raise ValueError("chunked assembly needs one species and three box lengths")
        from ._assemble import assemble_csr
        a = str(symbols[0])
        no_ = skset.norb(a)
        s_, p_ = skset.onsite[a]
        ons = np.array([s_] + [p_] * 3)[:no_]
        return tuple(assemble_csr(pos, lattice, skset.rcut, lambda d: list(skset.blocks(a, a, d)), no_, [ons, np.ones(no_)], kpoint, chunk))
    N = len(pos)
    no = np.array([skset.norb(s) for s in symbols])
    off = np.concatenate([[0], np.cumsum(no)])
    M = int(off[-1])
    I, J, d = _pairs(pos, lattice, skset.rcut)
    cplx = kpoint is not None
    dt = complex if cplx else float
    rows, cols, hv, sv = [], [], [], []
    ons = np.zeros(M)
    for ia in range(N):
        s_, p_ = skset.onsite[symbols[ia]]
        ons[off[ia]] = s_
        if no[ia] == 4:
            ons[off[ia] + 1:off[ia] + 4] = p_
    for a in skset.species:
        for b in skset.species:
            sel = np.where((symbols[I] == a) & (symbols[J] == b))[0]
            if len(sel) == 0:
                continue
            Hb, Sb = skset.blocks(a, b, d[sel])
            ph = np.exp(1j * (d[sel] @ np.asarray(kpoint, float))) if cplx else np.ones(len(sel))
            na, nb = Hb.shape[1], Hb.shape[2]
            ii, jj = I[sel], J[sel]
            ro = (off[ii][:, None, None] + np.arange(na)[None, :, None]).repeat(nb, 2).ravel()
            co = (off[jj][:, None, None] + np.arange(nb)[None, None, :]).repeat(na, 1).ravel()
            Hv = (Hb * ph[:, None, None]).astype(dt).ravel()
            Sv = (Sb * ph[:, None, None]).astype(dt).ravel()
            rows.append(ro); cols.append(co); hv.append(Hv); sv.append(Sv)
            # Hermitian conjugate block (j, i)
            rows.append(co); cols.append(ro)
            hv.append(np.conj(Hv)); sv.append(np.conj(Sv))
    rows = np.concatenate(rows + [np.arange(M)]); cols = np.concatenate(cols + [np.arange(M)])
    H = sp.coo_matrix((np.concatenate(hv + [ons.astype(dt)]), (rows, cols)), shape=(M, M)).tocsr()
    S = sp.coo_matrix((np.concatenate(sv + [np.ones(M, dtype=dt)]), (rows, cols)), shape=(M, M)).tocsr()
    return H, S
