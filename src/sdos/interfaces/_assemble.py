"""Chunked CSR assembly of two-centre tight-binding matrices for large single-species systems in an orthorhombic periodic box (minimum image).

``assemble_csr(pos, box, rcut, blocks_fn, norb, onsite)`` builds H (and S, ...) in blocks of ``chunk`` atoms: every ordered pair (i, j) is evaluated directly, so no COO copy of the whole
matrix is ever held; peak memory is the final CSR plus one chunk.  ``blocks_fn(d)`` returns a list of (n, norb, norb) arrays <i|M|j> for bond vectors d = x_j - x_i (one per matrix).
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


def assemble_csr(pos, box, rcut, blocks_fn, norb, onsite, kpoint=None, chunk=16000):
    box = np.asarray(box, float)
    if np.any(box < 2 * rcut):
        raise ValueError("every box edge must exceed 2 x the cutoff (minimum-image convention)")
    X = np.mod(np.asarray(pos, float), box)
    N = len(X)
    M = norb * N
    tree = cKDTree(X, boxsize=box)
    o = np.arange(norb)
    out = None
    for a0 in range(0, N, chunk):
        a1 = min(N, a0 + chunk)
        pr = cKDTree(X[a0:a1], boxsize=box).sparse_distance_matrix(tree, rcut, output_type="ndarray")
        i = pr["i"].astype(np.int64) + a0
        j = pr["j"].astype(np.int64)
        keep = i != j
        i, j = i[keep], j[keep]
        d = X[j] - X[i]
        d -= box * np.rint(d / box)
        Bs = blocks_fn(d)
        if out is None:
            out = [[] for _ in Bs]
        ph = np.exp(1j * (d @ np.asarray(kpoint, float)))[:, None, None] if kpoint is not None else 1.0
        rows = (norb * (i - a0)[:, None, None] + o[None, :, None]).repeat(norb, 2).ravel()
        cols = (norb * j[:, None, None] + o[None, None, :]).repeat(norb, 1).ravel()
        drow = np.arange(norb * (a1 - a0))
        dcol = norb * a0 + drow
        for m, B in enumerate(Bs):
            vals = (B * ph).ravel()
            dv = np.tile(onsite[m], a1 - a0).astype(vals.dtype)
            out[m].append(sp.coo_matrix((np.concatenate([vals, dv]), (np.concatenate([rows, drow]), np.concatenate([cols, dcol]))), shape=(norb * (a1 - a0), M)).tocsr())
    return [sp.vstack(ms, format="csr") for ms in out]
