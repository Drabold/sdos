"""H(k) and S(k) from a ``sisl.Hamiltonian`` (SIESTA, tight-binding built with sisl, ...), as SciPy sparse matrices.

Orthogonal Hamiltonians give (H, None); non-orthogonal ones give (H, S) for the generalized problem H x = lambda S x.  Matrices read from SIESTA files carry
round-off asymmetry; after verifying Hermiticity to 1e-8 they are symmetrized, (M + M^+)/2.  At k != 0 the matrices are
complex Hermitian.  Units are those of the sisl object (eV for SIESTA).  Needs the optional dependency ``sisl``.
"""

import numpy as np
import scipy.sparse as sp


def from_sisl(H, k=(0.0, 0.0, 0.0), gauge="cell"):
    """(Hk, Sk) as CSR matrices; Sk is None for an orthogonal basis."""
    try:
        import sisl  # noqa: F401
    except ImportError as exc:
        raise ImportError("from_sisl needs the optional dependency sisl (pip install sisl)") from exc
    Hk = sp.csr_matrix(H.Hk(k=k, format="csr", gauge=gauge))
    Sk = None if H.orthogonal else sp.csr_matrix(H.Sk(k=k, format="csr", gauge=gauge))
    out = []
    for name, M in (("H(k)", Hk), ("S(k)", Sk)):
        if M is not None:
            if abs(M - M.conj().T).max() > 1e-8 * max(abs(M).max(), 1.0):
                raise ValueError(f"{name} is not Hermitian to 1e-8: check the sisl object and k")
            M = (0.5 * (M + M.conj().T)).tocsr()       # remove the file-precision asymmetry (<= 1e-8 relative) so that downstream checks (1e-10) pass
        out.append(M)
    return out[0], out[1]


def sisl_dos(H, k=(0.0, 0.0, 0.0), **options):
    """``sdos.dos`` of a sisl Hamiltonian at k (generalized automatically if the basis is non-orthogonal)."""
    from ..api import dos
    Hk, Sk = from_sisl(H, k)
    return dos(Hk, B=Sk, **options)
