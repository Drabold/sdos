"""Parse the real-space H / S written by DFTB+ (Options { WriteRealHS = Yes }) into a Gamma-point dense matrix (eV for H)."""
import numpy as np

def read_real_hs(path, natom_hint=None):
    L = open(path).read().splitlines()
    n = int(L[1]); nn = []; no = []
    k = 3
    for _ in range(n):
        a = L[k].split(); nn.append(int(a[1])); no.append(int(a[2])); k += 1
    blocks = []
    while k < len(L):
        if L[k].startswith("#    IATOM1"):
            a = L[k + 1].split()
            i1, j2 = int(a[0]) - 1, int(a[2]) - 1
            cell = tuple(int(x) for x in a[3:6])
            m = np.array([[float(x) for x in L[k + 3 + r].split()] for r in range(no[i1])])
            blocks.append((i1, j2, cell, m))
            k += 3 + no[i1]
        else:
            k += 1
    return n, no, blocks

def dense_gamma(path):
    n, no, blocks = read_real_hs(path)
    off = np.concatenate([[0], np.cumsum(no)])
    D = np.zeros((off[-1], off[-1]))
    pairs = {}
    for i, j, cell, m in blocks:
        D[off[i]:off[i + 1], off[j]:off[j + 1]] += m
        pairs[(i, j, cell)] = pairs.get((i, j, cell), 0) + 1
    return D, blocks, pairs
