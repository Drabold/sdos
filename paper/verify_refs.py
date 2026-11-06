"""Look every reference up in Crossref and compare with what the report will cite (journal, volume, first page, year). Prints OK / MISMATCH with the DOI."""
import json, re, sys, time, urllib.parse, urllib.request
REFS = {  # key: (query, journal substring, volume, first page, year)
 "kpm": ("Weisse Wellein Alvermann Fehske The kernel polynomial method", "Reviews of Modern Physics", "78", "275", 2006),
 "silver1996": ("Silver Roeder Voter Kress Kernel polynomial approximations for densities of states and spectral functions", "Computational Physics", "124", "115", 1996),
 "silver1994": ("Silver Roeder Densities of states of mega-dimensional Hamiltonian matrices", "Modern Physics C", "5", "735", 1994),
 "drabold1993": ("Drabold Sankey Maximum entropy approach for linear scaling in the electronic structure problem", "Physical Review Letters", "70", "3631", 1993),
 "hutchinson": ("Hutchinson A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines", "Statistics", "19", "433", 1990),
 "mead": ("Mead Papanicolaou Maximum entropy in the problem of moments", "Mathematical Physics", "25", "2404", 1984),
 "goedecker": ("Goedecker Linear scaling electronic structure methods", "Reviews of Modern Physics", "71", "1085", 1999),
 "kwon": ("Kwon Biswas Wang Ho Soukoulis Transferable tight-binding models for silicon", "Physical Review B", "49", "7242", 1994),
 "xu": ("Xu Wang Chan Ho A transferable tight-binding potential for carbon", "Condensed Matter", "4", "6047", 1992),
 "deringer": ("Deringer Bernstein Csanyi Ben Mahmoud Ceriotti Wilson Drabold Elliott Origins of structural and electronic transitions in disordered silicon", "Nature", "589", "59", 2021),
 "ssc96": ("Drabold Ordejon Dong Martin Spectral properties of large fullerenes: from cluster to crystal", "Solid State Communications", "96", "833", 1995),
 "itoh": ("Itoh Ordejon Drabold Martin Structure and energetics of giant fullerenes: an order-N molecular-dynamics study", "Physical Review B", "53", "2132", 1996),
 "lcbop": ("Los Fasolino Intrinsic long-range bond-order potential for carbon: Performance in Monte Carlo simulations of graphitization", "Physical Review B", "68", "024107", 2003),
 "porezag": ("Porezag Frauenheim Kohler Seifert Kaschner Construction of tight-binding-like potentials on the basis of density-functional theory: Application to carbon", "Physical Review B", "51", "12947", 1995),
 "elstner": ("Elstner Porezag Jungnickel Elsner Haugk Frauenheim Suhai Seifert Self-consistent-charge density-functional tight-binding method for simulations of complex materials properties", "Physical Review B", "58", "7260", 1998),
 "dftbplus": ("Hourahine DFTB+, a software package for efficient approximate density functional theory based atomistic simulations", "Chemical Physics", "152", "124101", 2020),
 "siesta2002": ("Soler Artacho Gale Garcia Junquera Ordejon Sanchez-Portal The SIESTA method for ab initio order-N materials simulation", "Condensed Matter", "14", "2745", 2002),
 "siesta2020": ("Garcia Papior Akhtar Artacho Blum Bosoni Brandimarte Siesta: Recent developments and applications", "Chemical Physics", "152", "204108", 2020),
 "lammps": ("Thompson Aktulga Berger Bolintineanu Brown Crozier LAMMPS a flexible simulation tool for particle-based materials modeling at the atomic meso and continuum scales", "Computer Physics Communications", "271", "108171", 2022),
 "ace": ("Drautz Atomic cluster expansion for accurate and transferable interatomic potentials", "Physical Review B", "99", "014104", 2019),
 "pace": ("Lysogorskiy van der Oord Bochkarev Menon Rinaldi Hammerschmidt Mrovec Thompson Csanyi Ortner Drautz Performant implementation of the atomic cluster expansion (PACE) and application to copper and silicon", "npj Computational Materials", "7", "97", 2021),
 "sw": ("Stillinger Weber Computer simulation of local order in condensed phases of silicon", "Physical Review B", "31", "5262", 1985),
 "keating": ("Keating Effect of invariance requirements on the elastic strain energy of crystals with application to the diamond structure", "Physical Review", "145", "637", 1966),
 "dtw": ("Djordjevic Thorpe Wooten Computer model of tetrahedral amorphous diamond", "Physical Review B", "52", "5685", 1995),
 "www": ("Wooten Winer Weaire Computer generation of structural models of amorphous Si and Ge", "Physical Review Letters", "54", "1392", 1985),
 "zhou": ("Zhou Saad Tiago Chelikowsky Self-consistent-field calculations using Chebyshev-filtered subspace iteration", "Computational Physics", "219", "172", 2006),
 "meijerink": ("Meijerink van der Vorst An iterative solution method for linear systems of which the coefficient matrix is a symmetric M-matrix", "Mathematics of Computation", "31", "148", 1977),
 "manteuffel": ("Manteuffel An incomplete factorization technique for positive definite linear systems", "Mathematics of Computation", "34", "473", 1980),
 "kite": ("Joao Andelkovic Covaci Rappoport Lopes dos Santos Kite: high-performance accurate modelling of electronic structure and response functions of large molecules, disordered crystals and heterostructures", "Royal Society Open Science", "7", "191809", 2020),
 "kwant": ("Groth Wimmer Akhmerov Waintal Kwant: a software package for quantum transport", "New Journal of Physics", "16", "063065", 2014),
 "tbplas": ("Li Yuan Tight-binding propagation method TBPLaS", "Computer Physics Communications", "285", "108632", 2023),
 "scipy": ("Virtanen SciPy 1.0: fundamental algorithms for scientific computing in Python", "Nature Methods", "17", "261", 2020),
 "numpy": ("Harris Millman van der Walt Array programming with NumPy", "Nature", "585", "357", 2020),
 "mulliken": ("Mulliken Electronic population analysis on LCAO-MO molecular wave functions. I", "Chemical Physics", "23", "1833", 1955),
 "haydock": ("Haydock Heine Kelly Electronic structure based on the local atomic environment for tight-binding bands", "Physics C", "5", "2845", 1972),
 "tang": ("Tang Saad A probing method for computing the diagonal of a matrix inverse", "Numerical Linear Algebra", "19", "485", 2012),
 "goedeckercolombo": ("Goedecker Colombo Efficient linear scaling algorithm for tight-binding molecular dynamics", "Physical Review Letters", "73", "122", 1994),
 "jackson": ("Jackson On approximation by trigonometric sums and polynomials", "Transactions of the American Mathematical Society", "13", "491", 1912),
 "turek": ("Turek A maximum-entropy approach to the density-of-states recursion method", "Physics C", "21", "3251", 1988),
}
def q(key, query):
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode({"query.bibliographic": query, "rows": 3, "select": "title,author,container-title,volume,page,issued,DOI,article-number"})
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sdos-report-verify (mailto:drabold@ohio.edu)"}), timeout=30) as r:
        return json.load(r)["message"]["items"]
out = {}
for k, (query, jr, vol, pg, yr) in REFS.items():
    try:
        items = q(k, query)
    except Exception as e:
        print(f"{k:16s} LOOKUP FAILED {e}"); continue
    best = None
    for it in items:
        cj = " ".join(it.get("container-title", [])) ; v = str(it.get("volume", "")) ; p = str(it.get("page", it.get("article-number", ""))).split("-")[0]
        y = it.get("issued", {}).get("date-parts", [[None]])[0][0]
        score = (jr.lower() in cj.lower()) + (v == vol) + (p == pg) + (y == yr)
        if best is None or score > best[0]: best = (score, it, cj, v, p, y)
    score, it, cj, v, p, y = best
    au = ", ".join(a.get("family", "?") for a in it.get("author", [])[:4]) + (" et al." if len(it.get("author", [])) > 4 else "")
    status = "OK      " if score == 4 else f"MISMATCH({score}/4)"
    print(f"{k:16s} {status} {au} | {it['title'][0][:70]} | {cj[:38]} {v}, {p} ({y}) | doi:{it['DOI']}")
    out[k] = dict(status=status.strip(), doi=it["DOI"], title=it["title"][0], journal=cj, volume=v, page=p, year=y, authors=[(a.get("given", ""), a.get("family", "")) for a in it.get("author", [])])
    time.sleep(0.3)
json.dump(out, open("refs_crossref.json", "w"), indent=1)
