"""Write sdos.bib from the Crossref records of verify_refs.py (+ DOI lookups for the extra entries); manual entries are marked."""
import sys; sys.path.insert(0, '/tmp'); from protect import protect
import json, re, urllib.request, html
ref = json.load(open("refs_crossref.json"))
def clean(s): return protect(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html.unescape(s))).strip())
def fetch(doi):
    m = json.load(urllib.request.urlopen(urllib.request.Request("https://api.crossref.org/works/" + doi, headers={"User-Agent": "sdos (mailto:drabold@ohio.edu)"}), timeout=30))["message"]
    return dict(title=m["title"][0], journal=(m.get("container-title") or [""])[0], volume=str(m.get("volume", "")), page=str(m.get("page", "")).replace("--", "-"), year=m["issued"]["date-parts"][0][0], doi=doi,
                authors=[(a.get("given", ""), a.get("family", "")) for a in m.get("author", [])], status="OK")
extra = {"scipy": "10.1038/s41592-019-0686-2", "lin": "10.1137/130934283", "wang1994": "10.1103/physrevb.49.10154", "skilling": "10.1007/978-94-015-7860-8_48", "sieck2003": "10.1002/pssb.200301886"}
for k, d in extra.items(): ref[k] = fetch(d)
ref["tang"]["year"] = 2012                                  # Crossref carries the online-first year (2011); the volume 19 issue is 2012
ref["silver1994"]["volume"] = "5"
def authors(a):
    names = [f"{f}, {g}" for g, f in a]
    return " and ".join(names[:6] + (["others"] if len(names) > 8 else names[6:8]))
bib = []
for k, r in ref.items():
    if r["status"] != "OK" and k not in ("silver1994", "tang"): print("SKIP", k); continue
    if k == "skilling":
        bib.append(f"@incollection{{{k},\n author = {{{authors(r['authors'])}}},\n title = {{{clean(r['title'])}}},\n booktitle = {{Maximum Entropy and Bayesian Methods}},\n pages = {{{r['page'].replace('-', '--')}}},\n publisher = {{Springer}}, address = {{Dordrecht}},\n year = {{{r['year']}}},\n doi = {{{r['doi']}}}}}")
        continue
    jr = clean(r["journal"])
    bib.append(f"@article{{{k},\n author = {{{authors(r['authors'])}}},\n title = {{{clean(r['title'])}}},\n journal = {{{jr}}},\n volume = {{{r['volume'].lstrip('0')}}},\n pages = {{{r['page'].replace('-', '--')}}},\n year = {{{r['year']}}},\n doi = {{{r['doi']}}}}}")
bib.append("""@inproceedings{roder1996,
 author = {R{\\\"o}der, H. and Silver, R. N. and Kress, J. D.},
 title = {The kernel polynomial method for non-orthogonal electronic structure calculations},
 booktitle = {High Performance Computing 1996: Grand Challenges in Computer Simulation},
 editor = {Tentner, A.},
 publisher = {Society for Computer Simulation},
 address = {San Diego},
 year = {1996},
 note = {Report CONF-960482, OSTI ID 390779}}""")
bib.append("""@phdthesis{sieck2000,
 author = {Sieck, A.},
 title = {Doctoral thesis},
 school = {University of Paderborn},
 year = {2000},
 note = {Cited as required by the README of the pbc-0-3 Slater--Koster set}}""")
bib.append("""@misc{pbc03,
 author = {Frauenheim, Th.},
 title = {The \\texttt{pbc-0-3} Slater--Koster parameter set},
 howpublished = {Universit{\\\"a}t Bremen, CC BY-SA 4.0 (\\texttt{dftb.org})},
 year = {2017}}""")
bib.append("""@misc{sisl,
 author = {Papior, N.},
 title = {sisl: a Python package for tight-binding, DFT and SIESTA post-processing},
 howpublished = {\\url{https://github.com/zerothi/sisl}}}""")
open("sdos.bib", "w").write("\n\n".join(bib) + "\n")
print(len(bib), "entries")

# NOTE (2 Nov): sdos.bib was edited by hand after generation: silver1994 author/title case fixed; jaynes1978, fejer (Math. Ann. 58, 51 (1904), DOI 10.1007/bf01447779 checked in Crossref) and courant (book) appended. Re-running this script overwrites them.
# morrow2024 (verified in Crossref and Google Scholar; data: Zenodo 10.5281/zenodo.10794053, CC-BY-4.0) is in sdos.bib but not yet cited
