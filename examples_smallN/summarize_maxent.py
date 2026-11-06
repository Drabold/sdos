import json, glob
print("Maxent M=100 (E_BS errors per orbital = total/N).  cols: conv fraction, median Newton steps, bias/se, rmse/N, mean true cum err")
for f in sorted(glob.glob("results/e2maxent_*.json")):
    for blk in json.load(open(f)):
        N = blk["N"]; print(f.split("_")[1][:-5], "N =", N)
        for r in blk["rows"]:
            if r["floor"] == 0.0 and r["est"].startswith(("splice", "imp+")): continue
            s = f"   R={r['R']:2d} {r['est']:12s} {r['sigma']:7s} fl={r['floor']:g} n={r['nseeds']:2d} conv {r['converged_frac']:.2f} it {r['iters_median']:3.0f}"
            if "bias" in r: s += f" | bias/se {r['bias']/r['se']:+6.1f} rmse/N {r['rmse']/N:.2e} cum {r['cum_true_mean']:.3f}"
            print(s)
