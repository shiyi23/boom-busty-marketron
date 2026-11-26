"""Run the published-parameter reconstruction from a shell, on Windows or Linux."""
import argparse
import csv
import json
import platform
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from diagnostics import cycle_stages, drawdown_episodes, index_moments, lag_asymmetry, ou_entropy
from index_model import conditional_entropy, simulate_index
from model import simulate_bbm


ROOT = Path(__file__).resolve().parent
DAYS = {"A": 3777, "B": 3522}


def write_rows(filename, rows):
    if not rows:
        return
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with filename.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(filename):
    with filename.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sample_path(directory, name, path):
    columns = {k: path[k] for k in ("log_index", "log_return", "x", "y", "z", "top_share",
                                    "removals", "defaults", "member_defaults", "exposure",
                                    "within_factor_two", "holding", "bond_to_price", "fundamental_jump")
               if k in path}
    log_index = path["log_index"]
    columns["index_base_100"] = 100*np.exp(log_index - log_index[0])
    columns["drawdown"] = np.expm1(log_index - np.maximum.accumulate(log_index))
    columns["wyckoff_stage"] = cycle_stages(log_index)
    columns = {"day": np.arange(len(log_index)), **columns}
    with (directory/f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    episodes = drawdown_episodes(log_index)
    with (directory/f"{name}_episodes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, ("peak","breach","trough","rebound","depth"))
        writer.writeheader()
        writer.writerows(vars(event) for event in episodes)


def _worker(job):
    kind, window, scenario, number, seed, n, warmup, days, directory = job
    impact = scenario == "baseline"
    meta = dict(experiment=kind, window=window, scenario=scenario, run=number,
                seed=seed, assets=n, burn_years=warmup, days=days)
    model = "BBM" if kind.startswith("bbm") else "I-BBM"
    if model == "BBM":
        path = simulate_bbm(days/252, warmup=warmup, seed=seed, window={"A":"early","B":"late"}[window],
                            n_assets=n, impact=impact, keep_panel=kind=="bbm_window")
    else:
        path = simulate_index(days/252, warmup=warmup, seed=seed, impact=impact)
    meta["model"] = model
    states = np.column_stack([path[k] for k in ("x","y","z")])
    if not np.isfinite(states).all() or not np.isfinite(path["log_index"]).all():
        raise ArithmeticError(f"nonfinite path in {meta}")
    metrics = index_moments(path["log_index"])
    metrics["ou_entropy"] = ou_entropy(states)
    metrics["asymmetry_1d"] = lag_asymmetry(states, 1)[0]
    metrics["asymmetry_21d"] = lag_asymmetry(states, 21)[0]
    if model == "BBM":
        metrics["ou_entropy_no_replacements"] = ou_entropy(states, path["replaced"])
        metrics["within_factor_two"] = path["within_factor_two"][1:].mean()
        metrics["default_per_10000_firm_years"] = 10000*252*path["defaults"].sum()/path["exposure"].sum()
        metrics["annual_member_attrition"] = (path["removals"] + path["member_defaults"]).sum()/(n*days/252)
        metrics["max_budget_error"] = path["budget_residual"].max()
        metrics["minimum_cash_fraction"] = path["cash_fraction_min"][1:].min()
        metrics["max_posttrade_equity_fraction"] = path["post_trade_equity_share"][1:].max()
        if metrics["max_budget_error"] > 1e-10 or metrics["minimum_cash_fraction"] < -1e-9:
            raise ArithmeticError(f"book invariant failed for {meta}")
        if kind == "bbm_window":
            from diagnostics import cross_section_moments
            metrics.update(cross_section_moments(path["member_log_returns"], path["member_ids"],
                                               path["log_index"]))
    else:
        metrics["conditional_entropy"] = conditional_entropy(path, impact)
    extras = []
    if kind == "bbm_long":
        # Table 33 uses 150 years; Table 34 uses 200 years. Never pool their clocks.
        cutoff = min(len(path["log_index"]), 150*252+1)
        extra = {**meta, **index_moments(path["log_index"][:cutoff])}
        extra.update(experiment="bbm_cycles", days=cutoff-1)
        extras.append(extra)
        if number < 8 and impact:
            for start in range(0, cutoff-1, 25*252):
                block = path["log_index"][start:min(start+25*252+1, cutoff)]
                extras.append({**meta, **index_moments(block), "experiment":"blocks",
                               "start_year":start//252, "days":len(block)-1})
    if number == 0 and kind.endswith("_window"):
        sample_path(directory, f"{model}_{window}_path", path)
    return [{**meta, **metrics}, *extras]


def summarize(directory):
    records = []
    for f in sorted(directory.glob("*_runs.csv")):
        records.extend(read_rows(f))
    keys = ("experiment","model","window","scenario")
    fields = set(keys) | {"run","seed","days","assets","burn_years","start_year"}
    groups = defaultdict(list)
    for row in records:
        groups[tuple(row[k] for k in keys)+(row.get("start_year",""),)].append(row)
    collected = []
    for group, rows in groups.items():
        names = set.union(*(set(r) for r in rows)) - fields
        for metric in sorted(names):
            vals = np.array([float(r[metric]) for r in rows if r.get(metric,"") != ""])
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            sd = np.std(vals,ddof=1) if len(vals)>1 else np.nan
            q25, med, q75 = np.percentile(vals,[25,50,75])
            collected.append({**dict(zip(keys, group[:4])), "start_year":group[4], "metric":metric,
                              "n":len(vals), "days":rows[0]["days"], "mean":vals.mean(),
                              "sd":sd, "se":sd/np.sqrt(len(vals)), "median":med,
                              "iqr_scale":(q75-q25)/1.349})
    write_rows(directory/"summary.csv", collected)
    references = read_rows(ROOT/"data"/"paper_benchmarks.csv")
    by_key = {(r["experiment"],r["window"],r["scenario"],r["metric"]):r for r in collected}
    comparisons = []
    exp = {"5":"bbm_window","11":"index_window","33":"bbm_cycles","34":"bbm_long","28":"index_long"}
    for ref in references:
        if ref["table"] not in exp:
            continue
        row = by_key.get((exp[ref["table"]],ref["window"],ref["scenario"],ref["metric"]))
        if row is None:
            continue
        if ref["table"] == "5":
            center, spread, scheme = row["median"], row["iqr_scale"], "median and IQR/1.349"
        elif ref["table"] == "11":
            center, spread, scheme = row["mean"], row["sd"], "mean and SD"
        else:
            center, spread, scheme = row["mean"], row["se"], "mean and SE"
        low, high = ref["low"], ref["high"]
        overlaps = "" if not (low and high and np.isfinite(spread)) else (
            center+spread >= float(low) and center-spread <= float(high))
        comparisons.append({**ref, "runs":row["n"],"days_per_run":row["days"],
                            "reconstructed":center, "our_low":center-spread,
                            "our_high":center+spread,"our_band":scheme,
                            "center_minus_paper":center-float(ref["center"]),"bands_overlap":overlaps})
    write_rows(directory/"comparison.csv", comparisons)
    metrics_to_pair = {"bbm_cycles":("bears_20_per_century","bear_mean_depth"),
                       "bbm_long":("ou_entropy","ou_entropy_no_replacements")}
    paired = []
    for which, metric_names in metrics_to_pair.items():
        for metric in metric_names:
            pairs = defaultdict(dict)
            for r in records:
                if r["experiment"] == which:
                    pairs[r["seed"]][r["scenario"]] = float(r[metric])
            diffs = [pair["no_feedback"]-pair["baseline"] for pair in pairs.values()
                     if {"baseline","no_feedback"} <= pair.keys()]
            if diffs:
                paired.append(dict(experiment=which,metric=metric,n=len(diffs),
                                   no_feedback_minus_baseline=np.mean(diffs),
                                   se_paired=np.std(diffs,ddof=1)/np.sqrt(len(diffs))))
    write_rows(directory/"paired_effects.csv", paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=["all","bbm","index","long","report"], nargs="?", default="all")
    parser.add_argument("--profile", choices=["quick","full"], default="quick")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261126)
    args = parser.parse_args()
    directory = (args.out or ROOT/("results" if args.profile=="full" else "quick_results")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    full = args.profile == "full"
    for kind, switch in (("index_window","index"),("index_long","index"),("bbm_window","bbm"),
                         ("bbm_long","long")):
        if args.experiment not in ("all",switch):
            continue
        jobs = []
        if kind == "index_window":
            length, runs, burn, n = None, 20 if full else 4, 5, 1
        elif kind == "bbm_window":
            length, runs, burn, n = None, 16 if full else 2, 50 if full else 3, 500 if full else 100
        elif kind == "index_long":
            length, runs, burn, n = 200 if full else 20, 32 if full else 3, 5, 1
        else:
            length, runs, burn, n = 200 if full else 20, 8 if full else 2, 50 if full else 3, 500 if full else 100
        for win in (("A","B") if length is None else ("A",)):
            for scene in (("baseline","no_feedback") if kind=="bbm_long" else ("baseline",)):
                for idx in range(runs):
                    stream = ["index_window","index_long","bbm_window","bbm_long"].index(kind)
                    seed = int(np.random.SeedSequence([args.seed,stream,int(win=="B"),idx]).generate_state(1)[0])
                    jobs.append((kind,win,scene,idx,seed,n,burn,DAYS[win] if length is None else length*252,directory))
        rows = []
        if args.jobs == 1:
            for job in jobs:
                rows.extend(_worker(job))
        else:
            with ProcessPoolExecutor(max_workers=args.jobs) as pool:
                for result in pool.map(_worker, jobs):
                    rows.extend(result)
        write_rows(directory/f"{kind}_runs.csv",rows)
        print(f"{kind} finished: {len(jobs)} paths", flush=True)
    summarize(directory)
    if args.experiment in ("report","all"):
        from charts import make_figures
        make_figures(directory)
    if args.experiment == "all":
        from importlib.metadata import version
        manifest = dict(profile=args.profile,seed=args.seed,python=platform.python_version(),
                        packages={k:version(k) for k in ("numpy","numba","matplotlib","scipy")},
                        window_days=DAYS, jobs=args.jobs, os=platform.platform())
        (directory/"run_info.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
