"""Optional two-parameter SMM experiment on A's two survivor-basket volatilities.

Only the two group centers for the fixed stock shock scales h_i are estimated.
Usage: python fit_scales.py --jobs 4
"""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from diagnostics import index_moments
from inputs import H_CENTER
from model import simulate_bbm


ROOT = Path(__file__).resolve().parent
DAYS = {"A": 3777, "B": 3522}
TARGET = {
    "A": {"large_basket_vol": .196, "bulk_basket_vol": .210, "index_vol": .207,
          "bears_20_per_century": 26.688},
    "B": {"large_basket_vol": .178, "bulk_basket_vol": .170, "index_vol": .172,
          "bears_20_per_century": 14.310},
}
METRICS = tuple(TARGET["A"])


def basket_volatility(log_returns, opening_caps):
    """Point-in-time cap weights within a fixed, full-window survivor basket."""
    if log_returns.shape != opening_caps.shape or log_returns.shape[1] == 0:
        raise ValueError("empty or inconsistent survivor basket")
    simple = np.sum(opening_caps * np.expm1(log_returns), axis=1) / np.sum(opening_caps, axis=1)
    return float(np.std(np.log1p(simple), ddof=1) * np.sqrt(252))


def simulate(job):
    window, seed, centers = job
    # This ratio cancels the constructor's baseline exactly, including at entry.
    multipliers = np.asarray(centers)/np.asarray(H_CENTER)
    window_key = {"A":"early", "B":"late"}[window]
    path = simulate_bbm(years=DAYS[window]/252, warmup=50, seed=seed, window=window_key,
                        n_assets=500, keep_panel=True, group_vol_multipliers=multipliers)
    ids, caps, returns = (path[key] for key in ("member_ids", "member_caps", "member_log_returns"))
    survivor = np.all(ids == ids[0], axis=0)
    # The two size sleeves are fixed when the window opens. Slots 0:10 are
    # the ten largest; the simulator always replaces a name in its own slot.
    if caps[0, :10].min() < caps[0, 10:].max()*(1-1e-9):
        raise ArithmeticError("opening sleeve labels do not identify the ten largest names")
    large = np.arange(500) < 10
    row = {"window": window, "seed": seed, "h_large": centers[0], "h_bulk": centers[1],
           "days": DAYS[window]}
    for name, mask in (("large", large & survivor), ("bulk", (~large) & survivor)):
        row[f"{name}_basket_vol"] = basket_volatility(returns[1:, mask], caps[:-1, mask])
        row[f"{name}_survivors"] = int(mask.sum())
    index = index_moments(path["log_index"])
    row["index_vol"] = index["annual_volatility"]
    row["bears_20_per_century"] = index["bears_20_per_century"]
    return row


def seeds(master_seed, stream, count, window):
    return [int(np.random.SeedSequence([master_seed, 531, stream, int(window=="B"), i])
                .generate_state(1)[0]) for i in range(count)]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--train-runs", type=int, default=4)
    parser.add_argument("--eval-runs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261126)
    parser.add_argument("--out", type=Path, default=ROOT/"results")
    args = parser.parse_args()
    if min(args.train_runs,args.eval_runs) < 2 or args.jobs < 1:
        parser.error("use at least two seeds per stage and at least one worker")
    args.out.mkdir(parents=True, exist_ok=True)
    path_rows, summary = [], []
    centers = np.ones(2)
    wanted = np.array([TARGET["A"][key] for key in METRICS[:2]])
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        def batch(stage, candidate, window, group_centers, common_seeds):
            jobs = [(window, seed, tuple(group_centers)) for seed in common_seeds]
            batch_rows = list(pool.map(simulate, jobs))
            medians = []
            for row in batch_rows:
                path_rows.append({"stage":stage, "candidate":candidate, **row})
            for key in METRICS:
                vals = np.array([r[key] for r in batch_rows])
                median = float(np.median(vals))
                medians.append(median)
                target = TARGET[window][key]
                summary.append({
                    "stage":stage, "candidate":candidate, "window":window, "metric":key,
                    "paper_table": 6 if key in METRICS[:2] else 5,
                    "used_by_objective":stage=="train" and key in METRICS[:2],
                    "h_large":group_centers[0], "h_bulk":group_centers[1],
                    "n_paths":len(vals), "days_per_path":DAYS[window], "burn_years":50,
                    "median":median, "p25":np.percentile(vals,25), "p75":np.percentile(vals,75),
                    "mean":vals.mean(), "paper_market_center":target, "median_minus_market":median-target
                })
            print(stage, candidate, window, "centers", np.round(group_centers,4),
                  "median basket vol", np.round(medians[:2],4), flush=True)
            return np.array(medians[:2])

        train = seeds(args.seed, 0, args.train_runs, "A")
        for iteration in range(3):
            fitted = batch("train", f"iteration_{iteration}", "A", centers, train)
            if iteration < 2:
                # Multiplicative moment matching. No index, B or evaluation seed
                # enters either update. Common market shocks reduce sampling noise.
                centers = centers * wanted / fitted
        for window in DAYS:
            eval_seeds = seeds(args.seed, 1, args.eval_runs, window)
            for candidate, settings in [("baseline",H_CENTER),("A_fit",centers)]:
                batch("evaluation", candidate, window, settings, eval_seeds)

    write_csv(args.out/"scale_fit.csv", summary)
    write_csv(args.out/"scale_fit_paths.csv", path_rows)


if __name__ == "__main__":
    main()
