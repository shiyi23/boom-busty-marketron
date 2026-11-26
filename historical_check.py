"""Check index statistics against the public S&P 500 price and total-return series.

Run: python historical_check.py
Optional offline inputs: --price-json FILE --total-return-json FILE (Yahoo chart JSON)
The default output contains derived statistics, source URLs, units and date ranges.
"""
import argparse
import csv
from datetime import date, datetime, timezone
import json
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np

if __package__:
    from .diagnostics import index_moments
else:
    from diagnostics import index_moments


BASE = Path(__file__).resolve().parent
SERIES = {
    "SP500_price": "%5EGSPC",
    "SP500_total_return": "%5ESP500TR",
}
QUERY = "?period1=820454400&period2=1735689600&interval=1d&events=history"


def read_closes(source_url, local_file):
    if local_file is None:
        request = Request(source_url, headers={"User-Agent": "Mozilla/5.0"})
        with urlopen(request, timeout=30) as stream:
            obj = json.load(stream)
    else:
        obj = json.loads(local_file.read_text(encoding="utf-8-sig"))
    chart = obj["chart"]
    if chart.get("error") or not chart.get("result"):
        raise ValueError(f"Failed to obtain daily observations: {chart.get('error')}")
    data = chart["result"][0]
    dates = np.array([datetime.fromtimestamp(t, timezone.utc).date() for t in data["timestamp"]])
    close = np.array(data["indicators"]["quote"][0]["close"], dtype=float)
    if len(close) != len(dates) or not np.isfinite(close).all() or np.any(close <= 0):
        raise ValueError("Price file contains a missing or non-positive close")
    if not (dates[0] == date(1996, 1, 2) and dates[-1] == date(2024, 12, 31)
            and np.all(dates[:-1] < dates[1:])):
        raise ValueError("Unexpected historical date coverage or non-unique trading days")
    return dates, close


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-json", type=Path)
    parser.add_argument("--total-return-json", type=Path)
    parser.add_argument("--output", type=Path, default=BASE/"data"/"historical_crosscheck.csv")
    args = parser.parse_args()
    local_files = {"SP500_price": args.price_json, "SP500_total_return": args.total_return_json}
    with (BASE/"data"/"paper_benchmarks.csv").open(encoding="utf-8") as stream:
        benchmarks = [r for r in csv.DictReader(stream) if r["table"] == "5" and r["reference"] == "market"]
    output = []
    for series, symbol in SERIES.items():
        url = "https://query1.finance.yahoo.com/v8/finance/chart/" + symbol + QUERY
        dates, close = read_closes(url, local_files[series])
        split = int(np.searchsorted(dates, date(2011, 1, 1)))
        for name, block in [("A", slice(0, split)), ("B", slice(split - 1, None))]:
            subdates, levels = dates[block], np.log(close[block])
            stats = index_moments(levels)
            for row in benchmarks:
                key = row["metric"]
                if row["window"] != name or key not in stats:
                    continue
                output.append({
                    "series": series, "window": name, "opening_close": subdates[0],
                    "first_return_date": subdates[1], "last_close": subdates[-1],
                    "n_returns": len(levels)-1, "metric": key, "unit": row["unit"],
                    "computed": f"{stats[key]:.10g}", "paper_table5_market": row["center"],
                    "delta": f"{stats[key] - float(row['center']):.10g}", "source_url": url,
                    "note": "Published S&P index via Yahoo Finance. Table 5 uses the author's point-in-time CRSP constituent basket.",
                })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    print(f"Wrote {len(output)} index observations to {args.output}")


if __name__ == "__main__":
    main()
