"""Independent numerical and accounting checks. Run: python verify.py --out checks.json"""

import argparse
import json
from math import exp, log, log1p
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from index_model import index_parameters
from inputs import COMMON, GROUPS
from model import impact_flow, simulate_bbm


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def check_split():
    # With u=exp(x), du/dt = K (u - g(1-epsilon))/(u+epsilon*g).
    # Integrating in u allows scipy to detect the zero-price event directly.
    p = index_parameters()
    vectors = [(v["g"], v["epsilon"]) for v in GROUPS] + [(p.g, p.eps)]
    errors, event_errors = [], []
    for g, e in vectors:
        barrier = g * (1 - e)
        for ratio, impulse in [(2.5, 0.11), (2.5, -0.09), (0.7, -0.07),
                                (0.7, 0.008), (1.0, 0.015), (0.3, 0.95)]:
            u_initial = barrier * ratio

            def dynamics(t, u):
                return impulse * (u - barrier) / (u + e*g)

            def at_floor(t, u):
                return u[0]

            at_floor.terminal = True
            solution = solve_ivp(dynamics, (0, 1), [u_initial], method="DOP853",
                                 rtol=2e-12, atol=2e-14, events=at_floor)
            require(solution.success, solution.message)
            output = impact_flow(log(u_initial), impulse, g, e)
            if solution.t_events[0].size:
                require(np.isneginf(output), "missed intra-step absorption")
                exact = (g*log1p(u_initial / (barrier-u_initial)) - u_initial) / impulse
                event_errors.append(abs(exact - solution.t_events[0][0]))
            else:
                expected = log(solution.y[0, -1])
                errors.append(abs(output-expected))
    require(max(errors) < 2e-9 and max(event_errors) < 2e-9, "split-flow mismatch")
    bulk = GROUPS[1]
    g, e = bulk["g"], bulk["epsilon"]
    impulse_rate = bulk["c"] * bulk["m"]
    times = []
    for x, paper_days in [(-1.6, 48), (-2.0, 21), (-3.0, 5)]:
        u = exp(x)
        time = (g*log1p(u / (g*(1-e)-u)) - u) / impulse_rate
        require(np.isfinite(impact_flow(x, 0.99999*time*impulse_rate, g, e)) and
                np.isneginf(impact_flow(x, 1.00001*time*impulse_rate, g, e)),
                "split solver did not cross the boundary at the predicted time")
        days = time*252
        require(abs(days-paper_days) < 1, "section 3.8 absorption time mismatch")
        times.append(dict(x=x, calculated_days=days, paper_approximate_days=paper_days))
    return dict(parameter_sets=len(vectors), regular_ODE_cases=len(errors),
                absorbing_ODE_cases=len(event_errors), max_log_price_error=max(errors),
                max_error_in_absorption_fraction_of_step=max(event_errors),
                section_3_8_unit_bulk_absorption=times)


def landscape(g, epsilon, c, mu, kappa_x, rest):
    # For a fixed target, minimize the joint potential over y to obtain U(x).
    # Its maximum is the joint saddle. The turnover of bare V only changes V'.
    def bare(x):
        t = exp(-x)
        return t*(1 - 1/epsilon) + log1p(epsilon*g*t)/(epsilon**2*g)

    delta = 1e-5
    slope0 = (bare(delta) - bare(-delta)) / (2*delta)
    a = c * rest * slope0
    target = rest + c*bare(0)/mu

    def effective(x):
        v = bare(x)
        return kappa_x*x*x/2 - a*x + c*target*v - (c*v)**2/(2*mu)

    turn = log(g*(1-epsilon))
    saddle = minimize_scalar(lambda x: -effective(x), bounds=(turn-2, turn),
                             method="bounded", options={"xatol": 1e-10}).x
    minimum = minimize_scalar(effective, bounds=(-0.2, 0.2), method="bounded").x
    require(saddle < turn and abs(minimum) < 2e-5, "unexpected critical points")
    return dict(bare_turnover_x=turn, effective_saddle_x=saddle, effective_well_x=minimum,
                effective_barrier_height=effective(saddle)-effective(minimum),
                bare_height_at_turnover=bare(turn)-bare(0))


def check_landscapes():
    result = {}
    for name, p, printed in zip(("BBM_large", "BBM_bulk"), GROUPS, (-2.576, -1.159)):
        data = landscape(p["g"], p["epsilon"], p["c"], COMMON["mu"], p["kappa_x"], p["m"])
        require(abs(data["bare_turnover_x"] - printed) <= 5e-4, "Table 4 mismatch")
        result[name] = data
    p = index_parameters()
    data = landscape(p.g, p.eps, p.c, p.mu, p.kx, p.rest)
    require(abs(data["effective_saddle_x"] + 2.02) < 1e-5, "I-BBM saddle mismatch")
    # Turnover and saddle positions are reconstruction inputs. Figure 5's two
    # heights provide additional graphical cross-checks, to plotting precision.
    require(abs(data["effective_barrier_height"] - 2.3) < 0.1, "Figure 5 effective height")
    require(abs(data["bare_height_at_turnover"] - 0.43) < 0.04, "Figure 5 bare height")
    result["I_BBM"] = data
    return result


def check_accounting():
    spec = dict(years=2, warmup=0.5, n_assets=50, seed=711, window="early")
    first = simulate_bbm(**spec, keep_panel=True)
    second = simulate_bbm(**spec, keep_panel=False)
    require(all(np.array_equal(first[k], second[k]) for k in second), "seed repeatability")
    cap = first["member_caps"][:-1]
    weights = cap / cap.sum(axis=1, keepdims=True)
    rebuilt = np.log1p(np.sum(weights * np.expm1(first["member_log_returns"][1:]), axis=1))
    daily_error = np.max(np.abs(rebuilt - np.diff(first["log_index"])))
    total_error = np.max(np.abs(np.cumsum(rebuilt) - first["log_index"][1:]))
    changed = first["member_ids"][1:] != first["member_ids"][:-1]
    require(np.array_equal(changed.sum(axis=1), (first["removals"]+first["member_defaults"])[1:]),
            "membership roll does not balance")
    require(changed.any(), "test seed does not exercise replacement")
    residual = np.max(first["budget_residual"])
    min_cash = np.min(first["cash_fraction_min"][1:])
    require(max(daily_error, total_error, residual) < 1e-10 and min_cash >= -1e-10,
            "accounting identity failed")
    return dict(**spec, days=len(rebuilt), same_seed_identical=True, replaced_names=int(changed.sum()),
                max_daily_log_return_error=daily_error, max_cumulative_log_index_error=total_error,
                max_relative_budget_residual=residual, minimum_cash_fraction=min_cash)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, help="optional result file")
    args = parser.parse_args()
    result = dict(split_flow=check_split(), landscapes=check_landscapes(),
                  capitalization_and_budget=check_accounting(), passed=True)
    output = json.dumps(result, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
