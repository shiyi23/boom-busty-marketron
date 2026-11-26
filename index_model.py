"""Index-only Marketron. Units are years and natural logarithms throughout.

Table 10 defines a separate model. Its missing constants are reconstructed
explicitly by index_parameters(), not taken from the multi-stock simulator.
"""
from math import exp, expm1, isfinite, sqrt
from typing import NamedTuple

import numpy as np
from numba import njit

if __package__:
    from .inputs import COMMON, INDEX
    from .model import impact_flow, potential
else:
    from inputs import COMMON, INDEX
    from model import impact_flow, potential


class IndexParameters(NamedTuple):
    kx: float
    c: float
    g: float
    mu: float
    sx: float
    sy: float
    risk_price: float
    initial_holding: float
    sv: float
    kv: float
    rv: float
    ry: float
    eps: float
    rest: float
    gamma: float
    rb: float
    theta: float
    kmom: float
    risk_power: float
    kshort: float
    klong: float
    cap: float
    cost: float
    inflow: float
    jump_compensation: float


def index_parameters(impact=True, gamma=None):
    # Figure 5, right panel: x̂ ≈ −1.21. Section 7.4 gives xs ≈ −2.02.
    # Eq. (47) fixes a at fair value. Zero drift at xs then identifies rest.
    g, c, mu, kx = (INDEX[k] for k in ("g", "c", "mu", "kappa_x"))
    eps = 1.0 - exp(-1.21) / g
    saddle = -2.02
    v0, p0 = potential(0.0, g, eps)
    vs, ps = potential(saddle, g, eps)
    rest = (kx * saddle + c*c/mu * (v0-vs) * ps) / (c * (p0-ps))
    mix = INDEX["jump_weight_large"]
    mean_mark = mix * INDEX["j_M_large"] + (1.0-mix) * INDEX["j_M_bulk"]
    compensation = INDEX["lambda_M"] * mean_mark / (1.0-INDEX["alpha"])
    return IndexParameters(
        kx, c if impact else 0.0, g, mu, INDEX["sigma_x"], INDEX["sigma_y"],
        INDEX["Lambda"], INDEX["b"], INDEX["s_v"], INDEX["kappa_v"], INDEX["rho_v"],
        COMMON["rho_y"], eps, rest, COMMON["gamma"] if gamma is None else gamma,
        COMMON["r_b"], COMMON["theta"], COMMON["kappa"], COMMON["risk_exponent"],
        COMMON["kappa_sh"], COMMON["kappa_lg"], COMMON["equity_cap"], COMMON["chi"],
        COMMON["inflow"], compensation,
    )


def jump_series(days, rng):
    """Continuous-time exponential Hawkes process via immigrant/offspring clusters."""
    years = days / 252
    rate, alpha, beta = (INDEX[k] for k in ("lambda_M", "alpha", "beta_H"))
    events = rng.uniform(0.0, years, rng.poisson(rate * years)).tolist()
    cursor = 0
    while cursor < len(events):
        parent = events[cursor]
        cursor += 1
        children = parent + rng.exponential(1.0 / beta, rng.poisson(alpha))
        events.extend(children[children < years].tolist())
    marks = np.where(rng.random(len(events)) < INDEX["jump_weight_large"],
                     -INDEX["j_M_large"], -INDEX["j_M_bulk"])
    return np.bincount((np.asarray(events)*252).astype(int), weights=marks, minlength=days)


@njit(cache=True)
def _simulate(p, shocks, jumps):
    dt = 1 / 252
    out = np.empty((len(jumps)+1, 9))
    x, z, fundamental, signal = 0.0, 0.0, 0.0, 0.0
    y = p.rest
    holding = p.initial_holding
    bond = holding * (1.0 - p.cap) / p.cap  # bond market value per unit of stock price
    short_var = long_var = p.sx**2
    value_at_zero, slope_at_zero = potential(0.0, p.g, p.eps)
    rest_target = p.rest + p.c / p.mu * value_at_zero
    a = p.c * p.rest * slope_at_zero
    dz, dy = exp(-p.kv * dt), exp(-p.mu * dt)
    sz = p.sv * sqrt(1.0-dz*dz)
    sy = p.sy * sqrt((1.0-dy*dy)/(2*p.mu))
    out[0] = (0.0, x, y, z-p.sv**2/2, fundamental, holding, bond, 0.0, 0.0)
    for t in range(len(jumps)):
        price_shock, demand_shock, vol_shock = shocks[t]
        omega = (short_var / long_var)**p.risk_power
        excess = (p.gamma + p.theta * signal - p.rb) / (p.risk_price * omega)
        y_hat = rest_target + (excess-holding)/p.mu
        v, _ = potential(x, p.g, p.eps)
        desired = excess + p.mu*(rest_target-y) - p.c*v
        target = max(0.0, min(1.0, desired, p.cap*(holding+bond)))
        trade = (target - holding) * dt
        if trade > 0.0:
            headroom = (p.cap*(holding+bond)-holding)/(1+p.cap*p.cost)
            trade = min(trade, max(0.0, headroom))
        new_holding = holding + trade
        post_trade_bond = bond - trade - p.cost*abs(trade)
        x_next = impact_flow(x, p.c*y*dt, p.g, p.eps)
        if not isfinite(x_next):
            raise FloatingPointError("I-BBM reached the absorbing boundary")
        x_next += (a-p.kx*x)*dt + p.sx * exp(0.5*z-p.sv**2/4) * sqrt(dt) * price_shock
        y_next = dy*y + (1-dy)*(y_hat-p.c*v/p.mu) + sy*(
            p.ry*price_shock + sqrt(1-p.ry**2)*demand_shock)
        z_next = dz*z + sz*(p.rv*price_shock + sqrt(1-p.rv**2)*vol_shock)
        df = (p.gamma + p.jump_compensation)*dt + jumps[t]
        r = df + (x_next-x)
        signal += p.kmom * (x_next-x+jumps[t]-signal*dt)
        short_var += -expm1(-p.kshort*dt) * (r*r/dt-short_var)
        long_var += -expm1(-p.klong*dt) * (r*r/dt-long_var)
        bond = (post_trade_bond*exp(p.rb*dt) + p.inflow*dt)*exp(-r)
        fundamental += df
        x, y, z, holding = x_next, y_next, z_next, new_holding
        out[t+1] = (fundamental+x, x, y, z-p.sv**2/2, fundamental, holding, bond, y_hat, jumps[t])
    return out


def simulate_index(years=15, warmup=5, seed=20261126, impact=True, gamma=None):
    """One path including the scoring window's opening level; index is rebased to 1."""
    if years <= 0 or warmup < 0:
        raise ValueError("years must be positive and warmup must be nonnegative")
    burn, days = round(warmup*252), round(years*252)
    if days < 1:
        raise ValueError("the measurement window must contain at least one trading day")
    shock_seed, jump_seed = np.random.SeedSequence(seed).spawn(2)
    shocks = np.random.default_rng(shock_seed).normal(size=(burn+days, 3))
    jumps = jump_series(burn+days, np.random.default_rng(jump_seed))
    raw = _simulate(index_parameters(impact, gamma), shocks, jumps)[burn:].copy()
    raw[:, 0] -= raw[0, 0]
    names = ("log_index", "x", "y", "z", "log_fundamental", "holding", "bond_to_price",
             "y_hat_for_step_ending", "fundamental_jump")
    return dict(zip(names, raw.T))


def conditional_entropy(path, impact=True, gamma=None):
    """Eq. (53), midpoint force with the opening private-state target frozen per day."""
    p = index_parameters(impact, gamma)
    states = np.column_stack([path[k] for k in ("x","y","z")])
    x, y, log_v = ((states[1:] + states[:-1])/2).T
    scales = np.column_stack((p.sx*np.exp(log_v/2), np.full_like(x,p.sy),
                              np.full_like(x, sqrt(2*p.kv)*p.sv)))
    corr = np.array([[1, p.ry, p.rv], [p.ry, 1, p.ry*p.rv], [p.rv, p.ry*p.rv, 1]])
    # D = diag(scales) @ corr @ diag(scales)/2. Only D_xz varies with z.
    a = p.c*p.rest*potential(0.0,p.g,p.eps)[1]
    u = np.exp(x)
    vp = -(u-p.g*(1-p.eps))/(u*(u+p.eps*p.g))
    v = (1-1/p.eps)/u + np.log1p(p.eps*p.g/u)/(p.eps**2*p.g)
    drift = np.column_stack((a-p.kx*x-p.c*y*vp, p.mu*(path["y_hat_for_step_ending"][1:]-y)-p.c*v,
                             -p.kv*(log_v+p.sv**2/2)))
    drift[:, 0] -= 0.25 * scales[:,0]*scales[:,2]*p.rv
    force = 2 * ((drift/scales) @ np.linalg.inv(corr)) / scales
    return float(252*np.mean(np.sum(force*np.diff(states,axis=0),axis=1)))
