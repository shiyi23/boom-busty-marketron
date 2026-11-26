"""Published joint BBM parameters and reproducible proxy inputs, Halperin (2026).

Time is in years. group=0 is the large sleeve, group=1 the rest; sector codes
0,1,2 are sensitive, cyclical, defensive. For data lineage see data/inputs_zh.md.
"""

from statistics import NormalDist

import numpy as np


# Tables 3 and 4, including the first-stage vector.
COMMON = dict(
    inflow=0.00617, r_b=0.030, default_floor=0.10,
    gamma=0.1262, kappa_F=0.1, kappa_eta=42.37, kappa_s=1.0, kappa_ns=1.0,
    zeta=0.35, jump_span_days=1, d_mu=0.18, d_c=0.20, d_g=0.10,
    equity_cap=0.80, chi=0.0005, kappa_sh=3.0, kappa_lg=0.25, follow_years=5,
    rho_M=-0.586, rho_c=-0.444, w=0.549,
    sigma_y=0.0538, mu=8.4112, kappa_v=16.85, rho_y=-0.51,
    theta=1.782, kappa=0.0607, Lambda=0.158, risk_exponent=0.403,
)

GROUPS = (
    dict(
        g=0.0934, epsilon=0.1857, c=2.439, m=1.1400, sigma_x=0.1588,
        kappa_x=4.148, sigma_eta=0.0638, sigma_F=0.0606, beta_mean=0.6544,
        lambda_M=3.084, j_M=0.0168, alpha=0.2935, beta_H=26.97,
        lambda_S=11.788, j_S=0.0070, lambda_I=4.279, sigma_JI=0.0516,
        s_M=1.3866, s_c=0.3163, s_n=1.5480, s_lambda=0.6534,
        beta_sd=0.0470, d_sigma=0.1061, beta_sec=0.1158, delta_exit=0.0,
        rho_n=-0.303, w_n=0.189, rho_ns=-0.430,
    ),
    dict(
        g=0.4240, epsilon=0.2598, c=3.067, m=0.4092, sigma_x=0.2659,
        kappa_x=1.875, sigma_eta=0.0694, sigma_F=0.0341, beta_mean=0.4751,
        lambda_M=1.831, j_M=0.0239, alpha=0.1678, beta_H=43.26,
        lambda_S=9.991, j_S=0.0181, lambda_I=1.298, sigma_JI=0.0362,
        s_M=1.2359, s_c=0.7045, s_n=1.2799, s_lambda=0.4690,
        beta_sd=0.0806, d_sigma=0.1608, beta_sec=0.1887, delta_exit=0.4133,
        rho_n=-0.176, w_n=0.324, rho_ns=-0.286,
    ),
)

# Single-index fit (Table 10). Values inferred elsewhere are kept out of this table.
INDEX = dict(
    kappa_x=2.1051, c=1.7427, g=0.8427, mu=6.1680, sigma_x=0.1842,
    sigma_y=0.0725, Lambda=0.1605, b=0.3595, s_v=1.1674, kappa_v=5.9210,
    rho_v=-0.3455, lambda_M=0.7516, alpha=0.0257, j_M_large=0.0168,
    j_M_bulk=0.0239, jump_weight_large=0.2467, beta_H=39.24,
)

WINDOWS = {
    "1996-2010": dict(
        years=15, sector_share=(0.381, 0.338, 0.281),
        sector_beta=(1.06, 1.07, 0.66), downside=(0.88, 1.06, 0.99),
        sector_vol=(0.44, 0.41, 0.33), log_size_sd=(0.30, 1.06),
        entrant_median=0.82, entrant_mean=1.21, top_share=0.186,
        survivor_vol=(0.289, 0.371),
    ),
    "2011-2024": dict(
        years=14, sector_share=(0.426, 0.380, 0.194),
        sector_beta=(1.09, 1.04, 0.71), downside=(0.97, 1.02, 1.10),
        sector_vol=(0.33, 0.30, 0.26), log_size_sd=(0.25, 0.93),
        entrant_median=0.68, entrant_mean=1.11, top_share=0.202,
        survivor_vol=(0.235, 0.285),
    ),
}
ENTRY_QUANTILES = (13.38, 43.25, 178.78)
_NORMAL = NormalDist()


def canonical_window(window):
    window = {"early": "1996-2010", "late": "2011-2024"}.get(window, window)
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {list(WINDOWS)} or early/late")
    return window


def draw_entry_price(rng, size=None):
    """Split lognormal through the three stated empirical entry-price quantiles."""
    low, med, high = ENTRY_QUANTILES
    z = rng.standard_normal(size)
    side_scale = np.where(z < 0, np.log(med / low), np.log(high / med))
    return med * np.exp(z * side_scale / _NORMAL.inv_cdf(0.95))


def draw_entrant_ratio(rng, window, size=None):
    """Size divided by the median member size. Match Table 3's mean and median."""
    spec = WINDOWS[canonical_window(window)]
    median = spec["entrant_median"]
    log_sd = np.sqrt(2.0 * np.log(spec["entrant_mean"] / median))
    return rng.lognormal(mean=np.log(median), sigma=log_sd, size=size)


def unit_volatility(group):
    """Approximate annual spot return RMS at h=1 using E[variance factor]=1.

    This uses a representative stock's jump exposure beta=beta_mean and sector
    downside ratio of one; it ignores the O(dt) contribution of drift to returns.
    """
    p = GROUPS[group]
    beta_squared = p["beta_mean"] ** 2
    idio_squared = 1.0 - beta_squared - p["beta_sec"] ** 2
    rare = p["lambda_M"] / (1.0 - p["alpha"]) * 2.0 * p["j_M"] ** 2
    routine = p["lambda_S"] * 2.0 * p["j_S"] ** 2
    jitter = COMMON["zeta"]
    # E[max(0,1+zeta Z)^2] for the jittered market-event loading.
    jitter_ms = (1.0 + jitter**2) * _NORMAL.cdf(1.0 / jitter) + jitter * _NORMAL.pdf(1.0 / jitter)
    variance = p["sigma_x"] ** 2 + p["sigma_F"] ** 2 + p["sigma_eta"] ** 2
    variance += beta_squared * (rare + routine) * jitter_ms
    variance += idio_squared * p["lambda_I"] * p["sigma_JI"] ** 2
    return np.sqrt(variance)


H_CENTER = tuple(
    np.mean([w["survivor_vol"][group] for w in WINDOWS.values()]) / unit_volatility(group)
    for group in range(2)
)


def draw_loadings(rng, window, group, sector):
    """Draw name-level beta, both other event exposures, and the two scale factors."""
    spec = WINDOWS[canonical_window(window)]
    group, sector = np.broadcast_arrays(np.asarray(group, dtype=int), np.asarray(sector, dtype=int))
    sector_shape = np.asarray(spec["sector_beta"]) / np.dot(spec["sector_share"], spec["sector_beta"])
    vol_shape = np.asarray(spec["sector_vol"]) / np.dot(spec["sector_share"], spec["sector_vol"])
    def param(key):
        return np.asarray([p[key] for p in GROUPS])[group]

    beta_sec = param("beta_sec")
    tau = np.asarray(spec["downside"])[sector]
    beta = rng.normal(param("beta_mean") * sector_shape[sector], param("beta_sd"))
    bound = np.sqrt(1.0 - beta_sec ** 2) / np.maximum(tau, 1.0)
    beta = np.clip(beta, -bound + 1e-10, bound - 1e-10)
    h = np.asarray(H_CENTER)[group] * vol_shape[sector] * rng.lognormal(0, param("d_sigma"))
    return dict(
        beta=beta, beta_jump=beta * tau, beta_sec=beta_sec, h=h,
        lambda_i=rng.normal(1.0, param("s_lambda")),
    )


def _sleeve_sizes(n, log_sd, total, maximum=np.inf):
    """Finite-sample lognormal ranks, matched log spread and sleeve capitalization."""
    if n == 1:
        return np.array([total])
    z = np.array([_NORMAL.inv_cdf((i + 0.5) / n) for i in range(n)])

    def normalized(ranks):
        logs = (ranks - ranks.mean()) * (log_sd / ranks.std())
        sizes = np.exp(logs - logs.max())
        return sizes * (total / sizes.sum())

    sizes = normalized(z)
    if sizes.max() <= maximum:
        return sizes
    # Smoothly compress the upper tail until the largest bulk member is no
    # larger than the smallest member of the top sleeve. Keep the log SD exact.
    lo, hi = -15.0, 10.0
    for _ in range(55):
        midpoint = (lo + hi) / 2
        ranks = -np.logaddexp(0.0, midpoint - z)
        sizes = normalized(ranks)
        if sizes.max() < maximum:
            lo = midpoint
        else:
            hi = midpoint
    return normalized(-np.logaddexp(0.0, lo - z))


def _assign_sectors(sizes, fractions):
    """Assign ranks so large stocks and small stocks are represented in each sector."""
    allocated = np.zeros(3)
    sectors = np.zeros(len(sizes), dtype=np.int64)
    fractions = np.asarray(fractions)
    for i in np.argsort(-sizes):
        sector = np.argmin((allocated + 0.5 * sizes[i]) / fractions)
        sectors[i] = sector
        allocated[sector] += sizes[i]
    return sectors


def make_panel(n_assets=500, window="1996-2010", seed=1):
    """Construct a reproducible cross section. cap sums to one across both sleeves.

    Within each sleeve names appear in random order. The top sleeve is first.
    Firm labels describe synthetic slots and carry no historical security ID.
    """
    if not isinstance(n_assets, (int, np.integer)) or n_assets < 10:
        raise ValueError("n_assets must be an integer >= 10 to accommodate both size groups")
    window = canonical_window(window)
    spec = WINDOWS[window]
    rng = np.random.default_rng(seed)
    n_large = min(10, max(1, round(10 * n_assets / 500)))
    gidx = np.repeat([0, 1], [n_large, n_assets - n_large])
    large = _sleeve_sizes(n_large, spec["log_size_sd"][0], spec["top_share"])
    bulk = _sleeve_sizes(n_assets - n_large, spec["log_size_sd"][1], 1 - spec["top_share"], large.min())
    cap = np.concatenate([rng.permutation(large), rng.permutation(bulk)])
    sectors = _assign_sectors(cap, spec["sector_share"])
    panel = dict(
        group=gidx, sector=sectors, cap=cap, entry_price=draw_entry_price(rng, n_assets),
        gamma=np.full(n_assets, COMMON["gamma"]), epsilon=np.array([p["epsilon"] for p in GROUPS])[gidx],
    )
    for name in ("g", "c", "mu"):
        center = COMMON["mu"] if name == "mu" else np.array([p[name] for p in GROUPS])[gidx]
        panel[name] = center * rng.lognormal(0.0, COMMON["d_" + name], n_assets)
    panel.update(draw_loadings(rng, window, gidx, sectors))
    return panel
