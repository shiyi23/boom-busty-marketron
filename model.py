"""Daily multi-asset Boom-Busty Marketron, Eqs. (1)–(47).

Prices are in arbitrary capitalization units (one share per modeled firm).
The floor is converted from quoted dollars at each firm's entry. Two sleeves
share market, sector and exogenous volatility shocks but have separate books.
"""
from collections import defaultdict, namedtuple
from math import exp, expm1, log, log1p, sqrt

import numpy as np
from numba import njit

if __package__:
    from .inputs import (COMMON, GROUPS, canonical_window, draw_entry_price,
                         draw_entrant_ratio, draw_loadings, make_panel)
else:
    from inputs import (COMMON, GROUPS, canonical_window, draw_entry_price,
                        draw_entrant_ratio, draw_loadings, make_panel)


DAYS = 252
DT = 1.0 / DAYS
ROOT_DT = sqrt(DT)
HISTORY = 15 * DAYS


@njit(cache=True)
def potential(x, g, epsilon):
    """V and V' in Eq. (12)."""
    u = exp(x)
    value = (1 - 1 / epsilon) / u + log1p(epsilon * g / u) / (epsilon**2 * g)
    slope = -(u - g * (1 - epsilon)) / (u * (u + epsilon * g))
    return value, slope


@njit(cache=True)
def impact_flow(x, k_dt, g, epsilon):
    """Return exact split-flow log price, or -inf at finite-time absorption.

    In s=log(|w_new/w_old|), solve g*s+w_old*expm1(s)=K*dt. Its
    derivative on the physical branch is u_new+g*epsilon > 0, including
    on the lower branch. Bracketing prevents switching branches at the saddle.
    """
    u = exp(x)
    barrier = g * (1 - epsilon)
    w = u - barrier
    if k_dt == 0 or w == 0:
        return x
    if w < 0 and k_dt > 0:
        s_floor = log1p(u / (-w))
        if k_dt >= g * s_floor - u:
            return -np.inf
        low, high = 0.0, s_floor
    elif k_dt > 0:
        low, high = 0.0, k_dt / g
    elif w < 0:
        low, high = k_dt / (g + w), 0.0
    else:
        low, high = k_dt / g, 0.0
    s = min(max(k_dt / (g + w), low), high)
    for _ in range(50):
        if s > 700:
            high = s
            s = (low + high) / 2
            continue
        dw = w * expm1(s)
        residual = g * s + dw - k_dt
        if abs(residual) <= 3e-15 * (1 + abs(k_dt)):
            break
        if residual > 0:
            high = s
        else:
            low = s
        candidate = s - residual / (g + w + dw)
        s = candidate if low < candidate < high else (low + high) / 2
    new_u = u + w * expm1(s)
    return log(new_u) if new_u > 0 else -np.inf


# Static input and evolving state travel with a name when it leaves the index.
_int_fields = 'group sector firm_id last_day'
_float_fields = (
    'x y eta l phi momentum short_var long_var zn zns '
    'cap floor g epsilon c mu gamma beta beta_sec beta_jump h lambda_i '
    'psi psi_jump ybar drift_x memory_decay memory_gain memory_noise '
    'raw_target inv_risk'
)
_dtype = np.dtype([(n, np.int64) for n in _int_fields.split()] +
                  [(n, np.float64) for n in _float_fields.split()])
_constant_keys = ('r_b gamma inflow equity_cap chi theta kappa Lambda risk_exponent '
                  'rho_y sigma_y kappa_F kappa_eta kappa_sh kappa_lg kappa_v '
                  'kappa_s kappa_ns rho_M rho_c w zeta').split()
_group_keys = ('sigma_x kappa_x sigma_F sigma_eta m lambda_M alpha j_M '
               'lambda_S j_S lambda_I sigma_JI s_M s_c s_n rho_n rho_ns w_n').split()
Constants = namedtuple('Constants', _constant_keys)
Group = namedtuple('Group', _group_keys)


def _constants(leverage, demand_rho):
    common = {k: COMMON[k] for k in _constant_keys}
    common['rho_y'] = COMMON['rho_y'] if demand_rho is None else float(demand_rho)
    groups = [{k: group[k] for k in _group_keys} for group in GROUPS]
    if not leverage:
        common['rho_M'] = common['rho_c'] = 0.
        for group in groups:
            group['rho_n'] = group['rho_ns'] = 0.
    return Constants(**common), tuple(Group(**group) for group in groups)


def _initialize(firms, cap, entry_quote, relative_size, constants, groups):
    """Initialize new names, preserving their inherited structural constants."""
    firms['cap'] = cap
    firms['l'] = np.log(cap)
    firms['floor'] = cap * COMMON['default_floor'] / entry_quote
    firms['psi'] = np.sqrt(np.maximum(0, 1 - firms['beta']**2 - firms['beta_sec']**2))
    firms['psi_jump'] = np.sqrt(np.maximum(0, 1 - firms['beta_jump']**2 - firms['beta_sec']**2))
    mu, coupling = firms['mu'], firms['c']
    g, epsilon = firms['g'], firms['epsilon']
    value0 = 1 - 1 / epsilon + np.log1p(epsilon * g) / (epsilon**2 * g)
    slope0 = -(1 - g * (1 - epsilon)) / (1 + epsilon * g)
    group_id = firms['group']
    rest = np.array([gr.m for gr in groups])[group_id] * relative_size
    firms['y'] = rest
    firms['ybar'] = rest + coupling * value0 / mu
    firms['drift_x'] = coupling * rest * slope0
    firms['memory_decay'] = np.exp(-mu * DT)
    firms['memory_gain'] = -np.expm1(-mu * DT) / mu
    firms['memory_noise'] = constants.sigma_y * np.sqrt(-np.expm1(-2 * mu * DT) / (2 * mu))
    # Neutral, equal short/long variance at birth. Both estimates subsequently use realized returns.
    var = np.array([gr.sigma_x**2 + gr.sigma_F**2 + gr.sigma_eta**2
                    for gr in groups])[group_id] * firms['h']**2
    firms['short_var'] = var
    firms['long_var'] = var


def _market_events(rng, n_days, enabled):
    """Exact continuous-time marked thinning, with shared uniforms across sleeves.

    Each marginal has its own Hawkes baseline, decay and branching ratio.
    A common Poisson measure gives the lower-intensity sleeve a subset of the
    arrivals, including when the two conditional intensities cross.
    """
    event_days = defaultdict(list)
    if not enabled:
        return event_days
    baseline, decay, alpha, rare_size, routine, small_size = (
        np.array([p[key] for p in GROUPS]) for key in
        ('lambda_M', 'beta_H', 'alpha', 'j_M', 'lambda_S', 'j_S'))
    excess = np.zeros(2)
    t = 0.0
    while t < n_days / DAYS:
        bound = max(baseline + excess)
        wait = rng.exponential(1 / bound)
        t += wait
        if t >= n_days / DAYS:
            break
        excess *= np.exp(-decay * wait)
        accepted = rng.uniform(0, bound) <= baseline + excess
        if np.any(accepted):
            event_days[int(t * DAYS)].append(-rng.exponential() * rare_size * accepted)
            excess += alpha * decay * accepted
    t = 0.0
    while True:
        t += rng.exponential(1 / max(routine))
        if t >= n_days / DAYS:
            break
        accepted = rng.uniform(0, max(routine)) <= routine
        event_days[int(t * DAYS)].append(-rng.exponential() * small_size * accepted)
    return event_days


def _shared_drivers(rng, n_days, c, groups):
    # Two groups load on exactly the same standard OU paths, with different scales.
    shocks = rng.standard_normal((n_days, 11))
    f, slow = exp(-c.kappa_v * DT), exp(-c.kappa_s * DT)
    sf, ss = sqrt(1 - f*f), sqrt(1 - slow*slow)
    fast_path = slow_path = common_path = 0.
    log_v = np.empty((n_days + 1, 2))
    z_common = np.empty_like(log_v)
    fast_sd = np.array([p.s_M * sqrt(1-c.w) for p in groups])
    slow_sd = np.array([p.s_M * sqrt(c.w) for p in groups])
    common_sd = np.array([p.s_c for p in groups])
    correction = .5 * (fast_sd**2 + slow_sd**2) + (
        c.rho_M**2 * fast_sd * slow_sd * sf * ss / (1 - f * slow))
    correlated = shocks[:, 0] * c.rho_M
    other = sqrt(1 - c.rho_M**2)
    vol_innovations = np.column_stack((
        correlated + other * shocks[:, 8], correlated + other * shocks[:, 9],
        c.rho_c * shocks[:, 0] + sqrt(1-c.rho_c**2) * shocks[:, 10]))
    # Store opening log variance at [t]; closing is [t+1].
    modes = _ou_modes(vol_innovations, np.array([f, slow, f]), np.array([sf, ss, sf]))
    log_v[:] = modes[:, 0, None] * fast_sd + modes[:, 1, None] * slow_sd - correction
    z_common[:] = modes[:, 2, None] * common_sd
    return shocks, log_v, z_common


@njit(cache=True)
def _ou_modes(innovations, decay, sigma):
    modes = np.zeros((len(innovations) + 1, 3))
    for t in range(len(innovations)):
        for j in range(3):
            modes[t+1, j] = decay[j] * modes[t, j] + sigma[j] * innovations[t, j]
    return modes


@njit(cache=True)
def _advance(firms, n, c, groups, log_vm, zc, common_shock, noise, jump,
             events_enabled, close_cap, old_cap):
    """Trade on opening information; first cache its unconstrained target.

    The cash leg is computed separately, still at the opening marks. This
    function never overwrites phi, so the memory drift sees the opening book.
    """
    mean_l = np.zeros(2)
    count = np.zeros(2)
    for i in range(n):
        j = firms[i].group
        mean_l[j] += firms[i].l
        count[j] += 1
    mean_l /= count
    fv, sv = exp(-c.kappa_v * DT), exp(-c.kappa_ns * DT)
    sf, ss = sqrt(1 - fv*fv), sqrt(1 - sv*sv)
    short_gain, long_gain = -expm1(-c.kappa_sh * DT), -expm1(-c.kappa_lg * DT)
    eta_decay = exp(-c.kappa_eta * DT)
    eta_sd = sqrt(-expm1(-2 * c.kappa_eta * DT) / (2 * c.kappa_eta))
    sqrt_vm = np.exp(.5 * log_vm)
    for i in range(len(firms)):
        r = firms[i]
        j, sector = r.group, r.sector
        p = groups[j]
        sfast, sslow = p.s_n * sqrt(1-p.w_n), p.s_n * sqrt(p.w_n)
        cov = p.rho_n * p.rho_ns * sfast * sslow * sf * ss / (1-fv*sv)
        log_vi = (r.lambda_i * zc[j] + r.zn + r.zns
                  - .5 * ((r.lambda_i * p.s_c)**2 + p.s_n**2) - cov)
        idio = exp(.5 * log_vi) * r.psi
        scale = sqrt_vm[j]
        price_noise = scale * (r.beta * common_shock[0] + r.beta_sec * common_shock[1+sector]) + idio * noise[i, 0]
        fund_noise = scale * (r.beta * common_shock[4] + r.beta_sec * common_shock[5+sector]) + idio * noise[i, 1]
        price_sd = sqrt(scale**2 * (r.beta**2 + r.beta_sec**2) + idio**2)
        demand_noise = c.rho_y * price_noise / price_sd + sqrt(1-c.rho_y**2) * noise[i, 2]
        value, _ = potential(r.x, r.g, r.epsilon)
        r.inv_risk = (r.long_var / r.short_var)**c.risk_exponent / c.Lambda
        budget_free = (c.gamma + c.theta * r.momentum - c.r_b) * r.inv_risk
        r.raw_target = budget_free + r.mu * (r.ybar - r.y) - r.c * value
        x_old = r.x
        x_split = impact_flow(r.x, r.c * r.y * DT, r.g, r.epsilon)
        r.x = x_split + (r.drift_x - p.kappa_x * r.x) * DT + p.sigma_x * r.h * ROOT_DT * price_noise
        r.y += r.memory_gain * (r.raw_target - r.phi) + r.memory_noise * demand_noise
        compensator = r.h * r.beta_jump * (p.lambda_M*p.j_M/(1-p.alpha) + p.lambda_S*p.j_S) if events_enabled else 0.
        r.l += (r.gamma + compensator - c.kappa_F * (r.l - mean_l[j])) * DT + p.sigma_F * r.h * ROOT_DT * fund_noise + jump[i]
        r.eta = eta_decay * r.eta + p.sigma_eta * r.h * eta_sd * noise[i, 3]
        r.zn = fv * r.zn + sfast * sf * (p.rho_n * noise[i, 0] + sqrt(1-p.rho_n**2) * noise[i, 4])
        r.zns = sv * r.zns + sslow * ss * (p.rho_ns * noise[i, 0] + sqrt(1-p.rho_ns**2) * noise[i, 5])
        old_cap[i] = r.cap
        price = exp(r.l + r.x + r.eta)
        close_cap[i] = max(price, r.floor)
        if price > r.floor:
            log_r = log(close_cap[i] / r.cap)
            q = log_r**2 / DT
            r.short_var += short_gain * (q - r.short_var)
            r.long_var += long_gain * (q - r.long_var)
            r.momentum += c.kappa * (r.x - x_old - r.momentum * DT)


@njit(cache=True)
def _trade(firms, n, cash, c, close_cap):
    """Eqs. (18)–(24), including the upper bound on followed-name purchases."""
    opening_cap = np.zeros(2)
    equity = np.zeros(2)
    unconstrained = np.zeros(2)
    upper = np.zeros(2)
    for i in range(len(firms)):
        r = firms[i]
        j = r.group
        if i < n:
            opening_cap[j] += r.cap
        bound = 1. if i < n else r.phi
        equity[j] += r.phi * r.cap
        unconstrained[j] += min(bound, max(0., r.raw_target)) * r.cap
        upper[j] = max(upper[j], r.raw_target / r.inv_risk)
    wealth = equity + cash
    budget = c.equity_cap * wealth
    shift = np.zeros(2)
    for j in range(2):
        if unconstrained[j] > budget[j]:
            low, high = 0., upper[j]
            for _ in range(42):
                mid = (low + high) / 2
                total = 0.
                for i in range(len(firms)):
                    r = firms[i]
                    if r.group == j:
                        bound = 1. if i < n else r.phi
                        total += r.cap * min(bound, max(0., r.raw_target - mid*r.inv_risk))
                if total > budget[j]:
                    low = mid
                else:
                    high = mid
            shift[j] = high
    proposed = np.empty(len(firms))
    buys, sells = np.zeros(2), np.zeros(2)
    for i in range(len(firms)):
        r = firms[i]
        bound = 1. if i < n else r.phi
        target = min(bound, max(0., r.raw_target - shift[r.group] * r.inv_risk))
        proposed[i] = (target - r.phi) * DT
        value = proposed[i] * r.cap
        buys[r.group] += max(0., value)
        sells[r.group] += max(0., -value)
    fraction = np.ones(2)
    for j in range(2):
        if buys[j] > 0:
            headroom = c.equity_cap * (wealth[j] - c.chi*sells[j]) - (equity[j] - sells[j])
            fraction[j] = min(1., max(0., headroom / ((1+c.equity_cap*c.chi)*buys[j])))
    costs = c.chi * (buys*fraction + sells)
    cash_after = cash + sells - buys*fraction - costs
    eq_after = equity + buys*fraction - sells
    residual = np.max(np.abs(cash_after + eq_after - (wealth-costs)) / np.maximum(wealth, 1e-100))
    post_trade_share = eq_after / (eq_after + cash_after)
    for i in range(len(firms)):
        r = firms[i]
        r.phi += proposed[i] * (fraction[r.group] if proposed[i] > 0 else 1.)
        r.cap = close_cap[i]
    cash[:] = cash_after * exp(c.r_b * DT) + c.inflow * opening_cap * DT
    return residual, max(post_trade_share), min(cash_after / wealth)


@njit(cache=True)
def _update_high(relative, ring, high, high_day, born, day):
    """Exact 15-year rolling maximum. Rescan only when the maximum expires."""
    ring[day % HISTORY, :] = relative
    for i in range(len(high)):
        if relative[i] >= high[i]:
            high[i], high_day[i] = relative[i], day
        elif high_day[i] <= day - HISTORY:
            high[i] = relative[i]
            high_day[i] = day
            for past in range(max(born[i], day-HISTORY+1), day):
                v = ring[past % HISTORY, i]
                if v > high[i]:
                    high[i], high_day[i] = v, past


def _reseat(firms, n, panel, cash, high, ring, day):
    """Keep each sleeve's market cap and marked wealth fixed at reseating.

    Apply the rank-remapping as a unit change for a firm's floor and size
    history. Preserve the investor's dollar position, paying any holding in
    excess of one share into cash. Return the fixed inter-sleeve index scales.
    """
    members = firms[:n]
    weights = np.zeros(2)
    ratios = np.ones(n)
    relative_multiplier = np.ones(n)
    for j in (0, 1):
        idx = np.flatnonzero(members['group'] == j)
        opening = members['cap'][idx]
        target = np.sort(panel['cap'][idx])
        weights[j] = target.sum() / opening.sum()
        new_cap = np.empty_like(opening)
        new_cap[np.argsort(opening, kind='stable')] = opening.sum() * target / target.sum()
        scale = new_cap / opening
        ratios[idx] = scale
        relative_multiplier[idx] = scale * np.median(opening) / np.median(new_cap)
        raw_shares = members['phi'][idx] / scale
        shares = np.minimum(raw_shares, 1.)
        cash[j] += np.dot(raw_shares - shares, new_cap)
        members['phi'][idx] = shares
        members['cap'][idx] = new_cap
    members['l'] += np.log(ratios)
    members['floor'] *= ratios
    high *= relative_multiplier
    ring[:min(day + 1, HISTORY)] *= relative_multiplier
    return weights


_series = ('log_index log_return x y z top_share removals defaults member_defaults '
           'exposure within_factor_two post_trade_equity_share cash_fraction_min '
           'budget_residual').split()


def simulate_bbm(years=15, warmup=50, seed=20261126, window='early', n_assets=500,
                 impact=True, keep_panel=False, leverage=True, demand_rho=None,
                 jumps=True, group_vol_multipliers=(1., 1.)):
    """Simulate daily paths; return 1-D arrays with the opening state at row zero.

    `impact` multiplies every c_i and recomputes the anchors in Eq. (47);
    0, 0.5 and 1 are useful ablations. `z` uses the bulk sleeve's log vM,
    a fixed affine transform of the common two-mode market driver. Exposure
    counts firms alive at the open (members and leavers followed for 5 years).
    Default count includes both. Removals are solvent member deletions.
    group_vol_multipliers scales h_i for every initial and entering company,
    ordered (large, bulk). A pair of ones retains the baseline calibration.

    With keep_panel=True, member_log_returns[t,slot] belongs to the firm in
    member_ids[t-1,slot]. member_ids[t] and member_caps[t] refer to the
    post-replacement close. Day 0 has a zero return and zero event counts.
    """
    n, burn, days = n_assets, round(warmup * DAYS), round(years * DAYS)
    if not isinstance(n, (np.integer, int)) or n < 10 or burn < 0 or days < 1:
        raise ValueError('Need at least ten assets, one measurement day and nonnegative warm-up.')
    if not np.isfinite(impact) or float(impact) < 0:
        raise ValueError('impact must be finite and nonnegative')
    if demand_rho is not None and not -1 <= demand_rho <= 1:
        raise ValueError('demand_rho must be a correlation')
    vol_scale = np.asarray(group_vol_multipliers, dtype=float)
    if vol_scale.shape != (2,) or not np.all(np.isfinite(vol_scale) & (vol_scale > 0)):
        raise ValueError('group_vol_multipliers must contain two finite positive values (large, bulk).')
    window = canonical_window(window)
    c, groups = _constants(leverage, demand_rho)
    panel_seed, dyn_seed, events_seed, entry_seed, common_seed = np.random.SeedSequence(seed).spawn(5)
    rng = np.random.default_rng(dyn_seed)
    entry_rng = np.random.default_rng(entry_seed)
    panel = make_panel(n_assets=n, window=window, seed=int(panel_seed.generate_state(1)[0]))
    firms = np.zeros(n, dtype=_dtype)
    for key in ('group', 'sector', 'beta', 'beta_sec', 'beta_jump', 'h',
                'lambda_i', 'mu', 'c', 'g', 'epsilon', 'gamma'):
        firms[key] = panel[key]
    firms['firm_id'] = np.arange(n)
    firms['last_day'] = -1
    firms['c'] *= float(impact)
    ids = firms['group'].copy()
    firms['h'] *= vol_scale[ids]
    counts = np.bincount(ids, minlength=2)
    totals = np.bincount(ids, weights=panel['cap'], minlength=2)
    relative_size = panel['cap'] / (totals / counts)[ids]
    _initialize(firms, panel['cap'], panel['entry_price'], relative_size, c, groups)
    # A funded book, initially 50% invested in equity, each sleeve sized on its own cap.
    firms['phi'] = .25
    cash = .25 * totals
    dates = burn + days
    shocks, log_vm, zc = _shared_drivers(np.random.default_rng(common_seed), dates, c, groups)
    event_days = _market_events(np.random.default_rng(events_seed), dates, jumps)
    group_event_rate = np.array([g.lambda_I for g in groups]) * DT
    group_jump_scale = np.array([g.sigma_JI for g in groups])
    exit_fraction = np.array([g['delta_exit'] for g in GROUPS], dtype=float)[ids]
    ring = np.zeros((HISTORY, n))
    medians = np.array([np.median(firms['cap'][ids == j]) for j in (0, 1)])
    high = firms['cap'] / medians[ids]
    high_day, born = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    ring[0] = high
    next_id = n
    index_scale = np.ones(2)
    series = {key: np.zeros(days + 1) for key in _series}
    series['replaced'] = np.zeros(days + 1, dtype=bool)
    if keep_panel:
        series['member_caps'] = np.empty((days + 1, n))
        series['member_log_returns'] = np.zeros((days + 1, n))
        series['member_ids'] = np.empty((days + 1, n), dtype=np.int64)

    def snapshot(out, day):
        cap = firms['cap'][:n] * index_scale[ids]
        series['x'][out] = np.dot(cap, firms['x'][:n]) / cap.sum()
        series['y'][out] = firms['y'][:n].mean()
        series['z'][out] = log_vm[day, 1]
        series['top_share'][out] = cap[ids == 0].sum() / cap.sum()
        series['within_factor_two'][out] = np.mean(np.abs(firms['x'][:n]) < log(2.))
        if keep_panel:
            series['member_caps'][out] = cap
            series['member_ids'][out] = firms['firm_id'][:n]

    for day in range(dates):
        if day == burn:
            index_scale = _reseat(firms, n, panel, cash, high, ring, day)
            snapshot(0, day)
        size = len(firms)
        all_groups = firms['group']
        noise = rng.standard_normal((size, 6))
        jump = np.zeros(size)
        if jumps:
            # The sum of k independent Normal(0,sigma²) marks is Normal(0,k*sigma²).
            k = rng.poisson(group_event_rate[all_groups])
            hit = k > 0
            jump[hit] = (firms['h'][hit] * firms['psi_jump'][hit] * group_jump_scale[all_groups[hit]] *
                         np.sqrt(k[hit]) * rng.standard_normal(hit.sum()))
            for signed_sizes in event_days.get(day, ()):
                jitter = np.maximum(0., 1 + c.zeta * rng.standard_normal(size))
                jump += firms['h'] * firms['beta_jump'] * signed_sizes[all_groups] * jitter
        close_cap, old_cap = np.empty(size), np.empty(size)
        _advance(firms, n, c, groups, log_vm[day], zc[day], shocks[day], noise, jump,
                 jumps, close_cap, old_cap)
        residual, share, cash_ratio = _trade(firms, n, cash, c, close_cap)
        default = close_cap <= firms['floor']
        medians = np.array([np.median(close_cap[:n][ids == j]) for j in (0, 1)])
        relative = close_cap[:n] / medians[ids]
        _update_high(relative, ring, high, high_day, born, day + 1)
        remove = (relative < exit_fraction * high) & ~default[:n]
        replace = np.flatnonzero(remove | default[:n])

        # Index weights are opening cap. A departing firm contributes its mark or
        # floor before its successor enters. An entrant's cap does not create a return.
        out = day + 1 - burn
        if out > 0:
            mark = index_scale[ids]
            ret = log(np.dot(close_cap[:n], mark) / np.dot(old_cap[:n], mark))
            series['log_index'][out] = series['log_index'][out-1] + ret
            series['log_return'][out] = ret
            series['removals'][out] = remove.sum()
            series['defaults'][out] = default.sum()
            series['member_defaults'][out] = default[:n].sum()
            series['exposure'][out] = size
            series['replaced'][out] = bool(len(replace))
            series['post_trade_equity_share'][out] = share
            series['cash_fraction_min'][out] = cash_ratio
            series['budget_residual'][out] = residual
            if keep_panel:
                series['member_log_returns'][out] = np.log(close_cap[:n] / old_cap[:n])

        # Floor recoveries go into the owning sleeve at zero transaction cost.
        if default.any():
            cash += np.bincount(all_groups[default],
                                weights=close_cap[default]*firms['phi'][default], minlength=2)
        if size > n:
            expired = (firms['last_day'][n:] <= day + 1) & ~default[n:]
            if expired.any():
                sale = close_cap[n:][expired] * firms['phi'][n:][expired] * (1-c.chi)
                cash += np.bincount(all_groups[n:][expired], weights=sale, minlength=2)
            leaving = expired | default[n:]
            if leaving.any():
                firms = np.concatenate((firms[:n], firms[n:][~leaving]))
        if len(replace):
            followed = firms[:n][remove].copy()
            followed['last_day'] = day + 1 + round(COMMON['follow_years'] * DAYS)
            # Retain constants and sector, redraw the three loadings and entry quote.
            entrants = np.zeros(len(replace), dtype=_dtype)
            for key in ('group', 'sector', 'mu', 'c', 'g', 'epsilon', 'gamma'):
                entrants[key] = firms[key][replace]
            new_group = entrants['group']
            new_load = draw_loadings(entry_rng, window, new_group, entrants['sector'])
            for key in ('beta', 'beta_jump', 'beta_sec', 'h', 'lambda_i'):
                entrants[key] = new_load[key]
            entrants['h'] *= vol_scale[new_group]
            cap = medians[new_group] * draw_entrant_ratio(entry_rng, window, len(replace))
            entry_quote = draw_entry_price(entry_rng, len(replace))
            sleeve_mean = np.bincount(ids, weights=close_cap[:n], minlength=2) / counts
            _initialize(entrants, cap, entry_quote, cap / sleeve_mean[new_group], c, groups)
            entrants['firm_id'] = np.arange(next_id, next_id + len(replace))
            next_id += len(replace)
            entrants['last_day'] = -1
            firms[replace] = entrants
            if len(followed):
                firms = np.concatenate((firms, followed))
            high[replace] = cap / medians[new_group]
            high_day[replace] = born[replace] = day + 1
            ring[(day+1) % HISTORY, replace] = high[replace]
        if out > 0:
            snapshot(out, day + 1)
    return series
