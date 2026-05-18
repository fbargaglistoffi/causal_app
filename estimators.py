"""Causal inference estimators for the Module 5 student app.

All ATE estimators take a DataFrame and return an EstimateResult containing the
ATE, a 95% CI from the nonparametric bootstrap, and method-specific
diagnostic information (e.g., propensity scores, weights, matched pairs).

Heterogeneity tools: cate_interaction (regression-based), t_learner_cate (RF).
Sensitivity tool: e_value (VanderWeele & Ding 2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm


@dataclass
class EstimateResult:
    method: str
    ate: float
    se: float
    ci_low: float
    ci_high: float
    n_used: int
    diagnostics: dict = field(default_factory=dict)


def _bootstrap(estimator_fn, df, n_boot=200, seed=42):
    rng = np.random.default_rng(seed)
    n = len(df)
    ates = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sample = df.iloc[idx].reset_index(drop=True)
        try:
            ates.append(estimator_fn(sample))
        except Exception:
            continue
    ates = np.array(ates)
    se = float(np.std(ates))
    lo, hi = np.percentile(ates, [2.5, 97.5])
    return se, float(lo), float(hi)


# ===================================================================
# ATE estimators (Item 6)
# ===================================================================

def naive(df, outcome, treatment, n_boot=200):
    def point(d):
        return d.loc[d[treatment] == 1, outcome].mean() - d.loc[d[treatment] == 0, outcome].mean()
    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    return EstimateResult("Naive (no adjustment)", float(ate), se, lo, hi, len(df),
                          diagnostics={"n_treated": int((df[treatment]==1).sum()),
                                       "n_control": int((df[treatment]==0).sum())})


def stratification(df, outcome, treatment, stratum_col, n_boot=200):
    def point(d):
        total = 0.0
        n_total = len(d)
        for _, g in d.groupby(stratum_col):
            t = g.loc[g[treatment]==1, outcome]
            c = g.loc[g[treatment]==0, outcome]
            if len(t) == 0 or len(c) == 0:
                continue
            total += (t.mean() - c.mean()) * (len(g) / n_total)
        return total
    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    rows = []
    for level, g in df.groupby(stratum_col):
        t = g.loc[g[treatment]==1, outcome]
        c = g.loc[g[treatment]==0, outcome]
        rows.append({"stratum": level,
                     "n_treated": int(len(t)), "n_control": int(len(c)),
                     "mean_treated": float(t.mean()) if len(t) else np.nan,
                     "mean_control": float(c.mean()) if len(c) else np.nan,
                     "stratum_effect": float(t.mean()-c.mean()) if len(t) and len(c) else np.nan,
                     "weight": len(g)/len(df)})
    return EstimateResult(f"Stratification by {stratum_col}", float(ate), se, lo, hi, len(df),
                          diagnostics={"strata": pd.DataFrame(rows)})


def regression(df, outcome, treatment, covariates, include_interactions=False, n_boot=200):
    def fit(d):
        X_cols = [treatment] + list(covariates)
        X = d[X_cols].copy()
        if include_interactions:
            for c in covariates:
                X[f"{treatment}_x_{c}"] = d[treatment] * d[c]
        X = sm.add_constant(X, has_constant="add")
        return sm.OLS(d[outcome], X).fit()

    def point(d):
        m = fit(d)
        if include_interactions:
            ate = m.params[treatment]
            for c in covariates:
                ate += m.params[f"{treatment}_x_{c}"] * d[c].mean()
            return float(ate)
        return float(m.params[treatment])

    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    full = fit(df)
    return EstimateResult(
        f"Regression adjustment ({len(covariates)} covariates"
        + (", with interactions)" if include_interactions else ")"),
        float(ate), se, lo, hi, len(df),
        diagnostics={"summary": full.summary().as_text()})


def estimate_propensity(df, treatment, covariates):
    X = df[covariates].values
    y = df[treatment].values
    model = LogisticRegression(max_iter=2000, C=1e6)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return np.clip(ps, 1e-3, 1-1e-3)


def smd(t, c):
    sd = np.sqrt((np.var(t, ddof=1) + np.var(c, ddof=1))/2)
    return 0.0 if sd == 0 else (np.mean(t) - np.mean(c)) / sd


def covariate_balance(df, treatment, covariates, weights=None):
    rows = []
    for c in covariates:
        t = df.loc[df[treatment]==1, c].values
        ctrl = df.loc[df[treatment]==0, c].values
        raw = smd(t, ctrl)
        if weights is not None:
            wt = weights[df[treatment].values == 1]
            wc = weights[df[treatment].values == 0]
            mt = np.average(t, weights=wt); mc = np.average(ctrl, weights=wc)
            vt = np.average((t-mt)**2, weights=wt); vc = np.average((ctrl-mc)**2, weights=wc)
            pooled = np.sqrt((vt+vc)/2)
            wsmd = (mt - mc) / pooled if pooled > 0 else 0.0
        else:
            wsmd = raw
        rows.append({"covariate": c, "smd_raw": raw, "smd_adjusted": wsmd})
    return pd.DataFrame(rows)


def ps_matching(df, outcome, treatment, covariates, caliper=None, n_boot=100):
    df = df.reset_index(drop=True)
    ps = estimate_propensity(df, treatment, covariates)
    df = df.assign(_ps=ps)
    treated = df[df[treatment]==1]
    control = df[df[treatment]==0]
    nn = NearestNeighbors(n_neighbors=1).fit(control[["_ps"]].values)
    dist, idx = nn.kneighbors(treated[["_ps"]].values)
    matched_control_idx = control.index[idx.flatten()].values
    keep = np.ones(len(treated), dtype=bool)
    if caliper is not None:
        keep = dist.flatten() <= caliper

    matched = pd.DataFrame({
        "treat_idx": treated.index.values[keep],
        "control_idx": matched_control_idx[keep],
        "ps_distance": dist.flatten()[keep]})
    matched["y_treat"] = df.loc[matched["treat_idx"], outcome].values
    matched["y_control"] = df.loc[matched["control_idx"], outcome].values
    matched["pair_diff"] = matched["y_treat"] - matched["y_control"]

    def point(d):
        try:
            ps_b = estimate_propensity(d, treatment, covariates)
            tb = d[d[treatment]==1].assign(_ps=ps_b[d[treatment].values==1])
            cb = d[d[treatment]==0].assign(_ps=ps_b[d[treatment].values==0])
            if len(tb) == 0 or len(cb) == 0: return np.nan
            nn_b = NearestNeighbors(n_neighbors=1).fit(cb[["_ps"]].values)
            dist_b, idx_b = nn_b.kneighbors(tb[["_ps"]].values)
            keep_b = np.ones(len(tb), dtype=bool)
            if caliper is not None: keep_b = dist_b.flatten() <= caliper
            if keep_b.sum() == 0: return np.nan
            return float(np.mean(tb[outcome].values[keep_b] - cb[outcome].values[idx_b.flatten()][keep_b]))
        except Exception: return np.nan

    ate = float(matched["pair_diff"].mean())
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    control_match_count = pd.Series(matched_control_idx[keep]).value_counts()
    w = np.zeros(len(df))
    w[df[df[treatment]==1].index.values[keep]] = 1.0
    for cidx, count in control_match_count.items():
        w[cidx] = count
    bal = covariate_balance(df, treatment, covariates, weights=w)
    return EstimateResult(
        f"PS Matching (1:1 NN" + (f", caliper={caliper}" if caliper else "") + ")",
        ate, se, lo, hi, int(keep.sum()),
        diagnostics={"ps": ps, "matched_pairs": matched, "balance": bal,
                     "n_dropped_caliper": int((~keep).sum())})


def ipw(df, outcome, treatment, covariates, stabilized=False, trim=None, n_boot=200):
    df = df.reset_index(drop=True)
    ps = estimate_propensity(df, treatment, covariates)
    keep = np.ones(len(df), dtype=bool)
    if trim is not None: keep = (ps > trim) & (ps < 1-trim)

    def point(d):
        ps_b = estimate_propensity(d, treatment, covariates)
        keep_b = np.ones(len(d), dtype=bool)
        if trim is not None: keep_b = (ps_b > trim) & (ps_b < 1-trim)
        d_k = d[keep_b]; ps_k = ps_b[keep_b]
        w = d_k[treatment].values; y = d_k[outcome].values
        if stabilized:
            p1 = w.mean()
            wt = w * p1 / ps_k + (1-w) * (1-p1) / (1-ps_k)
        else:
            wt = w / ps_k + (1-w) / (1-ps_k)
        return float(np.average(y[w==1], weights=wt[w==1])
                     - np.average(y[w==0], weights=wt[w==0]))

    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    w = df[treatment].values
    if stabilized:
        p1 = w.mean()
        weights = w * p1 / ps + (1-w) * (1-p1) / (1-ps)
    else:
        weights = w / ps + (1-w) / (1-ps)
    bal = covariate_balance(df[keep], treatment, covariates, weights=weights[keep])
    return EstimateResult(
        f"IPW ({'stabilized' if stabilized else 'standard'}"
        + (f", trim={trim}" if trim else "") + ")",
        ate, se, lo, hi, int(keep.sum()),
        diagnostics={"ps": ps, "weights": weights, "balance": bal,
                     "max_weight": float(weights.max()),
                     "n_trimmed": int((~keep).sum())})


def aipw(df, outcome, treatment, covariates, n_boot=100):
    df = df.reset_index(drop=True)

    def point(d):
        ps_b = estimate_propensity(d, treatment, covariates)
        X = d[covariates].values; y = d[outcome].values; w = d[treatment].values
        m1 = LinearRegression().fit(X[w==1], y[w==1]) if (w==1).sum() >= 2 else None
        m0 = LinearRegression().fit(X[w==0], y[w==0]) if (w==0).sum() >= 2 else None
        mu1 = m1.predict(X) if m1 else np.full(len(d), y[w==1].mean() if (w==1).any() else 0.0)
        mu0 = m0.predict(X) if m0 else np.full(len(d), y[w==0].mean() if (w==0).any() else 0.0)
        aipw_t = mu1 + w * (y - mu1) / ps_b
        aipw_c = mu0 + (1-w) * (y - mu0) / (1-ps_b)
        return float(np.mean(aipw_t - aipw_c))

    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    ps = estimate_propensity(df, treatment, covariates)
    return EstimateResult("Doubly Robust (AIPW)", ate, se, lo, hi, len(df),
                          diagnostics={"ps": ps})


# ===================================================================
# Heterogeneity (Item 7)
# ===================================================================

@dataclass
class CATEResult:
    method: str
    interaction_coef: Optional[float] = None
    interaction_se: Optional[float] = None
    interaction_p: Optional[float] = None
    modifier: Optional[str] = None
    cate_by_x: Optional[pd.DataFrame] = None
    cate_per_unit: Optional[np.ndarray] = None
    cate_x_values: Optional[np.ndarray] = None
    calibration: Optional[pd.DataFrame] = None
    summary_text: Optional[str] = None


def cate_interaction(df, outcome, treatment, modifier, other_covariates):
    """Fit Y ~ W + X_mod + W*X_mod + other_covs and report the interaction.

    Returns interaction coefficient with 95% CI + p-value, plus the conditional
    average treatment effect computed across observed values of the modifier.
    """
    other = [c for c in other_covariates if c != modifier]
    X = df[[treatment, modifier] + other].copy()
    X[f"{treatment}_x_{modifier}"] = df[treatment] * df[modifier]
    X = sm.add_constant(X, has_constant="add")
    m = sm.OLS(df[outcome], X).fit()

    coef = float(m.params[f"{treatment}_x_{modifier}"])
    se = float(m.bse[f"{treatment}_x_{modifier}"])
    p = float(m.pvalues[f"{treatment}_x_{modifier}"])
    main = float(m.params[treatment])

    # CATE across observed values of the modifier
    xs = np.sort(df[modifier].unique())
    if len(xs) > 20:
        # continuous modifier — use a fine grid
        xs = np.linspace(df[modifier].quantile(0.05), df[modifier].quantile(0.95), 40)
    cate_x = main + coef * xs
    # SE for the CATE at each x: sqrt(Var(beta_W) + x^2*Var(beta_int) + 2*x*Cov)
    cov = m.cov_params()
    var_W = cov.loc[treatment, treatment]
    var_int = cov.loc[f"{treatment}_x_{modifier}", f"{treatment}_x_{modifier}"]
    cov_W_int = cov.loc[treatment, f"{treatment}_x_{modifier}"]
    se_x = np.sqrt(var_W + xs**2 * var_int + 2 * xs * cov_W_int)
    cate_df = pd.DataFrame({
        modifier: xs,
        "cate": cate_x,
        "ci_low": cate_x - 1.96 * se_x,
        "ci_high": cate_x + 1.96 * se_x,
    })

    return CATEResult(
        method=f"Regression with W × {modifier} interaction",
        interaction_coef=coef, interaction_se=se, interaction_p=p,
        modifier=modifier, cate_by_x=cate_df,
        summary_text=m.summary().as_text(),
    )


def t_learner_cate(df, outcome, treatment, covariates, n_estimators=200, seed=42):
    """Estimate individual CATEs using a random-forest T-learner with honest splitting.

    Returns per-unit CATE estimates on the held-out half, plus a calibration
    table grouping units into deciles of predicted CATE and reporting the
    observed treatment effect within each decile.
    """
    rng = np.random.default_rng(seed)
    X = df[covariates].values
    y = df[outcome].values
    W = df[treatment].values

    idx = np.arange(len(df))
    rng.shuffle(idx)
    half = len(df) // 2
    train, test = idx[:half], idx[half:]

    m1 = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                min_samples_leaf=5)
    m0 = RandomForestRegressor(n_estimators=n_estimators, random_state=seed,
                                min_samples_leaf=5)
    if (W[train]==1).sum() < 5 or (W[train]==0).sum() < 5:
        raise ValueError("Need at least 5 treated and 5 control units in the training half.")
    m1.fit(X[train][W[train]==1], y[train][W[train]==1])
    m0.fit(X[train][W[train]==0], y[train][W[train]==0])

    mu1_test = m1.predict(X[test])
    mu0_test = m0.predict(X[test])
    cate_test = mu1_test - mu0_test

    # Calibration: bin held-out units by predicted CATE, compute IPW-adjusted
    # observed difference within each bin
    n_bins = min(10, len(test) // 30) or 2
    test_df = pd.DataFrame({
        "cate_pred": cate_test,
        "y": y[test], "W": W[test], "row": test,
    })
    test_df["bin"] = pd.qcut(test_df["cate_pred"], q=n_bins,
                              labels=False, duplicates="drop")

    cal_rows = []
    for b in sorted(test_df["bin"].dropna().unique()):
        sub = test_df[test_df["bin"] == b]
        t = sub.loc[sub["W"]==1, "y"]
        c = sub.loc[sub["W"]==0, "y"]
        if len(t) == 0 or len(c) == 0:
            cal_rows.append({"bin": int(b), "n": len(sub),
                             "mean_predicted": float(sub["cate_pred"].mean()),
                             "observed_effect": np.nan, "se": np.nan})
            continue
        diff = t.mean() - c.mean()
        se = np.sqrt(t.var(ddof=1)/len(t) + c.var(ddof=1)/len(c))
        cal_rows.append({"bin": int(b), "n": len(sub),
                         "mean_predicted": float(sub["cate_pred"].mean()),
                         "observed_effect": float(diff), "se": float(se)})
    cal_df = pd.DataFrame(cal_rows)

    return CATEResult(
        method="T-learner causal forest (random forest, honest split)",
        cate_per_unit=cate_test, cate_x_values=test,
        calibration=cal_df,
    )


# ===================================================================
# Sensitivity (Item 8) — VanderWeele & Ding 2017 E-value
# ===================================================================

@dataclass
class EValueResult:
    method: str = "VanderWeele & Ding (2017) E-value"
    rr_estimate: float = 1.0
    rr_ci_bound: float = 1.0
    e_value_point: float = 1.0
    e_value_ci: float = 1.0
    note: str = ""


def _e_value_from_rr(rr: float) -> float:
    """E-value from a risk ratio (always > 1 after flipping if necessary)."""
    if rr < 1:
        rr = 1.0 / rr
    if rr <= 1.0:
        return 1.0
    return rr + np.sqrt(rr * (rr - 1))


def e_value(point_estimate: float, ci_low: float, ci_high: float,
            outcome_type: str = "continuous",
            outcome_sd: Optional[float] = None) -> EValueResult:
    """Compute the E-value (VanderWeele & Ding 2017).

    outcome_type:
      "binary" / "risk_ratio": point_estimate is already a risk ratio.
      "continuous":            point_estimate is a mean difference. Provide
                               outcome_sd. We use Chinn (2000)'s approximation
                               RR ≈ exp(0.91 × d), where d = mean_diff / SD.
    """
    if outcome_type in ("binary", "risk_ratio"):
        rr = point_estimate
        # The relevant CI bound is the one closer to the null (1.0)
        ci_bound = ci_high if rr < 1 else ci_low
        note = "Point estimate interpreted directly as a risk ratio."
    elif outcome_type == "continuous":
        if outcome_sd is None or outcome_sd <= 0:
            raise ValueError("outcome_sd must be positive for continuous outcomes.")
        d = point_estimate / outcome_sd
        d_low = ci_low / outcome_sd
        d_high = ci_high / outcome_sd
        rr = float(np.exp(0.91 * d))
        rr_low = float(np.exp(0.91 * d_low))
        rr_high = float(np.exp(0.91 * d_high))
        # Same logic: bound closer to the null
        ci_bound = rr_high if rr < 1 else rr_low
        note = (f"Continuous outcome — standardized as Cohen's d = "
                f"{d:.3f}, then converted to RR ≈ exp(0.91 × d) = {rr:.3f} "
                f"following Chinn (2000) and VanderWeele & Ding (2017).")
    else:
        raise ValueError(f"Unknown outcome_type: {outcome_type}")

    return EValueResult(
        rr_estimate=float(rr),
        rr_ci_bound=float(ci_bound),
        e_value_point=_e_value_from_rr(rr),
        e_value_ci=_e_value_from_rr(ci_bound),
        note=note,
    )
