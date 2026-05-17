"""Causal inference estimators for the Module 5 student app.

All estimators take a DataFrame and return an EstimateResult containing the
ATE, a 95% CI from the nonparametric bootstrap, and method-specific
diagnostic information (e.g., propensity scores, weights, matched pairs).

Implementations are intentionally readable rather than maximally efficient.
This is a teaching tool: students should be able to open the file and follow
each method end-to-end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors
import statsmodels.api as sm

RNG = np.random.default_rng(42)


@dataclass
class EstimateResult:
    method: str
    ate: float
    se: float
    ci_low: float
    ci_high: float
    n_used: int
    diagnostics: dict = field(default_factory=dict)


# -------------------- helper: bootstrap --------------------

def _bootstrap(estimator_fn, df, n_boot=200, seed=42):
    """Run estimator on B bootstrap resamples; return point + percentile CI."""
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


# -------------------- 1. Naive --------------------

def naive(df: pd.DataFrame, outcome: str, treatment: str, n_boot: int = 200) -> EstimateResult:
    """Unadjusted difference in means."""
    def point(d):
        return d.loc[d[treatment] == 1, outcome].mean() - d.loc[d[treatment] == 0, outcome].mean()
    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    return EstimateResult(
        method="Naive (no adjustment)",
        ate=float(ate), se=se, ci_low=lo, ci_high=hi,
        n_used=len(df),
        diagnostics={
            "n_treated": int((df[treatment] == 1).sum()),
            "n_control": int((df[treatment] == 0).sum()),
            "mean_treated": float(df.loc[df[treatment] == 1, outcome].mean()),
            "mean_control": float(df.loc[df[treatment] == 0, outcome].mean()),
        }
    )


# -------------------- 2. Stratification --------------------

def stratification(df: pd.DataFrame, outcome: str, treatment: str, stratum_col: str,
                   n_boot: int = 200) -> EstimateResult:
    """Average within-stratum treated-control differences, weighted by stratum size (ATE)."""
    def point(d):
        strata = d.groupby(stratum_col)
        total = 0.0
        n_total = len(d)
        for _, g in strata:
            t = g.loc[g[treatment] == 1, outcome]
            c = g.loc[g[treatment] == 0, outcome]
            if len(t) == 0 or len(c) == 0:
                continue  # skip empty strata
            diff = t.mean() - c.mean()
            total += diff * (len(g) / n_total)
        return total
    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)

    # Per-stratum diagnostics (point estimate only, no boot)
    rows = []
    for level, g in df.groupby(stratum_col):
        t = g.loc[g[treatment] == 1, outcome]
        c = g.loc[g[treatment] == 0, outcome]
        rows.append({
            "stratum": level,
            "n_treated": int(len(t)),
            "n_control": int(len(c)),
            "mean_treated": float(t.mean()) if len(t) else np.nan,
            "mean_control": float(c.mean()) if len(c) else np.nan,
            "stratum_effect": float(t.mean() - c.mean()) if len(t) and len(c) else np.nan,
            "weight": len(g) / len(df),
        })
    return EstimateResult(
        method=f"Stratification by {stratum_col}",
        ate=float(ate), se=se, ci_low=lo, ci_high=hi,
        n_used=len(df),
        diagnostics={"strata": pd.DataFrame(rows)},
    )


# -------------------- 3. Regression adjustment --------------------

def regression(df: pd.DataFrame, outcome: str, treatment: str, covariates: list,
               include_interactions: bool = False, n_boot: int = 200) -> EstimateResult:
    """OLS: outcome ~ treatment + covariates  (optionally with treatment*covariate interactions)."""
    def fit(d):
        X_cols = [treatment] + list(covariates)
        X = d[X_cols].copy()
        if include_interactions:
            for c in covariates:
                X[f"{treatment}_x_{c}"] = d[treatment] * d[c]
        X = sm.add_constant(X, has_constant="add")
        y = d[outcome]
        return sm.OLS(y, X).fit()

    def point(d):
        m = fit(d)
        if include_interactions:
            # ATE = beta_treat + sum_c beta_int_c * mean(c) over the whole sample
            beta_t = m.params[treatment]
            ate = beta_t
            for c in covariates:
                ate += m.params[f"{treatment}_x_{c}"] * d[c].mean()
            return float(ate)
        return float(m.params[treatment])

    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)
    full = fit(df)
    return EstimateResult(
        method=f"Regression adjustment ({len(covariates)} covariates"
               + (", with interactions)" if include_interactions else ")"),
        ate=float(ate), se=se, ci_low=lo, ci_high=hi,
        n_used=len(df),
        diagnostics={"summary": full.summary().as_text()},
    )


# -------------------- 4. Propensity score (shared) --------------------

def estimate_propensity(df: pd.DataFrame, treatment: str, covariates: list) -> np.ndarray:
    """Logistic regression for P(W=1|X). Clipped to (0.001, 0.999)."""
    X = df[covariates].values
    y = df[treatment].values
    model = LogisticRegression(max_iter=2000, C=1e6)
    model.fit(X, y)
    ps = model.predict_proba(X)[:, 1]
    return np.clip(ps, 1e-3, 1 - 1e-3)


def smd(treated: np.ndarray, control: np.ndarray) -> float:
    """Standardized mean difference."""
    pooled_sd = np.sqrt((np.var(treated, ddof=1) + np.var(control, ddof=1)) / 2)
    if pooled_sd == 0:
        return 0.0
    return (np.mean(treated) - np.mean(control)) / pooled_sd


def covariate_balance(df: pd.DataFrame, treatment: str, covariates: list,
                      weights: Optional[np.ndarray] = None) -> pd.DataFrame:
    """Per-covariate SMD, optionally with weights."""
    rows = []
    for c in covariates:
        t = df.loc[df[treatment] == 1, c].values
        ctrl = df.loc[df[treatment] == 0, c].values
        raw = smd(t, ctrl)
        if weights is not None:
            wt = weights[df[treatment].values == 1]
            wc = weights[df[treatment].values == 0]
            mt = np.average(t, weights=wt)
            mc = np.average(ctrl, weights=wc)
            vt = np.average((t - mt) ** 2, weights=wt)
            vc = np.average((ctrl - mc) ** 2, weights=wc)
            pooled = np.sqrt((vt + vc) / 2)
            wsmd = (mt - mc) / pooled if pooled > 0 else 0.0
        else:
            wsmd = raw
        rows.append({"covariate": c, "smd_raw": raw, "smd_adjusted": wsmd})
    return pd.DataFrame(rows)


# -------------------- 5. Propensity score matching --------------------

def ps_matching(df: pd.DataFrame, outcome: str, treatment: str, covariates: list,
                caliper: Optional[float] = None, n_boot: int = 100) -> EstimateResult:
    """1:1 nearest-neighbor matching on the propensity score (with replacement).

    Estimates the ATT (the conventional matching estimand).
    """
    df = df.reset_index(drop=True)
    ps = estimate_propensity(df, treatment, covariates)
    df = df.assign(_ps=ps)
    treated = df[df[treatment] == 1]
    control = df[df[treatment] == 0]

    nn = NearestNeighbors(n_neighbors=1).fit(control[["_ps"]].values)
    dist, idx = nn.kneighbors(treated[["_ps"]].values)
    matched_control_idx = control.index[idx.flatten()].values

    keep = np.ones(len(treated), dtype=bool)
    if caliper is not None:
        keep = dist.flatten() <= caliper
    matched = pd.DataFrame({
        "treat_idx": treated.index.values[keep],
        "control_idx": matched_control_idx[keep],
        "ps_distance": dist.flatten()[keep],
    })
    matched["y_treat"] = df.loc[matched["treat_idx"], outcome].values
    matched["y_control"] = df.loc[matched["control_idx"], outcome].values
    matched["pair_diff"] = matched["y_treat"] - matched["y_control"]

    def point(d):
        # Bootstrap version: resample the data, refit PS, rematch
        try:
            ps_b = estimate_propensity(d, treatment, covariates)
            tb = d[d[treatment] == 1].assign(_ps=ps_b[d[treatment].values == 1])
            cb = d[d[treatment] == 0].assign(_ps=ps_b[d[treatment].values == 0])
            if len(tb) == 0 or len(cb) == 0:
                return np.nan
            nn_b = NearestNeighbors(n_neighbors=1).fit(cb[["_ps"]].values)
            dist_b, idx_b = nn_b.kneighbors(tb[["_ps"]].values)
            keep_b = np.ones(len(tb), dtype=bool)
            if caliper is not None:
                keep_b = dist_b.flatten() <= caliper
            if keep_b.sum() == 0:
                return np.nan
            y_treat = tb[outcome].values[keep_b]
            y_control = cb[outcome].values[idx_b.flatten()][keep_b]
            return float(np.mean(y_treat - y_control))
        except Exception:
            return np.nan

    ate = float(matched["pair_diff"].mean())
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)

    # Compute weighted balance (each control gets weight = times matched)
    control_match_count = pd.Series(matched_control_idx[keep]).value_counts()
    w = np.zeros(len(df))
    w[df[df[treatment] == 1].index.values[keep]] = 1.0
    for cidx, count in control_match_count.items():
        w[cidx] = count
    bal = covariate_balance(df, treatment, covariates, weights=w)

    return EstimateResult(
        method=f"PS Matching (1:1 NN" + (f", caliper={caliper}" if caliper else "") + ")",
        ate=ate, se=se, ci_low=lo, ci_high=hi,
        n_used=int(keep.sum()),
        diagnostics={
            "ps": ps,
            "matched_pairs": matched,
            "balance": bal,
            "n_dropped_caliper": int((~keep).sum()),
        }
    )


# -------------------- 6. IPW --------------------

def ipw(df: pd.DataFrame, outcome: str, treatment: str, covariates: list,
        stabilized: bool = False, trim: Optional[float] = None,
        n_boot: int = 200) -> EstimateResult:
    """Inverse probability weighting (Horvitz-Thompson or stabilized)."""
    df = df.reset_index(drop=True)
    ps = estimate_propensity(df, treatment, covariates)
    keep = np.ones(len(df), dtype=bool)
    if trim is not None:
        keep = (ps > trim) & (ps < 1 - trim)

    def point(d):
        ps_b = estimate_propensity(d, treatment, covariates)
        keep_b = np.ones(len(d), dtype=bool)
        if trim is not None:
            keep_b = (ps_b > trim) & (ps_b < 1 - trim)
        d_k = d[keep_b]
        ps_k = ps_b[keep_b]
        w = d_k[treatment].values
        y = d_k[outcome].values
        if stabilized:
            p1 = w.mean()
            wt = w * p1 / ps_k + (1 - w) * (1 - p1) / (1 - ps_k)
        else:
            wt = w / ps_k + (1 - w) / (1 - ps_k)
        # Hajek estimator (always — it's more stable than Horvitz-Thompson)
        treated_w = wt[w == 1]
        control_w = wt[w == 0]
        return float(
            np.average(y[w == 1], weights=treated_w)
            - np.average(y[w == 0], weights=control_w)
        )

    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)

    # Diagnostics: weights on the original sample
    w = df[treatment].values
    if stabilized:
        p1 = w.mean()
        weights = w * p1 / ps + (1 - w) * (1 - p1) / (1 - ps)
    else:
        weights = w / ps + (1 - w) / (1 - ps)
    bal = covariate_balance(df[keep], treatment, covariates, weights=weights[keep])

    return EstimateResult(
        method=f"IPW ({'stabilized' if stabilized else 'standard'}"
               + (f", trim={trim}" if trim else "") + ")",
        ate=ate, se=se, ci_low=lo, ci_high=hi,
        n_used=int(keep.sum()),
        diagnostics={
            "ps": ps,
            "weights": weights,
            "balance": bal,
            "max_weight": float(weights.max()),
            "n_trimmed": int((~keep).sum()),
        }
    )


# -------------------- 7. Doubly robust (AIPW) --------------------

def aipw(df: pd.DataFrame, outcome: str, treatment: str, covariates: list,
         n_boot: int = 100) -> EstimateResult:
    """Augmented IPW (doubly robust). Outcome model: linear regression."""
    df = df.reset_index(drop=True)

    def point(d):
        ps_b = estimate_propensity(d, treatment, covariates)
        X = d[covariates].values
        y = d[outcome].values
        w = d[treatment].values

        # Outcome models for E[Y | W=1, X] and E[Y | W=0, X]
        if (w == 1).sum() >= 2:
            m1 = LinearRegression().fit(X[w == 1], y[w == 1])
            mu1 = m1.predict(X)
        else:
            mu1 = np.full(len(d), y[w == 1].mean() if (w == 1).any() else 0.0)
        if (w == 0).sum() >= 2:
            m0 = LinearRegression().fit(X[w == 0], y[w == 0])
            mu0 = m0.predict(X)
        else:
            mu0 = np.full(len(d), y[w == 0].mean() if (w == 0).any() else 0.0)

        aipw_t = mu1 + w * (y - mu1) / ps_b
        aipw_c = mu0 + (1 - w) * (y - mu0) / (1 - ps_b)
        return float(np.mean(aipw_t - aipw_c))

    ate = point(df)
    se, lo, hi = _bootstrap(point, df, n_boot=n_boot)

    ps = estimate_propensity(df, treatment, covariates)
    return EstimateResult(
        method="Doubly Robust (AIPW)",
        ate=ate, se=se, ci_low=lo, ci_high=hi,
        n_used=len(df),
        diagnostics={"ps": ps, "max_weight": float(max(1/ps[df[treatment]==1].min(),
                                                       1/(1-ps[df[treatment]==0].max())))}
    )
