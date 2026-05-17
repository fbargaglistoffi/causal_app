"""Streamlit app for Module 5: students play with causal-inference estimators
on the LaLonde / NSW dataset and compare results to the RCT benchmark.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from estimators import (
    EstimateResult,
    naive, stratification, regression, ps_matching, ipw, aipw,
    estimate_propensity, covariate_balance,
)

# ---------------------------------------------------------------------------
# Page config & global style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Causal Inference Playground — Module 5",
                   page_icon="📊", layout="wide")

NAVY    = "#1E2761"
COLORS  = {
    "treated":   "#1f77b4",
    "control":   "#d62728",
    "rct":       "#2ca02c",
    "estimate":  "#1E2761",
    "ok":        "#2ca02c",
    "warn":      "#ff7f0e",
    "bad":       "#d62728",
}

st.markdown(
    f"""
    <style>
    h1 {{ color: {NAVY}; }}
    h2, h3 {{ color: {NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
LALONDE_COVARIATES = ["age", "educ", "black", "hisp", "marr", "nodegree",
                      "re74", "re75"]


@st.cache_data
def load_lalonde():
    exp = pd.read_csv("data/nsw_experimental.csv")
    obs = pd.read_csv("data/nsw_observational.csv")
    return exp, obs


@st.cache_data
def rct_benchmark(_exp_df: pd.DataFrame, treatment: str, outcome: str) -> EstimateResult:
    return naive(_exp_df, outcome, treatment, n_boot=300)


@st.cache_data
def parse_uploaded(file_bytes: bytes) -> pd.DataFrame:
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("📊 Causal Inference Playground")
st.caption("Module 5 · From Identification to Estimation")

exp_df, obs_df = load_lalonde()

# ---------------------------------------------------------------------------
# Sidebar — data source picker
# ---------------------------------------------------------------------------
st.sidebar.header("Data source")

source = st.sidebar.radio(
    "Where does your data come from?",
    ["LaLonde / NSW (built-in)", "Upload your own CSV"],
    key="data_source",
)

# These get set differently depending on the source
df = None
TREATMENT = None
OUTCOME = None
ALL_COVARIATES = []
has_truth = False
rct = None

if source.startswith("LaLonde"):
    st.sidebar.divider()
    dataset_choice = st.sidebar.radio(
        "Dataset",
        ["Observational (NSW treated + CPS controls)", "Experimental (RCT)"],
        help="The observational dataset combines NSW participants with non-random "
             "CPS controls — confounding is severe. The experimental dataset is "
             "the RCT and gives the 'true' causal effect.",
    )
    df = obs_df if dataset_choice.startswith("Observational") else exp_df
    TREATMENT = "treat"
    OUTCOME = "re78"
    ALL_COVARIATES = LALONDE_COVARIATES
    rct = rct_benchmark(exp_df, TREATMENT, OUTCOME)
    has_truth = True

else:
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV", type=["csv"],
        help="Your CSV needs at least one binary 0/1 column (treatment), "
             "one numeric column (outcome), and one or more covariate columns. "
             "All columns must be numeric — encode categoricals as 0/1 dummies "
             "before uploading.",
    )

    if uploaded is None:
        st.sidebar.info("Pick a CSV to begin. The file is read once and held "
                        "in memory for the rest of the session.")
        st.info("👈 Upload a CSV in the sidebar to start.")
        st.markdown("""
### What your CSV should look like

A simple wide table where every row is a unit (person, household, firm, …):

| treat | outcome | age | sex | … |
|-------|---------|-----|-----|---|
| 1     | 12500   | 35  | 1   | … |
| 0     | 8400    | 42  | 0   | … |
| …     | …       | …   | …   | … |

- **One column** is the binary treatment (0/1)
- **One column** is the numeric outcome you want the causal effect on
- **The remaining columns** are candidate confounders (numeric — encode
  any categorical variables as 0/1 dummies first)

You can download the LaLonde dataset as a template:
""")
        with open("data/nsw_observational.csv", "rb") as f:
            st.download_button("⬇️ Download LaLonde observational CSV",
                               f.read(), file_name="nsw_observational.csv",
                               mime="text/csv")
        st.stop()

    # Parse and validate
    try:
        df = parse_uploaded(uploaded.getvalue())
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        st.stop()

    if len(df) < 20:
        st.sidebar.error(f"CSV has {len(df)} rows — too small. Need at least 20.")
        st.stop()

    # Subsample large datasets for performance (with a notice)
    if len(df) > 5000:
        n_orig = len(df)
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        st.sidebar.info(f"Subsampled to 5,000 rows from {n_orig:,} for "
                        f"interactive performance.")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.sidebar.error("No numeric columns found. All columns must be numeric.")
        st.stop()

    binary_cols = [c for c in numeric_cols if set(df[c].dropna().unique()) <= {0, 1}]
    if not binary_cols:
        st.sidebar.error("No binary 0/1 column found — need one for treatment.")
        st.stop()

    st.sidebar.divider()
    st.sidebar.subheader("Map your columns")

    TREATMENT = st.sidebar.selectbox(
        "Treatment column (binary 0/1)",
        binary_cols,
        index=0,
        key="upload_treat",
    )

    outcome_options = [c for c in numeric_cols if c != TREATMENT]
    OUTCOME = st.sidebar.selectbox(
        "Outcome column (numeric)",
        outcome_options,
        index=0,
        key="upload_outcome",
    )

    cov_options = [c for c in numeric_cols if c not in (TREATMENT, OUTCOME)]
    ALL_COVARIATES = st.sidebar.multiselect(
        "Available covariates",
        cov_options,
        default=cov_options,
        key="upload_covs",
        help="Variables you might want to use to adjust for confounding. "
             "You'll still pick which ones to actually include in each tab.",
    )

    if not ALL_COVARIATES:
        st.sidebar.error("Pick at least one covariate.")
        st.stop()

    has_truth = False  # No RCT benchmark for uploaded data

st.sidebar.divider()
st.sidebar.markdown(f"**Sample**: {len(df):,} rows · "
                    f"{(df[TREATMENT]==1).sum()} treated · "
                    f"{(df[TREATMENT]==0).sum()} control")
st.sidebar.markdown(f"**Treatment**: `{TREATMENT}`   "
                    f"**Outcome**: `{OUTCOME}`")

if has_truth and rct is not None:
    st.sidebar.divider()
    st.sidebar.markdown("### RCT benchmark (truth)")
    st.sidebar.metric("ATE on earnings (1978 USD)",
                      f"${rct.ate:,.0f}",
                      delta=f"95% CI [${rct.ci_low:,.0f}, ${rct.ci_high:,.0f}]",
                      delta_color="off")
    st.sidebar.caption("This is the 'right answer' you are trying to recover "
                       "with observational methods.")
else:
    st.sidebar.divider()
    st.sidebar.caption("No experimental benchmark available for this dataset — "
                       "you cannot validate your estimate against a 'true' ATE.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_about, tab_explore, tab_method, tab_compare = st.tabs(
    ["1 · About the data",
     "2 · Explore the data",
     "3 · Try a method",
     "4 · Compare all methods"]
)


# =========================================================================
# TAB 1 — ABOUT
# =========================================================================
with tab_about:
    if has_truth:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("The LaLonde / National Supported Work (NSW) experiment")
            st.markdown("""
The **National Supported Work Demonstration (1975–79)** was a randomised
job-training programme for disadvantaged workers (former drug addicts,
ex-offenders, high-school dropouts, long-term welfare recipients).

**Treatment ($W$):** assignment to the training programme.
**Outcome ($Y$):** real earnings in 1978 (post-treatment).
**Confounders ($X$):** age, education, race, marital status, prior earnings, etc.

**Why this dataset is famous in causal inference**

In 1986, **Robert LaLonde** showed that when you replace the experimental
controls with non-experimental controls from the **CPS** (Current Population
Survey), standard regression methods fail badly — they suggest the programme
*hurt* earnings.

In 1999, **Dehejia & Wahba** revisited the data with propensity-score methods
and recovered estimates close to the experimental truth. This became the
canonical demonstration that **how you adjust for confounders matters**.

**Your job in this app**

You have two datasets:

1. **Experimental dataset** — the RCT. The treated–control difference is the
   true causal effect (about **\$1,794**).
2. **Observational dataset** — same NSW treated units, but the controls come
   from the CPS. Your job: use the methods from Module 5 to recover the truth.
""")
        with col2:
            st.subheader("Truth (from the RCT)")
            st.metric("Average Treatment Effect", f"${rct.ate:,.0f}",
                      help="True causal effect of the NSW programme on 1978 earnings.")
            st.write(f"**95% CI**: [${rct.ci_low:,.0f}, ${rct.ci_high:,.0f}]")
            st.write(f"**n** = {len(exp_df)}")
            st.divider()
            st.subheader("Naive observational estimate")
            naive_obs = naive(obs_df, OUTCOME, TREATMENT, n_boot=200)
            st.metric("Treated − Control (no adjustment)",
                      f"${naive_obs.ate:,.0f}",
                      delta=f"vs RCT: ${naive_obs.ate - rct.ate:+,.0f}",
                      delta_color="inverse")
            st.caption("Without adjusting for confounders, the observational data "
                       "suggests the programme *reduced* earnings by ~\$8,500. "
                       "Confounding has flipped the sign of the effect.")
    else:
        st.subheader("Your uploaded dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows",     f"{len(df):,}")
            st.metric("Treated",  f"{int((df[TREATMENT]==1).sum()):,}")
            st.metric("Control",  f"{int((df[TREATMENT]==0).sum()):,}")
        with col2:
            st.metric("Outcome (mean, treated)",
                      f"{df.loc[df[TREATMENT]==1, OUTCOME].mean():,.2f}")
            st.metric("Outcome (mean, control)",
                      f"{df.loc[df[TREATMENT]==0, OUTCOME].mean():,.2f}")
            naive_diff = (df.loc[df[TREATMENT]==1, OUTCOME].mean()
                          - df.loc[df[TREATMENT]==0, OUTCOME].mean())
            st.metric("Naive treated − control", f"{naive_diff:+,.2f}",
                      help="Unadjusted difference in means. This is what the "
                           "Module 5 methods will adjust.")
        st.divider()
        st.markdown(f"**Treatment column:** `{TREATMENT}`  \n"
                    f"**Outcome column:** `{OUTCOME}`  \n"
                    f"**Candidate covariates:** {', '.join(f'`{c}`' for c in ALL_COVARIATES)}")
        st.markdown("---")
        st.markdown("##### First 5 rows")
        st.dataframe(df[[TREATMENT, OUTCOME] + ALL_COVARIATES].head(),
                     use_container_width=True)
        st.warning("⚠️ No experimental benchmark is available for this dataset. "
                   "You'll get an estimate from each method, but you cannot "
                   "validate it against a known 'true' ATE the way you can "
                   "with LaLonde.")


# =========================================================================
# TAB 2 — EXPLORE
# =========================================================================
with tab_explore:
    st.subheader("Distribution of confounders by treatment status")

    selected = st.multiselect("Variables to compare",
                              ALL_COVARIATES, default=ALL_COVARIATES,
                              key="explore_vars")

    if selected:
        # Summary stats by treatment
        summary = (df.groupby(TREATMENT)[selected].mean().T)
        summary.columns = ["Control mean", "Treated mean"]
        bal = covariate_balance(df, TREATMENT, selected)
        summary["SMD (raw)"] = bal.set_index("covariate")["smd_raw"]
        summary["|SMD| > 0.1"] = summary["SMD (raw)"].abs() > 0.1

        def style_smd(val):
            if isinstance(val, bool):
                return "background-color: #ffe5e5" if val else ""
            return ""

        st.dataframe(
            summary.style.format({"Control mean": "{:.2f}",
                                  "Treated mean": "{:.2f}",
                                  "SMD (raw)": "{:+.3f}"})
                         .map(style_smd, subset=["|SMD| > 0.1"]),
            use_container_width=True,
        )
        st.caption("**SMD** = standardized mean difference. Values with "
                   "$|SMD| > 0.1$ (highlighted) indicate meaningful imbalance "
                   "between treated and control groups before any adjustment.")

    st.divider()
    st.subheader("Distribution of pre-treatment earnings (re75)")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(df.loc[df[TREATMENT]==0, "re75"], bins=30, alpha=0.6,
            color=COLORS["control"], label=f"Control (n={(df[TREATMENT]==0).sum()})")
    ax.hist(df.loc[df[TREATMENT]==1, "re75"], bins=30, alpha=0.6,
            color=COLORS["treated"], label=f"Treated (n={(df[TREATMENT]==1).sum()})")
    ax.set_xlabel("1975 earnings (USD)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    st.caption("If the two distributions look very different, the naive "
               "comparison is comparing apples and oranges. The role of the "
               "methods in Module 5 is to make these populations comparable.")


# =========================================================================
# TAB 3 — TRY A METHOD
# =========================================================================
with tab_method:
    col_l, col_r = st.columns([1, 2])

    with col_l:
        st.subheader("Choose a method")
        method = st.selectbox(
            "Estimator",
            ["Naive (no adjustment)",
             "Stratification",
             "Regression adjustment",
             "Propensity Score Matching",
             "Inverse Probability Weighting (IPW)",
             "Doubly Robust (AIPW)"],
        )

        st.subheader("Adjustment set")
        covariates = st.multiselect(
            "Confounders to adjust for",
            ALL_COVARIATES, default=ALL_COVARIATES,
            help="The variables that you believe satisfy the backdoor "
                 "criterion (Module 4). Try removing some and watch the "
                 "estimate move.",
        )

        # method-specific options
        kwargs = {}
        if method == "Stratification":
            # Only offer columns with a small number of distinct values
            stratum_options = [c for c in ALL_COVARIATES
                               if df[c].nunique() <= 20]
            if not stratum_options:
                st.warning("No covariate has ≤ 20 distinct values — "
                           "stratification needs a discrete variable.")
                kwargs["stratum_col"] = ALL_COVARIATES[0]
            else:
                kwargs["stratum_col"] = st.selectbox("Stratify by", stratum_options)
        elif method == "Regression adjustment":
            kwargs["include_interactions"] = st.checkbox(
                "Include treatment × covariate interactions", value=False)
        elif method == "Propensity Score Matching":
            use_caliper = st.checkbox("Use caliper (drop poor matches)",
                                      value=False)
            if use_caliper:
                kwargs["caliper"] = st.slider(
                    "Caliper width (PS distance)", 0.001, 0.2, 0.05, 0.005)
        elif method == "Inverse Probability Weighting (IPW)":
            kwargs["stabilized"] = st.checkbox("Stabilized weights", value=True)
            use_trim = st.checkbox("Trim extreme propensity scores", value=True)
            if use_trim:
                kwargs["trim"] = st.slider(
                    "Trim threshold", 0.001, 0.2, 0.05, 0.005)

        run = st.button("Estimate ▶", type="primary", use_container_width=True)

    with col_r:
        if run:
            if not covariates and method != "Naive (no adjustment)":
                st.error("Pick at least one covariate to adjust for.")
                st.stop()

            with st.spinner(f"Running {method}…"):
                try:
                    if method == "Naive (no adjustment)":
                        result = naive(df, OUTCOME, TREATMENT)
                    elif method == "Stratification":
                        result = stratification(df, OUTCOME, TREATMENT, **kwargs)
                    elif method == "Regression adjustment":
                        result = regression(df, OUTCOME, TREATMENT, covariates, **kwargs)
                    elif method == "Propensity Score Matching":
                        result = ps_matching(df, OUTCOME, TREATMENT, covariates,
                                             n_boot=50, **kwargs)
                    elif method == "Inverse Probability Weighting (IPW)":
                        result = ipw(df, OUTCOME, TREATMENT, covariates, **kwargs)
                    elif method == "Doubly Robust (AIPW)":
                        result = aipw(df, OUTCOME, TREATMENT, covariates, n_boot=50)
                except Exception as e:
                    st.error(f"Estimation failed: {e}")
                    st.stop()

            # ----- top-line metric -----
            st.subheader("Estimated Average Treatment Effect")

            if has_truth:
                cols = st.columns(3)
                cols[0].metric("Your estimate", f"{result.ate:,.2f}")
                cols[1].metric("Bootstrap 95% CI",
                               f"[{result.ci_low:,.2f},  {result.ci_high:,.2f}]")
                gap = result.ate - rct.ate
                cols[2].metric("Distance from RCT truth",
                               f"{gap:+,.2f}",
                               delta=f"RCT: {rct.ate:,.2f}",
                               delta_color="off")
            else:
                cols = st.columns(3)
                cols[0].metric("Your estimate", f"{result.ate:,.2f}")
                cols[1].metric("Bootstrap 95% CI",
                               f"[{result.ci_low:,.2f},  {result.ci_high:,.2f}]")
                cols[2].metric("CI width",
                               f"{result.ci_high - result.ci_low:,.2f}")

            # ----- estimate visual -----
            fig, ax = plt.subplots(figsize=(8, 1.8))
            if has_truth:
                ax.axvline(rct.ate, color=COLORS["rct"], linestyle="--",
                           label=f"RCT truth ({rct.ate:,.2f})", linewidth=2)
                ax.axvspan(rct.ci_low, rct.ci_high, alpha=0.15, color=COLORS["rct"])
            ax.errorbar([result.ate], [0],
                        xerr=[[result.ate - result.ci_low],
                              [result.ci_high - result.ate]],
                        fmt="o", color=COLORS["estimate"], markersize=10,
                        capsize=5, linewidth=2.5,
                        label=f"Your estimate ({result.ate:,.2f})")
            ax.set_yticks([])
            ax.set_xlabel(f"Estimated ATE on `{OUTCOME}`")
            ax.legend(loc="upper right", framealpha=0.95)
            ax.spines[["top", "right", "left"]].set_visible(False)
            st.pyplot(fig)

            # ----- method-specific diagnostics -----
            d = result.diagnostics

            if method == "Stratification" and "strata" in d:
                st.subheader("Per-stratum estimates")
                st.dataframe(
                    d["strata"].style.format({
                        "mean_treated": "{:,.2f}",
                        "mean_control": "{:,.2f}",
                        "stratum_effect": "{:+,.2f}",
                        "weight": "{:.2%}"
                    }),
                    use_container_width=True,
                )
                st.caption("ATE is the weighted average of stratum-specific "
                           "effects. Strata where one of treated/control is "
                           "empty are dropped (positivity violation).")

            if method == "Regression adjustment" and "summary" in d:
                with st.expander("📋 Full regression output (advanced)"):
                    st.text(d["summary"])

            if "ps" in d:
                st.subheader("Propensity score overlap")
                fig, ax = plt.subplots(figsize=(8, 3))
                ax.hist(d["ps"][df[TREATMENT].values == 0], bins=40, alpha=0.6,
                        color=COLORS["control"], label="Control")
                ax.hist(d["ps"][df[TREATMENT].values == 1], bins=40, alpha=0.6,
                        color=COLORS["treated"], label="Treated")
                ax.set_xlabel("Estimated propensity score $\hat{e}(X)$")
                ax.set_ylabel("Count")
                ax.legend()
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                st.caption("Where the two distributions overlap, you can compare "
                           "treated vs control. Where they don't, any estimate "
                           "relies on extrapolation.")

            if "weights" in d:
                st.subheader("Distribution of IPW weights")
                fig, ax = plt.subplots(figsize=(8, 2.5))
                w = d["weights"]
                ax.hist(w, bins=40, color=COLORS["estimate"], alpha=0.8)
                ax.axvline(w.mean(), color="black", linestyle="--",
                           label=f"Mean = {w.mean():.2f}")
                ax.axvline(w.max(), color=COLORS["bad"], linestyle="--",
                           label=f"Max = {w.max():.2f}")
                ax.set_xlabel("Weight")
                ax.set_ylabel("Count")
                ax.legend()
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                m_a, m_b, m_c = st.columns(3)
                m_a.metric("Mean weight", f"{w.mean():.2f}")
                m_b.metric("Max weight",  f"{d['max_weight']:.1f}")
                m_c.metric("Units dropped (trim)",
                           f"{d.get('n_trimmed', 0)}")

            if "balance" in d:
                st.subheader("Covariate balance (Love plot)")
                bal = d["balance"]
                fig, ax = plt.subplots(figsize=(8, 0.45 * len(bal) + 1))
                y = np.arange(len(bal))
                ax.scatter(bal["smd_raw"].abs(), y, color=COLORS["bad"],
                           s=60, label="Before adjustment")
                ax.scatter(bal["smd_adjusted"].abs(), y, color=COLORS["ok"],
                           s=60, label="After adjustment")
                for i, row in bal.reset_index(drop=True).iterrows():
                    ax.plot([abs(row["smd_raw"]), abs(row["smd_adjusted"])],
                            [i, i], color="lightgray", zorder=0)
                ax.axvline(0.1, color="black", linestyle=":", alpha=0.6,
                           label="Threshold (|SMD| = 0.1)")
                ax.set_yticks(y)
                ax.set_yticklabels(bal["covariate"])
                ax.set_xlabel("|Standardized Mean Difference|")
                ax.legend(loc="lower right")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                worst = bal["smd_adjusted"].abs().max()
                n_above = int((bal["smd_adjusted"].abs() > 0.1).sum())
                b_a, b_b, b_c = st.columns(3)
                b_a.metric("Max |SMD| after",       f"{worst:.3f}")
                b_b.metric("Mean |SMD| after",      f"{bal['smd_adjusted'].abs().mean():.3f}")
                b_c.metric("# covariates with |SMD|>0.1", f"{n_above} / {len(bal)}")

            if "matched_pairs" in d:
                with st.expander("🔗 First 10 matched pairs"):
                    st.dataframe(d["matched_pairs"].head(10), use_container_width=True)
                if d.get("n_dropped_caliper", 0) > 0:
                    st.info(f"ℹ️ {d['n_dropped_caliper']} treated units could "
                            f"not be matched within the caliper.")

        else:
            st.info("👈 Configure a method on the left and hit **Estimate**.")


# =========================================================================
# TAB 4 — COMPARE ALL METHODS
# =========================================================================
with tab_compare:
    st.subheader("Run every method on the same dataset")
    st.write("This runs all six adjustment methods on your selected covariate "
             "set, then plots them side by side"
             + (" against the RCT benchmark." if has_truth else "."))

    covariates_all = st.multiselect(
        "Covariates to adjust for (used by every method)",
        ALL_COVARIATES, default=ALL_COVARIATES,
        key="compare_covariates",
    )
    stratum_options = [c for c in ALL_COVARIATES if df[c].nunique() <= 20]
    if stratum_options:
        stratum = st.selectbox("Stratification variable", stratum_options,
                               key="compare_stratum")
    else:
        stratum = ALL_COVARIATES[0]
        st.caption(f"Stratifying by `{stratum}` (no low-cardinality covariates).")

    if st.button("Run all methods ▶", type="primary"):
        if not covariates_all:
            st.error("Pick at least one covariate.")
            st.stop()

        results = []
        progress = st.progress(0.0, text="Starting…")
        steps = [
            ("Naive",            lambda: naive(df, OUTCOME, TREATMENT)),
            ("Stratification",   lambda: stratification(df, OUTCOME, TREATMENT, stratum)),
            ("Regression",       lambda: regression(df, OUTCOME, TREATMENT, covariates_all)),
            ("PS Matching",      lambda: ps_matching(df, OUTCOME, TREATMENT,
                                                     covariates_all, n_boot=50)),
            ("IPW (stabilized)", lambda: ipw(df, OUTCOME, TREATMENT,
                                             covariates_all, stabilized=True, trim=0.05)),
            ("Doubly Robust",    lambda: aipw(df, OUTCOME, TREATMENT,
                                              covariates_all, n_boot=50)),
        ]
        for i, (name, fn) in enumerate(steps):
            progress.progress((i + 1) / len(steps), text=f"Running {name}…")
            try:
                r = fn()
                results.append((name, r.ate, r.ci_low, r.ci_high))
            except Exception as e:
                st.warning(f"{name} failed: {e}")
        progress.empty()

        # Forest plot
        st.subheader("Forest plot — every method on this dataset")
        fig, ax = plt.subplots(figsize=(9, 0.55 * len(results) + 1.5))
        names = [r[0] for r in results]
        ates  = np.array([r[1] for r in results])
        los   = np.array([r[2] for r in results])
        his   = np.array([r[3] for r in results])
        y     = np.arange(len(results))[::-1]

        if has_truth:
            ax.axvspan(rct.ci_low, rct.ci_high, color=COLORS["rct"], alpha=0.15,
                       label="RCT 95% CI")
            ax.axvline(rct.ate, color=COLORS["rct"], linestyle="--", linewidth=2,
                       label=f"RCT truth ({rct.ate:,.2f})")
        ax.errorbar(ates, y, xerr=[ates - los, his - ates],
                    fmt="o", color=COLORS["estimate"], markersize=9,
                    capsize=5, linewidth=2)
        for i, (n, a, lo, hi) in enumerate(results):
            ax.text(max(his) + (max(his) - min(los)) * 0.05, y[i],
                    f"{a:,.2f}  [{lo:,.2f}, {hi:,.2f}]",
                    va="center", fontsize=9, color="#222222")

        ax.axvline(0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel(f"Estimated ATE on `{OUTCOME}`")
        if has_truth:
            ax.legend(loc="upper left")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)

        st.subheader("Summary table")
        df_results = pd.DataFrame(results, columns=["Method", "ATE", "CI low", "CI high"])
        df_results["CI width"] = df_results["CI high"] - df_results["CI low"]
        if has_truth:
            df_results["Distance from RCT"] = df_results["ATE"] - rct.ate
        format_dict = {"ATE": "{:,.2f}", "CI low": "{:,.2f}",
                       "CI high": "{:,.2f}", "CI width": "{:,.2f}"}
        if has_truth:
            format_dict["Distance from RCT"] = "{:+,.2f}"
        st.dataframe(
            df_results.style.format(format_dict),
            use_container_width=True,
        )

        if has_truth:
            st.markdown("""
**Discussion prompts**

- Which method got closest to the RCT truth?
- Did any method fail dramatically? Why might that be?
- What happens if you remove `re74` and `re75` (prior earnings) from the
  covariate set and re-run? These are the strongest known confounders.
- Does Doubly Robust give the smallest CI? The most accurate point estimate?
""")
        else:
            st.markdown("""
**Discussion prompts**

- How much do the six estimates disagree? A wide spread suggests fragility
  in the identification argument, not just the estimator.
- Try removing your strongest suspected confounder and re-running. Which
  methods are most sensitive to its inclusion?
- Which method gives the narrowest CI? Does narrower mean more credible?
""")
    else:
        st.info("👈 Pick covariates and hit **Run all methods** to compare.")


