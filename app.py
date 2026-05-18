"""Causal Inference Playground — final project app.

Tab layout maps onto the credibility checklist:
  1. About the data
  2. Balance diagnostics                (Item 3 — Is the comparison fair?)
  3. Estimate the ATE                   (Item 6 — Estimation strategy)
  4. Heterogeneity (CATE)               (Item 7 — Effect modification)
  5. Sensitivity analysis (E-value)     (Item 8 — Other threats)
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from estimators import (
    EstimateResult, CATEResult, EValueResult,
    naive, stratification, regression, ps_matching, ipw, aipw,
    estimate_propensity, covariate_balance,
    cate_interaction, t_learner_cate, e_value,
)

# ---------------------------------------------------------------------------
# Page config & style
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Causal Inference Playground",
                   page_icon="📊", layout="wide")

NAVY = "#1E2761"
COLORS = {
    "treated":  "#1f77b4",
    "control":  "#d62728",
    "rct":      "#2ca02c",
    "estimate": "#1E2761",
    "ok":       "#2ca02c",
    "warn":     "#ff7f0e",
    "bad":      "#d62728",
}

st.markdown(
    f"""<style> h1 {{color:{NAVY};}} h2,h3 {{color:{NAVY};}} </style>""",
    unsafe_allow_html=True,
)

LALONDE_COVARIATES = ["age", "educ", "black", "hisp", "marr", "nodegree",
                      "re74", "re75"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_lalonde():
    exp = pd.read_csv("data/nsw_experimental.csv")
    obs = pd.read_csv("data/nsw_observational.csv")
    return exp, obs


@st.cache_data
def rct_benchmark(_exp_df, treatment, outcome):
    return naive(_exp_df, outcome, treatment, n_boot=300)


@st.cache_data
def parse_uploaded(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))


# ---------------------------------------------------------------------------
# Header & sidebar
# ---------------------------------------------------------------------------
st.title("📊 Causal Inference Playground")
st.caption("End-to-end workflow · Balance → Estimate → Heterogeneity → Sensitivity")

exp_df, obs_df = load_lalonde()

st.sidebar.header("Data source")
source = st.sidebar.radio(
    "Where does your data come from?",
    ["LaLonde / NSW (built-in)", "Upload your own CSV"],
    key="data_source",
)

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
        help="Observational data confounding is severe. Experimental is the "
             "RCT and gives the 'true' causal effect.",
    )
    df = obs_df if dataset_choice.startswith("Observational") else exp_df
    TREATMENT = "treat"; OUTCOME = "re78"
    ALL_COVARIATES = LALONDE_COVARIATES
    rct = rct_benchmark(exp_df, TREATMENT, OUTCOME)
    has_truth = True
else:
    uploaded = st.sidebar.file_uploader(
        "Upload a CSV", type=["csv"],
        help="Needs one binary 0/1 column (treatment), one numeric column "
             "(outcome), and one or more numeric covariate columns. Encode "
             "categoricals as 0/1 dummies before uploading.",
    )
    if uploaded is None:
        st.sidebar.info("Pick a CSV to begin.")
        st.info("👈 Upload a CSV in the sidebar to start.")
        st.markdown("""
### What your CSV should look like

A wide table where every row is a unit:

| treat | outcome | age | sex | … |
|-------|---------|-----|-----|---|
| 1     | 12500   | 35  | 1   | … |
| 0     | 8400    | 42  | 0   | … |

- One column is the **binary treatment** (0/1)
- One column is the **numeric outcome**
- Remaining columns are candidate confounders (numeric only)

You can download the LaLonde observational dataset as a template:
""")
        with open("data/nsw_observational.csv", "rb") as f:
            st.download_button("⬇️ Download LaLonde observational CSV",
                               f.read(), file_name="nsw_observational.csv",
                               mime="text/csv")
        st.stop()

    try:
        df = parse_uploaded(uploaded.getvalue())
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        st.stop()
    if len(df) < 20:
        st.sidebar.error(f"CSV has {len(df)} rows — too small (need ≥ 20).")
        st.stop()
    if len(df) > 5000:
        n_orig = len(df)
        df = df.sample(n=5000, random_state=42).reset_index(drop=True)
        st.sidebar.info(f"Subsampled to 5,000 from {n_orig:,} for performance.")

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
    TREATMENT = st.sidebar.selectbox("Treatment column (binary 0/1)",
                                     binary_cols, key="upload_treat")
    outcome_options = [c for c in numeric_cols if c != TREATMENT]
    OUTCOME = st.sidebar.selectbox("Outcome column (numeric)",
                                    outcome_options, key="upload_outcome")
    cov_options = [c for c in numeric_cols if c not in (TREATMENT, OUTCOME)]
    ALL_COVARIATES = st.sidebar.multiselect(
        "Available covariates", cov_options, default=cov_options,
        key="upload_covs",
        help="Variables you might use to adjust for confounding.",
    )
    if not ALL_COVARIATES:
        st.sidebar.error("Pick at least one covariate.")
        st.stop()
    has_truth = False

st.sidebar.divider()
st.sidebar.markdown(f"**Sample**: {len(df):,} rows · "
                    f"{(df[TREATMENT]==1).sum()} treated · "
                    f"{(df[TREATMENT]==0).sum()} control")
st.sidebar.markdown(f"**Treatment**: `{TREATMENT}`   **Outcome**: `{OUTCOME}`")

if has_truth and rct is not None:
    st.sidebar.divider()
    st.sidebar.markdown("### RCT benchmark (truth)")
    st.sidebar.metric("ATE on earnings (1978 USD)",
                      f"${rct.ate:,.0f}",
                      delta=f"95% CI [${rct.ci_low:,.0f}, ${rct.ci_high:,.0f}]",
                      delta_color="off")
    st.sidebar.caption("The 'right answer' you are trying to recover with "
                       "observational methods.")
else:
    st.sidebar.divider()
    st.sidebar.caption("No experimental benchmark — you cannot validate your "
                       "estimate against a known 'true' ATE.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_about, tab_balance, tab_estimate, tab_hete, tab_sens = st.tabs([
    "1 · About the data",
    "2 · Balance diagnostics",
    "3 · Estimate the ATE",
    "4 · Heterogeneity",
    "5 · Sensitivity",
])


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
job-training programme for disadvantaged workers.

**Treatment ($W$):** assignment to training.
**Outcome ($Y$):** real earnings in 1978.
**Confounders ($X$):** age, education, race, marital status, prior earnings.

In 1986, **Robert LaLonde** showed that when you replace the experimental
controls with non-experimental controls from the **CPS**, standard regression
methods suggest the programme *hurt* earnings.

In 1999, **Dehejia & Wahba** revisited the data with propensity-score methods
and recovered estimates close to the experimental truth.

You have two datasets in this app:

1. **Experimental** — the RCT, true effect ≈ **\\$1,794**
2. **Observational** — same NSW treated, CPS controls. Your job: recover the truth.
""")
        with col2:
            st.subheader("Truth (from the RCT)")
            st.metric("Average Treatment Effect", f"${rct.ate:,.0f}")
            st.write(f"**95% CI**: [${rct.ci_low:,.0f}, ${rct.ci_high:,.0f}]")
            st.write(f"**n** = {len(exp_df)}")
            st.divider()
            naive_obs = naive(obs_df, OUTCOME, TREATMENT, n_boot=200)
            st.subheader("Naive observational estimate")
            st.metric("Treated − Control (no adjustment)",
                      f"${naive_obs.ate:,.0f}",
                      delta=f"vs RCT: ${naive_obs.ate - rct.ate:+,.0f}",
                      delta_color="inverse")
            st.caption("Confounding has flipped the sign of the effect.")
    else:
        st.subheader("Your uploaded dataset")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Rows", f"{len(df):,}")
            st.metric("Treated", f"{int((df[TREATMENT]==1).sum()):,}")
            st.metric("Control", f"{int((df[TREATMENT]==0).sum()):,}")
        with col2:
            st.metric("Outcome (mean, treated)",
                      f"{df.loc[df[TREATMENT]==1, OUTCOME].mean():,.2f}")
            st.metric("Outcome (mean, control)",
                      f"{df.loc[df[TREATMENT]==0, OUTCOME].mean():,.2f}")
            naive_diff = (df.loc[df[TREATMENT]==1, OUTCOME].mean()
                          - df.loc[df[TREATMENT]==0, OUTCOME].mean())
            st.metric("Naive treated − control", f"{naive_diff:+,.2f}",
                      help="Unadjusted difference in means.")
        st.divider()
        st.markdown(f"**Treatment column:** `{TREATMENT}`  \n"
                    f"**Outcome column:** `{OUTCOME}`  \n"
                    f"**Candidate covariates:** {', '.join(f'`{c}`' for c in ALL_COVARIATES)}")
        st.markdown("##### First 5 rows")
        st.dataframe(df[[TREATMENT, OUTCOME] + ALL_COVARIATES].head(),
                     use_container_width=True)


# =========================================================================
# TAB 2 — BALANCE DIAGNOSTICS (Item 3)
# =========================================================================
with tab_balance:
    st.subheader("Standardized Mean Differences (raw, before adjustment)")
    selected = st.multiselect("Covariates to inspect",
                              ALL_COVARIATES, default=ALL_COVARIATES,
                              key="balance_vars")
    if selected:
        summary = df.groupby(TREATMENT)[selected].mean().T
        summary.columns = ["Control mean", "Treated mean"]
        bal = covariate_balance(df, TREATMENT, selected)
        summary["SMD (raw)"] = bal.set_index("covariate")["smd_raw"]
        st.dataframe(
            summary.style.format({"Control mean": "{:.3f}",
                                  "Treated mean": "{:.3f}",
                                  "SMD (raw)": "{:+.3f}"}),
            use_container_width=True,
        )
        st.caption("**SMD** = standardized mean difference. Convention: |SMD| > 0.1 "
                   "suggests meaningful imbalance, |SMD| > 0.25 is severe.")

    st.divider()
    st.subheader("Love plot — raw imbalance for every covariate")
    bal_all = covariate_balance(df, TREATMENT, ALL_COVARIATES)
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(bal_all) + 1.2))
    y = np.arange(len(bal_all))
    ax.scatter(bal_all["smd_raw"].abs(), y, color=COLORS["bad"], s=70)
    ax.axvline(0.1, color="black", linestyle=":", alpha=0.6, label="|SMD| = 0.1")
    ax.axvline(0.25, color="gray", linestyle=":", alpha=0.4, label="|SMD| = 0.25")
    ax.set_yticks(y); ax.set_yticklabels(bal_all["covariate"])
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)

    st.divider()
    st.subheader("Propensity score overlap")
    st.caption("Logistic regression of treatment on every covariate. Where "
               "treated and control distributions overlap, you can compare "
               "them. Where they don't, the methods extrapolate.")
    overlap_covs = st.multiselect(
        "Covariates in the propensity score model",
        ALL_COVARIATES, default=ALL_COVARIATES, key="overlap_covs",
    )
    if overlap_covs:
        ps = estimate_propensity(df, TREATMENT, overlap_covs)
        ps_t = ps[df[TREATMENT].values == 1]
        ps_c = ps[df[TREATMENT].values == 0]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Treated, $\\hat{e}<0.05$", f"{(ps_t<0.05).mean():.0%}")
        m2.metric("Treated, $\\hat{e}>0.95$", f"{(ps_t>0.95).mean():.0%}")
        m3.metric("Control, $\\hat{e}<0.05$", f"{(ps_c<0.05).mean():.0%}")
        m4.metric("Control, $\\hat{e}>0.95$", f"{(ps_c>0.95).mean():.0%}")
        st.caption(f"Min/max PS — treated: [{ps_t.min():.3f}, {ps_t.max():.3f}], "
                   f"control: [{ps_c.min():.3f}, {ps_c.max():.3f}].")

        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.hist(ps_c, bins=40, alpha=0.6, color=COLORS["control"], label="Control")
        ax.hist(ps_t, bins=40, alpha=0.6, color=COLORS["treated"], label="Treated")
        ax.axvline(0.05, color="black", linestyle=":", alpha=0.5)
        ax.axvline(0.95, color="black", linestyle=":", alpha=0.5)
        ax.set_xlabel(r"Estimated propensity score $\hat{e}(X)$")
        ax.set_ylabel("Count"); ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)


# =========================================================================
# TAB 3 — ESTIMATE THE ATE (Item 6)
# =========================================================================
with tab_estimate:
    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.subheader("Choose a method")
        method = st.selectbox(
            "Estimator",
            ["Regression adjustment", "Propensity Score Matching",
             "Inverse Probability Weighting (IPW)",
             "Doubly Robust (AIPW)"],
            key="estimate_method",
        )
        st.subheader("Adjustment set")
        covariates = st.multiselect(
            "Confounders to adjust for", ALL_COVARIATES,
            default=ALL_COVARIATES, key="estimate_covs",
        )
        kwargs = {}
        if method == "Propensity Score Matching":
            use_caliper = st.checkbox("Use caliper (drop poor matches)", value=False)
            if use_caliper:
                kwargs["caliper"] = st.slider("Caliper width", 0.001, 0.2, 0.05, 0.005)
        elif method == "Inverse Probability Weighting (IPW)":
            kwargs["stabilized"] = st.checkbox("Stabilized weights", value=True)
            use_trim = st.checkbox("Trim extreme propensity scores", value=True)
            if use_trim:
                kwargs["trim"] = st.slider("Trim threshold", 0.001, 0.2, 0.05, 0.005)
        run = st.button("Estimate ▶", type="primary",
                        use_container_width=True, key="estimate_run")

    with col_r:
        if not run:
            st.info("👈 Configure a method on the left and hit **Estimate**.")
        elif not covariates:
            st.error("Pick at least one covariate.")
        else:
            with st.spinner(f"Running {method}…"):
                if method == "Regression adjustment":
                    result = regression(df, OUTCOME, TREATMENT, covariates)
                elif method == "Propensity Score Matching":
                    result = ps_matching(df, OUTCOME, TREATMENT, covariates,
                                         n_boot=50, **kwargs)
                elif method == "Inverse Probability Weighting (IPW)":
                    result = ipw(df, OUTCOME, TREATMENT, covariates, **kwargs)
                else:
                    result = aipw(df, OUTCOME, TREATMENT, covariates, n_boot=50)

            # Stash the result so the Sensitivity tab can default to it
            st.session_state["last_result"] = {
                "method": result.method,
                "ate": result.ate,
                "ci_low": result.ci_low,
                "ci_high": result.ci_high,
                "outcome_sd": float(df[OUTCOME].std()),
            }

            st.subheader("Estimated Average Treatment Effect")
            if has_truth:
                c1, c2, c3 = st.columns(3)
                c1.metric("Your estimate", f"{result.ate:,.2f}")
                c2.metric("Bootstrap 95% CI",
                          f"[{result.ci_low:,.2f}, {result.ci_high:,.2f}]")
                c3.metric("Distance from RCT truth",
                          f"{result.ate - rct.ate:+,.2f}",
                          delta=f"RCT: {rct.ate:,.2f}",
                          delta_color="off")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Your estimate", f"{result.ate:,.2f}")
                c2.metric("Bootstrap 95% CI",
                          f"[{result.ci_low:,.2f}, {result.ci_high:,.2f}]")
                c3.metric("CI width",
                          f"{result.ci_high - result.ci_low:,.2f}")

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
            ax.set_xlabel(f"ATE on `{OUTCOME}`")
            ax.legend(loc="upper right")
            ax.spines[["top", "right", "left"]].set_visible(False)
            st.pyplot(fig)

            d = result.diagnostics
            if "ps" in d:
                st.subheader("Propensity score overlap (post-fit)")
                fig, ax = plt.subplots(figsize=(8, 2.5))
                ax.hist(d["ps"][df[TREATMENT].values==0], bins=40, alpha=0.6,
                        color=COLORS["control"], label="Control")
                ax.hist(d["ps"][df[TREATMENT].values==1], bins=40, alpha=0.6,
                        color=COLORS["treated"], label="Treated")
                ax.set_xlabel(r"$\hat{e}(X)$"); ax.set_ylabel("Count"); ax.legend()
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)

            if "weights" in d:
                st.subheader("IPW weight distribution")
                fig, ax = plt.subplots(figsize=(8, 2.5))
                w = d["weights"]
                ax.hist(w, bins=40, color=COLORS["estimate"], alpha=0.8)
                ax.set_xlabel("Weight"); ax.set_ylabel("Count")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                a, b, c = st.columns(3)
                a.metric("Mean weight", f"{w.mean():.2f}")
                b.metric("Max weight",  f"{d['max_weight']:.1f}")
                c.metric("Units dropped (trim)", f"{d.get('n_trimmed', 0)}")

            if "balance" in d:
                st.subheader("Covariate balance (Love plot)")
                bal = d["balance"]
                fig, ax = plt.subplots(figsize=(8, 0.35*len(bal)+1.2))
                y = np.arange(len(bal))
                ax.scatter(bal["smd_raw"].abs(), y, color=COLORS["bad"],
                           s=70, label="Before adjustment", zorder=3)
                ax.scatter(bal["smd_adjusted"].abs(), y, color=COLORS["ok"],
                           s=70, label="After adjustment", zorder=3)
                for i, row in bal.reset_index(drop=True).iterrows():
                    ax.plot([abs(row["smd_raw"]), abs(row["smd_adjusted"])],
                            [i, i], color="lightgray", zorder=1)
                ax.axvline(0.1, color="black", linestyle=":", alpha=0.6,
                           label="|SMD| = 0.1")
                ax.set_yticks(y); ax.set_yticklabels(bal["covariate"])
                ax.set_xlabel("|Standardized Mean Difference|")
                ax.legend(loc="lower right")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                worst = bal["smd_adjusted"].abs().max()
                n_above = int((bal["smd_adjusted"].abs() > 0.1).sum())
                a, b, c = st.columns(3)
                a.metric("Max |SMD| after",       f"{worst:.3f}")
                b.metric("Mean |SMD| after",      f"{bal['smd_adjusted'].abs().mean():.3f}")
                c.metric("# covariates with |SMD|>0.1", f"{n_above} / {len(bal)}")


# =========================================================================
# TAB 4 — HETEROGENEITY (Item 7)
# =========================================================================
with tab_hete:
    st.subheader("Does the treatment effect vary across subgroups?")
    st.markdown(
        "Two ways to investigate. The **interaction term** is the canonical "
        "approach when you have a pre-specified modifier. The **causal forest** "
        "is a flexible exploratory tool that searches across all covariates "
        "at once."
    )

    approach = st.radio(
        "Approach",
        ["Regression with interaction term", "T-learner causal forest"],
        horizontal=True, key="hete_approach",
    )

    hete_covs = st.multiselect(
        "Covariates to adjust for", ALL_COVARIATES,
        default=ALL_COVARIATES, key="hete_covs",
    )

    if approach == "Regression with interaction term":
        modifier_options = [c for c in hete_covs]
        if not modifier_options:
            st.warning("Pick at least one covariate.")
            st.stop()
        modifier = st.selectbox(
            "Effect modifier (the variable you suspect changes the treatment effect)",
            modifier_options, key="hete_modifier",
        )

        if st.button("Estimate interaction ▶", type="primary", key="hete_run"):
            with st.spinner("Fitting…"):
                r = cate_interaction(df, OUTCOME, TREATMENT, modifier, hete_covs)
            c1, c2, c3 = st.columns(3)
            c1.metric("Interaction coefficient",
                      f"{r.interaction_coef:+,.3f}",
                      help=f"Change in treatment effect per unit increase in `{modifier}`.")
            c2.metric("Standard error", f"{r.interaction_se:,.3f}")
            c3.metric("p-value", f"{r.interaction_p:.4f}")

            st.markdown(f"##### Conditional treatment effect across `{modifier}`")
            cate_df = r.cate_by_x
            fig, ax = plt.subplots(figsize=(8, 3.2))
            ax.fill_between(cate_df[modifier], cate_df["ci_low"], cate_df["ci_high"],
                            alpha=0.2, color=COLORS["estimate"], label="95% CI")
            ax.plot(cate_df[modifier], cate_df["cate"],
                    color=COLORS["estimate"], linewidth=2.5, label="CATE")
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.set_xlabel(f"`{modifier}`")
            ax.set_ylabel(f"CATE on `{OUTCOME}`")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

            with st.expander("📋 Full regression output"):
                st.text(r.summary_text)

    else:  # T-learner causal forest
        st.caption("Fits separate random forests for treated and control on a "
                   "training half, then predicts CATEs on a held-out half "
                   "(honest splitting).")
        if st.button("Fit causal forest ▶", type="primary", key="hete_forest_run"):
            if (df[TREATMENT]==1).sum() < 30 or (df[TREATMENT]==0).sum() < 30:
                st.error("Causal forest needs at least 30 treated and 30 control units.")
                st.stop()
            with st.spinner("Fitting…"):
                try:
                    r = t_learner_cate(df, OUTCOME, TREATMENT, hete_covs)
                except Exception as e:
                    st.error(f"Fit failed: {e}")
                    st.stop()

            cate = r.cate_per_unit
            c1, c2, c3 = st.columns(3)
            c1.metric("Mean CATE", f"{cate.mean():,.2f}")
            c2.metric("Min CATE",  f"{cate.min():,.2f}")
            c3.metric("Max CATE",  f"{cate.max():,.2f}")

            st.markdown("##### Distribution of individual treatment-effect estimates")
            fig, ax = plt.subplots(figsize=(8, 2.8))
            ax.hist(cate, bins=40, color=COLORS["estimate"], alpha=0.8)
            ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
            ax.axvline(cate.mean(), color=COLORS["rct"], linestyle="--",
                       linewidth=2, label=f"Mean = {cate.mean():,.2f}")
            ax.set_xlabel(f"Predicted CATE on `{OUTCOME}`"); ax.set_ylabel("Count")
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)

            st.markdown("##### Calibration plot")
            st.caption("Units on the held-out half are sorted into deciles by "
                       "predicted CATE; the observed treated-control difference "
                       "in each decile should track the predictions.")
            cal = r.calibration.dropna(subset=["observed_effect"])
            if len(cal) < 2:
                st.warning("Too few overlap bins for a calibration plot. "
                           "Try a different covariate set or a larger sample.")
            else:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.errorbar(cal["mean_predicted"], cal["observed_effect"],
                            yerr=cal["se"].fillna(0).clip(lower=0),
                            fmt="o", color=COLORS["estimate"], markersize=8,
                            capsize=4)
                lo = float(min(cal["mean_predicted"].min(),
                               cal["observed_effect"].min()))
                hi = float(max(cal["mean_predicted"].max(),
                               cal["observed_effect"].max()))
                ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect calibration")
                ax.set_xlabel(f"Mean predicted CATE in decile")
                ax.set_ylabel(f"Observed treated − control mean in decile")
                ax.legend()
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)

                with st.expander("📋 Calibration table"):
                    st.dataframe(cal.style.format({
                        "mean_predicted": "{:,.2f}",
                        "observed_effect": "{:+,.2f}",
                        "se": "{:,.2f}",
                    }), use_container_width=True)


# =========================================================================
# TAB 5 — SENSITIVITY (Item 8)
# =========================================================================
with tab_sens:
    st.subheader("E-value sensitivity analysis (VanderWeele & Ding, 2017)")
    st.markdown(
        "The **E-value** is the minimum strength an unmeasured confounder "
        "would need (measured as a risk ratio with both treatment and outcome) "
        "to fully explain away your observed association. Larger E-value → "
        "more robust to hidden confounding."
    )

    last = st.session_state.get("last_result")
    if last:
        st.success(f"Loaded your most recent estimate from Tab 3: "
                   f"**{last['method']}**, ATE = {last['ate']:,.2f}, "
                   f"95% CI = [{last['ci_low']:,.2f}, {last['ci_high']:,.2f}].")
    else:
        st.info("Run a method on Tab 3 first, or enter your own numbers below.")

    c1, c2, c3, c4 = st.columns(4)
    point = c1.number_input("Point estimate",
                            value=float(last["ate"]) if last else 1000.0,
                            key="sens_point")
    ci_low = c2.number_input("CI lower bound",
                             value=float(last["ci_low"]) if last else 100.0,
                             key="sens_lo")
    ci_high = c3.number_input("CI upper bound",
                              value=float(last["ci_high"]) if last else 1900.0,
                              key="sens_hi")
    outcome_type = c4.selectbox("Outcome scale",
                                ["continuous", "binary"],
                                index=0, key="sens_type",
                                help="'continuous' = mean difference. "
                                     "'binary' = risk ratio.")

    sd_default = float(last["outcome_sd"]) if last else float(df[OUTCOME].std())
    if outcome_type == "continuous":
        outcome_sd = st.number_input(
            f"SD of outcome `{OUTCOME}` (for the Chinn approximation)",
            value=sd_default, min_value=1e-6,
            key="sens_sd",
            help="Standard deviation of the outcome in the full sample. The "
                 "Chinn (2000) approximation standardises the mean difference "
                 "by this SD, then converts to a risk ratio.")
    else:
        outcome_sd = None

    if st.button("Compute E-value ▶", type="primary", key="sens_run"):
        try:
            ev = e_value(point, ci_low, ci_high,
                         outcome_type=outcome_type,
                         outcome_sd=outcome_sd)
        except Exception as exc:
            st.error(f"Failed: {exc}")
            st.stop()

        st.subheader("Result")
        c1, c2 = st.columns(2)
        c1.metric("E-value for point estimate", f"{ev.e_value_point:.2f}")
        c2.metric("E-value for CI bound (closer to null)",
                  f"{ev.e_value_ci:.2f}")
        st.markdown(
            f"**Approx. risk ratio**: point = `{ev.rr_estimate:.3f}`, "
            f"relevant CI bound = `{ev.rr_ci_bound:.3f}`.  \n"
            f"{ev.note}"
        )
        st.markdown(
            f"**How to read it.** An unmeasured confounder $U$ would need to be "
            f"associated with **both** treatment and outcome by a risk ratio of "
            f"at least **{ev.e_value_point:.2f}** (each) to fully explain away "
            f"your point estimate. To shift the confidence interval to include "
            f"the null, a confounder of strength **{ev.e_value_ci:.2f}** would "
            f"suffice. Compare to: the strongest measured covariate in your "
            f"adjustment set probably has a much smaller association with "
            f"treatment and outcome than this. If the E-value is close to 1, "
            f"the finding is fragile."
        )

        # Tipping-point curve: what combinations of confounder-treatment and
        # confounder-outcome RRs would explain away the point estimate?
        st.subheader("Tipping-point curve")
        st.caption("Every point on this curve represents a combination of "
                   "(confounder→treatment) and (confounder→outcome) risk ratios "
                   "that would, together, fully explain away your point estimate. "
                   "Anything weaker than this curve cannot.")
        rr_obs = ev.rr_estimate if ev.rr_estimate > 1 else 1.0 / ev.rr_estimate
        if rr_obs > 1.0:
            rr_uy = np.linspace(rr_obs + 0.01, rr_obs * 5, 200)
            rr_uw = (rr_obs * (rr_uy - 1)) / (rr_uy - rr_obs)
            mask = (rr_uw > 0) & np.isfinite(rr_uw)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(rr_uy[mask], rr_uw[mask], color=COLORS["estimate"], linewidth=2)
            ax.fill_between(rr_uy[mask], 1, rr_uw[mask], alpha=0.15,
                            color=COLORS["bad"], label="Cannot explain away")
            ax.scatter([ev.e_value_point], [ev.e_value_point],
                       color=COLORS["bad"], s=80, zorder=5,
                       label=f"E-value = {ev.e_value_point:.2f}")
            ax.set_xlabel("RR(confounder → outcome)")
            ax.set_ylabel("RR(confounder → treatment)")
            ax.set_xlim(left=1)
            ax.set_ylim(bottom=1)
            ax.legend()
            ax.spines[["top", "right"]].set_visible(False)
            st.pyplot(fig)
        else:
            st.caption("Tipping-point curve only shown for non-null effects.")

    st.divider()
    st.markdown(
        "**DAG editing** is best done in dedicated tools such as "
        "[DAGitty](http://www.dagitty.net/), which also reports the minimum "
        "sufficient adjustment set automatically."
    )
