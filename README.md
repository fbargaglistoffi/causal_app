# Causal Inference Playground

A Streamlit app for bachelor students to work through an end-to-end causal
inference analysis on any tabular dataset. Built around the credibility
checklist used in the course.

## Capabilities (mapped to the checklist)

| Checklist item | App location |
|---|---|
| Item 3 — Is the comparison fair? (balance diagnostics) | **Tab 2 · Balance**: per-covariate SMD table, Love plot, propensity-score overlap histogram |
| Item 6 — Estimation strategy | **Tab 3 · Estimate**: four methods (Regression adjustment, PS Matching, IPW, AIPW / Doubly Robust). Point estimate + 95% bootstrap CI + method-specific diagnostics |
| Item 7 — Effect modification / CATE | **Tab 4 · Heterogeneity**: regression with $W \times X$ interaction term (CATE curve + interaction coefficient with CI and p-value), plus T-learner causal forest with calibration plot |
| Item 8 — Sensitivity to unmeasured confounding | **Tab 5 · Sensitivity**: VanderWeele & Ding (2017) E-value for the point estimate and the CI bound, tipping-point curve. DAG editing is delegated to [DAGitty](http://www.dagitty.net/) |

Two data sources:

1. **LaLonde / NSW (built-in)** — students compare observational estimates
   against the RCT benchmark ($1,794).
2. **Upload your own CSV** — students bring any dataset; sidebar maps the
   treatment, outcome, and covariate columns.

## Files

- `app.py` — Streamlit app (5 tabs)
- `estimators.py` — every method, plus heterogeneity (`cate_interaction`, `t_learner_cate`) and sensitivity (`e_value`)
- `data/nsw_experimental.csv` and `data/nsw_observational.csv` — Dehejia–Wahba
- `requirements.txt`
- `.gitignore`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)

Push the folder to a public GitHub repo, then at <https://share.streamlit.io>
click **Create app → Deploy a public app from GitHub** and point at the repo
with `app.py` as the main file. Free tier limits (1 GB RAM, sleep after 7
days of inactivity) are comfortable for a class.

## Upload requirements

Students' CSVs need:

- One row per unit
- One **binary 0/1 column** for treatment
- One **numeric column** for the outcome
- One or more numeric covariate columns (categoricals must be 0/1 dummies)
- Files larger than 5,000 rows are auto-subsampled for interactive performance

## Suggested final-project flow

1. Each group is assigned (or picks) a dataset and a research question
2. **Tab 1** — describe the dataset and the causal estimand
3. **Tab 2** — report raw imbalance and overlap; comment on the positivity assumption
4. **Tab 3** — pick a primary estimator, justify the choice, report ATE + CI + balance after adjustment
5. **Tab 4** — pick at least one effect modifier; run the interaction; optionally use the causal forest as a discovery tool
6. **Tab 5** — report the E-value for the point estimate and for the CI bound; interpret in plain language

## Customisation

- **Add a method**: write a function in `estimators.py` that returns an
  `EstimateResult` and add it to the dropdown in `app.py` Tab 3.
- **Add a built-in dataset**: drop a CSV into `data/` and add a branch in the
  sidebar (mirror the LaLonde branch).
- **Bootstrap counts** in each method are tuned for ~2,500-row datasets;
  reduce `n_boot=` for larger ones to keep interactive speed.

## References

- LaLonde, R. J. (1986). *Evaluating the Econometric Evaluations of Training Programs with Experimental Data.* AER, 76(4):604–620.
- Dehejia, R. H., & Wahba, S. (1999). *Causal Effects in Non-experimental Studies.* JASA, 94(448):1053–1062.
- Rosenbaum, P. R., & Rubin, D. B. (1983). *The Central Role of the Propensity Score in Observational Studies for Causal Effects.* Biometrika, 70(1):41–55.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). *Estimation of Regression Coefficients When Some Regressors Are Not Always Observed.* JASA, 89(427):846–866.
- VanderWeele, T. J., & Ding, P. (2017). *Sensitivity Analysis in Observational Research: Introducing the E-Value.* Annals of Internal Medicine, 167(4):268–274.
- Chinn, S. (2000). *A simple method for converting an odds ratio to effect size for use in meta-analysis.* Stat Med, 19(22):3127–3131.
- Künzel, S. R., Sekhon, J. S., Bickel, P. J., & Yu, B. (2019). *Metalearners for estimating heterogeneous treatment effects using machine learning.* PNAS, 116(10):4156–4165.
