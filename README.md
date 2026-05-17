# Causal Inference Playground — Module 5

A Streamlit app for bachelor students to play with the four estimation
methods from Module 5 (stratification, regression, propensity-score
matching, IPW) plus a Doubly Robust (AIPW) estimator.

Two data sources are supported:

1. **LaLonde / NSW (built-in)** — students compare their observational
   estimates against the **RCT benchmark** ($1,794) and watch confounding
   play out in real time.
2. **Upload your own CSV** — students bring any dataset they like, pick
   the treatment / outcome / covariate columns, and run all six estimators.

## Files

- `app.py` — Streamlit app with 4 tabs (About → Explore → Estimate → Compare)
- `estimators.py` — Implementations of all 6 estimators with bootstrap CIs
- `data/nsw_experimental.csv` — Dehejia–Wahba experimental subset (n = 445)
- `data/nsw_observational.csv` — NSW treated + 2,500 CPS controls (n = 2,685)
- `requirements.txt` — Python dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app opens at <http://localhost:8501>.

## Deploy free on Streamlit Community Cloud

This is the path that gives your students a public URL with no setup.

1. **Create a GitHub repo** with this folder's contents
   (`app.py`, `estimators.py`, `requirements.txt`, `data/*.csv`)
2. **Sign in** at <https://share.streamlit.io> with your GitHub account
3. Click **New app** → pick the repo → set the main file to `app.py`
4. Click **Deploy** — you get a public URL like
   `https://your-app-name.streamlit.app`

Free tier limits (more than enough for a class):

- Up to 1 GB RAM per app (this app uses ~150 MB)
- Apps sleep after 7 days of inactivity (one click wakes them)
- Public apps are visible to anyone with the URL

## Suggested in-class flow (60–75 min)

| Time   | Activity                                                                 |
|--------|--------------------------------------------------------------------------|
| 5 min  | Students open the URL on their laptops; you walk through the **About** tab |
| 10 min | Each group explores **Tab 2** — what does the imbalance look like?       |
| 25 min | In **Tab 3**, each group is assigned ONE method to investigate in depth  |
| 15 min | Spokesperson reports method, estimate, diagnostics                       |
| 15 min | Class runs **Tab 4** together, discusses the forest plot                 |

## Suggested group questions

1. **Naive group**: How wrong is the naive estimate? Why?
2. **Stratification group**: Which stratification variable works best? Why are some strata empty?
3. **Regression group**: Does adding `re74` and `re75` change the estimate? Why are these the most powerful confounders?
4. **PS Matching group**: Look at the Love plot. Did matching achieve balance?
5. **IPW group**: Inspect the weight distribution. What happens with and without trimming?
6. **AIPW group**: Did "doubly robust" actually help? Why might it have failed?

## Pedagogical highlights

- **The naive estimate has the wrong sign** (−$8,500 vs the truth of +$1,800).
  This is the famous LaLonde finding and gives the entire course its punch.
- **The propensity-score overlap plot** (in Tab 3) shows visually why the
  observational analysis is hard — most CPS controls have $\hat{e}(X) \approx 0$.
- **The Love plot** turns covariate balance from an abstraction into something
  students can look at and judge.
- **Doubly Robust can fail** when overlap is poor (LaLonde is a famous case).
  This is a great lesson about when DR's "extra protection" actually helps.

## Uploading your own dataset

In the sidebar, switch the **Data source** radio to "Upload your own CSV".
Requirements for the CSV:

- One row per unit (person, household, firm, …)
- One **binary 0/1 column** for the treatment indicator
- One **numeric column** for the outcome
- One or more numeric columns for candidate confounders
- All categorical variables must be encoded as 0/1 dummies before upload
- Files larger than 5,000 rows are auto-subsampled for interactive performance

Once uploaded, the sidebar shows column pickers (treatment, outcome,
candidate covariates), and every tab adapts to your dataset. Note that
the RCT benchmark line in the forest plot is hidden — without an
experimental subset, there is no "true" ATE to compare against.

## Customisation

- **Change the bootstrap count** in `app.py` for faster/slower CIs (`n_boot=` arg)
- **Pre-load a built-in dataset** by dropping a CSV into `data/`, then
  adding a branch in the sidebar (similar to the LaLonde branch in `app.py`)
- **Add a new method** by writing a function in `estimators.py` that returns
  an `EstimateResult` and adding it to the dropdown in `app.py`

## Source of the data

- LaLonde, R. J. (1986). "Evaluating the Econometric Evaluations of Training
  Programs with Experimental Data." *American Economic Review*, 76(4):604–620.
- Dehejia, R. H., & Wahba, S. (1999). "Causal Effects in Non-experimental
  Studies: Reevaluating the Evaluation of Training Programs." *JASA*,
  94(448):1053–1062.

The CSVs in `data/` are derived from the Mostly Harmless Econometrics
versions packaged in the `causaldata` Python library (Dehejia–Wahba subset).
