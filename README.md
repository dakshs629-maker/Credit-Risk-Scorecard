# Credit Risk & Portfolio Analytics System

An end-to-end credit risk platform built on the Home Credit Default Risk dataset (307,511 loan applications). Combines applicant-level default prediction with portfolio-level loss simulation and risk-return analysis, wrapped in a single Streamlit dashboard with AI-generated risk narratives.

**Live app:** [credit-portfolio-risk-analytics.streamlit.app](https://credit-portfolio-risk-analytics-hhrbkmnkqvtgwfrh5u5rtp.streamlit.app)

---

## What this does

Two connected modules, one dashboard:

1. **Credit Risk Scorecard** — scores an individual applicant's probability of default
2. **Portfolio Risk Analysis** — takes those same PD scores across the full loan book and answers: *if we hold this portfolio, how much could we lose, and which segments are actually profitable?*

The second module reuses the first module's trained model, so the whole system is one pipeline rather than two disconnected notebooks.

---

## Module 1: Credit Risk Scorecard

**Problem:** given an applicant's financial and demographic profile, predict probability of default.

**Pipeline:**
- Cleaned 307K rows, dropped features with >40% missing, imputed remainder (mode for categorical, median for numeric)
- Feature selection via Information Value (IV) and Weight of Evidence (WoE), narrowed to 24 features
- Trained XGBoost (AUC 0.751) against a Logistic Regression baseline (AUC 0.739)
- Model diagnostics: KS statistic 0.37, Gini 0.50
- SHAP for feature-level explainability
- Claude API call converts the raw score + SHAP output into a plain-English risk narrative and recommendation

**In the dashboard:** enter applicant details (credit bureau scores, income, loan amount, employment history), get a risk tier (Low/Medium/High) with default probability and an AI-generated explanation of the top drivers.

---

## Module 2: Portfolio Risk Analysis

**Problem:** given PD scores for the whole loan book, what's the portfolio's expected and unexpected loss, and how should exposure be priced across segments?

**Pipeline:**
- Runs the Module 1 model across all 307,511 loans to get per-loan PD
- **EAD** (Exposure at Default): reconstructed from log-transformed credit amount
- **LGD** (Loss Given Default): fixed at 0.45, the Basel II standard assumption for unsecured retail credit
- **Expected Loss:** EL = PD × LGD × EAD, computed per loan and aggregated
- **Unexpected loss / tail risk:** single-factor Vasicek Monte Carlo simulation (asset correlation ρ = 0.15) generates a portfolio loss distribution, from which VaR 95%, VaR 99%, and Economic Capital (VaR − Expected Loss) are derived
- **Efficient frontier:** loans grouped by income segment, each segment's mean PD (risk) plotted against mean net yield (return: assumed gross yield by contract type minus PD × LGD), identifying which segments are economically viable to underwrite at current pricing
- Claude API generates a banking-analyst-style narrative summarizing segment profitability and pricing implications

**In the dashboard:**
- *Monte Carlo Simulation* tab — loss distribution, VaR/Economic Capital metrics, adjustable correlation/LGD/simulation count (live if `loan_risk_data.parquet` is present; falls back to static pre-computed results otherwise)
- *Efficient Frontier* tab — risk-return scatter by income segment, adjustable yield assumptions
- *AI Risk Summary* tab — narrative writeup, regenerable on demand from current frontier data

**Why Basel II framing (not III):** the core loss formula (PD × LGD × EAD) and single-factor asset correlation model are unchanged between Basel II and III. Basel III's main additions are institutional capital buffer requirements on top of this — outside the scope of a single-portfolio EL/VaR exercise, so Basel II is the accurate frame here.

---

## Tech stack

| Layer | Tools |
|---|---|
| Modeling | Python, pandas, NumPy, scikit-learn, XGBoost, SciPy |
| Explainability | SHAP |
| AI narratives | Claude API (Anthropic) |
| Simulation | Vasicek single-factor Monte Carlo (custom implementation) |
| Dashboard | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Repo structure

```
├── app.py                          # Unified Streamlit dashboard (both modules)
├── Credit Risk Scorecard.ipynb     # Module 1: cleaning → modeling → SHAP → scoring
├── Portfolio Risk Analysis.ipynb   # Module 2: PD scoring → EL → Monte Carlo → frontier
├── xgb_model.pkl                   # Trained XGBoost model (shared by both modules)
├── scaler.pkl                      # Feature scaler
├── loan_risk_data.parquet          # Per-loan PD/EAD, powers live portfolio simulation
├── portfolio_losses.npy            # Pre-computed loss distribution (static fallback)
├── frontier_data.csv               # Pre-computed segment risk-return data (static fallback)
├── risk_narrative.txt              # Pre-computed AI risk narrative (static fallback)
├── iv_scores.csv                   # IV/WoE feature selection output
└── requirements.txt
```

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Requires an `ANTHROPIC_API_KEY` environment variable for the AI narrative features. Everything else runs without it.

---

## Notes on scale

Loss figures (mean loss, VaR) are computed across the full 307,511-loan book with real credit amounts, so they read in the billions — this reflects the size of the dataset, not a units error. The average PD from the trained model (~40%) is elevated relative to typical retail lending, since Home Credit Default Risk oversamples for the classification exercise; treat absolute loss figures as illustrative of the methodology rather than as a realistic book.
