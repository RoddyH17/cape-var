# VAR-Enhanced Shiller CAPE Ratio

> Enhanced the Shiller-CAPE ratio for long-run U.S. equity return forecasting via a Vector Autoregression (VAR) incorporating earning yields and a capital index. Achieved **~40% RMSE reduction** vs. Shiller's baseline OLS for 10-year nominal annualized returns.

---

## Research Summary

**Problem**: Shiller's CAPE is a strong long-run predictor but has two key weaknesses:
1. **Interest rate blindness**: CAPE ignores the opportunity cost of equities vs. bonds
2. **Payout structure changes**: Post-1990 shift from dividends to buybacks distorts historical earnings comparisons
3. **Univariate specification**: Single-equation OLS misses co-movement dynamics between valuation, rates, and capital returns

**Solution**: Replace simple OLS with a VAR system:

```
Y_t = [EY_t, CapIdx_t, (Y10_t)]'
Y_t = c + A₁Y_{t-1} + A₂Y_{t-2} + ... + AₚY_{t-p} + εt

where:
  EY_t    = 1/CAPE_t  (earning yield — inverse of CAPE)
  CapIdx_t = dividend yield + buyback yield (total capital return)
  Y10_t   = 10-yr Treasury yield (optional: interest rate regime)
```

**10-year return forecast**: Derived from VAR-implied EY path with mean-reversion correction:
```
r̂_10yr ≈ EY_VAR + g_correction
```

**Result**: ~40% RMSE reduction for 10-year nominal annualized return forecasts vs. Shiller baseline OLS.

---

## Architecture

```
cape_var/
├── src/
│   ├── cape_var_model.py      # VAR model + baseline comparison + metrics
│   └── analysis.py            # Original CAPE analysis (Shiller reconstruction)
├── notebooks/
│   ├── CAPE Ratio Regression.ipynb   # Original CAPE regression analysis
│   ├── CAPE Visualization.ipynb      # CAPE historical plots
│   ├── accounting.ipynb              # GAAP accounting adjustment
│   ├── interest rate regression.ipynb # Interest rate regime analysis
│   ├── market index.ipynb            # Index composition changes
│   └── taxes.ipynb                   # Tax adjustment to CAPE
├── data/
│   ├── ie_data.xls             # Shiller's original dataset (1871-present)
│   ├── df.pkl                  # Preprocessed DataFrame
│   └── [accounting/tax supplements]
├── research/
│   ├── Kenourgios2021_Article_OnThePredictivePowerOfCAPEOrSh.pdf
│   ├── GIC.docx                # Original Monet research report
│   └── *.bib                   # Literature references
└── reports/
```

---

## Model Details

### Baseline: Shiller OLS (Campbell & Shiller 1988)

```
r̂_{t+10} = α + β·log(CAPE_t) + ε
```

- Significant at 1% level
- R² ≈ 0.30-0.40 for 10-year horizon (in-sample)
- Out-of-sample performance degrades post-2000 due to regime changes

### VAR-CAPE (This Model)

**Why VAR?**
- EY (earning yield) is cointegrated with capital returns over long horizons
- Interest rates and equity valuations exhibit VAR dynamics (Fed Model)
- VAR captures the cyclical adjustment missing from univariate CAPE

**Lag selection**: AIC-optimal (typically 2-6 months for monthly data)

**Key improvement mechanisms**:
1. EY mean-reversion captured by VAR autocovariance structure
2. Capital index stabilizes prediction when payout policy shifts
3. Cross-variable dynamics (rate-equity correlation) reduce forecast bias

---

## CAPE Adjustments (Notebooks)

The notebooks implement four CAPE adjustments from the GIC report:

| Adjustment | Notebook | Effect |
|-----------|----------|--------|
| GAAP accounting changes | `accounting.ipynb` | Corrects for post-2000 goodwill write-offs |
| Interest rate regime | `interest rate regression.ipynb` | Fair-Value CAPE (Asness/Siegel) |
| Market index composition | `market index.ipynb` | Tech-heavy index → higher justified P/E |
| Tax rate changes | `taxes.ipynb` | Post-TCJA earnings adjustment |

---

## Quick Start

```bash
pip install -r requirements.txt

# Run VAR model comparison
python -c "
import pandas as pd
from src.cape_var_model import load_shiller_data, compare_models

df = load_shiller_data('data/ie_data.xls')
results = compare_models(df, target='fwd_10yr_nominal', train_end='2000-01-01')
print(results)
"

# Or run notebooks
jupyter notebook notebooks/
```

---

## References

- Campbell, J., & Shiller, R. (1988). Stock prices, earnings, and expected dividends. *Journal of Finance*.
- Asness, C. (2012). An old friend: The stock market's Shiller P/E. *AQR*.
- Siegel, J. (2016). The Shiller CAPE ratio: A new look. *FAJ*.
- Kenourgios, D. et al. (2021). On the predictive power of CAPE or Shiller's PE. *International Review of Financial Analysis*.
- Shiller, R. (2000). *Irrational Exuberance*. Princeton University Press.
