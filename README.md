# Bike Sharing Regression

The two scripts predict daily bike rentals (`cnt`) from temperature and humidity in the [UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

Run from the project root with Python, Pandas and Plotly installed:

```sh
python src/MultipleRegression.py
python src/PolynomialRegression.py
```

Both scripts print coefficients and evaluation metrics. The polynomial script also opens an interactive Plotly surface.

## Inputs and methods

The local `daily.csv` has no header. After dropping columns `[0, 1]`, temperature is `days.iloc[:, 8]`, humidity is `days.iloc[:, 10]`, and total rentals are the last column. The previous feature positions selected humidity and casual rentals.

The CSV weather values are already normalized. Training and `predict(beta, temperature, humidity)` both use normalized temperature followed by normalized humidity, without further scaling. `predict()` does not accept raw Celsius or humidity percentages.

Multiple regression uses `[1, temperature, humidity]`. Polynomial regression uses `[1, temperature, humidity, temperature**2, temperature*humidity, humidity**2]`.

Both solve the normal equations:

$$
(X^TX)\beta = X^Ty
$$

Matrix multiplication, transposition and gaussian elimination with partial pivoting are implemented from scratch. Before division, pivots at or below `1e-12` times the largest absolute entry in the original coefficient matrix raise a singular or numerically unstable system error. Normal equations can amplify numerical conditioning problems; the pivot check is a safeguard, not a full condition estimate.

The polynomial surface uses separate observed minimum-to-maximum ranges for temperature and humidity. Its axes identify the normalized inputs and predicted daily rentals.

## Evaluation

The 731 CSV observations are already in chronological order and are split without shuffling:

| Portion | Observations | Dates |
| --- | ---: | --- |
| Training | 511 | 2011-01-01 to 2012-05-25 |
| Validation | 110 | 2012-05-26 to 2012-09-12 |
| Test | 110 | 2012-09-13 to 2012-12-31 |

The split boundaries are `int(0.70 * n)` and `int(0.85 * n)`. Only training observations determine the coefficients. Validation evaluates both fixed models; testing provides the final comparison without refitting. The baseline always predicts the training-set mean of 3794.21 rentals on the same validation and test observations.

MAE is mean absolute error, RMSE is root mean squared error, and R² is `1 - sum((actual - predicted)**2) / sum((actual - evaluation_mean)**2)`. MAE and RMSE are measured in rentals; lower is better. Higher R² is better.

Measured results from the included CSV:

| Portion | Model | MAE | RMSE | R² |
| --- | --- | ---: | ---: | ---: |
| Validation | Multiple regression | 1613.48 | 1810.24 | -3.4951 |
| Validation | Polynomial regression | 1945.54 | 2047.60 | -4.7512 |
| Validation | Training-mean baseline | 2961.04 | 3081.68 | -12.0270 |
| Test | Multiple regression | 2346.18 | 2550.84 | -0.6261 |
| Test | Polynomial regression | 1990.00 | 2179.40 | -0.1870 |
| Test | Training-mean baseline | 2352.94 | 2663.19 | -0.7725 |

Multiple regression performs better on validation; polynomial regression performs better in the final test comparison. Both beat the training-mean baseline on both portions. Negative R² means their squared errors exceed those of predicting each evaluation portion's own mean, which differs from the training-mean baseline.
