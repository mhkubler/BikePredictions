# Bike Sharing Regression

A from-scratch implementation of **multiple linear regression** using the Bike Sharing dataset.

The model predicts daily bike rentals (`cnt`) from:

* Temperature (`temp`)
* Humidity (`hum`)

### Method

The regression coefficients are calculated using the normal equation:

$$
\beta = (X^TX)^{-1}X^Ty
$$

Matrix operations and **Gaussian elimination** are implemented manually without using a machine-learning library.

The resulting regression plane is visualized interactively using Plotly.

### Future Development

The project will be extended to support **polynomial regression**, allowing the model to capture nonlinear relationships between temperature, humidity, and bike rentals while retaining the same underlying regression framework.

### Technologies

* Python
* Pandas
* Plotly

### Dataset

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)
