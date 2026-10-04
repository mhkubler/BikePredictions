import pandas as pd
import plotly.graph_objects as go


days = pd.read_csv("./daily.csv", header=None)
days.drop(labels=[0, 1], axis=1, inplace=True)

x1 = days.iloc[:, 8].to_list()
x2 = days.iloc[:, 10].to_list()

y = days.iloc[:, -1].to_list()

points = [(x1[i], x2[i], y[i]) for i in range(len(x1))]
train_end = int(len(points) * 0.7)
validation_end = int(len(points) * 0.85)
training_points = points[:train_end]
validation_points = points[train_end:validation_end]
test_points = points[validation_end:]

def matrix_multiply(A, B):
    result = []

    for i in range(len(A)):
        row = []

        for j in range(len(B[0])):
            value = 0

            for k in range(len(B)):
                value += A[i][k] * B[k][j]

            row.append(value)

        result.append(row)

    return result

def transpose(A):
    result = []
    for j in range(len(A[0])):
        row = []
        for i in range(len(A)):
            row.append(A[i][j])
        result.append(row)

    return result

def construct_polynomial_matrices(points):
    X = []
    y = []

    for point in points:
        x1, x2 = point[:-1]
        X.append([1, x1, x2, x1**2, x1*x2, x2**2])
        y.append([point[-1]])
    return X, y

def gaussian_elimination(A, b):

    matrix = []

    for i in range(len(A)):
        matrix.append(A[i] + [b[i][0]])

    n = len(matrix)
    # Scale the pivot tolerance to the original coefficient matrix.
    tolerance = 1e-12 * max(abs(value) for row in A for value in row)

    for i in range(n):

        pivot_row = max(
            range(i, n),
            key=lambda k: abs(matrix[k][i])
        )

        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]
        pivot = matrix[i][i]
        if abs(pivot) <= tolerance:
            raise ValueError("Singular or numerically unstable system: zero or near-zero pivot.")

        for j in range(i, n + 1):
            matrix[i][j] /= pivot
        for k in range(i + 1, n):
            factor = matrix[k][i]
            for j in range(i, n + 1):
                matrix[k][j] -= factor * matrix[i][j]

    x = [0] * n

    for i in range(n - 1, -1, -1):
        x[i] = matrix[i][n]
        for j in range(i + 1, n):
            x[i] -= matrix[i][j] * x[j]

    return x

def multiple_regression(X, y):
    XT = transpose(X)
    XTX = matrix_multiply(XT, X)
    XTy = matrix_multiply(XT, y)

    beta = gaussian_elimination(XTX, XTy)

    return beta

X, y = construct_polynomial_matrices(training_points)
beta = multiple_regression(X, y)

print("Betas:", beta)
equation = (
    f"y = {beta[0]} "
    f"+ ({beta[1]})x1 "
    f"+ ({beta[2]})x2 "
    f"+ ({beta[3]})x1² "
    f"+ ({beta[4]})x1x2 "
    f"+ ({beta[5]})x2²"
)

print("equation:", equation)

def r_squared(y, predictions):

    y_values = [row[0] for row in y]
    prediction_values = [row[0] for row in predictions]

    y_mean = sum(y_values) / len(y_values)

    ss_res = 0
    ss_tot = 0

    for i in range(len(y_values)):
        ss_res += (y_values[i] - prediction_values[i]) ** 2
        ss_tot += (y_values[i] - y_mean) ** 2

    return 1 - (ss_res / ss_tot)

def evaluate(y, predictions):
    errors = [y[i][0] - predictions[i][0] for i in range(len(y))]
    mae = sum(abs(error) for error in errors) / len(errors)
    rmse = (sum(error**2 for error in errors) / len(errors)) ** 0.5
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R^2: {r_squared(y, predictions):.4f}")

def predict(beta, temperature, humidity):

    return (
        beta[0]
        + beta[1] * temperature
        + beta[2] * humidity
        + beta[3] * temperature**2
        + beta[4] * temperature * humidity
        + beta[5] * humidity**2
    )
training_mean = sum(point[-1] for point in training_points) / len(training_points)
print("Training/validation/test observations:", len(training_points), len(validation_points), len(test_points))
print("Training-set mean:", training_mean)

for name, evaluation_points in [("Validation", validation_points), ("Test", test_points)]:
    y = [[point[-1]] for point in evaluation_points]
    predictions = [[predict(beta, point[0], point[1])] for point in evaluation_points]
    print(name, "polynomial regression")
    evaluate(y, predictions)
    print(name, "training-mean baseline")
    evaluate(y, [[training_mean] for point in evaluation_points])

def f(x1, x2):
    return predict(beta, x1, x2)

def generateGrid(f, temperature_min, temperature_max, humidity_min, humidity_max):
    temperature_step = (temperature_max - temperature_min)/(150-1)
    humidity_step = (humidity_max - humidity_min)/(150-1)
    X = []
    Y = []
    for i in range(150):
        X.append(temperature_min + i*temperature_step)
        Y.append(humidity_min + i*humidity_step)
    Z = []
    for y in Y:
        row = []
        for x in X:
            row.append(f(x, y))
        Z.append(row)
    return X, Y, Z

X, Y, Z = generateGrid(f, min(x1), max(x1), min(x2), max(x2))
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.update_layout(scene={
    "xaxis_title": "Temperature (normalized)",
    "yaxis_title": "Humidity (normalized)",
    "zaxis_title": "Predicted daily rentals"
})
fig.show()
