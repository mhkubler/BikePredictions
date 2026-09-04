import pandas as pd

days = pd.read_csv("./daily.csv", header=None)
days.drop(labels=[0, 1], axis=1, inplace=True)
print(days)

x1 = days.iloc[:, 10].to_list()
x2 = days.iloc[:, 12].to_list()

y = days.iloc[:, -1].to_list()

points = [(x1[i], x2[i], y[i]) for i in range(len(x1))]

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

def construct_matrices(points):
    X = []
    y = []

    for point in points:
        X.append([1] + list(point[:-1]))
        y.append([point[-1]])

    return X, y


def gaussian_elimination(A, b):

    matrix = []

    for i in range(len(A)):
        matrix.append(A[i] + [b[i][0]])

    n = len(matrix)

    for i in range(n):

        pivot_row = max(
            range(i, n),
            key=lambda k: abs(matrix[k][i])
        )

        matrix[i], matrix[pivot_row] = matrix[pivot_row], matrix[i]

        pivot = matrix[i][i]

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

X, y = construct_matrices(points)

def multiple_regression(X, y):
    XT = transpose(X)
    XTX = matrix_multiply(XT, X)
    XTy = matrix_multiply(XT, y)

    beta = gaussian_elimination(XTX, XTy)

    return beta

beta = multiple_regression(X, y)

print("Betas:", beta)

predictions = matrix_multiply(
    X,
    [[b] for b in beta]
)

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

print("R^2:", r_squared(y, predictions))

def predict(beta, temperature, humidity):

    return (
        beta[0]
        + beta[1] * (temperature / 41)
        + beta[2] * (humidity / 100)
    )

print("Prediction:", predict(beta, 20, 27))