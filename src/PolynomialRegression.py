import pandas as pd
import plotly.graph_objects as go


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

def multiple_regression(X, y):
    XT = transpose(X)
    XTX = matrix_multiply(XT, X)
    XTy = matrix_multiply(XT, y)

    beta = gaussian_elimination(XTX, XTy)

    return beta

X, y = construct_polynomial_matrices(points)
beta = multiple_regression(X, y)

print(beta)
equation = (
    f"y = {beta[0]} "
    f"+ ({beta[1]})x1 "
    f"+ ({beta[2]})x2 "
    f"+ ({beta[3]})x1² "
    f"+ ({beta[4]})x1x2 "
    f"+ ({beta[5]})x2²"
)

def f(x1, x2):
    return beta[0] + beta[1] * x1 + beta[2] * x2 + beta[3] * x1**2 + beta[4] * x1*x2 + beta[5] * x2**2

print("equation:", equation)

def predict(beta, temperature, humidity):

    temperature /= 41
    humidity /= 100

    return (
        beta[0]
        + beta[1] * temperature
        + beta[2] * humidity
        + beta[3] * temperature**2
        + beta[4] * temperature * humidity
        + beta[5] * humidity**2
    )
print(predict(beta, 20, 27))

def generateGrid(f, m):
    n = 2*m/(150-1)
    X = []
    Y = []
    for i in range(150):
        X.append(-m + i*n)
        Y.append(-m + i*n)
    Z = []
    for y in Y:
        row = []
        for x in X:
            row.append(f(x, y))
        Z.append(row)
    return X, Y, Z

X, Y, Z = generateGrid(f, 9)
fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
fig.show()