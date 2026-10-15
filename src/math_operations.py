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


def predict_multiple(beta, temperature, humidity):

    return (
        beta[0]
        + beta[1] * temperature
        + beta[2] * humidity
    )


def predict_polynomial(beta, temperature, humidity):

    return (
        beta[0]
        + beta[1] * temperature
        + beta[2] * humidity
        + beta[3] * temperature**2
        + beta[4] * temperature * humidity
        + beta[5] * humidity**2
    )


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
