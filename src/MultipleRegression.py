from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import math_operations as mo

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

X_train, y_train = mo.construct_matrices(training_points)
beta = mo.multiple_regression(X_train, y_train)

print("Betas:", beta)

training_mean = sum(point[-1] for point in training_points) / len(training_points)
print("Training/validation/test observations:", len(training_points), len(validation_points), len(test_points))
print("Training-set mean:", training_mean)

for name, evaluation_points in [("Validation", validation_points), ("Test", test_points)]:
    y = [[point[-1]] for point in evaluation_points]
    predictions = [[mo.predict_multiple(beta, point[0], point[1])] for point in evaluation_points]
    print(name, "multiple regression")
    mo.evaluate(y, predictions)
    print(name, "training-mean baseline")
    mo.evaluate(y, [[training_mean] for point in evaluation_points])

X_np = np.array(X_train, dtype=float)
y_np = np.array(y_train, dtype=float).ravel()

numpy_beta = np.linalg.lstsq(X_np, y_np, rcond=None)[0]

print("My coefficients:", beta)
print("NumPy coefficients:", numpy_beta)

np.testing.assert_allclose(
    beta, numpy_beta,
    rtol=1e-6, atol=1e-8
)
print("Coefficient comparison passed")

for name, evaluation_points in [("Validation", validation_points), ("Test", test_points)]:
    x_eval, y_eval = mo.construct_matrices(evaluation_points)

    numpy_predictions = (np.array(x_eval, dtype=float) @ numpy_beta).reshape(-1, 1).tolist()
    my_prediction = [[mo.predict_multiple(beta, point[0], point[1])] for point in evaluation_points]

    numpy_values = [point[0] for point in numpy_predictions]
    my_values = [point[0] for point in my_prediction]

    print(f"\n{name} — my implementation")
    mo.evaluate(y_eval, my_prediction)

    print(f"{name} — NumPy")
    mo.evaluate(y_eval, numpy_predictions)

    differences = np.abs(
        np.array(my_prediction) - np.array(numpy_predictions)
    )
    print(f"Largest prediction difference: {differences.max():.10f}")

    np.testing.assert_allclose(
        my_prediction, numpy_predictions,
        rtol=1e-6, atol=1e-8
    )

    plt.scatter(numpy_values, my_values, label=name)
plt.axline((0,0), slope=1, color="blue", linewidth=0.5)
plt.xlabel("numpy predictions")
plt.ylabel("my predictions")
plt.grid(True)
plt.legend()
plt.xlim(1500, 7000)
plt.ylim(1500, 7000)

Path("images").mkdir(exist_ok=True)
plt.savefig("images/multiple_regression.png", dpi=200, bbox_inches="tight")

plt.show()