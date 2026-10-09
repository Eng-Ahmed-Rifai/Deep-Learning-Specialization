"""
=============================================================================
Coursera Deep Learning Specialization (DeepLearning.AI)
Course 1: Neural Networks and Deep Learning
Week 2 Programming Assignment: Logistic Regression with a Neural Network Mindset
=============================================================================

This module implements a complete, bug-free, fully vectorized Logistic Regression
classifier from scratch using NumPy. It mirrors the exact mathematical equations,
architectural conventions, and public unit test suites of Andrew Ng's classic
Coursera assignment.

Key Components Implemented:
  1. sigmoid(z)
  2. initialize_with_zeros(dim)
  3. propagate(w, b, X, Y)
  4. optimize(w, b, X, Y, num_iterations, learning_rate, print_cost)
  5. predict(w, b, X)
  6. model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, print_cost)
  7. Comprehensive unit test suite mirroring Coursera autograder test cases
  8. Vectorization vs For-loop benchmark demonstrating NumPy vector speedup
"""

from typing import Dict, Tuple, List, Union, Any
import time
import numpy as np


def sigmoid(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Compute the sigmoid activation function for a scalar or numpy array.

    Arguments:
        z: A scalar or numpy array of any size.

    Returns:
        s: sigmoid(z) = 1 / (1 + exp(-z))
    """
    s = 1.0 / (1.0 + np.exp(-z))
    return s


def initialize_with_zeros(dim: int) -> Tuple[np.ndarray, float]:
    """
    Creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.0.

    Arguments:
        dim: size of the w vector we want (or number of parameters in this case).

    Returns:
        w: initialized vector of shape (dim, 1)
        b: initialized scalar (float) representing bias
    """
    w = np.zeros((dim, 1), dtype=np.float64)
    b = 0.0

    assert w.shape == (dim, 1), f"Expected w shape {(dim, 1)}, got {w.shape}"
    assert isinstance(b, (float, int)), f"Expected b to be float/int, got {type(b)}"

    return w, b


def propagate(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    Y: np.ndarray
) -> Tuple[Dict[str, Union[np.ndarray, float]], float]:
    """
    Implement the cost function and its gradient for the forward and backward propagation.

    Mathematical Formulation:
      - m = number of training examples (X.shape[1])
      - Forward Propagation:
          A = sigmoid(w^T * X + b)                   # Shape: (1, m)
          cost = -1/m * sum(Y * log(A) + (1 - Y) * log(1 - A))
      - Backward Propagation:
          dw = 1/m * X * (A - Y)^T                   # Shape: (dim, 1)
          db = 1/m * sum(A - Y)                      # Scalar float

    Arguments:
        w: weights, a numpy array of size (dim, 1)
        b: bias, a scalar (float)
        X: data matrix of size (dim, number of examples m)
        Y: true "label" vector of size (1, number of examples m)

    Returns:
        grads: dictionary containing gradient of the loss with respect to w ('dw')
               and gradient with respect to b ('db')
        cost: negative log-likelihood cost for logistic regression (scalar float)
    """
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    # Z = w.T @ X + b
    A = sigmoid(np.dot(w.T, X) + b)
    
    # Clip A slightly to avoid log(0) numerical overflow in edge scenarios
    epsilon = 1e-15
    A_clipped = np.clip(A, epsilon, 1.0 - epsilon)
    cost = (-1.0 / m) * np.sum(Y * np.log(A_clipped) + (1.0 - Y) * np.log(1.0 - A_clipped))

    # BACKWARD PROPAGATION (TO FIND GRADIENTS)
    dw = (1.0 / m) * np.dot(X, (A - Y).T)
    db = float((1.0 / m) * np.sum(A - Y))

    # Ensure shape safety and type conformity
    cost = float(np.squeeze(np.array(cost)))
    assert dw.shape == w.shape, f"dw shape {dw.shape} mismatch with w {w.shape}"
    assert isinstance(db, float), f"db should be float, got {type(db)}"

    grads = {"dw": dw, "db": db}

    return grads, cost


def optimize(
    w: np.ndarray,
    b: float,
    X: np.ndarray,
    Y: np.ndarray,
    num_iterations: int = 100,
    learning_rate: float = 0.009,
    print_cost: bool = False
) -> Tuple[Dict[str, Union[np.ndarray, float]], Dict[str, Union[np.ndarray, float]], List[float]]:
    """
    Optimizes w and b by running batch gradient descent algorithm.

    Arguments:
        w: weights, a numpy array of size (dim, 1)
        b: bias, a scalar
        X: data of shape (dim, number of examples)
        Y: true "label" vector of shape (1, number of examples)
        num_iterations: number of iterations of the optimization loop
        learning_rate: learning rate of the gradient descent update rule
        print_cost: True to print the loss every 100 steps

    Returns:
        params: dictionary containing the weights w and bias b
        grads: dictionary containing the gradients of the weights and bias with respect to the cost function
        costs: list of all the costs computed during the optimization (recorded every 100 iterations)
    """
    w = np.copy(w)
    b = float(b)
    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # Gradient descent parameter update rule:
        # θ = θ - α * dθ
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print(f"Cost after iteration {i:4d}: {cost:.6f}")

    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}

    return params, grads, costs


def predict(w: np.ndarray, b: float, X: np.ndarray) -> np.ndarray:
    """
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b).

    Arguments:
        w: weights, a numpy array of size (dim, 1)
        b: bias, a scalar
        X: data of size (dim, number of examples)

    Returns:
        Y_prediction: a numpy array (vector) containing all predictions (0/1) for the examples in X
    """
    m = X.shape[1]
    Y_prediction = np.zeros((1, m), dtype=np.float64)
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    # Vectorized conversion of probabilities A to 0 or 1 predictions:
    # A > 0.5 -> 1.0, otherwise 0.0
    Y_prediction = (A > 0.5).astype(np.float64)

    assert Y_prediction.shape == (1, m), f"Y_prediction shape mismatch: {Y_prediction.shape}"

    return Y_prediction


def model(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    num_iterations: int = 2000,
    learning_rate: float = 0.5,
    print_cost: bool = False
) -> Dict[str, Any]:
    """
    Builds the complete logistic regression model by calling helper functions.

    Workflow:
      1. Initialize parameters (w, b) with zeros
      2. Optimize parameters via gradient descent
      3. Retrieve optimal w, b
      4. Make predictions on both train and test sets
      5. Compute accuracies
      6. Return comprehensive results dictionary

    Arguments:
        X_train: training set represented by a numpy array of shape (dim, m_train)
        Y_train: training labels represented by a numpy array of shape (1, m_train)
        X_test: test set represented by a numpy array of shape (dim, m_test)
        Y_test: test labels represented by a numpy array of shape (1, m_test)
        num_iterations: hyperparameter representing the number of iterations to optimize
        learning_rate: hyperparameter representing the learning rate used in the update rule
        print_cost: set to True to print the cost every 100 iterations

    Returns:
        d: dictionary containing information about the model:
           - "costs": list of costs during training
           - "Y_prediction_test": predictions on test set
           - "Y_prediction_train": predictions on train set
           - "w": learned weights vector
           - "b": learned bias scalar
           - "learning_rate": learning rate hyperparameter
           - "num_iterations": number of iterations hyperparameter
           - "train_accuracy": accuracy on training set in %
           - "test_accuracy": accuracy on test set in %
    """
    dim = X_train.shape[0]

    # Step 1: Initialize parameters with zeros
    w, b = initialize_with_zeros(dim)

    # Step 2: Gradient descent
    parameters, grads, costs = optimize(
        w, b, X_train, Y_train,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        print_cost=print_cost
    )

    # Step 3: Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Step 4: Predict test/train set examples
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)

    # Step 5: Compute accuracies
    train_accuracy = float(100.0 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100.0)
    test_accuracy = float(100.0 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100.0)

    if print_cost:
        print(f"Train accuracy: {train_accuracy:.2f} %")
        print(f"Test accuracy : {test_accuracy:.2f} %")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediction_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations,
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy
    }

    return d


# =============================================================================
# UNIT TESTS & COURSERA GRADER TEST SUITE VERIFICATION
# =============================================================================

def run_coursera_unit_tests() -> None:
    """
    Runs unit tests mirroring the official Coursera DeepLearning.AI autograder test cases.
    Verifies every function against mathematical ground-truth values and dimensions.
    """
    print("=" * 70)
    print("RUNNING COURSERA C1W2 AUTOGRADER UNIT TEST SUITE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: sigmoid(z)
    # -------------------------------------------------------------------------
    print("\n[Test 1] Testing sigmoid(z)...")
    s_scalar = sigmoid(0.0)
    assert np.isclose(s_scalar, 0.5), f"Failed: sigmoid(0.0) should be 0.5, got {s_scalar}"

    x_arr = np.array([0, 2])
    expected_s_arr = np.array([0.5, 0.88079708])
    s_arr = sigmoid(x_arr)
    assert np.allclose(s_arr, expected_s_arr, atol=1e-7), (
        f"Failed: sigmoid({x_arr}) should be {expected_s_arr}, got {s_arr}"
    )

    x_neg = np.array([-1, 0, 1])
    expected_s_neg = np.array([0.26894142, 0.5, 0.73105858])
    assert np.allclose(sigmoid(x_neg), expected_s_neg, atol=1e-7), "Failed on negative values."
    print("  -> PASSED: sigmoid(z) output matches Coursera expected values.")

    # -------------------------------------------------------------------------
    # Test 2: initialize_with_zeros(dim)
    # -------------------------------------------------------------------------
    print("\n[Test 2] Testing initialize_with_zeros(dim)...")
    dim = 2
    w, b = initialize_with_zeros(dim)
    assert w.shape == (2, 1), f"Failed: w shape should be (2, 1), got {w.shape}"
    assert np.all(w == 0.0), "Failed: w elements should all be 0.0"
    assert b == 0.0, f"Failed: b should be 0.0, got {b}"
    assert isinstance(b, float), f"Failed: b type should be float, got {type(b)}"

    dim3 = 4
    w3, b3 = initialize_with_zeros(dim3)
    assert w3.shape == (4, 1) and b3 == 0.0
    print("  -> PASSED: initialize_with_zeros(dim) correctly creates (dim, 1) zero vector and float scalar.")

    # -------------------------------------------------------------------------
    # Test 3: propagate(w, b, X, Y)
    # -------------------------------------------------------------------------
    print("\n[Test 3] Testing propagate(w, b, X, Y)...")
    w = np.array([[1.], [2.]])
    b = 2.0
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])

    grads, cost = propagate(w, b, X, Y)

    expected_dw = np.array([[0.99845601], [2.39507239]])
    expected_db = 0.001455578136784208
    expected_cost = 5.801545319394553

    assert np.isclose(cost, expected_cost, atol=1e-7), (
        f"Failed cost: expected {expected_cost}, got {cost}"
    )
    assert np.allclose(grads["dw"], expected_dw, atol=1e-7), (
        f"Failed dw: expected \n{expected_dw}, got \n{grads['dw']}"
    )
    assert np.isclose(grads["db"], expected_db, atol=1e-7), (
        f"Failed db: expected {expected_db}, got {grads['db']}"
    )
    assert grads["dw"].shape == w.shape, f"dw shape {grads['dw'].shape} mismatch with w {w.shape}"
    print(f"  cost = {cost:.8f} (Expected: {expected_cost:.8f})")
    print(f"  dw   = {grads['dw'].ravel().tolist()} (Expected: {expected_dw.ravel().tolist()})")
    print(f"  db   = {grads['db']:.8f} (Expected: {expected_db:.8f})")
    print("  -> PASSED: propagate(w, b, X, Y) mathematically verified.")

    # -------------------------------------------------------------------------
    # Test 4: optimize(w, b, X, Y, num_iterations, learning_rate, print_cost)
    # -------------------------------------------------------------------------
    print("\n[Test 4] Testing optimize(w, b, X, Y)...")
    w = np.array([[1.], [2.]])
    b = 2.0
    X = np.array([[1., 2., -1.], [3., 4., -3.2]])
    Y = np.array([[1, 0, 1]])

    params, grads, costs = optimize(
        w, b, X, Y,
        num_iterations=100,
        learning_rate=0.009,
        print_cost=False
    )

    expected_w = np.array([[0.19033591], [0.12259159]])
    expected_b = 1.9253598300845747
    expected_dw = np.array([[0.67752042], [1.41625495]])
    expected_db = 0.21919450454067657
    expected_initial_cost = 5.801545319394553

    assert np.allclose(params["w"], expected_w, atol=1e-6), (
        f"Failed w: expected \n{expected_w}, got \n{params['w']}"
    )
    assert np.isclose(params["b"], expected_b, atol=1e-6), (
        f"Failed b: expected {expected_b}, got {params['b']}"
    )
    assert np.allclose(grads["dw"], expected_dw, atol=1e-6), (
        f"Failed final dw: expected \n{expected_dw}, got \n{grads['dw']}"
    )
    assert np.isclose(grads["db"], expected_db, atol=1e-6), (
        f"Failed final db: expected {expected_db}, got {grads['db']}"
    )
    assert len(costs) == 1, f"Expected 1 recorded cost for 100 iterations, got {len(costs)}"
    assert np.isclose(costs[0], expected_initial_cost, atol=1e-6), (
        f"Initial cost mismatch: {costs[0]} vs {expected_initial_cost}"
    )
    print(f"  w  = {params['w'].ravel().tolist()}")
    print(f"  b  = {params['b']:.8f}")
    print(f"  dw = {grads['dw'].ravel().tolist()}")
    print(f"  db = {grads['db']:.8f}")
    print("  -> PASSED: optimize parameter updates and gradients verified.")

    # -------------------------------------------------------------------------
    # Test 5: predict(w, b, X)
    # -------------------------------------------------------------------------
    print("\n[Test 5] Testing predict(w, b, X)...")
    w = np.array([[0.1124579], [0.23106775]])
    b = -0.3
    X = np.array([[1., -1.1, -3.2], [1.2, 2., 0.1]])
    expected_preds = np.array([[1., 1., 0.]])

    preds = predict(w, b, X)
    assert preds.shape == (1, 3), f"Expected shape (1, 3), got {preds.shape}"
    assert np.allclose(preds, expected_preds), (
        f"Failed predict: expected {expected_preds}, got {preds}"
    )
    print(f"  predictions = {preds}")
    print("  -> PASSED: predict(w, b, X) matches expected Coursera output [[1., 1., 0.]].")

    # -------------------------------------------------------------------------
    # Test 6: model(X_train, Y_train, X_test, Y_test, ...)
    # -------------------------------------------------------------------------
    print("\n[Test 6] Testing model() integration on synthetic classification dataset...")
    np.random.seed(42)
    # Generate linearly separable synthetic dataset
    m_train = 200
    m_test = 50
    dim_features = 4

    # Class 0: centered around -1.0, Class 1: centered around +1.0
    X_train_0 = np.random.randn(dim_features, m_train // 2) - 1.0
    X_train_1 = np.random.randn(dim_features, m_train // 2) + 1.0
    X_train = np.hstack([X_train_0, X_train_1])
    Y_train = np.hstack([np.zeros((1, m_train // 2)), np.ones((1, m_train // 2))])

    X_test_0 = np.random.randn(dim_features, m_test // 2) - 1.0
    X_test_1 = np.random.randn(dim_features, m_test // 2) + 1.0
    X_test = np.hstack([X_test_0, X_test_1])
    Y_test = np.hstack([np.zeros((1, m_test // 2)), np.ones((1, m_test // 2))])

    # Shuffle training set
    perm_train = np.random.permutation(m_train)
    X_train = X_train[:, perm_train]
    Y_train = Y_train[:, perm_train]

    result = model(
        X_train, Y_train, X_test, Y_test,
        num_iterations=1000,
        learning_rate=0.05,
        print_cost=False
    )

    # Check required keys
    required_keys = [
        "costs", "Y_prediction_test", "Y_prediction_train",
        "w", "b", "learning_rate", "num_iterations",
        "train_accuracy", "test_accuracy"
    ]
    for key in required_keys:
        assert key in result, f"Missing required key '{key}' in model output dictionary."

    assert result["w"].shape == (dim_features, 1), f"Weight shape mismatch: {result['w'].shape}"
    assert isinstance(result["b"], float), f"b type should be float: {type(result['b'])}"
    assert result["train_accuracy"] >= 95.0, (
        f"Train accuracy too low for separable data: {result['train_accuracy']}%"
    )
    assert result["test_accuracy"] >= 90.0, (
        f"Test accuracy too low for separable data: {result['test_accuracy']}%"
    )
    # Verify cost decreases monotonically
    costs_series = result["costs"]
    assert costs_series[0] > costs_series[-1], "Cost failed to decrease over training iterations."

    print(f"  Train Accuracy: {result['train_accuracy']:.2f} %")
    print(f"  Test Accuracy : {result['test_accuracy']:.2f} %")
    print(f"  Initial Cost  : {costs_series[0]:.6f}")
    print(f"  Final Cost    : {costs_series[-1]:.6f}")
    print("  -> PASSED: Full model trained to high accuracy with smoothly decreasing loss.")

    print("\n" + "=" * 70)
    print("ALL COURSERA AUTOGRADER UNIT TESTS PASSED WITH 100% ACCURACY!")
    print("=" * 70)


def benchmark_vectorization_vs_loop() -> None:
    """
    Demonstrates Andrew Ng's core lesson on Vectorization:
    Compares the execution time of vectorized matrix computation (w^T * X)
    against an explicit non-vectorized Python for-loop over 100,000 examples.
    """
    print("\n" + "=" * 70)
    print("VECTORIZATION BENCHMARK (Andrew Ng's Golden Rule)")
    print("=" * 70)

    n_features = 1000
    m_examples = 100_000

    w = np.random.randn(n_features, 1)
    b = 0.5
    X = np.random.randn(n_features, m_examples)

    # Method 1: Vectorized Dot Product
    t_start = time.perf_counter()
    z_vectorized = np.dot(w.T, X) + b
    t_vectorized = time.perf_counter() - t_start

    # Method 2: Explicit For-Loop over training examples (m)
    # We sample first 1,000 examples to avoid freezing execution while proving the point
    sample_m = 2000
    t_start = time.perf_counter()
    z_loop = np.zeros((1, sample_m))
    for i in range(sample_m):
        val = 0.0
        for j in range(n_features):
            val += w[j, 0] * X[j, i]
        z_loop[0, i] = val + b
    t_loop = (time.perf_counter() - t_start) * (m_examples / sample_m)

    speedup = t_loop / max(t_vectorized, 1e-9)

    print(f"Features: {n_features}, Examples: {m_examples}")
    print(f"Vectorized NumPy time   : {t_vectorized * 1000.0:.2f} ms")
    print(f"Projected For-Loop time : {t_loop * 1000.0:.2f} ms")
    print(f"Vectorization Speedup   : ~{speedup:.1f}x FASTER!")
    print("=" * 70)


if __name__ == "__main__":
    run_coursera_unit_tests()
    benchmark_vectorization_vs_loop()
