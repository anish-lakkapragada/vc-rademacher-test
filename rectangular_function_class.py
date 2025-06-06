# %% 
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

VC_DIM = 4
M_TRAIN = 10 ** 4
M_TEST = 10 ** 4
NUM_TRIALS_RAD = 100
DELTA = 0.05

def generate_data(m): 
    a, b, c, d = 1, 3, 2, 4  # Class +1 is inside this rectangle
    X = np.random.randn(m, 2) * 2
    noise = np.random.randn(m) * 0.0025

    inside = (X[:, 0] >= a) & (X[:, 0] <= b) & (X[:, 1] >= c) & (X[:, 1] <= d)
    y = np.where(inside, 1, -1).astype(float)

    flip_mask = (np.random.rand(m) < np.abs(noise))
    y[flip_mask] *= -1

    return X, y

def train_rectangle(X, y):
    class1 = X[y == 1]
    h, v = class1[:, 0], class1[:, 1]
    a = np.percentile(h, 5)
    b = np.percentile(h, 95)
    c = np.percentile(v, 5)
    d = np.percentile(v, 95)
    return a, b, c, d

def predict_rectangle(X, rect):
    a, b, c, d = rect
    inside = (X[:, 0] >= a) & (X[:, 0] <= b) & (X[:, 1] >= c) & (X[:, 1] <= d)
    return np.where(inside, 1, -1)

def compute_emp_risk(X, y, rect_params):
    preds = predict_rectangle(X, rect_params)
    return np.mean(preds != y)

def compute_vc_bound(r_emp, m, d, delta=DELTA):
    term = 2 * np.sqrt(2 * ((d * np.log(2 * math.e * m / d) + np.log(4 / delta)) / m))
    return r_emp + term

def compute_rademacher_bound(r_emp, rad_complexity, m, delta=DELTA):
    slack = 3 * np.sqrt(np.log(2 / delta) / m)
    return r_emp + rad_complexity + slack

def estimate_emp_rademacher_complexity(X, T=NUM_TRIALS_RAD):
    m = X.shape[0]
    total = 0

    for _ in tqdm(range(T), desc="Estimating Rademacher complexity"):
        sigma = np.random.choice([-1, 1], size=m)
        rect_params = train_rectangle(X, sigma)
        preds = predict_rectangle(X, rect_params)
        total += np.sum(sigma * preds)

    return total / (T * m)

# %% 

X_train, y_train = generate_data(M_TRAIN)
X_test, y_test = generate_data(M_TEST)

rect_params = train_rectangle(X_train, y_train)

emp_risk_train = compute_emp_risk(X_train, y_train, rect_params)
emp_risk_test = compute_emp_risk(X_test, y_test, rect_params)

rad_complexity = estimate_emp_rademacher_complexity(X_train)
rad_bound = compute_rademacher_bound(emp_risk_train, rad_complexity, M_TRAIN)
vc_bound = compute_vc_bound(emp_risk_train, M_TRAIN, VC_DIM)

print("\n--- Rectangle Classifier ---")
print(f"Empirical Risk (Train): {emp_risk_train:.4f}")
print(f"Empirical Risk (Test): {emp_risk_test:.4f}")
print(f"Empirical Rademacher Complexity: {rad_complexity:.4f}")
print(f"Rademacher Bound (95%): {rad_bound:.4f}")
print(f"VC Bound (95%): {vc_bound:.4f}")


# %%
