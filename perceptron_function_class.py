"""
Function class for binary classification perceptron: sign(w^Tx)
"""
# %%
import numpy as np 
import matplotlib.pyplot as plt 
import math 
from tqdm import tqdm 

VC_DIM = 2
M_TRAIN = 10 ** 4
M_TEST = 10 ** 4
NUM_TRIALS = 100

def generate_data(m): 
    w_true = np.array([1, 2])
    X = np.random.randn(m, 2)
    noise = np.random.randn(m) * 0.025
    y_true = np.sign(X.dot(w_true) + noise)
    y_true[y_true == 0] = 1
    return X, y_true 

def compute_emp_risk(X, y, weights): 
    sum_wrong = 0 
    for xi, yi in zip(X, y): 
       if yi * np.dot(weights, xi) < 0: # wrong
            sum_wrong += 1
    return sum_wrong / X.shape[0]

def compute_vc_bound(r_emp, delta=0.05): 
    other_term = 2 * np.sqrt(2 * ((VC_DIM * np.log(2 * math.e * M_TRAIN / VC_DIM) + np.log(4 / delta)) / M_TRAIN))
    return r_emp + other_term 

def estimate_emp_rademacher_complexity(X, T=100, alpha=0.1, iters=1000):
    m = X.shape[0]
    total = 0

    for _ in tqdm(range(T)):
        sigma = np.random.choice([-1, 1], size=m)

        w = np.zeros(X.shape[1])
        for _ in range(iters):
            for xi, si in zip(X, sigma):
                if si * np.dot(w, xi) <= 0:
                    w += alpha * si * xi

        preds = np.sign(X @ w)
        preds[preds == 0] = 1  
        total += np.sum(sigma * preds)

    return total / (T * m)

emp_rad_complexity = estimate_emp_rademacher_complexity(generate_data(M_TRAIN)[0], iters=1000)


# %%

def compute_rad_bound(emp_risk, delta=0.05): 
    other_term = 3 * np.sqrt(np.log(2 / delta) / M_TRAIN)
    return emp_risk + other_term + emp_rad_complexity

def create_plot(X, y, X_test, y_test, alpha, iters):
    w = np.zeros(2) # initialize weights
    vc_bounds = []
    rad_bounds = []
    test_estimated_risks = []
    for _ in range(iters):
        for xi, yi in zip(X, y):
            if yi * np.dot(w, xi) <= 0: # wrong 
                w += alpha * yi * xi # update

            # now compute the empirical risk 
        test_estimated_risks += [compute_emp_risk(X_test, y_test, w)]
        emp_risk = compute_emp_risk(X, y, w)
        vc_bounds += [compute_vc_bound(emp_risk)] 
        rad_bounds += [compute_rad_bound(emp_risk)]
        print(w)
            
    plt.plot(vc_bounds, label=r"95% VC Bound on $R^{\text{true}}(f)$")
    plt.plot(rad_bounds, label=r"95% Rademacher Bound on $R^{\text{true}}(f)$")
    plt.plot(test_estimated_risks, label=r"$R^{\text{test}}(f)$")
    plt.xlabel("Iteration")
    plt.ylabel("Risk")
    plt.title(r"95% VC & Rademacher Bounds on $R^{\text{true}}(f)$")
    plt.legend()
    plt.grid(True)
    plt.show()

    return w

X, y = generate_data(M_TRAIN)
X_test, y_test = generate_data(M_TEST)
create_plot(X, y, X_test, y_test, 0.10, 1000)

# %%
