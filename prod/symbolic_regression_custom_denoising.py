import time
import pandas as pd
import numpy as np
import os
os.environ["JULIA_NUM_THREADS"] = "8"  # Use 8 threads (adjust as needed)
from pysr import PySRRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def perform_simple_sr( 
        independent_vars: np.ndarray, 
        dependent_var: np.ndarray, 
        n_iterations: int=128, 
        maxsize: int=40
    ):
    model = PySRRegressor(
        niterations=n_iterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
        elementwise_loss="loss(x, y) = (x - y)^2",
        verbosity=1,
        maxsize=maxsize
    )
    model.fit(independent_vars, dependent_var)
    return model

def bound_denoise(
        X_independent_vars: np.ndarray, 
        y_dependent_var: np.ndarray,
        length_scale_bounds: str | tuple[float, float] = (1e-3, 30),
        noise_level_bounds: str | tuple[float, float] = (1e-5, 1e3)
    ):
    kernel = (ConstantKernel(1.0) * RBF(length_scale=0.1, length_scale_bounds=length_scale_bounds) +
              WhiteKernel(noise_level=1.0, noise_level_bounds=noise_level_bounds))
    print('Running GPR...')
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=1)
    gp.fit(X_independent_vars, y_dependent_var)
    print(gp.kernel_)
    y_denoised_dependent = gp.predict(X_independent_vars)
    return y_denoised_dependent

if __name__ == "__main__":
    start_time = time.time()

    df = pd.read_csv('C:/Users/svenl/vs_code_projects/hyperRealDataDescriber/data/credit_score/cleaned_credit_score_v2.csv')
    df = df.sample(n=9600, random_state=42)

    independents = ['Credit_History_Age', 'Delay_from_due_date', 'Num_Bank_Accounts', 
                    'Num_Credit_Inquiries', 'Num_of_Delayed_Payment', 'Outstanding_Debt']
    dependent = 'Interest_Rate'

    X = df[independents].to_numpy()
    y = df[dependent].to_numpy()

    y_denoised = bound_denoise(X, y, length_scale_bounds=(1e-3, 30), noise_level_bounds=(1e-5, 20))

    model = perform_simple_sr(X, y_denoised, n_iterations=256)
    print(model)

    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total execution time: {minutes} minutes and {seconds} seconds")
