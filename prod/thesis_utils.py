# import os
# os.environ["JULIA_NUM_THREADS"] = "8"  # Use 8 threads (adjust as needed)

import pandas as pd
# from pysr import PySRRegressor
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from distfit import distfit
import copy
import json
import bnlearn as bn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sympy import sympify, lambdify
from collections import defaultdict
from scipy.stats import norm, gamma, lognorm, norm as normal_dist, gaussian_kde
from scipy.special import softmax
from numpy.linalg import cholesky, eigh
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Define a reusable type alias
ArrayLike = Union[list, np.ndarray, pd.Series]


# def perform_simple_sr( 
#         independent_vars: np.ndarray, 
#         dependent_var: np.ndarray, 
#         n_iterations: int=128, 
#         maxsize: int=40
#     ):

#     # Use PySR to find the symbolic relationship
#     model = PySRRegressor(
#         niterations=n_iterations,  # Number of iterations to search for equations
#         binary_operators=["+", "-", "*", "/"],
#         unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
#         elementwise_loss="loss(x, y) = (x - y)^2",  # Define loss function (mean squared error)
#         verbosity=1,
#         maxsize=maxsize
#     )

#     # Fit the model
#     model.fit(independent_vars, dependent_var)

#     return model


def auto_denoise(
        X_independent_data: np.ndarray, 
        y_dependent_data: np.ndarray
    ):
    custom_kernel = ConstantKernel(1.0) * RBF() + WhiteKernel()
    gp = GaussianProcessRegressor(kernel=custom_kernel, n_restarts_optimizer=5)
    gp.fit(X_independent_data, y_dependent_data)
    # Generate denoised targets using custom GP.
    y_denoised_dependent = gp.predict(X_independent_data)
    return y_denoised_dependent


def bound_denoise(
        X_independent_data: np.ndarray, 
        y_dependent_data: np.ndarray,
        length_scale_bounds: str | tuple[float, float] = (1e-3, 30),
        noise_level_bounds: str | tuple[float, float] = (1e-5, 1e3),
        show_denoising_effect: bool = False
    ):
    kernel = (ConstantKernel(1.0) *
          RBF(length_scale=0.1, length_scale_bounds=length_scale_bounds) +
          WhiteKernel(noise_level=1.0, noise_level_bounds=noise_level_bounds))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=1)
    gp.fit(X_independent_data, y_dependent_data)
    print(gp.kernel_)
    y_denoised_dependent = gp.predict(X_independent_data)

    if show_denoising_effect:
        plt.figure(figsize=(8,4))
        plt.scatter(y_dependent_data, y_denoised_dependent, alpha=0.5)
        plt.xlabel("Original y")
        plt.ylabel("Denoised y")
        plt.title("Custom Denoising Effect")
        plt.show()
    
    return y_denoised_dependent


# def perform_auto_denoised_sr(
#         dataset: pd.DataFrame, 
#         independent_vars: list, 
#         dependent_var: str, 
#         n_iterations: int=128, 
#         maxsize: int=40
#     ):
#     X = dataset[independent_vars]
#     y = dataset[dependent_var]

#     y_denoised = auto_denoise(X, y)

#     model = perform_simple_sr(dataset, X, y_denoised, n_iterations, maxsize)

#     return model, y_denoised


# def perform_bound_denoised_sr(
#         dataset: pd.DataFrame, 
#         independent_vars: list, 
#         dependent_var: str, 
#         n_iterations: int=128, 
#         maxsize: int=40,
#         length_scale_bounds: str | tuple[float, float] = (1e-3, 30),
#         noise_level_bounds: str | tuple[float, float] = (1e-5, 1e3)
#     ):
#     X = dataset[independent_vars]
#     y = dataset[dependent_var]

#     y_denoised = bound_denoise(X, y, length_scale_bounds, noise_level_bounds)

#     model = perform_simple_sr(dataset, X, y_denoised, n_iterations, maxsize)

#     return model, y_denoised


def generate_candidate_function(selected_eq_str):
    """
    Converts a string representation of an equation into a callable function.
    
    Parameters:
    - selected_eq_str (str): The candidate equation as a string, using variables like x0, x1, etc.
    
    Returns:
    - candidate_function (function): A function that takes a NumPy array X and evaluates the equation.
    """
    f_sympy = sympify(selected_eq_str)
    free_syms = sorted(f_sympy.free_symbols, key=lambda s: s.name)
    f_callable = lambdify(free_syms, f_sympy, 'numpy')

    def candidate_function(X):
        # Assumes ordering: first free symbol -> first column, etc.
        if len(free_syms) == 1:
            return f_callable(X[:, 0])
        elif len(free_syms) == 2:
            return f_callable(X[:, 0], X[:, 1])
        else:
            args = [X[:, i] for i in range(len(free_syms))]
            return f_callable(*args)
    
    return candidate_function


def generate_noise(
    residuals,
    method='bootstrap',
    rng=None,
    y_sr=None,
    stratify_bins=10
):
    """
    Generate noise according to the chosen method.
    
    - 'bootstrap': simple residual bootstrap from `residuals`.
    - 'std': Gaussian noise N(0, std(residuals)).
    - 'stratified': bin-wise bootstrap of `residuals` based on y_denoised.
    
    `rng` may be an integer seed or a numpy Generator.
    """
    # Ensure rng is a Generator
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    
    n = len(residuals)
    
    if method == 'bootstrap':
        return rng.choice(residuals, size=n, replace=True)
    
    if method == 'std':
        noise_std = np.std(residuals)
        return rng.normal(0, noise_std, size=n)
    
    if method == 'stratified':
        if y_sr is None:
            raise ValueError("y_denoised must be provided for stratified sampling")
        # 1) Build bins on y_denoised
        edges = np.quantile(y_sr, np.linspace(0, 1, stratify_bins+1))
        bin_idx = np.digitize(y_sr, edges[1:-1])  # 0..stratify_bins-1
        
        # 2) Group residuals by bin
        res_by_bin = {
            b: residuals[bin_idx == b]
            for b in range(stratify_bins)
        }
        
        # 3) Sample
        noise = np.empty(n)
        for i in range(n):
            pool = res_by_bin.get(bin_idx[i])
            if pool is None or pool.size == 0:
                pool = residuals
            noise[i] = rng.choice(pool)
        return noise
    
    raise ValueError("Unknown method: " + method)


def compute_renoised_error(
    y,
    y_sr,
    residuals,
    amp,
    method,
    bins,
    use_direct_mse,
    rng,
    clip_lower=None,
    clip_upper=None,
    tail_replace=False,
    lower_percentile=25,
    upper_percentile=75,
    stratify_bins=10
):
    """
    Generate one re-noised prediction and its MSE.
    """
    # 1) Generate noise
    noise = generate_noise(
        residuals,
        method=method,
        rng=rng,
        y_sr=y_sr,
        stratify_bins=stratify_bins
    )
    # 2) Apply amplification
    y_pred = y_sr + amp * noise

    # 3) Handle out-of-bounds
    if tail_replace and (clip_lower is not None or clip_upper is not None):
        # precompute tails from the original y
        if clip_lower is not None:
            q_low = np.percentile(y, lower_percentile)
            lower_pool = y[y <= q_low]
            mask = y_pred < clip_lower
            if mask.any() and lower_pool.size:
                y_pred[mask] = rng.choice(lower_pool, size=mask.sum())
        if clip_upper is not None:
            q_high = np.percentile(y, upper_percentile)
            upper_pool = y[y >= q_high]
            mask = y_pred > clip_upper
            if mask.any() and upper_pool.size:
                y_pred[mask] = rng.choice(upper_pool, size=mask.sum())
    else:
        # hard clipping
        if clip_lower is not None:
            y_pred = np.maximum(y_pred, clip_lower)
        if clip_upper is not None:
            y_pred = np.minimum(y_pred, clip_upper)

    # 4) Compute MSE
    if use_direct_mse:
        error = np.mean((y - y_pred) ** 2)
    else:
        hist_y, edges = np.histogram(y, bins=bins, density=True)
        hist_pred, _ = np.histogram(y_pred, bins=edges, density=True)
        error = np.mean((hist_y - hist_pred) ** 2)

    return error, y_pred


def correct_predictions(
    y,
    y_sr,
    original_residuals=None,
    method='bootstrap',
    amplification_factor=None,
    amplification_grid=np.linspace(0.5, 2.0, 20),
    bins=30,
    seed=None,
    use_direct_mse=False,
    clip_lower=None,
    clip_upper=None,
    tail_replace=False,
    lower_percentile=25,
    upper_percentile=75,
    stratify_bins=10
):
    """
    Reintroduce noise into y_denoised and return the best match to y.
    
    Returns:
      y_renoised, best_amplification, errors_list, used_residuals
    """
    rng = np.random.default_rng(seed)
    # ** Choose which residuals to use **
    if original_residuals is None:
        residuals = y - y_sr
    else:
        residuals = original_residuals
    errors = []
    preds  = []

    # Grid search over amplification if needed
    if amplification_factor is None:
        for amp in amplification_grid:
            err, yp = compute_renoised_error(
                y, y_sr, residuals, amp,
                method, bins, use_direct_mse, rng,
                clip_lower, clip_upper,
                tail_replace, lower_percentile, upper_percentile,
                stratify_bins
            )
            errors.append(err)
            preds.append(yp)
        best_idx = int(np.argmin(errors))
        best_amp = amplification_grid[best_idx]
        y_renoised = preds[best_idx]
    else:
        best_amp = amplification_factor
        _, y_renoised = compute_renoised_error(
            y, y_sr, residuals, best_amp,
            method, bins, use_direct_mse, rng,
            clip_lower, clip_upper,
            tail_replace, lower_percentile, upper_percentile,
            stratify_bins
        )
        errors = None

    # Return also the residuals used (for your “freeze residuals” experiments)
    return y_renoised, best_amp, errors, residuals


def correct_sr_inference(
        dataset: pd.DataFrame, 
        features: list, 
        target: str, 
        sr_prediction: pd.Series
    ):

    features_df = dataset[features]
    residuals = dataset[target] - sr_prediction
    residuals = residuals.reset_index(drop=True)

    # Combine SR prediction as an additional feature
    X = pd.concat([features_df, sr_prediction.rename("SR_pred")], axis=1)
    y = residuals  # target: residuals computed from (actual - SR prediction)

    # Split data for training and testing (if desired)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create XGBoost DMatrix objects
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Set up parameters for regression
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6
    }

    num_rounds = 10000
    model_xgb = xgb.train(params, dtrain, num_rounds)

    # Evaluate on test set
    y_pred_test = model_xgb.predict(dtest)
    mse_test = mean_squared_error(y_test, y_pred_test)
    print(f"XGBoost Residual Correction Test MSE: {mse_test:.4f}")

    # For final predictions, use the full dataset:
    dall = xgb.DMatrix(X)
    predicted_residual_correction = model_xgb.predict(dall)

    # Compute final prediction as the sum of the SR prediction and the predicted residual correction
    final_prediction = sr_prediction + predicted_residual_correction

    # Optionally, compute overall MSE against actual values
    final_mse = mean_squared_error(dataset[target], final_prediction)
    print(f"Final MSE after XGBoost residual correction: {final_mse:.4f}")

    return final_prediction


def plot_histograms(
        datasets: list[tuple[np.ndarray, str, str]],
        xlabel: str,
        ylabel: str = 'Frequency',
        rows: int = 1,
        cols: int = None,
        bins: int = 30,
        figsize_per_plot: tuple[int, int] = (5, 5),
        stack: bool = False
    ):
    import numpy as np
    import matplotlib.pyplot as plt

    n = len(datasets)
    global_min = min(data.min() for data, _, _ in datasets)
    global_max = max(data.max() for data, _, _ in datasets)
    all_data = np.concatenate([data for data, _, _ in datasets])
    bin_edges = np.histogram_bin_edges(all_data, bins=bins)

    # Consistent max frequency across all datasets
    max_freq = max(np.histogram(data, bins=bin_edges)[0].max() for data, _, _ in datasets)

    if stack:
        plt.figure(figsize=figsize_per_plot)
        for data, label, color in datasets:
            plt.hist(data, bins=bin_edges, alpha=0.5, label=label, color=color)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Stacked Histogram of All Distributions')
        plt.legend()
        plt.xlim(global_min, global_max)
        plt.ylim(0, max_freq * 1.1)
        plt.tight_layout()
        plt.show()

    else:
        if cols is None:
            cols = int(np.ceil(n / rows))
        fig_width = figsize_per_plot[0] * cols
        fig_height = figsize_per_plot[1] * rows
        plt.figure(figsize=(fig_width, fig_height))

        for i, (data, label, color) in enumerate(datasets):
            plt.subplot(rows, cols, i + 1)
            plt.hist(data, bins=bin_edges, alpha=0.7, color=color, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f'{label} Distribution')
            plt.legend()
            plt.xlim(global_min, global_max)
            plt.ylim(0, max_freq * 1.1)

        plt.tight_layout()
        plt.show()


def plot_densities(
    datasets: list[tuple[np.ndarray, str, str]],
    xlabel: str,
    ylabel: str = 'Density',
    rows: int = 1,
    cols: int = None,
    figsize_per_plot: tuple[int, int] = (5, 5),
    stack: bool = False,
    bw_adjust: float = 1.0
):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    n = len(datasets)

    global_min = min(np.min(data) for data, _, _ in datasets)
    global_max = max(np.max(data) for data, _, _ in datasets)

    # Determine true max y across all KDEs using seaborn
    max_density = 0
    x_vals = np.linspace(global_min, global_max, 1000)

    for data, _, _ in datasets:
        kde_line = sns.kdeplot(data, bw_adjust=bw_adjust).get_lines()[0]
        y_vals = kde_line.get_ydata()
        max_density = max(max_density, np.max(y_vals))
        plt.clf()  # Clear the temp plot

    if stack:
        fig, ax = plt.subplots(figsize=figsize_per_plot)
        for data, label, color in datasets:
            sns.kdeplot(data, fill=True, color=color, label=label, bw_adjust=bw_adjust, ax=ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Stacked Density Plot of All Distributions')
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(0, max_density * 1.1)
        ax.legend()
        plt.tight_layout()
        plt.show()

    else:
        if cols is None:
            cols = int(np.ceil(n / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_plot[0] * cols, figsize_per_plot[1] * rows))
        axes = np.array(axes).flatten()

        for ax, (data, label, color) in zip(axes, datasets):
            sns.kdeplot(data, fill=True, color=color, label=label, bw_adjust=bw_adjust, ax=ax)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{label} Density')
            ax.set_xlim(global_min, global_max)
            ax.set_ylim(0, max_density * 1.1)
            ax.legend()

        for i in range(len(datasets), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()
        plt.show()


def fit_distributions(
        dataset: pd.DataFrame, 
        target_distributions: list=['norm', 'lognorm', 'gamma']
    ):
    # Dictionary to store results for each numeric column
    fitted_results = {}

    # Get a list of numeric columns in your DataFrame
    numeric_cols = dataset.select_dtypes(include='number').columns

    for col in numeric_cols:
        # print(f"Processing column: {col}")
        
        # Extract and clean the data for the current column
        data = dataset[col].dropna().values
        
        # Initialize a distfit object, restricting to only the desired distributions.
        # Setting verbose=0 will suppress the printed log.
        dfit = distfit(distr=target_distributions, verbose=0)
        
        # Fit the distributions on the data.
        dfit.fit_transform(data)
        
        # Retrieve the summary DataFrame that contains the fit results.
        summary_df = dfit.summary
        
        if not summary_df.empty:
            # Choose the best fit as the one with the lowest RSS (score)
            best_dist = summary_df['score'].idxmin()
            best_params = summary_df.loc[best_dist].to_dict()
            
            # Store the result for this column
            fitted_results[col] = {
                'best_distribution': best_dist,
                'parameters': best_params
            }
        else:
            fitted_results[col] = None
    
    return fitted_results


def distributions_to_profile_wf(
        distr_report: dict, 
        to_json: bool=True
    ):

    distr_profiles = []

    for col, result in distr_report.items():

        distr_elem = {
            'name': str(col),
            'type': 'num'
        }

        if result is not None:

            col_details = copy.deepcopy(result['parameters'])

            if col_details['name'] == 'norm':
                distr_details = {
                    'type': 'GAUSSIAN',
                    'mean': col_details['params'][0],
                    'deviation': col_details['params'][1]
                }
                distr_elem['wf.hr'] = distr_details
            elif col_details['name'] == 'lognorm':
                distr_details = {
                    'type': 'LOGNORM',
                    'mean': np.log(col_details['params'][2]),
                    'deviation': col_details['params'][0] 
                }
                distr_elem['wf.hr'] = distr_details
            elif col_details['name'] == 'gamma':
                distr_details = {
                    'type': 'GAMMA',
                    'alpha': col_details['params'][0],
                    'theta': col_details['params'][2] 
                }
                distr_elem['wf.hr'] = distr_details
            else:
                distr_elem['wf.hr'] = 'ERROR'
        else:
            distr_elem['message'] = 'ERROR'

        distr_profiles.append(distr_elem)
    
    if to_json:
        return json.dumps(distr_profiles, indent=4)
    else:
        return distr_profiles


def anonymize_column_names(dataset: pd.DataFrame):
    rename_map = {col: f"X{i+1}" for i, col in enumerate(dataset.columns)}
    df_anonymized = dataset.rename(columns=rename_map)
    return df_anonymized, rename_map


def deanonymize_column_names(
        dataset_anonymized: pd.DataFrame, 
        rename_map: dict[str, str]
    ):
    reverse_map = {v: k for k, v in rename_map.items()}
    df_original = dataset_anonymized.rename(columns=reverse_map)
    return df_original


def get_correlation_statements(correlation_matrix: pd.DataFrame):
    corr_statements = []
    corr_permutations = []

    for column in correlation_matrix:
        for i in range(len(correlation_matrix)):
            current = correlation_matrix[column][i]
            if current != 1 and np.abs(current) >= 0.2:
                first_arg = column.split(' ')[0]
                second_arg = (list(correlation_matrix.columns)[i]).split(' ')[0]
                r = current.round(2)
                corr_permutations.append([second_arg, first_arg, r])
                if [first_arg, second_arg, r] not in corr_permutations:
                    corr_statements.append(f"Columns {first_arg} and {second_arg} (r = {r})")
    
    return corr_statements


def get_discrete_numerical_columns(dataset: pd.DataFrame):
    # Select only numeric columns
    numeric_df = dataset.select_dtypes(include=['number'])

    # Define discrete columns as those where every non-null value is an integer.
    # Using np.allclose to be robust against floating point rounding errors.
    discrete_cols = [
        col for col in numeric_df.columns
        if np.allclose(numeric_df[col].dropna() % 1, 0)
    ]

    return discrete_cols


def get_categorical_columns(dataset: pd.DataFrame):
    all_cols = dataset.columns.to_list()
    # Select only numeric columns
    numeric_df = dataset.select_dtypes(include=['number'])
    numeric_cols = numeric_df.columns.to_list()
    categorical_cols = [item for item in all_cols if item not in numeric_cols]
    return categorical_cols


# TODO add min-max values and handle discrete numeric columns
def get_distribution_statements(distr_report: dict, dataset: pd.DataFrame):
    discrete_columns = get_discrete_numerical_columns(dataset)
    distr_statements = []
    statement = None
    for col in dataset:
        if col in distr_report and distr_report[col] is not None:
            result = distr_report[col]
            col_min = round(dataset[col].min(), 3)
            col_max = round(dataset[col].max(), 3)
            col_details = result['parameters']
            if col_details['name'] == 'norm':
                statement = (f'Column \"{col}\": type = numerical, distribution = gaussian, loc = {round(col_details["params"][0], 3)}, scale = {round(col_details["params"][1], 3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
            elif col_details['name'] == 'lognorm':
                statement = (f'Column \"{col}\": type = numerical, distribution = lognorm, s = {round(col_details["params"][0], 3)}, loc = {round(col_details["params"][1], 3)}, scale = {round(col_details["params"][2], 3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
            elif col_details['name'] == 'gamma':
                statement = (f'Column \"{col}\": type = numerical, distribution = gamma, a = {round(col_details["params"][0], 3)}, loc = {round(col_details["params"][1], 3)}, scale = {round(col_details["params"][2], 3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
        else:
            categories = json.dumps(dataset[col].unique().tolist())
            statement = (f'Column \"{col}\": type = categorical, categories = {categories}')
        distr_statements.append(statement)
    return distr_statements


def auto_preprocess_bn(
        dataset: pd.DataFrame, 
        bins: int = 5, 
        discrete_threshold: int = 10
    ):
    df_processed = dataset.copy()
    for col in df_processed.columns:
        if pd.api.types.is_numeric_dtype(df_processed[col]):
            # Heuristic: if float type or many unique values, then consider it continuous.
            if df_processed[col].dtype in [np.float64, np.float32] or df_processed[col].nunique() > discrete_threshold:
                try:
                    # Use 'duplicates="drop"' to handle cases with non-unique bin edges.
                    df_processed[col] = pd.qcut(df_processed[col], q=bins, labels=False, duplicates='drop')
                except Exception as e:
                    print(f"Error discretizing column {col}: {e}")
        elif df_processed[col].dtype == 'object':
            # Convert string columns to categorical type.
            df_processed[col] = df_processed[col].astype('category')
    return df_processed


def get_bn_statements(model: (dict | list)):
    bn_statements = []
    try:
        for edge in model['model_edges']:
            bn_statements.append(f'{edge[0]} -> {edge[1]}')
        return bn_statements
    except KeyError:
        return None


def get_llm_generation_prompt(
        dataset: pd.DataFrame, 
        correlation_matrix: pd.DataFrame,
        include_bn: bool = False, 
        enforce_positive_nums: bool = False,
        n_rows: int = None,
        discrete_threshold: int = 10
    ):
    n_columns = len(dataset.columns)
    if not n_rows:
        n_rows = len(dataset)

    prompt = f'Generate a table with {n_columns} columns and {n_rows} rows with the following properties:\n\n'

    distributions = fit_distributions(dataset)
    distribution_staements = get_distribution_statements(distributions, dataset)
    for statement in distribution_staements:
        prompt += statement + '\n'

    prompt += '\nWith correlations:\n'

    correlation_statements = get_correlation_statements(correlation_matrix)
    for statement in correlation_statements:
        prompt += statement + '\n'
    
    if include_bn:
        prompt += '\nWith Bayesian Network structure:\n'
        df_preprocessed = auto_preprocess_bn(dataset, discrete_threshold=discrete_threshold)
        model = bn.structure_learning.fit(df_preprocessed, methodtype='hc', scoretype='bic', verbose=0)
        bn_statements = get_bn_statements(model)
        for statement in bn_statements:
            prompt += statement + '\n'

    prompt += '\n'

    if enforce_positive_nums:
        prompt += 'All numbers in the table must be positive. '

    prompt += ('Re-check the data to ensure that all conditions are met before displaying. '
               'Every condition must be met exactly. Do not output until exactly correct. '
               'Provide the data in a downloadable Excel file.')

    return prompt


def generate_bn_generation_profile(df, categorical_cols, bn_edges, distfit_dict=None, bins=5, discrete_threshold=10):
    """
    Generates a generation profile from data and a Bayesian network structure.
    
    Parameters:
        df (pd.DataFrame): The original dataset.
        categorical_cols (list): List of column names that are categorical.
        bn_edges (list of tuples): BN structure as (parent, child) edges.
        distfit_dict (dict, optional): Dictionary with best-fitting distribution details
            for numerical variables (e.g., output from distfit).
        bins (int, optional): Number of bins to use when discretizing continuous parent columns.
        discrete_threshold (int, optional): Threshold for unique values to consider a numeric column as continuous.
    
    Returns:
        dict: Generation profile containing marginal and conditional parameters.
    """
    profile = {"variables": {}, "structure": bn_edges}
    all_vars = list(df.columns)
    
    # Process unconditional/marginal profiles using original continuous data.
    for col in all_vars:
        if col in categorical_cols:
            marginal = df[col].value_counts(normalize=True).to_dict()
            levels = list(marginal.keys())
            profile["variables"][col] = {
                "type": "categorical",
                "levels": levels,
                "marginal": marginal
            }
        else:
            col_min = round(df[col].min(), 3)
            col_max = round(df[col].max(), 3)
            if distfit_dict is not None and col in distfit_dict and distfit_dict[col] is not None:
                col_details = distfit_dict[col]['parameters']
                if col_details['name'] == 'norm':
                    distribution_name = "Gaussian"
                    parameters = {
                        "mean": round(col_details["params"][0], 3),
                        "deviation": round(col_details["params"][1], 3),
                        "min": col_min,
                        "max": col_max
                    }
                elif col_details['name'] == 'lognorm':
                    distribution_name = "Lognormal"
                    parameters = {
                        "mean": round(np.log(col_details["params"][2]), 3),
                        "deviation": round(col_details["params"][0], 3),
                        "min": col_min,
                        "max": col_max
                    }
                elif col_details['name'] == 'gamma':
                    distribution_name = "Gamma"
                    parameters = {
                        "alpha": round(col_details["params"][0], 3),
                        "theta": round(col_details["params"][2], 3),
                        "min": col_min,
                        "max": col_max
                    }
                else:
                    distribution_name = col_details['name'].capitalize()
                    parameters = {
                        "params": [round(p, 3) for p in col_details["params"]],
                        "min": col_min,
                        "max": col_max
                    }
                profile["variables"][col] = {
                    "type": "numerical",
                    "distribution": distribution_name,
                    "parameters": parameters
                }
            else:
                mean_val = round(df[col].mean(), 3)
                std_val = round(df[col].std(), 3)
                profile["variables"][col] = {
                    "type": "numerical",
                    "distribution": "Gaussian",
                    "parameters": {
                        "mean": mean_val,
                        "deviation": std_val,
                        "min": col_min,
                        "max": col_max
                    }
                }
    
    # Build mapping from each child to its parent(s) from the BN edges.
    parent_dict = defaultdict(list)
    for parent, child in bn_edges:
        parent_dict[child].append(parent)
    
    # Compute conditional distributions.
    for child, parents in parent_dict.items():
        if child not in df.columns:
            continue
        
        # Create a copy of df for grouping and bin the parent columns into intervals.
        df_group = df.copy()
        for parent in parents:
            if parent not in categorical_cols and pd.api.types.is_numeric_dtype(df_group[parent]):
                # If too many unique values, bin into quantile-based intervals.
                if df_group[parent].nunique() > discrete_threshold or df_group[parent].dtype in [np.float64, np.float32]:
                    try:
                        # Compute bin edges based on quantiles.
                        quantiles = np.linspace(0, 1, bins + 1)
                        bin_edges = df_group[parent].quantile(quantiles).unique()
                        bin_edges = np.sort(bin_edges)
                        # Only bin if there is more than one unique edge.
                        if len(bin_edges) > 1:
                            # Use pd.cut with right=False for left-closed, right-open intervals.
                            df_group[parent] = pd.cut(df_group[parent], bins=bin_edges, include_lowest=True, right=False)
                        # Else leave the column as is.
                    except Exception as e:
                        print(f"Error discretizing column {parent}: {e}")
        
        grouped = df_group.groupby(parents)
        cond_table = {}
        
        # Determine the unconditional distribution type of the child, if available.
        if distfit_dict is not None and child in distfit_dict and distfit_dict[child] is not None:
            child_dist = distfit_dict[child]['parameters']['name']
        else:
            child_dist = 'norm'
            
        for group_vals, group_df in grouped:
            # For a single parent, group_vals is an Interval; for multiple parents, it's a tuple.
            if not isinstance(group_vals, tuple):
                config_key = f"{parents[0]}={group_vals}"
            else:
                config_key = ",".join([f"{p}={v}" for p, v in zip(parents, group_vals)])
            
            if child in categorical_cols:
                cond_prob = group_df[child].value_counts(normalize=True).to_dict()
                cond_table[config_key] = cond_prob
            else:
                n = group_df[child].shape[0]
                base_deviation = 0.0 if n < 2 else group_df[child].std()
                if child_dist == 'lognorm':
                    valid = group_df[child] > 0
                    if valid.sum() >= 2:
                        log_vals = np.log(group_df[child][valid])
                        cond_mean = log_vals.mean()
                        cond_std = log_vals.std()
                    else:
                        cond_mean, cond_std = np.nan, np.nan
                    cond_table[config_key] = {
                        "mean": round(cond_mean, 3),
                        "deviation": round(cond_std, 3)
                    }
                elif child_dist == 'gamma':
                    m = group_df[child].mean()
                    s = base_deviation
                    if m > 0 and s != 0:
                        shape = (m/s)**2
                        scale = (s**2)/m
                    else:
                        shape, scale = 0.0, 0.0
                    cond_table[config_key] = {
                        "alpha": round(shape, 3),
                        "theta": round(scale, 3)
                    }
                else:
                    m = group_df[child].mean()
                    s = base_deviation
                    cond_table[config_key] = {
                        "mean": round(m, 3),
                        "deviation": round(s, 3)
                    }
        if child in profile["variables"]:
            profile["variables"][child]["conditional"] = {
                "parents": parents,
                "table": cond_table
            }
    
    return profile


def convert_distfit_to_marginals(distr_report: dict, dataset: pd.DataFrame) -> dict:
    discrete_columns = get_discrete_numerical_columns(dataset)
    marginals = {}

    for col in dataset.columns:
        if col in distr_report and distr_report[col] is not None:
            result = distr_report[col]
            dist_name = result['parameters']['name']
            params_list = result['parameters']['params']
            col_min = round(dataset[col].min(), 3)
            col_max = round(dataset[col].max(), 3)

            # Initialize param_dict
            param_dict = {}

            # Handle known distributions
            if dist_name == 'norm':
                param_dict = {
                    'loc': round(params_list[0], 3),
                    'scale': round(params_list[1], 3)
                }

            elif dist_name == 'lognorm':
                param_dict = {
                    's': round(params_list[0], 3),
                    'loc': round(params_list[1], 3),
                    'scale': round(params_list[2], 3)
                }

            elif dist_name == 'gamma':
                param_dict = {
                    'a': round(params_list[0], 3),
                    'loc': round(params_list[1], 3),
                    'scale': round(params_list[2], 3)
                }

            else:
                raise ValueError(f"Unsupported distribution '{dist_name}' for column '{col}'")

            marginals[col] = {
                'name': dist_name,
                'params': param_dict,
                'range': (col_min, col_max),
                'round': col in discrete_columns
            }

    return marginals


def nearest_pd(A):
    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)
    H = V.T @ np.diag(s) @ V
    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while np.min(np.real(np.linalg.eigvals(A3))) < 0:
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1
    return A3



def _make_positive_definite(corr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Force a symmetric matrix to be positive-definite by eigen-decomposition.
    Clips negative eigenvalues to 'eps' and re-constructs the matrix with unit diagonal.
    """
    # Symmetrize
    A = (corr + corr.T) / 2
    # Eigen-decomposition
    vals, vecs = eigh(A)
    # Clip eigenvalues
    vals_clipped = np.clip(vals, a_min=eps, a_max=None)
    # Reconstruct
    A_pd = vecs @ np.diag(vals_clipped) @ vecs.T
    # Enforce exact unit diagonal
    np.fill_diagonal(A_pd, 1.0)
    return A_pd


def generate_synthetic_dataset(
    original_data: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    categorical_columns: list,
    marginals: dict,
    n_rows: int = 10000,
    conditional_method: str = "quantile",
    correlation_threshold: float = 0.25,
    noise_level: float = 0.0,
    noise_type: str = "gaussian",
    edge_strategy: str = "clip",
    histogram_jitter: bool = False,
    sampling_strategy: str = "sorted",
    chunk_size: int = None,
) -> pd.DataFrame:
    """
    Generate synthetic data using a Gaussian copula with empirical marginals.
    This version ensures the correlation matrix is strictly positive-definite before sampling.
    """
    # Identify numeric columns
    numeric_cols = [c for c in correlation_matrix.columns if c not in categorical_columns]
    # Threshold small correlations
    reduced_corr = correlation_matrix.loc[numeric_cols, numeric_cols].copy()
    reduced_corr[reduced_corr.abs() < correlation_threshold] = 0.0

    # Convert to numpy and enforce PD
    corr_array = reduced_corr.values
    corr_array = _make_positive_definite(corr_array)

    # Try Cholesky, with jitter fallback
    jitter = 0.0
    while True:
        try:
            L = cholesky(corr_array + jitter * np.eye(corr_array.shape[0]))
            break
        except np.linalg.LinAlgError:
            jitter = max(1e-10, (jitter or 1e-8) * 10)
    
    # Sample Gaussian copula
    Z = np.random.normal(size=(n_rows, len(numeric_cols)))
    Z_corr = Z @ L.T

    # Build DataFrame for numeric variables
    df_num = pd.DataFrame(index=range(n_rows))
    for i, col in enumerate(numeric_cols):
        info = marginals[col]
        dist_name = info['name']
        params = info['params']

        # Simulate marginal
        if dist_name == 'gamma':
            base = gamma(a=params['a'], scale=params['scale'])
            samples = base.rvs(n_rows) + params.get('loc', 0)
        elif dist_name == 'lognorm':
            base = lognorm(s=params['s'], scale=params['scale'])
            samples = base.rvs(n_rows) + params.get('loc', 0)
        elif dist_name == 'norm':
            base = np.random.normal(loc=params['loc'], scale=params['scale'], size=n_rows)
            samples = base
        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")

        minval, maxval = info['range']

        # Sampling strategies
        if sampling_strategy == 'kde':
            kde = gaussian_kde(samples)
            empirical = kde.resample(n_rows).flatten()
        elif sampling_strategy == 'resample':
            empirical = np.random.choice(samples, size=n_rows, replace=True)
        elif sampling_strategy == 'local-chunks':
            if chunk_size is None:
                chunk_size = max(50, int(n_rows * 0.02))
            sorted_samps = np.sort(samples)
            empirical = np.zeros(n_rows)
            for j in range(0, n_rows, chunk_size):
                chunk = sorted_samps[j % len(sorted_samps):][:chunk_size]
                z_chunk = Z_corr[j:j+chunk_size, i]
                ranks = np.argsort(np.argsort(z_chunk))
                empirical[j:j+chunk_size] = np.sort(chunk)[ranks]
        else:
            empirical = np.sort(samples)

        # Rank-match
        ord_indices = Z_corr[:, i].argsort()
        inv_ord = np.argsort(ord_indices)
        values = empirical[inv_ord]

        # Histogram jitter
        if histogram_jitter:
            values += 0.01 * np.std(values) * np.sin(np.linspace(0, 12 * np.pi, n_rows))

        # Post-copula noise
        if noise_level > 0:
            std_val = np.std(values)
            if noise_type == 'laplace':
                noise = np.random.laplace(0, noise_level * std_val, n_rows)
            else:
                noise = np.random.normal(0, noise_level * std_val, n_rows)
            values += noise

        # Edge handling
        if edge_strategy == 'random':
            mask = (values < minval) | (values > maxval)
            values[mask] = np.random.uniform(minval, maxval, mask.sum())
        else:
            values = np.clip(values, minval, maxval)

        # Rounding
        if info.get('round', False):
            values = np.round(values)

        df_num[col] = values

    # Sample categorical columns
    df_cat = pd.DataFrame(index=range(n_rows))
    for cat in categorical_columns:
        vals, counts = np.unique(original_data[cat].dropna(), return_counts=True)
        props = counts / counts.sum()
        corrs = correlation_matrix.loc[cat, numeric_cols].abs()
        strong = corrs[corrs >= correlation_threshold]
        if not strong.empty:
            weights = strong / strong.sum()
            score = sum(Z_corr[:, numeric_cols.index(c)] * w for c, w in weights.items())
            if conditional_method == 'quantile':
                cum = np.cumsum(props)
                bins = np.quantile(score, cum[:-1])
                df_cat[cat] = pd.cut(score, bins=[-np.inf, *bins, np.inf], labels=vals)
            else:
                invf = 1 / props
                logits = (invf - invf.mean()) / invf.std() * 2
                s_scaled = (score - score.min()) / (score.max() - score.min())
                pmat = softmax(np.outer(s_scaled, logits), axis=1)
                df_cat[cat] = [np.random.choice(vals, p=p) for p in pmat]
        else:
            df_cat[cat] = np.random.choice(vals, size=n_rows, p=props)

    return pd.concat([df_num, df_cat], axis=1)


def plot_correlation_matrices(corr1: pd.DataFrame, corr2: pd.DataFrame, title1: str = "Matrix 1", title2: str = "Matrix 2"):
    """
    Plot two correlation matrices side by side with hoverable heatmaps.
    
    Args:
        corr1 (pd.DataFrame): First correlation matrix.
        corr2 (pd.DataFrame): Second correlation matrix.
        title1 (str): Title for the first heatmap.
        title2 (str): Title for the second heatmap.
    """
    if corr1.shape != corr2.shape or not all(corr1.columns == corr2.columns):
        raise ValueError("Correlation matrices must have the same shape and column labels.")
    
    features = corr1.columns.tolist()
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=(title1, title2), horizontal_spacing=0.2, vertical_spacing=0.2)

    # First heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr1.values,
            x=features,
            y=features,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            colorbar=dict(title="Correlation", len=0.9),
            hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>"
        ),
        row=1, col=1
    )

    # Second heatmap
    fig.add_trace(
        go.Heatmap(
            z=corr2.values,
            x=features,
            y=features,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            showscale=False,
            hovertemplate="X: %{x}<br>Y: %{y}<br>Corr: %{z:.2f}<extra></extra>"
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text="Comparison of Correlation Matrices",
        height=800,
        width=1400,
        showlegend=False
    )

    fig.show()
