import pandas as pd
import os
from pysr import PySRRegressor
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
from collections import defaultdict


# Define a reusable type alias
ArrayLike = Union[list, np.ndarray, pd.Series]


def perform_simple_sr(
        dataset: pd.DataFrame, 
        independent_vars: list, 
        dependent_var: str, 
        n_iterations: int=128, 
        maxsize: int=40
    ):

    os.environ["JULIA_NUM_THREADS"] = "8"  # Use 8 threads (adjust as needed)

    X = dataset[independent_vars].to_numpy()  # Features
    y = dataset[dependent_var].to_numpy()  # Target

    # Use PySR to find the symbolic relationship
    model = PySRRegressor(
        niterations=n_iterations,  # Number of iterations to search for equations
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
        elementwise_loss="loss(x, y) = (x - y)^2",  # Define loss function (mean squared error)
        verbosity=1,
        maxsize=maxsize
    )

    # Fit the model
    model.fit(X, y)

    return model


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

    num_rounds = 5000
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
        datasets: list[tuple[ArrayLike, str, str]],
        xlabel: str,
        ylabel: str = 'Frequency',
        rows: int = 1,
        cols: int = None,
        bins: int = 30,
        figsize_per_plot: tuple[int, int] = (5, 5),
        stack: bool = False
    ):

    n = len(datasets)

    global_min = min(data.min() for data, _, _ in datasets)
    global_max = max(data.max() for data, _, _ in datasets)
    bin_edges = np.linspace(global_min, global_max, bins + 1)

    if stack:
        plt.figure(figsize=figsize_per_plot)
        max_freq = 0
        for data, label, color in datasets:
            freq, _ = np.histogram(data, bins=bin_edges)
            max_freq = max(max_freq, freq.max())
            plt.hist(data, bins=bin_edges, alpha=0.5, label=label, color=color)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title('Stacked Histogram of All Distributions')
        plt.legend()
        plt.xlim(global_min, global_max)
        plt.ylim(0, max_freq)
        plt.tight_layout()
        plt.show()

    else:
        if cols is None:
            cols = int(np.ceil(n / rows))

        fig_width = figsize_per_plot[0] * cols
        fig_height = figsize_per_plot[1] * rows
        plt.figure(figsize=(fig_width, fig_height))

        max_freq = 0
        for data, _, _ in datasets:
            freq, _ = np.histogram(data, bins=bin_edges)
            max_freq = max(max_freq, freq.max())

        for i, (data, label, color) in enumerate(datasets):
            plt.subplot(rows, cols, i + 1)
            plt.hist(data, bins=bin_edges, alpha=0.7, color=color, label=label)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(f'{label} Distribution')
            plt.legend()
            plt.xlim(global_min, global_max)
            plt.ylim(0, max_freq)

        plt.tight_layout()
        plt.show()


def plot_densities(
    datasets: list[tuple[ArrayLike, str, str]],
    xlabel: str,
    ylabel: str = 'Density',
    rows: int = 1,
    cols: int = None,
    figsize_per_plot: tuple[int, int] = (5, 5),
    stack: bool = False,
    bw_adjust: float = 1.0
):
    n = len(datasets)

    # Compute global min and max for x-axis
    global_min = min(np.min(data) for data, _, _ in datasets)
    global_max = max(np.max(data) for data, _, _ in datasets)
    x_vals = np.linspace(global_min, global_max, 1000)

    # Compute max density using gaussian_kde
    max_density = 0
    for data, _, _ in datasets:
        kde = gaussian_kde(data, bw_method=bw_adjust)
        density = kde(x_vals)
        max_density = max(max_density, np.max(density))

    # Plot stacked or individual
    if stack:
        fig, ax = plt.subplots(figsize=figsize_per_plot)
        for data, label, color in datasets:
            sns.kdeplot(data, fill=True, color=color, label=label, bw_adjust=bw_adjust, ax=ax)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title('Stacked Density Plot of All Distributions')
        # ax.set_xlim(global_min, global_max)
        # ax.set_ylim(0, max_density)
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

        # Hide unused subplots if any
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


# TODO add min-max values and handle discrete numeric columns
def get_distribution_statements(distr_report: dict, dataset: pd.DataFrame):
    discrete_columns = get_discrete_numerical_columns(dataset)
    distr_statements = []
    statement = None
    for col, result in distr_report.items():
        if result is not None:
            col_min = round(dataset[col].min(), 3)
            col_max = round(dataset[col].max(), 3)
            col_details = result['parameters']
            if col_details['name'] == 'norm':
                statement = (f'Column \"{col}\": distribution = gaussian, mean = {(col_details["params"][0]).round(3)}, deviation = {(col_details["params"][1]).round(3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
            elif col_details['name'] == 'lognorm':
                statement = (f'Column \"{col}\": distribution = lognorm, mean = {(np.log(col_details["params"][2])).round(3)}, deviation = {(col_details["params"][0]).round(3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
            elif col_details['name'] == 'gamma':
                statement = (f'Column \"{col}\": distribution = gamma, alpha = {(col_details["params"][0]).round(3)}, theta = {(col_details["params"][2]).round(3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
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
