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
from distfit import distfit
import copy
import json


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

        plt.xlabel('Price')
        plt.ylabel('Frequency')
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
            plt.xlabel('Price')
            plt.ylabel('Frequency')
            plt.title(f'{label} Distribution')
            plt.legend()
            plt.xlim(global_min, global_max)
            plt.ylim(0, max_freq)

        plt.tight_layout()
        plt.show()


def plot_densities(
        datasets: list[tuple[ArrayLike, str, str]],
        rows: int = 1,
        cols: int = None,
        figsize_per_plot: tuple[int, int] = (5, 5),
        stack: bool = False,
        bw_adjust: float = 1.0
    ):

    n = len(datasets)

    global_min = min(data.min() for data, _, _ in datasets)
    global_max = max(data.max() for data, _, _ in datasets)

    x_vals = np.linspace(global_min, global_max, 1000)
    max_density = 0
    for data, _, _ in datasets:
        kde = sns.kdeplot(data, bw_adjust=bw_adjust).get_lines()[0].get_data()
        max_density = max(max_density, max(kde[1]))
    plt.clf()

    if stack:
        plt.figure(figsize=figsize_per_plot)
        for data, label, color in datasets:
            sns.kdeplot(data, fill=True, color=color, label=label, bw_adjust=bw_adjust)

        plt.xlabel('Price')
        plt.ylabel('Density')
        plt.title('Stacked Density Plot of All Distributions')
        plt.legend()
        plt.xlim(global_min, global_max)
        plt.ylim(0, max_density)
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
            sns.kdeplot(data, fill=True, color=color, label=label, bw_adjust=bw_adjust)
            plt.xlabel('Price')
            plt.ylabel('Density')
            plt.title(f'{label} Density')
            plt.legend()
            plt.xlim(global_min, global_max)
            plt.ylim(0, max_density)

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
        print(f"Processing column: {col}")
        
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
            col_min = dataset[col].min().round(3)
            col_max = dataset[col].max().round(3)
            col_details = result['parameters']
            if col_details['name'] == 'norm':
                statement = (f'Column \"{col}\": distribution = gaussian, mean = {(col_details['params'][0]).round(3)}, deviation = {(col_details['params'][1]).round(3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
            elif col_details['name'] == 'lognorm':
                statement = (f'Column \"{col}\": distribution = lognorm, mean = {(np.log(col_details['params'][2])).round(3)}, deviation = {(col_details['params'][0]).round(3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
            elif col_details['name'] == 'gamma':
                statement = (f'Column \"{col}\": distribution = gaussian, alpha = {(col_details['params'][0]).round(3)}, theta = {(col_details['params'][2]).round(3)}, range = {col_min} to {col_max}')
                if col in discrete_columns: statement += ' (with only whole numbers allowed)'
            distr_statements.append(statement)
    return distr_statements
