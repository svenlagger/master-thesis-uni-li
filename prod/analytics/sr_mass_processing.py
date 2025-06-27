import os
import glob
import pandas as pd
from datetime import datetime
from pysr import PySRRegressor
import contextlib

# === Environment setup ===
# JuliaCall configuration for multithreading & signal handling
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_OPTLEVEL"] = "3"

# === File patterns under denoising/data ===
dataset_patterns = [
    # "denoising/data/real_estate_valuation_cleaned*.csv",
    "denoising/data/insurance_original_syn.csv"
    # "denoising/data/customer_churn*.csv",
    # "denoising/data/credit_score_cleaned*.csv",
    # "denoising/data/air_quality_cleaned*.csv",
]

# === Collect files ===
files = []
for pat in dataset_patterns:
    files.extend(sorted(glob.glob(pat)))
total_runs = len(files) * 10
print(f"Files identified! {total_runs} files in total to process...")

# === Main processing loop ===
run_counter = 0
for filepath in files:
    # Derive dataset name and output directory
    dataset_name = os.path.splitext(os.path.basename(filepath))[0]
    parent_dir = os.path.dirname(filepath)
    outdir = os.path.join(parent_dir, dataset_name)
    os.makedirs(outdir, exist_ok=True)

    # Load data
    data = pd.read_csv(filepath)
    # MOD: Drop rows with any NaN and print counts
    rows_before = data.shape[0]               # MOD
    data = data.dropna()                      # MOD
    rows_after = data.shape[0]                # MOD
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {dataset_name}: Dropped {rows_before - rows_after} rows with NaN, {rows_after} rows remaining.")  # MOD

    # Dataset-specific preprocessing
    if dataset_name.startswith("real_estate_valuation_cleaned"):
        indep = [
            'X2 distance MRT station',
            'X3 number convenience stores',
            'X4 lat'
        ]
        dep = 'X6 price'

    elif dataset_name.startswith("insurance_original"):
        # data['smoker'] = data['smoker'].map({'yes': 1, 'no': 0})
        indep = ['age', 'smoker']
        dep = 'charges'

    elif dataset_name.startswith("customer_churn"):
        indep = [
            'Distinct Called Numbers',
            'Frequency of SMS',
            'Frequency of use',
            'Seconds of Use'
        ]
        dep = 'Customer Value'

    elif dataset_name.startswith("credit_score_cleaned"):
        data = data.sample(n=9600, random_state=42)
        indep = [
            'Credit_History_Age',
            'Delay_from_due_date',
            'Num_Bank_Accounts',
            'Num_Credit_Inquiries',
            'Num_of_Delayed_Payment',
            'Outstanding_Debt'
        ]
        dep = 'Interest_Rate'

    elif dataset_name.startswith("air_quality_cleaned"):
        indep = ['AH', 'PT08.S4(NO2)', 'RH']
        dep = 'T'

    else:
        raise ValueError(f"Unrecognized dataset: {dataset_name}")

    X = data[indep].to_numpy()
    y = data[dep].to_numpy()

    # Run 10 Symbolic Regression fits per file
    for i in range(1, 11):
        run_counter += 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Starting run {i}/10 for {dataset_name} ({run_counter}/{total_runs})")

        sr = PySRRegressor(
            niterations=1024,
            binary_operators=["+", "-", "*", "/"],
            unary_operators=["log", "abs", "sqrt"],
            maxdepth=10,
            elementwise_loss="loss(x, y) = (x - y)^2",
            verbosity=0,
            maxsize=50,
            parallelism="multithreading",
            output_directory=outdir,
        )
        # Suppress Julia warnings (e.g., maxsize > 40 warning)
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stderr(devnull):
            sr.fit(X, y)

print("All runs complete!")
