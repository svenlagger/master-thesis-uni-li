import pathlib, time, warnings, re
import numpy as np, pandas as pd
from scipy import stats
from sdmetrics.reports.single_table import QualityReport

###############################################################################
# 1) CONFIG – your paths & metadata remain unchanged
###############################################################################
BASE_SYNTH_DIR = pathlib.Path('denoising/data/_synth_samples')

REAL_DATA = {
    'credit_score':   pathlib.Path('denoising/data/credit_score_cleaned.csv'),
    'air_quality':    pathlib.Path('denoising/data/air_quality_cleaned.csv'),
    'customer_churn': pathlib.Path('denoising/data/customer_churn.csv'),
    'insurance':      pathlib.Path('denoising/data/insurance_original.csv'),
    'real_estate':    pathlib.Path('denoising/data/real_estate_valuation_cleaned.csv')
}

PREPROCESS_DROP = {
    'credit_score': ['ID', 'Customer_ID', 'Name', 'SSN',
                     'Type_of_Loan', 'Payment_Behaviour'],
    'air_quality':  ['Date', 'Time']
}

# Paste the SDMetrics metadata *dict* for each table here ---------------------
METADATA = {
    'credit_score': {
        "columns": {
            "Month": {"sdtype": "categorical"},
            "Age": {"sdtype": "numerical"},
            "Occupation": {"sdtype": "categorical"},
            "Annual_Income": {"sdtype": "numerical"},
            "Monthly_Inhand_Salary": {"sdtype": "numerical"},
            "Num_Bank_Accounts": {"sdtype": "numerical"},
            "Num_Credit_Card": {"sdtype": "numerical"},
            "Interest_Rate": {"sdtype": "numerical"},
            "Num_of_Loan": {"sdtype": "numerical"},
            "Delay_from_due_date": {"sdtype": "numerical"},
            "Num_of_Delayed_Payment": {"sdtype": "numerical"},
            "Changed_Credit_Limit": {"sdtype": "numerical"},
            "Num_Credit_Inquiries": {"sdtype": "numerical"},
            "Credit_Mix": {"sdtype": "categorical"},
            "Outstanding_Debt": {"sdtype": "numerical"},
            "Credit_Utilization_Ratio": {"sdtype": "numerical"},
            "Credit_History_Age": {"sdtype": "numerical"},
            "Payment_of_Min_Amount": {"sdtype": "categorical"},
            "Total_EMI_per_month": {"sdtype": "numerical"},
            "Amount_invested_monthly": {"sdtype": "numerical"},
            "Monthly_Balance": {"sdtype": "numerical"},
        }
    },
    'air_quality': {
        "columns": {
            "CO(GT)": {"sdtype": "numerical"},
            "PT08.S1(CO)": {"sdtype": "numerical"},
            "C6H6(GT)": {"sdtype": "numerical"},
            "PT08.S2(NMHC)": {"sdtype": "numerical"},
            "NOx(GT)": {"sdtype": "numerical"},
            "PT08.S3(NOx)": {"sdtype": "numerical"},
            "NO2(GT)": {"sdtype": "numerical"},
            "PT08.S4(NO2)": {"sdtype": "numerical"},
            "PT08.S5(O3)": {"sdtype": "numerical"},
            "T": {"sdtype": "numerical"},
            "RH": {"sdtype": "numerical"},
            "AH": {"sdtype": "numerical"}
        }
    },
    'customer_churn': {
        "columns": {
            "Call Failure": {"sdtype": "numerical"},
            "Complains": {"sdtype": "numerical"},
            "Subscription Length": {"sdtype": "numerical"},
            "Charge Amount": {"sdtype": "numerical"},
            "Seconds of Use": {"sdtype": "numerical"},
            "Frequency of use": {"sdtype": "numerical"},
            "Frequency of SMS": {"sdtype": "numerical"},
            "Distinct Called Numbers": {"sdtype": "numerical"},
            "Age Group": {"sdtype": "numerical"},
            "Tariff Plan": {"sdtype": "numerical"},
            "Status": {"sdtype": "numerical"},
            "Age": {"sdtype": "numerical"},
            "Customer Value": {"sdtype": "numerical"},
            "Churn": {"sdtype": "numerical"}
        }
    },
    'insurance': {
        "columns": {
            "age": {"sdtype": "numerical"},
            "sex": {"sdtype": "categorical"},
            "bmi": {"sdtype": "numerical"},
            "children": {"sdtype": "numerical"},
            "smoker": {"sdtype": "categorical"},
            "region": {"sdtype": "categorical"},
            "charges": {"sdtype": "numerical"},
        }
    },
    'real_estate': {
        "columns": {
            "X1 house age": {"sdtype": "numerical"},
            "X2 distance MRT station": {"sdtype": "numerical"},
            "X3 number convenience stores": {"sdtype": "numerical"},
            "X4 lat": {"sdtype": "numerical"},
            "X5 long": {"sdtype": "numerical"},
            "X6 price": {"sdtype": "numerical"}
        }
    }
}


###############################################################################
# 2)  NEW HELPERS  ────────────────────────────────────────────────────────────
###############################################################################
def canon(col: str) -> str:
    """Canonical form: lower-case, no underscores, no spaces."""
    return re.sub(r'[\s_]+', '', col).lower()

def rename_tabula(df: pd.DataFrame, target_cols) -> pd.DataFrame:
    """
    TabuLa removes spaces (and often adds underscores).  Build a 1-to-1 map
    from the *canonical* version in df.columns to the original target_cols,
    then rename.  Columns that already match are left untouched.
    """
    mapper = {}
    remaining = set(target_cols)
    for col in df.columns:
        matches = [t for t in remaining if canon(t) == canon(col)]
        if matches:
            mapper[col] = matches[0]
            remaining.remove(matches[0])

    if mapper:                                           # do nothing if empty
        df = df.rename(columns=mapper)                   # pandas one-liner :contentReference[oaicite:1]{index=1}
    return df

def mean_ci(vals, alpha=.05):
    vals = np.asarray(vals, float)
    h = stats.sem(vals) * stats.t.ppf(1 - alpha/2, len(vals)-1)
    return vals.mean(), h

def quality_scores(real_df, synth_df, metadata):
    rpt = QualityReport()
    rpt.generate(real_df, synth_df, metadata, verbose=False)
    props = rpt.get_properties()
    if 'Property' in props.columns:                      # index fix
        props = props.set_index('Property')
    return (rpt.get_score(),
            props.loc['Column Shapes',      'Score'],
            props.loc['Column Pair Trends', 'Score'])

###############################################################################
# 3)  MAIN LOOP  (unchanged except for one call to rename_tabula)
###############################################################################
start = time.time()
rows  = []
model_folders = {
    'SDV-CTGAN':      '_ctgan',
    'SDV-Copula':     '_sdv-copula',
    'TabuLa':         '_tabula',
    'Copula Cloning': '_copula-cloning'
}

for dataset, real_path in REAL_DATA.items():
    print(f'\n=== DATASET: {dataset} ===')
    real_df = pd.read_csv(real_path).drop(PREPROCESS_DROP.get(dataset, []), axis=1)
    expected_cols = list(METADATA[dataset]['columns'])
    real_df = real_df[expected_cols]                     # enforce column order

    scores = {m: {'overall': [], 'col': [], 'pair': []} for m in model_folders}

    for model, folder in model_folders.items():
        synth_dir   = BASE_SYNTH_DIR / folder / dataset
        synth_files = list(synth_dir.glob('*.csv'))
        print(f'  {model}: {len(synth_files)} files')

        for f in synth_files:
            print(f'    → {f.name}', end=' ... ', flush=True)
            synth_df = pd.read_csv(f)

            # ──► NEW: restore spaces for TabuLa-generated headers
            if model == 'TabuLa' and dataset in ('real_estate', 'customer_churn'):
                synth_df = rename_tabula(synth_df, expected_cols)

            # Check column set
            missing = [c for c in expected_cols if c not in synth_df.columns]
            if missing:
                print(f'SKIPPED (missing {len(missing)} cols)')
                continue

            synth_df = synth_df[expected_cols]

            try:
                o, c, p = quality_scores(real_df, synth_df, METADATA[dataset])
                scores[model]['overall'].append(o)
                scores[model]['col'].append(c)
                scores[model]['pair'].append(p)
                print('OK')
            except Exception as err:
                print(f'FAILED ({err})')

    for label, key in [('Columns Shapes', 'col'),
                       ('Column Pair Trends', 'pair'),
                       ('Overall',           'overall')]:
        row = {'Dataset / Metric':
               f'{dataset.replace("_"," ").title()}\n{label}'}
        for model in model_folders:
            vals = scores[model][key]
            row[model] = (f'{mean_ci(vals)[0]*100:5.2f}% ± {mean_ci(vals)[1]*100:4.2f}%'
                          if vals else '—')
        rows.append(row)

###############################################################################
# 4)  RESULTS
###############################################################################
table = (pd.DataFrame(rows)
         .set_index('Dataset / Metric')
         .rename_axis(None, axis=1))

pd.set_option('display.width', 200)
print('\n\nRESULT TABLE\n-------------')
print(table)
print(f"\nFinished in {time.time()-start:,.1f} s")

