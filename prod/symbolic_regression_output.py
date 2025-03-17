import pandas as pd
import os
os.environ["JULIA_NUM_THREADS"] = "8"  # Use 8 threads (adjust as needed)
from pysr import PySRRegressor

df = pd.read_csv('C:/Users/svenl/vs_code_projects/hyperRealDataDescriber/data/credit_score/cleaned_credit_score.csv')

df_clean = df.dropna(subset=['Delay_from_due_date', 'Num_Bank_Accounts', 'Num_Credit_Inquiries', 'Interest_Rate'])
# df_clean = df_clean[df_clean['Interest_Rate'] <= 32]

df_sample = df_clean.sample(n=9600, random_state=42)

X = df_sample[['Credit_History_Age', 'Delay_from_due_date', 'Num_Bank_Accounts', 'Num_Credit_Inquiries', 'Num_of_Delayed_Payment', 'Outstanding_Debt']].to_numpy()  # Features: B and C
y = df_sample['Interest_Rate'].to_numpy()  # Target: A

# Use PySR to find the symbolic relationship
model = PySRRegressor(
    niterations=128,  # Number of iterations to search for equations
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
    elementwise_loss="loss(x, y) = (x - y)^2",  # Define loss function (mean squared error)
    verbosity=1,
    maxsize=40
)

# Fit the model
model.fit(X, y)

print(model)