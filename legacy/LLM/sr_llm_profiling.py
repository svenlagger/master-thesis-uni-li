import pandas as pd
import os
os.environ["JULIA_NUM_THREADS"] = "8"  # Use 8 threads (adjust as needed)
from pysr import PySRRegressor

df = pd.read_csv('C:/Users/svenl/vs_code_projects/hyperRealDataDescriber/data/real_estate/real_estate_valuation_cleaned.csv')

X = df[['X2 distance MRT station', 'X3 number convenience stores']].to_numpy()  # Features
y = df['X6 price'].to_numpy()  # Target

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