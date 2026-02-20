import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

import structural_penalties_icp
from structural_penalties_icp import ICPWrapper, InductiveConformalPredictor



print("Successfully imported structural_penalties_icp!")
print(f"Package Version: {structural_penalties_icp.__version__}")

# 1. Load your data

print("Loading yeast dataset...")
X = pd.read_csv("data/yeast/X_yeast.csv")
y = pd.read_csv("data/yeast/y_yeast.csv")

# First, separate out the Test set (10%)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Then, split the remaining data into Proper Train and Calibration (30%)
X_train, X_calib, y_train, y_calib = train_test_split(X_temp, y_temp, test_size=0.3, random_state=42)

# 2. Initialize your class to prove it works
print("Initializing ICP Wrapper...")
base_model = MultiOutputClassifier(RandomForestClassifier())
wrapper = ICPWrapper(base_model, weight_hamming=2.0, weight_cardinality=1.5, device='cpu')
args = {'estimator__n_estimators': 5}
wrapper.fit(X_train, y_train, **args)

print("It works perfectly!")