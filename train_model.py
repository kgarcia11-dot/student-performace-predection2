# train_model.py
# Student Performance Prediction System - Data Mining Laboratory
# Part 1: retrieve the dataset (Activity A)
# Part 2: preprocess, train and evaluate (Activity B)

import json
import pandas as pd
from ucimlrepo import fetch_ucirepo

# ---------------------------------------------------------------
# ACTIVITY A - Retrieve and inspect the dataset
# ---------------------------------------------------------------
# Fetch UCI Student Performance dataset (ID 320)
student = fetch_ucirepo(id=320)

# Combine student features and grade target
df = pd.concat(
    [student.data.features, student.data.targets],
    axis=1
)

# Remove any accidental duplicate column names (e.g., G1/G2 listed twice)
df = df.loc[:, ~df.columns.duplicated()]

# Save the records as a CSV file
df.to_csv("student_data.csv", index=False)

print("Rows and columns:", df.shape)
print(df.head())
print(df.dtypes)
print("Missing values per column:")
print(df.isnull().sum())

# Checkpoint: G1, G2 and G3 must exist
assert {"G1", "G2", "G3"}.issubset(df.columns), "G1, G2 and G3 are required."

# ---------------------------------------------------------------
# ACTIVITY B - Preprocess, train, and evaluate
# ---------------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Target: 1 = at risk (G3 < 10); 0 = not at risk
df = df.drop_duplicates()
y = (df["G3"] < 10).astype(int)

# Never use G3 as a predictor
X = df.drop(columns=["G3"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipe, numeric_cols),
    ("categorical", categorical_pipe, categorical_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", round(accuracy, 3))
print(classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0
))

joblib.dump(model, "student_model.pkl")
print("Saved student_model.pkl")

# Save the evaluation summary so the Streamlit app can display it
report = classification_report(
    y_test, predictions,
    target_names=["Not at risk", "At risk"],
    zero_division=0,
    output_dict=True
)
with open("model_metrics.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "precision_at_risk": report["At risk"]["precision"],
        "recall_at_risk": report["At risk"]["recall"],
        "f1_at_risk": report["At risk"]["f1-score"],
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }, f, indent=2)
print("Saved model_metrics.json")
