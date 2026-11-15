
import joblib
import shap
import pandas as pd
import numpy as np

# Load dataset and trained pipeline
data = pd.read_csv("models/diabetes.csv")
pipeline = joblib.load("models/diabetes_rf_pipeline.pkl")

imputer = pipeline.named_steps["imputer"]
rf_model = pipeline.named_steps["model"]

features = list(pipeline.feature_names_in_)

# Separate input features
X = data[features].copy()

# Replace invalid zero values with missing values
missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[missing_columns] = X[missing_columns].replace(0, np.nan)

# Apply the exact trained imputer
transformed = imputer.transform(X)

# Get all 13 feature names
transformed_features = imputer.get_feature_names_out(features)

X_imputed = pd.DataFrame(
    transformed,
    columns=transformed_features
)

# Select background data
background = X_imputed.sample(
    n=100,
    random_state=42
)

# Create SHAP explainer
explainer = shap.TreeExplainer(
    rf_model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability"
)

# Explain one sample
sample = X_imputed.iloc[[0]]
explanation = explainer(sample)

print("SHAP test successful!")
print("Number of features:", len(transformed_features))
print("SHAP values shape:", explanation.values.shape)
print("Base values:", explanation.base_values)
print("SHAP values:", explanation.values)