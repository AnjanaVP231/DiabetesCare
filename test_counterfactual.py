
import joblib
import pandas as pd
import numpy as np

# Load model and dataset
model = joblib.load("models/diabetes_rf_pipeline.pkl")
data = pd.read_csv("models/diabetes.csv")

features = list(model.feature_names_in_)
imputer = model.named_steps["imputer"]

# Prepare dataset
X = data[features].copy()

missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[missing_columns] = X[missing_columns].replace(0, np.nan)

# Example user input
sample = pd.DataFrame([{
    "Pregnancies": 2,
    "Glucose": 120,
    "BloodPressure": 70,
    "SkinThickness": 25,
    "Insulin": 80,
    "BMI": 28,
    "DiabetesPedigreeFunction": np.nan,
    "Age": 30
}])

# Get current prediction
current_prediction = int(model.predict(sample)[0])

# Get model predictions for dataset records
candidate_predictions = model.predict(X)

# Find records with opposite predictions
opposite_records = X[
    candidate_predictions != current_prediction
].copy()

if opposite_records.empty:
    print("No opposite prediction examples found.")

else:
    # Apply the trained imputer
    sample_imputed = imputer.transform(sample)[:, :8][0]

    candidates_imputed = imputer.transform(
        opposite_records
    )[:, :8]

    # Normalize distances using feature IQR
    scale = np.percentile(
        imputer.transform(X)[:, :8], 75, axis=0
    ) - np.percentile(
        imputer.transform(X)[:, :8], 25, axis=0
    )

    scale[scale == 0] = 1

    distances = np.sqrt(
        np.mean(
            ((candidates_imputed - sample_imputed) / scale) ** 2,
            axis=1
        )
    )

    # Select nearest opposite prediction
    best_index = np.argmin(distances)

    counterfactual = candidates_imputed[best_index]

    print("\nCurrent prediction:", current_prediction)
    print("Counterfactual prediction:", 1 - current_prediction)

    print("\nNearest counterfactual example:")

    for feature, old, new in zip(
        features, sample_imputed, counterfactual
    ):
        if not np.isclose(old, new):
            print(f"{feature}: {old:.2f} -> {new:.2f}")

    print("\nNormalized distance:", round(distances[best_index], 4))