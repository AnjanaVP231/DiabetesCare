
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import os
import joblib
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user,
    logout_user, login_required, current_user
)
from flask_bcrypt import Bcrypt
from flask_wtf.csrf import CSRFProtect
import mimetypes
from datetime import datetime
from zoneinfo import ZoneInfo


mimetypes.add_type("text/css", ".css", True)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "diabetes_rf_pipeline.pkl"

# Flask configuration
app = Flask(__name__)


app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "temporary-development-key"
)
app.config["SQLALCHEMY_DATABASE_URI"] = (
    "sqlite:///" + str(BASE_DIR / "database" / "diabetes.db")
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
csrf = CSRFProtect(app)

login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"

# Load ML model
model = joblib.load(MODEL_PATH)
print("Diabetes ML model loaded successfully!")

# Initialize SHAP explainer

SHAP_DATA_PATH = BASE_DIR / "models" / "diabetes.csv"

imputer = model.named_steps["imputer"]
rf_model = model.named_steps["model"]

feature_names = list(model.feature_names_in_)

# Load background dataset
shap_data = pd.read_csv(SHAP_DATA_PATH)

X_background = shap_data[feature_names].copy()

# Handle invalid zero values
missing_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X_background[missing_columns] = (
    X_background[missing_columns].replace(0, np.nan)
)

# Apply trained imputer
transformed_background = imputer.transform(X_background)

shap_feature_names = imputer.get_feature_names_out(feature_names)

background = pd.DataFrame(
    transformed_background,
    columns=shap_feature_names
)

background = background.sample(
    n=100,
    random_state=42
)

# Create SHAP explainer
shap_explainer = shap.TreeExplainer(
    rf_model,
    data=background,
    feature_perturbation="interventional",
    model_output="probability"
)

print("SHAP explainer loaded successfully.")


def generate_counterfactual(input_data):

    # Load original dataset features
    X = shap_data[feature_names].copy()

    missing_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    X[missing_columns] = X[missing_columns].replace(0, np.nan)

    # Find predictions opposite to the user's prediction
    current_prediction = int(model.predict(input_data)[0])
    candidate_predictions = model.predict(X)

    opposite_indices = np.where(
        candidate_predictions != current_prediction
    )[0]

    if len(opposite_indices) == 0:
        return None

    candidates = X.iloc[opposite_indices]

    # Impute values using the trained pipeline's imputer
    sample_imputed = imputer.transform(input_data)[:, :8][0]
    candidates_imputed = imputer.transform(candidates)[:, :8]

    # Calculate IQR-based normalized distances
    all_imputed = imputer.transform(X)[:, :8]

    scale = (
        np.percentile(all_imputed, 75, axis=0)
        - np.percentile(all_imputed, 25, axis=0)
    )

    scale[scale == 0] = 1

    distances = np.sqrt(
        np.mean(
            ((candidates_imputed - sample_imputed) / scale) ** 2,
            axis=1
        )
    )

    # Select nearest opposite-prediction example
    best_index = int(np.argmin(distances))

    counterfactual_values = candidates_imputed[best_index]

    changes = []

    for feature, old, new in zip(
        feature_names,
        sample_imputed,
        counterfactual_values
    ):
        if not np.isclose(old, new):
            changes.append({
                "feature": feature,
                "current": float(old),
                "counterfactual": float(new)
            })

    return {
        "prediction": int(candidate_predictions[opposite_indices[best_index]]),
        "changes": changes,
        "distance": float(distances[best_index])
    }

# User database table
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)

# Prediction history table
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("user.id"), nullable=False
    )
    result = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(
        db.DateTime,
        default=lambda: datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
    )

# Load logged-in user
@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

# Home page
@app.route("/")
def home():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

# Registration
@app.route("/register", methods=["GET", "POST"])
def register():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not email or not password:
            flash("All fields are required.", "danger")

        elif len(username) > 100 or len(email) > 150:
            flash("Username or email is too long.", "danger")

        elif "@" not in email or "." not in email.split("@")[-1]:
            flash("Enter a valid email address.", "danger")

        elif len(password) < 8:
            flash("Password must contain at least 8 characters.", "danger")

        elif password != confirm_password:
            flash("Passwords do not match.", "danger")

        elif User.query.filter_by(username=username).first():
            flash("Username already exists.", "danger")

        elif User.query.filter_by(email=email).first():
            flash("Email already registered.", "danger")

        else:
            hashed_password = bcrypt.generate_password_hash(
                password
            ).decode("utf-8")

            new_user = User(
                username=username,
                email=email,
                password=hashed_password
            )

            db.session.add(new_user)
            db.session.commit()

            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")

# Login
@app.route("/login", methods=["GET", "POST"])
def login():

    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":

        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if user and bcrypt.check_password_hash(
            user.password, password
        ):
            login_user(user)
            flash("Welcome back!", "success")

            next_page = request.args.get("next")

            if next_page and next_page.startswith("/") and not next_page.startswith("//"):
                return redirect(next_page)

            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")

# Dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():

    if request.method == "POST":

        try:
            # Collect input values in the model's expected order
            values = {
                "Pregnancies": int(request.form["pregnancies"]),
                "Glucose": float(request.form["glucose"]),
                "BloodPressure": float(request.form["blood_pressure"]),
                "SkinThickness": float(request.form["skin_thickness"]),
                "Insulin": float(request.form["insulin"]),
                "BMI": float(request.form["bmi"]),
                "DiabetesPedigreeFunction": (
                    float(request.form["dpf"])
                    if request.form.get("dpf", "").strip()
                    else np.nan
                ),
                "Age": int(request.form["age"])
            }

            
            # Basic validation
            limits = {
                "Pregnancies": (0, 20),
                "Glucose": (0, 300),
                "BloodPressure": (0, 200),
                "SkinThickness": (0, 100),
                "Insulin": (0, 1000),
                "BMI": (0, 100),
                "Age": (1, 120)
            }

            for feature, (minimum, maximum) in limits.items():
                value = values[feature]

                if not np.isfinite(value) or value < minimum or value > maximum:
                    raise ValueError(
                        f"{feature} must be between {minimum} and {maximum}."
                    )

            # Validate Diabetes Pedigree Function
            dpf = values["DiabetesPedigreeFunction"]

            if not np.isnan(dpf):
                if not np.isfinite(dpf) or dpf < 0 or dpf > 3:
                    raise ValueError("DPF must be between 0 and 3.")

            # Prepare input for the Random Forest pipeline
            input_data = pd.DataFrame([values])

            # Treat invalid zero measurements as missing values
            missing_columns = [
                "Glucose",
                "BloodPressure",
                "SkinThickness",
                "Insulin",
                "BMI"
            ]

            for column in missing_columns:
                if input_data.loc[0, column] == 0:
                    input_data.loc[0, column] = np.nan

            # Generate prediction and probability
            prediction = int(model.predict(input_data)[0])
            probabilities = model.predict_proba(input_data)[0]

            positive_index = list(rf_model.classes_).index(1)
            probability = float(probabilities[positive_index])

            # Generate SHAP explanation
            transformed_input = imputer.transform(input_data)

            shap_input = pd.DataFrame(
                transformed_input,
                columns=shap_feature_names
            )

            shap_result = shap_explainer(shap_input)

            # Extract contributions for the positive class
            shap_values = shap_result.values[0, :, positive_index]

            base_probability = float(
                shap_result.base_values[0, positive_index]
            )

            # Prepare readable feature explanations
            shap_explanations = []

            for feature, contribution in zip(shap_feature_names, shap_values):

                display_name = feature.replace(
                    "missingindicator_",
                    "Missing value indicator: "
                )

                shap_explanations.append({
                    "feature": display_name,
                    "value": float(contribution)
                })

            # Show the most influential features first
            shap_explanations.sort(
                key=lambda item: abs(item["value"]),
                reverse=True
            )

            # Generate counterfactual explanation
            counterfactual = generate_counterfactual(input_data)
            
            # Store prediction in the database
            new_prediction = Prediction(
                user_id=current_user.id,
                result=prediction,
                probability=probability
            )

            db.session.add(new_prediction)
            db.session.commit()

            return render_template(
                "result.html",
                prediction=prediction,
                probability=probability,
                family_history=request.form.getlist("family_history"),
                dpf_imputed=np.isnan(values["DiabetesPedigreeFunction"]),
                shap_explanations=shap_explanations,
                base_probability=base_probability ,
                counterfactual=counterfactual
            )

        except (ValueError, TypeError, KeyError) as error:
            flash(str(error), "error")

        except Exception:
            db.session.rollback()
            flash("Something went wrong while generating the prediction.", "error")

    return render_template("predict.html")


@app.route("/history")
@login_required
def history():

    predictions = Prediction.query.filter_by(
        user_id=current_user.id
    ).order_by(
        Prediction.created_at.desc()
    ).all()

    return render_template(
        "history.html",
        predictions=predictions
    )

# Logout
@app.route("/logout", methods=["POST"])
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

# Application health check
@app.route("/health")
def health():
    return {
        "application": "running",
        "model_loaded": model is not None,
        "prediction_available": hasattr(model, "predict"),
        "probability_available": hasattr(model, "predict_proba")
    }

if __name__ == "__main__":

    with app.app_context():
        db.create_all()

    app.run(debug=True)