# Import necessary libraries
import mlflow
import mlflow.sklearn
import dagshub  # For logging to DagsHub
import pandas as pd
import joblib  # For saving models
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 🔹 1. Initialize DagsHub for MLflow Logging
dagshub.init(repo_owner='8754148482azar', repo_name='mlflow_with_dagshub', mlflow=True)

# 🔹 2. Set DagsHub MLflow Tracking URI
MLFLOW_TRACKING_URI = "https://dagshub.com/8754148482azar/mlflow_with_dagshub.mlflow"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Athlete_Injury_Risk")

# 🔹 3. Load the dataset
vald_data = pd.read_csv("data/vald_data_for_modelling.csv")

# 🔹 4. Define features and target variable
risk_mapping = {'Low Risk': 0, 'Medium Risk': 1, 'High Risk': 2}
X = vald_data[['ForceSymmetry', 'MaxForceSymmetry', 'TorqueSymmetry']]  # Features
y = vald_data['RiskCategory'].map(risk_mapping)  # Encode RiskCategory to numerical values

# 🔹 5. Function to Train, Log, and Register Models
def train_and_log_model(model_name, model, X_train, X_test, y_train, y_test):
    """
    Trains the model, logs all necessary details to MLflow via DagsHub, and registers the best model.
    """
    with mlflow.start_run(run_name=f"{model_name}_Run"):  # Unique run name for each model
        model.fit(X_train, y_train)  # Train model
        y_pred = model.predict(X_test)  # Make predictions
        accuracy = accuracy_score(y_test, y_pred)  # Compute accuracy

        # 🔹 5.1 Log Parameters and Metrics
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)

        # 🔹 5.2 Log Classification Report Metrics
        report = classification_report(y_test, y_pred, target_names=risk_mapping.keys(), output_dict=True)
        for category, metrics in report.items():
            if isinstance(metrics, dict):  # Ignore overall accuracy keys
                mlflow.log_metric(f"precision_{category}", metrics["precision"])
                mlflow.log_metric(f"recall_{category}", metrics["recall"])
                mlflow.log_metric(f"f1_{category}", metrics["f1-score"])

        # 🔹 5.3 Save and Log Model
        mlflow.sklearn.log_model(model, model_name)  # Log model in MLflow (DagsHub)
        joblib.dump(model, f"models/{model_name}.pkl")  # Save model locally for deployment

        # 🔹 5.4 Register Best Model
        if accuracy > 0.95:  # Change threshold if needed
            mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
                name="Athlete_Injury_Risk_Model"
            )

        print(f"{model_name} logged with Accuracy: {accuracy:.4f}")

# 🔹 6. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 7. Train Different Models

# Model 1: SMOTE Oversampling (Handles Class Imbalance)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)  # Oversample minority class
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)
rf_model_1 = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_log_model("RF_SMOTE", rf_model_1, X_train_s, X_test_s, y_train_s, y_test_s)

# Model 2: No Balancing (Train on Raw Data)
rf_model_2 = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_log_model("RF_No_Balancing", rf_model_2, X_train, X_test, y_train, y_test)

# Model 3: SMOTEENN (Hybrid Resampling for Noise Reduction)
smoteenn = SMOTEENN(random_state=42)
X_resampled_se, y_resampled_se = smoteenn.fit_resample(X, y)  # SMOTE + Edited Nearest Neighbors
X_train_se, X_test_se, y_train_se, y_test_se = train_test_split(X_resampled_se, y_resampled_se, test_size=0.2, stratify=y_resampled_se, random_state=42)
rf_model_3 = RandomForestClassifier(n_estimators=100, random_state=42)
train_and_log_model("RF_SMOTEENN", rf_model_3, X_train_se, X_test_se, y_train_se, y_test_se)

# Model 4: Class Weights (Handles Imbalance via Weight Adjustments)
rf_model_4 = RandomForestClassifier(n_estimators=100, random_state=42, class_weight={0: 1, 1: 2, 2: 3})
train_and_log_model("RF_Class_Weights", rf_model_4, X_train, X_test, y_train, y_test)

print("✅ All models trained, logged, and saved successfully in DagsHub!")
