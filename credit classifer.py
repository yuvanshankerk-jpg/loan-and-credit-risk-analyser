import antigravity
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import joblib

# ---------------------------------------------------------
# 1. GENERATE SYNTHETIC DATASET (Mimicking Kaggle Loan Data)
# ---------------------------------------------------------
def generate_sample_data(n_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    income = np.random.normal(loc=50000, scale=18000, size=n_samples).clip(15000, 150000)
    loan_amount = np.random.normal(loc=15000, scale=7000, size=n_samples).clip(2000, 50000)
    credit_score = np.random.normal(loc=680, scale=80, size=n_samples).clip(300, 850)
    age = np.random.randint(21, 65, size=n_samples)
    
    employment_status = np.random.choice(["Employed", "Self-Employed", "Unemployed"], size=n_samples, p=[0.7, 0.2, 0.1])
    education = np.random.choice(["Graduate", "Not Graduate"], size=n_samples, p=[0.75, 0.25])
    loan_purpose = np.random.choice(["Home", "Education", "Debt_Consolidation", "Personal"], size=n_samples)
    
    # Calculate a synthetic probability of default / rejection based on key risk factors
    # High debt-to-income ratio and low credit score increase rejection chance
    dti = loan_amount / income
    risk_score = (
        (dti * 3.0) 
        + (850 - credit_score) / 150 
        + (employment_status == "Unemployed") * 2.0 
        - (education == "Graduate") * 0.4
    )
    
    prob_approved = 1 / (1 + np.exp(risk_score - 3.2))
    loan_status = np.random.binomial(1, prob_approved)  # 1 = Approved, 0 = Rejected

    df = pd.DataFrame({
        "Age": age,
        "Income": np.round(income, 2),
        "LoanAmount": np.round(loan_amount, 2),
        "CreditScore": np.round(credit_score),
        "EmploymentStatus": employment_status,
        "Education": education,
        "LoanPurpose": loan_purpose,
        "LoanStatus": loan_status
    })
    
    # Introduce random missing values to test preprocessing imputers
    mask = np.random.rand(*df.shape) < 0.03
    df = df.mask(mask & (df.columns != "LoanStatus"))
    
    return df

# ---------------------------------------------------------
# 2. BUILD MODEL PIPELINE
# ---------------------------------------------------------
def create_loan_pipeline(numerical_cols, categorical_cols):
    # Numerical: Impute median, then standardize
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    # Categorical: Impute most frequent, then one-hot encode
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    # Bundle preprocessing
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transformer, numerical_cols),
        ("cat", cat_transformer, categorical_cols)
    ])
    
    # Full pipeline with Random Forest Classifier
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150, 
            max_depth=7, 
            class_weight="balanced", 
            random_state=42
        ))
    ])
    
    return pipeline

# ---------------------------------------------------------
# 3. TRAINING & EVALUATION
# ---------------------------------------------------------
def train_and_evaluate():
    print("Generating dataset...")
    df = generate_sample_data(n_samples=1200)
    
    # Feature / Target split
    X = df.drop(columns=["LoanStatus"])
    y = df["LoanStatus"]
    
    numerical_cols = ["Age", "Income", "LoanAmount", "CreditScore"]
    categorical_cols = ["EmploymentStatus", "Education", "LoanPurpose"]
    
    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples | Testing set: {X_test.shape[0]} samples")
    
    # Build & Train
    model = create_loan_pipeline(numerical_cols, categorical_cols)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # Save Model Artifact
    model_filename = "loan_classifier_model.joblib"
    joblib.dump(model, model_filename)
    print(f"\nModel pipeline saved to '{model_filename}'")
    
    return model

# ---------------------------------------------------------
# 4. INFERENCE DEMO
# ---------------------------------------------------------
def predict_loan_application(model, application_data: dict):
    input_df = pd.DataFrame([application_data])
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    status = "APPROVED" if prediction == 1 else "REJECTED / HIGH RISK"
    
    return {
        "Status": status,
        "ApprovalProbability": round(float(probability), 4),
        "RiskScore": round(float(1 - probability), 4)
    }

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    # Train model
    trained_model = train_and_evaluate()
    
    # Test with sample profiles
    print("\n" + "="*50)
    print("TESTING INFERENCE")
    print("="*50)
    
    # High-risk profile
    high_risk_profile = {
        "Age": 24,
        "Income": 25000.0,
        "LoanAmount": 30000.0,
        "CreditScore": 540.0,
        "EmploymentStatus": "Unemployed",
        "Education": "Not Graduate",
        "LoanPurpose": "Personal"
    }
    
    # Low-risk profile
    low_risk_profile = {
        "Age": 38,
        "Income": 95000.0,
        "LoanAmount": 12000.0,
        "CreditScore": 760.0,
        "EmploymentStatus": "Employed",
        "Education": "Graduate",
        "LoanPurpose": "Home"
    }
    
    print("High Risk Profile Result:", predict_loan_application(trained_model, high_risk_profile))
    print("Low Risk Profile Result: ", predict_loan_application(trained_model, low_risk_profile))