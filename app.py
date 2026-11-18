import os
from flask import Flask, request, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load model using absolute path for Vercel serverless environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "loan_classifier_model.joblib"))

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Loan Risk Classifier</title>
    <style>
        body { font-family: system-ui, sans-serif; background: #0f172a; color: #f8fafc; display: flex; justify-content: center; padding: 40px 20px; }
        .card { background: #1e293b; padding: 30px; border-radius: 12px; width: 100%; max-width: 460px; box-shadow: 0 10px 25px rgba(0,0,0,0.4); }
        h2 { margin-top: 0; color: #38bdf8; }
        label { display: block; margin-top: 12px; font-size: 14px; color: #cbd5e1; }
        input, select { width: 100%; padding: 10px; margin-top: 5px; border-radius: 6px; border: 1px solid #334155; background: #0f172a; color: #fff; box-sizing: border-box; }
        button { width: 100%; margin-top: 20px; padding: 12px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); color: white; border: none; border-radius: 6px; font-weight: bold; cursor: pointer; }
        .result { margin-top: 20px; padding: 15px; border-radius: 8px; text-align: center; font-weight: bold; }
        .approved { background: #064e3b; color: #6ee7b7; }
        .rejected { background: #7f1d1d; color: #fca5a5; }
    </style>
</head>
<body>
    <div class="card">
        <h2>💳 Loan Approval Predictor</h2>
        <form method="POST">
            <label>Age</label>
            <input type="number" name="Age" value="30" required>
            
            <label>Annual Income ($)</label>
            <input type="number" step="any" name="Income" value="55000" required>
            
            <label>Loan Amount ($)</label>
            <input type="number" step="any" name="LoanAmount" value="15000" required>
            
            <label>Credit Score (300 - 850)</label>
            <input type="number" name="CreditScore" min="300" max="850" value="680" required>
            
            <label>Employment Status</label>
            <select name="EmploymentStatus">
                <option value="Employed">Employed</option>
                <option value="Self-Employed">Self-Employed</option>
                <option value="Unemployed">Unemployed</option>
            </select>
            
            <label>Education</label>
            <select name="Education">
                <option value="Graduate">Graduate</option>
                <option value="Not Graduate">Not Graduate</option>
            </select>
            
            <label>Loan Purpose</label>
            <select name="LoanPurpose">
                <option value="Home">Home</option>
                <option value="Education">Education</option>
                <option value="Debt_Consolidation">Debt Consolidation</option>
                <option value="Personal">Personal</option>
            </select>
            
            <button type="submit">Evaluate Application</button>
        </form>

        {% if result %}
            <div class="result {{ css_class }}">
                {{ result }}<br>
                <small>{{ detail }}</small>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = detail = css_class = None
    if request.method == "POST":
        data = pd.DataFrame([{
            "Age": int(request.form["Age"]),
            "Income": float(request.form["Income"]),
            "LoanAmount": float(request.form["LoanAmount"]),
            "CreditScore": float(request.form["CreditScore"]),
            "EmploymentStatus": request.form["EmploymentStatus"],
            "Education": request.form["Education"],
            "LoanPurpose": request.form["LoanPurpose"]
        }])
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        if pred == 1:
            result = "✅ APPROVED"
            detail = f"Approval Probability: {prob * 100:.1f}%"
            css_class = "approved"
        else:
            result = "❌ REJECTED / HIGH RISK"
            detail = f"Calculated Risk Score: {(1 - prob) * 100:.1f}%"
            css_class = "rejected"

    return render_template_string(HTML_TEMPLATE, result=result, detail=detail, css_class=css_class)

if __name__ == "__main__":
    app.run(debug=True)
