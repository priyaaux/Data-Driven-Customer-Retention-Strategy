from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load model & scaler
with open("model/churn_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get inputs
    features = [
        float(request.form["tenure"]),
        float(request.form["MonthlyCharges"]),
        float(request.form["TotalCharges"]),
        float(request.form["Contract"]),
        float(request.form["PaymentMethod"]),
        float(request.form["InternetService"])
    ]

    # Scale features
    final_features = scaler.transform([features])

    # Prediction
    prediction = model.predict(final_features)[0]

    # Probability
    probability = model.predict_proba(final_features)[0][1] * 100

    # Result text
    if prediction == 1:
        result = "Customer WILL Churn ❌"
    else:
        result = "Customer WILL NOT Churn ✅"

    # Retention strategy
    if probability > 70:
        strategy = "⚠️ High Risk: Offer discount, priority support, loyalty benefits."
    elif probability > 40:
        strategy = "⚠️ Medium Risk: Personalized offers & engagement."
    else:
        strategy = "✅ Low Risk: Maintain relationship with regular communication."

    return render_template(
        "index.html",
        prediction_text=result,
        probability_text=f"Churn Probability: {probability:.2f}%",
        strategy_text=strategy
    )

if __name__ == "__main__":
    app.run(debug=True)
