from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("credit_card_model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        time = float(request.form["Time"])
        features = [float(request.form[f"V{i}"]) for i in range(1, 29)]
        amount = float(request.form["Amount"])

        data = np.array([time] + features + [amount]).reshape(1, -1)
        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Fraudulent Transaction"
        else:
            result = "Legitimate Transaction"

        return render_template("index.html", prediction=result)

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)