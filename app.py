from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import time
import joblib
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "fraud_detection_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
# Load your trained model and scaler
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

@app.route('/predict', methods=['POST'])
def predict_api():
    data = request.get_json()
    amount = data.get("amount")
    if amount is None:
        return jsonify({"error": "Missing amount"}), 400

    # Use current time for prediction
    current_time = time.time()
    transaction_df = pd.DataFrame([[current_time, amount]], columns=["Time", "Amount"])
    transaction_df[["Time", "Amount"]] = scaler.transform(transaction_df[["Time", "Amount"]])
    prediction = model.predict(transaction_df)
    is_fraud = int(prediction[0] == -1)
    return jsonify({"is_fraud": is_fraud})

# REMOVE the following block for production/gunicorn:
# if __name__ == '__main__':
#     app.run(debug=True)