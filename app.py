from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import time
import joblib

app = Flask(__name__)
CORS(app)

# Load your trained model and scaler
model = joblib.load('C:\\Users\\kesha\\Documents\\fraud-detection-system\\backend\\model\\fraud_detection_model.pkl')      # Make sure this path is correct
scaler = joblib.load('C:\\Users\\kesha\\Documents\\fraud-detection-system\\backend\\model\\scaler.pkl')    # Make sure this path is correct

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