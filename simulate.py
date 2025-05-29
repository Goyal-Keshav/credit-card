import requests
import time
import random
import pandas as pd

# Load the dataset
data = pd.read_csv("C:\\Users\\kesha\\Documents\\fraud-detection-system\\data\\creditcard.csv")

# Only use Time and Amount for simulation, as per the updated model
test_data = data[["Time", "Amount", "Class"]].sample(n=20, random_state=42)  # Smaller sample for demo

print("Columns in test data:", test_data.columns.tolist())

for index, row in test_data.iterrows():
    try:
        # Prepare payload with only Time and Amount
        payload = {
            "Time": row["Time"],
            "Amount": row["Amount"]
        }
        # Send POST request to the root endpoint (form expects form data, but let's assume you have a /predict endpoint for API)
        response = requests.post("http://localhost:5000/predict", json=payload)
        
        # Print the raw response content
        print("Raw response:", response.text)
        
        # Parse JSON only if status code is 200
        if response.status_code == 200:
            result = response.json()
            if result.get("fraud") == 1:
                print(f"ALERT: Fraud detected! Index: {index}")
            else:
                print(f"Legitimate transaction: {index}")
        else:
            print(f"Error: HTTP {response.status_code}")
        
        time.sleep(1)
    
    except Exception as e:
        print(f"Error: {e}")

# If you want to test the web form, you must enter Amount manually in the browser.
