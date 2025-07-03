from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib  # If you saved scalers with joblib

# Create the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("house_price_model.h5", compile=False)


# Load the scalers
x_scaler = joblib.load("x_scaler.pkl")
y_scaler = joblib.load("y_scaler.pkl")

@app.route("/")
def index():
    return "✅ House Price Prediction API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Parse JSON input
        input_data = request.get_json()
        # Example: {"features": [3, 1200, 2, 1, 0, 1, 0]} 
        features = np.array(input_data["features"]).reshape(1, -1)

        # 2. Scale input
        features_scaled = x_scaler.transform(features)

        # 3. Predict scaled price
        y_pred_scaled = model.predict(features_scaled)

        # 4. Inverse-transform to get original price
        y_pred = y_scaler.inverse_transform(y_pred_scaled).ravel()[0]

        # 5. Return prediction
        return jsonify({
            "predicted_price": float(y_pred)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
