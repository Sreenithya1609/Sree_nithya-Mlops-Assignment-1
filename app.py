from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")
    

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from frontend
        data = request.json
        features = data["features"]
        
        # Create DataFrame with input values
        x_new = pd.DataFrame({
            'JobLevel': [features[0]],
            'MonthlyIncome': [features[1]],
            'TotalWorkingYears': [features[2]],
            'PercentSalaryHike': [features[3]],
            'PerformanceRating': [features[4]],
            'YearsAtCompany': [features[5]],
            'YearsInCurrentRole': [features[6]],
            'YearsWithCurrManager': [features[7]]
        })
        # Make prediction
        prediction = model.predict(x_new)[0]
        probability = model.predict_proba(x_new)[0][1]  # Assuming a classifier

        # Return result
        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400
        
if __name__ == "__main__":
    app.run(debug=True)

return jsonify({"error": str(e)}), 400
