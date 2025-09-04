
from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("Cancer_prediction.joblib")

@app.route('/')
def home():
    return render_template("home.html")  # Home Page

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            features = [float(request.form[f'feature{i}']) for i in range(1, 30)]
            input_data = np.array([features])

            # Make prediction
            prediction = model.predict(input_data)
            result = "Malignant (Cancer Detected)" if prediction[0] == 1 else "Benign (No Cancer)"

            return render_template("result.html", prediction_text=result)

        except Exception as e:
            return render_template("predict.html", error="⚠️ Invalid input! Enter numbers only.")

    return render_template("predict.html")  # Show the form

if __name__ == "__main__":
    app.run(debug=True)
