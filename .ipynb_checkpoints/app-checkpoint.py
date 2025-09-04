from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load("Cancer_prediction.joblib")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect and validate input values
        features = []
        for i in range(1, 31):
            value = request.form.get(f'feature{i}')
            if not value or not value.replace('.', '', 1).isdigit():
                return render_template("index.html", prediction_text="⚠️ Please enter valid numeric values!")
            features.append(float(value))

        # Convert to NumPy array and predict
        input_data = np.array([features])
        prediction = model.predict(input_data)
        
        # Return the result
        result = "Malignant (Cancer Detected)" if prediction[0] == 1 else "Benign (No Cancer)"
        return render_template("index.html", prediction_text=result)

    except Exception as e:
        print(f"Error: {e}")  # Debugging
        return render_template("index.html", prediction_text="⚠️ Unexpected error! Try again.")

if __name__ == "__main__":
    app.run(debug=True)

