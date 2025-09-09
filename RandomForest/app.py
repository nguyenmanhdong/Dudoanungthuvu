from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load mô hình
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cancer_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model, features = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    values = {}
    if request.method == "POST":
        try:
            inputs = []
            for feat in features:
                val = float(request.form[feat])
                values[feat] = val
                inputs.append(val)

            prediction = model.predict([inputs])[0]
            result = "✅ Khối u LÀNH TÍNH" if prediction == 1 else "❌ Khối u ÁC TÍNH"
        except Exception as e:
            result = f"Lỗi: {e}"

    return render_template("index.html", features=features, result=result, values=values)

if __name__ == "__main__":
    app.run(debug=True)
