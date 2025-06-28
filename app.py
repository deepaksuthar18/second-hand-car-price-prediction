from flask import Flask, render_template, request
import numpy as np
import sqlite3
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime

app = Flask(__name__)

# Load model and scaler
model = load_model("car_price_model.h5", compile=False)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
scaler = joblib.load("scaler.pkl")

# Predict function
def predict_price(features):
    input_array = np.array([features], dtype=float)
    scaled_input = scaler.transform(input_array)
    prediction = model.predict(scaled_input)[0][0]
    return round(prediction, 2)

# Save prediction in DB
def insert_into_db(data):
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            old_price REAL,
            now_price REAL,
            years REAL,
            km REAL,
            rating REAL,
            condition REAL,
            economy REAL,
            top_speed REAL,
            hp REAL,
            torque REAL,
            predicted_price REAL,
            timestamp TEXT
        )
    """)
    cursor.execute("""
        INSERT INTO predictions (
            old_price, now_price, years, km, rating, condition, economy,
            top_speed, hp, torque, predicted_price, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (*data, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Read inputs
            old_price = float(request.form["old_price"])
            now_price = float(request.form["now_price"])
            years = float(request.form["years"])
            km = float(request.form["km"])
            rating = float(request.form["rating"])
            condition = float(request.form["condition"])
            economy = float(request.form["economy"])
            top_speed = float(request.form["top_speed"])
            hp = float(request.form["hp"])
            torque = float(request.form["torque"])

            # Match feature order
            features = [
                old_price, now_price, years, km, rating,
                condition, economy, top_speed, hp, torque
            ]

            prediction = predict_price(features)

            insert_into_db(features + [prediction])

            return render_template("index.html", prediction=prediction)
        except ValueError:
            return render_template("index.html", prediction="Invalid input")

    return render_template("index.html", prediction=None)



if __name__ == "__main__":
    app.run(debug=True)
