from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained Machine Learning model
model = joblib.load("house_price_model.pkl")


# Home Page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction Page
@app.route("/predict", methods=["POST"])
def predict():

    overall_quality = int(request.form["OverallQual"])
    living_area = int(request.form["GrLivArea"])
    garage_cars = int(request.form["GarageCars"])
    garage_area = int(request.form["GarageArea"])
    total_bsmt_sf = int(request.form["TotalBsmtSF"])
    first_floor_sf = int(request.form["1stFlrSF"])
    full_bath = int(request.form["FullBath"])
    year_built = int(request.form["YearBuilt"])
    year_remod = int(request.form["YearRemodAdd"])
    lot_area = int(request.form["LotArea"])

    sample = [[
        overall_quality,
        living_area,
        garage_cars,
        garage_area,
        total_bsmt_sf,
        first_floor_sf,
        full_bath,
        year_built,
        year_remod,
        lot_area
    ]]

    prediction = model.predict(sample)

    print(prediction)

    return render_template(
    "index.html",
    prediction=round(prediction[0], 2)
)

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)