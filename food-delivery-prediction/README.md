# Food Delivery Time Predictor

A single-page Streamlit app that estimates how many minutes a food delivery will take. It asks for the seven inputs the saved Random Forest was trained on, encodes the categories with the saved label encoders, and shows that model's prediction.

The app does not train a model and does not estimate anything other than delivery time.

## What you enter

- Distance (km)
- Weather
- Traffic level
- Time of day
- Vehicle type
- Preparation time (minutes)
- Courier experience (years)

The result is `Delivery_Time_min`.

## Project layout

```
food-delivery-prediction/
    app.py
    model/
        best_random_forest_model.pkl
        label_encoders.pkl
    data/
        Food_Delivery_Times.csv
    utils/
        prediction.py
    requirements.txt
    README.md
```

`model/best_random_forest_model.pkl` is the RandomForestRegressor saved from training (scikit-learn 1.6.1). `model/label_encoders.pkl` is a dictionary of LabelEncoders for Weather, Traffic_Level, Time_of_Day, and Vehicle_Type. The app loads each file once and does not scale the inputs.

`data/Food_Delivery_Times.csv` is the public 1,000-row Food Delivery Times table (the NTUA Data Analysis exercise file, with these same columns). The page uses it only to set slider ranges. It is not sent to the model.

## Run it

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8654 --server.address 0.0.0.0
```

On Windows, activate the environment with `.venv\Scripts\activate` and use the same `pip` and `streamlit` commands.

Then open http://localhost:8654.

`requirements.txt` pins scikit-learn 1.6.1 so the pickle loads with the version that created it.

## Check the saved model from the terminal

With the virtual environment active and both pickle files in `model/`:

```bash
python -m utils.prediction
```

That prints one delivery-time prediction, in minutes, for a fixed sample order (7.93 km, Windy, Low traffic, Afternoon, Scooter, 12 minutes of preparation, 1 year of experience). The number comes from `model.predict`.

Run the checks for encoding and column order with:

```bash
python -m unittest tests.test_prediction
```

## If prediction is unavailable

The Predict button needs both pickle files in `model/` under the names above. If either file is missing, or a chosen category was not in the encoder, the page shows the reason and does not display a delivery time.
