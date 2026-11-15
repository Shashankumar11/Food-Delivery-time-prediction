"""Checks for encoding, feature order, and validation.

The in-memory forest used here only exercises this project's code. The app
itself still predicts with model/best_random_forest_model.pkl.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.prediction import (  # noqa: E402
    CATEGORICAL_FEATURES,
    FEATURE_ORDER,
    PREFERRED_OPTIONS,
    PredictionError,
    build_feature_frame,
    dataset_bounds,
    load_encoders,
    load_model,
    predict_delivery_time,
)


def _encoders() -> dict:
    encoders = {}
    for column, labels in PREFERRED_OPTIONS.items():
        encoder = LabelEncoder()
        encoder.fit(labels)
        encoders[column] = encoder
    return encoders


def _training_frame(encoders: dict, rows: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = {
        "Distance_km": rng.uniform(1, 20, rows),
        "Preparation_Time_min": rng.integers(5, 30, rows),
        "Courier_Experience_yrs": rng.integers(0, 10, rows),
    }
    for column, encoder in encoders.items():
        picks = rng.integers(0, len(encoder.classes_), rows)
        data[column] = encoder.classes_[picks]
        data[column] = encoder.transform(data[column])
    return pd.DataFrame(data)


def _fit_forest(columns: list[str]) -> tuple[RandomForestRegressor, dict]:
    encoders = _encoders()
    frame = _training_frame(encoders)
    target = (
        frame["Distance_km"] * 2
        + frame["Preparation_Time_min"]
        + frame["Weather"]
        - frame["Courier_Experience_yrs"]
    )
    model = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0)
    model.fit(frame[columns], target)
    return model, encoders


def _sample() -> dict:
    return {
        "Distance_km": 7.93,
        "Weather": "Windy",
        "Traffic_Level": "Low",
        "Time_of_Day": "Afternoon",
        "Vehicle_Type": "Scooter",
        "Preparation_Time_min": 12,
        "Courier_Experience_yrs": 1,
    }


class PredictionPipelineTests(unittest.TestCase):
    def test_feature_frame_uses_the_model_column_order(self) -> None:
        order = list(reversed(FEATURE_ORDER))
        model, encoders = _fit_forest(order)
        frame = build_feature_frame(model, encoders, _sample())
        self.assertEqual(list(frame.columns), order)
        self.assertNotIn("Delivery_Time_min", frame.columns)
        self.assertEqual(len(frame), 1)

    def test_prediction_is_the_model_output(self) -> None:
        model, encoders = _fit_forest(list(FEATURE_ORDER))
        values = _sample()
        minutes = predict_delivery_time(model, encoders, values)
        frame = build_feature_frame(model, encoders, values)
        expected = float(model.predict(frame)[0])
        self.assertTrue(np.isfinite(minutes))
        self.assertAlmostEqual(minutes, expected, places=6)

    def test_target_in_the_input_dict_is_ignored(self) -> None:
        model, encoders = _fit_forest(list(FEATURE_ORDER))
        values = _sample()
        values["Delivery_Time_min"] = 999
        frame = build_feature_frame(model, encoders, values)
        self.assertNotIn("Delivery_Time_min", frame.columns)

    def test_unseen_category_is_rejected(self) -> None:
        model, encoders = _fit_forest(list(FEATURE_ORDER))
        values = _sample()
        values["Weather"] = "Hail"
        with self.assertRaises(PredictionError) as caught:
            predict_delivery_time(model, encoders, values)
        message = str(caught.exception)
        self.assertIn("Hail", message)
        self.assertIn("not seen", message)

    def test_invalid_numbers_are_rejected(self) -> None:
        model, encoders = _fit_forest(list(FEATURE_ORDER))
        for column, bad in (
            ("Distance_km", 0),
            ("Preparation_Time_min", -5),
            ("Courier_Experience_yrs", -1),
        ):
            values = _sample()
            values[column] = bad
            with self.assertRaises(PredictionError):
                predict_delivery_time(model, encoders, values)

    def test_dataset_bounds_follow_the_csv(self) -> None:
        bounds = dataset_bounds()
        self.assertLessEqual(bounds["Distance_km"]["min"], 0.59)
        self.assertGreaterEqual(bounds["Distance_km"]["max"], 19.99)
        self.assertEqual(bounds["Preparation_Time_min"]["default"], 17.0)
        self.assertEqual(bounds["Courier_Experience_yrs"]["default"], 5.0)

    def test_saved_files_round_trip_from_a_temp_directory(self) -> None:
        model, encoders = _fit_forest(list(FEATURE_ORDER))
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            model_path = folder_path / "best_random_forest_model.pkl"
            encoder_path = folder_path / "label_encoders.pkl"
            import pickle

            with model_path.open("wb") as handle:
                pickle.dump(model, handle)
            with encoder_path.open("wb") as handle:
                pickle.dump(encoders, handle)

            loaded_model = load_model(model_path)
            loaded_encoders = load_encoders(encoder_path)
            minutes = predict_delivery_time(loaded_model, loaded_encoders, _sample())
            self.assertTrue(np.isfinite(minutes))
            self.assertEqual(set(loaded_encoders), set(CATEGORICAL_FEATURES))


if __name__ == "__main__":
    unittest.main()
