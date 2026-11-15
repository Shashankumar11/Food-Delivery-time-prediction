"""Load the saved Random Forest and turn the seven form inputs into one delivery time.

The model and the label encoders are the files produced in training. This module
never fits a new model, never scales the inputs, and never fills in a category
the encoder has not seen.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # noqa: F401  (needed so the pickle can resolve)
from sklearn.preprocessing import LabelEncoder  # noqa: F401

# Directory layout: food-delivery-prediction/{utils,model,data}
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT_DIR / "model"
DATA_PATH = ROOT_DIR / "data" / "Food_Delivery_Times.csv"

MODEL_FILENAME = "best_random_forest_model.pkl"
ENCODER_FILENAME = "label_encoders.pkl"

# The regressor was trained on these columns, in this order.
# Delivery_Time_min is the target, so it is intentionally absent here.
FEATURE_ORDER = [
    "Distance_km",
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

CATEGORICAL_FEATURES = [
    "Weather",
    "Traffic_Level",
    "Time_of_Day",
    "Vehicle_Type",
]

NUMERIC_FEATURES = [
    "Distance_km",
    "Preparation_Time_min",
    "Courier_Experience_yrs",
]

# Shown in the form in this order. The encoder's own class order is only used
# to turn the chosen label into the integer the forest expects.
PREFERRED_OPTIONS = {
    "Weather": ["Clear", "Foggy", "Rainy", "Snowy", "Windy"],
    "Traffic_Level": ["Low", "Medium", "High"],
    "Time_of_Day": ["Morning", "Afternoon", "Evening", "Night"],
    "Vehicle_Type": ["Bike", "Car", "Scooter"],
}

# Used when the CSV cannot be read. They match the public training table.
FALLBACK_BOUNDS = {
    "Distance_km": {"min": 0.5, "max": 20.0, "default": 10.2, "step": 0.1},
    "Preparation_Time_min": {"min": 5.0, "max": 29.0, "default": 17.0, "step": 1.0},
    "Courier_Experience_yrs": {"min": 0.0, "max": 9.0, "default": 5.0, "step": 1.0},
}

# Hard limits for the prediction function. The sliders stay inside the dataset
# range; these checks reject values that cannot describe a real delivery.
NUMERIC_LIMITS = {
    "Distance_km": (0.1, 100.0, "Distance must be between 0.1 and 100 km."),
    "Preparation_Time_min": (
        1.0,
        180.0,
        "Preparation time must be between 1 and 180 minutes.",
    ),
    "Courier_Experience_yrs": (
        0.0,
        40.0,
        "Courier experience must be between 0 and 40 years.",
    ),
}

# Optional copies of the uploaded artifacts, checked only if model/ is empty.
_UPLOAD_CANDIDATES = {
    MODEL_FILENAME: [
        Path("/workspace/uploads/best_random_forest_model_066e.pkl"),
        Path("/workspace/uploads/best_random_forest_model.pkl"),
    ],
    ENCODER_FILENAME: [
        Path("/workspace/uploads/label_encoders_be84.pkl"),
        Path("/workspace/uploads/label_encoders.pkl"),
    ],
}


class PredictionError(Exception):
    """A problem the page can show directly to the person using the form."""


def bundled_artifact_path(filename: str) -> Path:
    """Return the on-disk path for one of the two saved training files."""
    if filename not in (MODEL_FILENAME, ENCODER_FILENAME):
        raise PredictionError(f"Unknown artifact '{filename}'.")

    primary = MODEL_DIR / filename
    if primary.is_file():
        return primary

    for candidate in _UPLOAD_CANDIDATES[filename]:
        if candidate.is_file():
            return candidate

    raise PredictionError(
        f"Could not find {filename}. Add that file to the model folder, "
        "using the same file name, and restart the app."
    )


def load_artifact(path: Path):
    """Read a pickle that was saved from the training notebook.

    scikit-learn 1.6.1 wrote these files. Both pickle and joblib can read that
    format, so joblib is only a fallback when pickle cannot open the file.
    """
    if path.name not in (MODEL_FILENAME, ENCODER_FILENAME) and path.name not in {
        candidate.name
        for names in _UPLOAD_CANDIDATES.values()
        for candidate in names
    }:
        raise PredictionError(f"Refusing to load unexpected file '{path.name}'.")

    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as pickle_error:
        try:
            import joblib
        except ImportError as import_error:
            raise PredictionError(
                f"Could not read {path.name}. Install scikit-learn 1.6.1 and try again."
            ) from import_error
        try:
            return joblib.load(path)
        except Exception as joblib_error:
            raise PredictionError(
                f"Could not read {path.name}. It should be the original file "
                "saved with scikit-learn 1.6.1. "
                f"Details: {pickle_error}"
            ) from joblib_error


def load_model(path: Path | None = None):
    """Load the saved RandomForestRegressor. Does not train anything."""
    model_path = path or bundled_artifact_path(MODEL_FILENAME)
    model = load_artifact(model_path)
    class_name = type(model).__name__
    if class_name != "RandomForestRegressor" or not hasattr(model, "predict"):
        raise PredictionError(
            "The model file is not a RandomForestRegressor "
            f"(found {type(model).__module__}.{class_name})."
        )
    return model


def load_encoders(path: Path | None = None) -> dict:
    """Load the dictionary of LabelEncoders for the four categorical features."""
    encoder_path = path or bundled_artifact_path(ENCODER_FILENAME)
    encoders = load_artifact(encoder_path)
    if not isinstance(encoders, dict):
        raise PredictionError(
            "label_encoders.pkl should be a dictionary with one LabelEncoder "
            "for Weather, Traffic_Level, Time_of_Day, and Vehicle_Type."
        )

    missing = [name for name in CATEGORICAL_FEATURES if name not in encoders]
    if missing:
        found = ", ".join(str(key) for key in encoders.keys()) or "none"
        raise PredictionError(
            "The encoder file is missing "
            + ", ".join(missing)
            + f". Keys found: {found}."
        )
    return encoders


def feature_order(model) -> list[str]:
    """Column order the saved forest expects.

    When the forest was fitted on a DataFrame it remembers that order in
    ``feature_names_in_``. Otherwise the original training order is used.
    """
    names = getattr(model, "feature_names_in_", None)
    if names is None:
        return list(FEATURE_ORDER)

    order = [str(name) for name in list(names)]
    if order == list(FEATURE_ORDER):
        return order
    if sorted(order) == sorted(FEATURE_ORDER) and len(order) == len(set(order)):
        return order

    raise PredictionError(
        "This model does not use the seven delivery features. "
        f"It expects: {', '.join(order)}."
    )


def encoder_classes(encoder) -> list[str]:
    """Labels a saved encoder can accept, as strings."""
    if hasattr(encoder, "classes_"):
        classes = []
        for label in list(encoder.classes_):
            text = str(label).strip()
            if text and text.lower() != "nan":
                classes.append(text)
        return classes
    if isinstance(encoder, dict):
        return [str(label) for label in encoder.keys()]
    raise PredictionError("A saved encoder is not a LabelEncoder.")


def ordered_options(column: str, classes: list[str]) -> list[str]:
    """Preferred display order, limited to labels the encoder actually knows."""
    preferred = PREFERRED_OPTIONS[column]
    known = set(classes)
    ordered = [label for label in preferred if label in known]
    extras = [label for label in classes if label not in ordered]
    return ordered + extras


def dataset_bounds(path: Path | None = None) -> dict:
    """Slider min, max, and default from the training CSV.

    Defaults are the median of each numeric column. Missing experience values
    are skipped. If the file cannot be read, fixed bounds from that same table
    are used so the form can still render.
    """
    csv_path = path or DATA_PATH
    bounds = {key: dict(value) for key, value in FALLBACK_BOUNDS.items()}
    if not csv_path.is_file():
        return bounds

    frame = pd.read_csv(csv_path)
    for column, spec in bounds.items():
        if column not in frame.columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce").dropna()
        if series.empty:
            continue
        step = spec["step"]
        low = np.floor(float(series.min()) / step) * step
        high = np.ceil(float(series.max()) / step) * step
        default = float(series.median())
        low = round(float(low), 2)
        high = round(float(high), 2)
        if high < low:
            high = low
        steps_from_min = round((default - low) / step)
        default = round(low + steps_from_min * step, 2)
        if default < low:
            default = low
        if default > high:
            default = high
        spec["min"] = low
        spec["max"] = high
        spec["default"] = default
    return bounds


def _as_python_number(value):
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise PredictionError("The encoder returned more than one code.")
    number = array[0]
    if isinstance(number, np.generic):
        number = number.item()
    return number


def _encode_category(encoder, column: str, raw_value) -> object:
    """Map one label through its saved encoder.

    Unknown labels raise. They are not replaced with a nearby class.
    """
    text = str(raw_value).strip()
    if hasattr(encoder, "classes_") and hasattr(encoder, "transform"):
        classes = encoder_classes(encoder)
        if text not in classes:
            known = ", ".join(classes) if classes else "none"
            raise PredictionError(
                f"{column.replace('_', ' ')} '{text}' was not seen in training. "
                f"Choose one of: {known}."
            )
        # Pass the original class object so the encoder's own comparison matches.
        original = next(label for label in list(encoder.classes_) if str(label) == text)
        try:
            encoded = encoder.transform([original])
        except ValueError as error:
            raise PredictionError(
                f"Could not encode {column.replace('_', ' ')} '{text}'."
            ) from error
        return _as_python_number(encoded)

    if isinstance(encoder, dict):
        if text not in encoder and raw_value not in encoder:
            known = ", ".join(str(label) for label in encoder.keys())
            raise PredictionError(
                f"{column.replace('_', ' ')} '{text}' was not seen in training. "
                f"Choose one of: {known}."
            )
        return encoder[text] if text in encoder else encoder[raw_value]

    raise PredictionError(
        f"The saved encoder for {column} is not a LabelEncoder."
    )


def validate_numbers(values: dict) -> dict[str, float]:
    """Check the three numeric inputs and return them as floats."""
    checked = {}
    for column, (low, high, message) in NUMERIC_LIMITS.items():
        raw = values.get(column)
        try:
            number = float(raw)
        except (TypeError, ValueError) as error:
            raise PredictionError(message) from error
        if not np.isfinite(number) or number < low or number > high:
            raise PredictionError(message)
        checked[column] = number
    return checked


def build_feature_frame(model, encoders: dict, values: dict) -> pd.DataFrame:
    """One-row DataFrame with the names and order the forest expects."""
    numbers = validate_numbers(values)
    order = feature_order(model)
    encoded = {}
    for column in CATEGORICAL_FEATURES:
        if column not in values or values[column] is None or str(values[column]).strip() == "":
            raise PredictionError(f"Choose a value for {column.replace('_', ' ')}.")
        encoded[column] = _encode_category(encoders[column], column, values[column])

    row = {}
    for column in order:
        if column in CATEGORICAL_FEATURES:
            row[column] = encoded[column]
        elif column in numbers:
            row[column] = numbers[column]
        else:
            raise PredictionError(f"Missing model input '{column}'.")

    frame = pd.DataFrame([row], columns=order)
    if "Delivery_Time_min" in frame.columns:
        raise PredictionError("Delivery time cannot be used as an input.")
    return frame


def predict_delivery_time(model, encoders: dict, values: dict) -> float:
    """Return the forest's delivery-time prediction in minutes."""
    frame = build_feature_frame(model, encoders, values)
    try:
        raw = model.predict(frame)
    except Exception as error:
        raise PredictionError(
            "The model could not estimate this delivery. "
            f"Details: {error}"
        ) from error

    prediction = _as_python_number(raw)
    try:
        minutes = float(prediction)
    except (TypeError, ValueError) as error:
        raise PredictionError("The model returned a value that is not a number.") from error
    if not np.isfinite(minutes):
        raise PredictionError("The model returned a value that is not a number.")
    return minutes


def main() -> None:
    """Print one prediction from the saved model, for a quick terminal check."""
    sample = {
        "Distance_km": 7.93,
        "Weather": "Windy",
        "Traffic_Level": "Low",
        "Time_of_Day": "Afternoon",
        "Vehicle_Type": "Scooter",
        "Preparation_Time_min": 12,
        "Courier_Experience_yrs": 1,
    }
    model = load_model()
    encoders = load_encoders()
    minutes = predict_delivery_time(model, encoders, sample)
    print(f"{minutes:.4f}")


if __name__ == "__main__":
    try:
        main()
    except PredictionError as error:
        raise SystemExit(str(error)) from error
