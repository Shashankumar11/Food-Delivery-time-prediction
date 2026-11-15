"""Food Delivery Time Predictor.

A single page that estimates delivery time in minutes with the saved
Random Forest. Nothing on this page trains a model or invents a number.
"""

from __future__ import annotations

import html
import math
import sys
import textwrap
from pathlib import Path

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils.prediction import (  # noqa: E402
    PREFERRED_OPTIONS,
    PredictionError,
    dataset_bounds,
    encoder_classes,
    load_encoders,
    load_model,
    ordered_options,
    predict_delivery_time,
)

ASSETS_DIR = APP_DIR / "assets"


def show_html(body: str) -> None:
    """Render an HTML fragment.

    Blank lines end a Markdown HTML block, which would print the rest as text.
    Drop those lines so the whole fragment stays HTML.
    """
    cleaned = textwrap.dedent(body).strip()
    cleaned = "\n".join(line for line in cleaned.splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def inject_styles() -> None:
    """Load the page CSS once per run."""
    css = (ASSETS_DIR / "styles.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def init_state() -> None:
    """Remember the latest estimate and which version of the form is showing."""
    if "form_nonce" not in st.session_state:
        st.session_state.form_nonce = 0
    if "result" not in st.session_state:
        st.session_state.result = None
    if "form_error" not in st.session_state:
        st.session_state.form_error = None


@st.cache_resource(show_spinner=False)
def cached_model():
    """Load the Random Forest once per Streamlit process."""
    return load_model()


@st.cache_resource(show_spinner=False)
def cached_encoders():
    """Load the label encoders once per Streamlit process."""
    return load_encoders()


@st.cache_data(show_spinner=False)
def cached_bounds():
    """Read slider ranges from the training CSV once."""
    return dataset_bounds()


def open_artifacts():
    """Return (model, encoders, error message)."""
    try:
        return cached_model(), cached_encoders(), None
    except PredictionError as error:
        return None, None, str(error)
    except Exception as error:
        return None, None, f"The saved model could not be loaded. Details: {error}"


def category_options(column: str, encoders) -> list[str]:
    """Dropdown labels. Encoder classes win, so the form cannot offer an unknown label."""
    preferred = PREFERRED_OPTIONS[column]
    if encoders is None:
        return list(preferred)
    try:
        options = ordered_options(column, encoder_classes(encoders[column]))
    except PredictionError:
        return list(preferred)
    return options or list(preferred)


def render_header() -> None:
    show_html(
        """
        <section class="hero">
          <div class="hero-row">
            <div class="brand">
              <div class="mark" aria-hidden="true">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="8" stroke="white" stroke-width="2"/>
                  <path d="M12 8v4.5L15 15" stroke="white" stroke-width="2" stroke-linecap="round"/>
                </svg>
              </div>
              <div>
                <h1>Food Delivery Time Predictor</h1>
                <p class="subtitle">AI-powered delivery time estimation using Machine Learning.</p>
              </div>
            </div>
            <div class="badge"><span class="badge-dot"></span>Powered by Random Forest ML</div>
          </div>
        </section>
        """
    )


def render_section_title(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        (
            f'<p class="section-kicker">{html.escape(kicker)}</p>'
            f'<h2 class="section-title">{html.escape(title)}</h2>'
            f'<p class="section-copy">{html.escape(copy)}</p>'
        ),
        unsafe_allow_html=True,
    )


def collect_inputs(bounds: dict, encoders, nonce: int) -> dict:
    """Draw the two form sections and return the seven model inputs."""
    distance = bounds["Distance_km"]
    preparation = bounds["Preparation_Time_min"]
    experience = bounds["Courier_Experience_yrs"]

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            render_section_title(
                "Delivery details",
                "Delivery Details",
                "Where the order is going, and what the courier will meet on the way.",
            )
            distance_km = st.slider(
                "Distance (km)",
                min_value=float(distance["min"]),
                max_value=float(distance["max"]),
                value=float(distance["default"]),
                step=float(distance["step"]),
                help="Road distance from the restaurant to the customer, in kilometres.",
                key=f"distance_{nonce}",
            )
            weather = st.selectbox(
                "Weather",
                category_options("Weather", encoders),
                help="Conditions during the trip.",
                key=f"weather_{nonce}",
            )
            traffic = st.selectbox(
                "Traffic Level",
                category_options("Traffic_Level", encoders),
                index=_default_index(category_options("Traffic_Level", encoders), "Medium"),
                help="How busy the roads are.",
                key=f"traffic_{nonce}",
            )
            time_of_day = st.selectbox(
                "Time of Day",
                category_options("Time_of_Day", encoders),
                index=_default_index(category_options("Time_of_Day", encoders), "Evening"),
                help="When the order leaves the restaurant.",
                key=f"time_{nonce}",
            )
            vehicle = st.selectbox(
                "Vehicle Type",
                category_options("Vehicle_Type", encoders),
                help="How the courier is travelling.",
                key=f"vehicle_{nonce}",
            )

    with right:
        with st.container(border=True):
            render_section_title(
                "Kitchen and courier",
                "Preparation and Courier Details",
                "How long the kitchen needs, and how experienced the courier is.",
            )
            preparation_min = st.slider(
                "Preparation Time (minutes)",
                min_value=float(preparation["min"]),
                max_value=float(preparation["max"]),
                value=float(preparation["default"]),
                step=float(preparation["step"]),
                help="Minutes from order acceptance until the food is ready for pickup.",
                key=f"prep_{nonce}",
            )
            experience_yrs = st.slider(
                "Courier Experience (years)",
                min_value=float(experience["min"]),
                max_value=float(experience["max"]),
                value=float(experience["default"]),
                step=float(experience["step"]),
                help="Years the courier has been delivering.",
                key=f"experience_{nonce}",
            )
            st.markdown(
                "<p class='section-copy'>The estimate uses these seven inputs only. "
                "Delivery time is the result, never an input.</p>",
                unsafe_allow_html=True,
            )

    return {
        "Distance_km": distance_km,
        "Weather": weather,
        "Traffic_Level": traffic,
        "Time_of_Day": time_of_day,
        "Vehicle_Type": vehicle,
        "Preparation_Time_min": preparation_min,
        "Courier_Experience_yrs": experience_yrs,
    }


def _default_index(options: list[str], preferred: str) -> int:
    if preferred in options:
        return options.index(preferred)
    return 0


def format_minutes(minutes: float) -> str:
    """Whole minutes when the value is effectively an integer, otherwise one decimal."""
    rounded = round(float(minutes), 1)
    if abs(rounded - round(rounded)) < 1e-9:
        return str(int(round(rounded)))
    return f"{rounded:.1f}"


def duration_visual(minutes: float) -> str:
    """A bar that shows the predicted length against a two-hour scale."""
    scale = 120.0
    if minutes > scale:
        scale = float(math.ceil(minutes / 15.0) * 15)
    ratio = 0.0 if minutes <= 0 else min(minutes / scale, 1.0)
    width = f"{ratio * 100:.2f}"
    scale_label = html.escape(f"{scale:.0f}")
    return (
        '<div class="duration">'
        '<div class="track" role="img" '
        f'aria-label="Duration bar filled to {html.escape(format_minutes(minutes))} minutes '
        f'on a {scale_label} minute scale">'
        f'<div class="fill" style="width: {width}%;"></div>'
        "</div>"
        '<div class="track-labels"><span>0 min</span>'
        f"<span>{scale_label} min</span></div>"
        "</div>"
    )


def chip(text: str) -> str:
    return f'<span class="chip">{html.escape(text)}</span>'


def render_result(result: dict) -> None:
    minutes = float(result["minutes"])
    inputs = result["inputs"]
    headline = html.escape(format_minutes(minutes))
    exact = html.escape(f"{minutes:.2f}")
    caution = ""
    if minutes < 0:
        caution = (
            '<p class="caution">The model returned a negative time. '
            "The figure above is that raw prediction.</p>"
        )
    elif minutes > 180:
        caution = (
            '<p class="caution">This estimate is longer than three hours. '
            "The figure above is the raw model output.</p>"
        )

    experience = float(inputs["Courier_Experience_yrs"])
    experience_text = (
        f"{experience:.0f} yrs experience"
        if float(experience).is_integer()
        else f"{experience:.1f} yrs experience"
    )
    chips = "".join(
        [
            chip(f"{float(inputs['Distance_km']):.1f} km"),
            chip(str(inputs["Weather"])),
            chip(f"{inputs['Traffic_Level']} traffic"),
            chip(str(inputs["Time_of_Day"])),
            chip(str(inputs["Vehicle_Type"])),
            chip(f"{float(inputs['Preparation_Time_min']):.0f} min prep"),
            chip(experience_text),
        ]
    )

    show_html(
        f"""
        <article class="result-card">
          <div>
            <p class="result-kicker">
              <span class="clock" aria-hidden="true">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                  <circle cx="12" cy="12" r="8" stroke="#FF6B1A" stroke-width="2"/>
                  <path d="M12 8v5l3 2" stroke="#FF6B1A" stroke-width="2" stroke-linecap="round"/>
                </svg>
              </span>
              Estimated Delivery Time
            </p>
            <h2>Your estimated delivery time is ready!</h2>
            <div class="minutes"><strong>{headline}</strong><span>Minutes</span></div>
            <p class="exact">Model output: {exact} minutes</p>
            {duration_visual(minutes)}
            {caution}
            <div class="chips">{chips}</div>
          </div>
          <div class="ring-wrap">
            {_clock_ring(minutes)}
            <p class="ring-caption">The orange arc fills as the estimated wait gets longer.</p>
          </div>
        </article>
        """
    )


def _clock_ring(minutes: float) -> str:
    """A clock-style ring filled in proportion to the prediction."""
    scale = 120.0
    if minutes > scale:
        scale = float(math.ceil(minutes / 15.0) * 15)
    ratio = 0.0 if minutes <= 0 else min(minutes / scale, 1.0)
    radius = 52
    circumference = 2 * math.pi * radius
    offset = circumference * (1 - ratio)
    return (
        '<svg width="148" height="148" viewBox="0 0 148 148" aria-hidden="true">'
        f'<circle cx="74" cy="74" r="{radius}" fill="#FFF8F3" stroke="#EEF2F7" stroke-width="12"/>'
        f'<circle cx="74" cy="74" r="{radius}" fill="none" stroke="#FF6B1A" stroke-width="12" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.2f}" '
        f'stroke-dashoffset="{offset:.2f}" transform="rotate(-90 74 74)"/>'
        '<circle cx="74" cy="74" r="3" fill="#0E1C36"/>'
        f'<line x1="74" y1="74" x2="74" y2="40" stroke="#0E1C36" stroke-width="3" stroke-linecap="round" '
        f'transform="rotate({ratio * 360:.2f} 74 74)"/>'
        "</svg>"
    )


def render_empty() -> None:
    show_html(
        """
        <div class="empty-card">
          <strong>The estimate appears here</strong>
          Choose the trip details, then press Predict Delivery Time.
          The page waits for that click before it asks the model.
        </div>
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Food Delivery Time Predictor",
        page_icon="⏱️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    init_state()
    render_header()

    model, encoders, load_error = open_artifacts()
    bounds = cached_bounds()
    if load_error:
        st.error(load_error)

    nonce = st.session_state.form_nonce
    with st.form(key=f"delivery_form_{nonce}", border=False):
        values = collect_inputs(bounds, encoders, nonce)
        st.write("")
        predict_col, reset_col = st.columns([3, 1])
        with predict_col:
            predict_clicked = st.form_submit_button(
                "Predict Delivery Time",
                type="primary",
                use_container_width=True,
            )
        with reset_col:
            reset_clicked = st.form_submit_button(
                "Reset",
                type="secondary",
                use_container_width=True,
            )

    if reset_clicked:
        st.session_state.form_nonce += 1
        st.session_state.result = None
        st.session_state.form_error = None
        st.rerun()

    if predict_clicked:
        st.session_state.form_error = None
        if model is None or encoders is None:
            st.session_state.result = None
            if not load_error:
                st.session_state.form_error = "The saved model is unavailable."
        else:
            try:
                with st.spinner("Estimating delivery time..."):
                    minutes = predict_delivery_time(model, encoders, values)
                st.session_state.result = {"minutes": minutes, "inputs": values}
            except PredictionError as error:
                st.session_state.result = None
                st.session_state.form_error = str(error)
            except Exception as error:
                st.session_state.result = None
                st.session_state.form_error = (
                    "Something went wrong while estimating this delivery. "
                    f"Details: {error}"
                )

    if st.session_state.form_error:
        st.error(st.session_state.form_error)

    if st.session_state.result:
        render_result(st.session_state.result)
    else:
        render_empty()

    st.markdown(
        "<p class='footnote'>Estimates come from the saved Random Forest model "
        "and the label encoders trained with it.</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
