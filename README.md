# Food Delivery Time Predictor

Streamlit app that estimates food delivery time in minutes with a saved Random Forest model.

Setup, the seven inputs, and the run commands are in [food-delivery-prediction/README.md](food-delivery-prediction/README.md).

From the repository root:

```bash
cd food-delivery-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.port 8654 --server.address 0.0.0.0
```

Open http://localhost:8654.
