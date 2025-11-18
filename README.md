# AML_model

Hybrid AML transaction monitoring tailored for Jubilee Insurance in Kenya combines transparent rule scoring with explainable machine learning (XGBoost plus Isolation Forest). The notebook synthesizes policy data, applies the hybrid pipeline, and documents results for compliance review.

## Setup

- Create and activate a Python 3.11 virtual environment (the workspace already configures `.venv`).
- Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

- Regenerate the notebook or refresh the calculations:

```powershell
python scripts/generate_notebook.py
```
- Open `notebooks/hybrid_aml_jubilee.ipynb` in Jupyter or VS Code for interactive execution, auditing rule scores, inspecting XGBoost metrics, and reviewing anomaly flags.

## Project structure

- `scripts/generate_notebook.py`: builds the notebook described above.
- `notebooks/hybrid_aml_jubilee.ipynb`: final hybrid AML notebook combining rule scoring, XGBoost, and Isolation Forest.

Additional analysis can extend the synthetic ledger with real Jubilee policy data or plug in live transaction streams for online scoring.