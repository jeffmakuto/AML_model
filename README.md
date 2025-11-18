# AML_model

Hybrid AML transaction monitoring tailored for Jubilee Insurance in Kenya combines transparent rule scoring with explainable machine learning (XGBoost plus Isolation Forest). The notebook ingests the Jubilee transaction ledger, applies the hybrid pipeline, and documents results for compliance review.

## Setup

- Create and activate a Python 3.11 virtual environment (the workspace already configures `.venv`).
- Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

- Regenerate the dataset and rebuild the notebook:

```powershell
python scripts/generate_jubilee_data.py
python scripts/generate_notebook.py
```
- Open `notebooks/hybrid_aml_jubilee.ipynb` in Jupyter or VS Code for interactive execution, auditing rule scores, inspecting XGBoost metrics, and reviewing anomaly flags.

## Streamlit dashboard

- Launch the interactive dashboard to explore the hybrid scores and transaction mix:

```powershell
streamlit run streamlit_app.py
```
The app loads `data/jubilee_transactions.csv`, provides filter controls, and surfaces the top critical transactions along with risk-tier visualizations.

## Data

- `data/jubilee_transactions.csv`: 64,000+ transactions for 10,000 Jubilee Insurance customers (with at least five transactions per person) spanning Kenya, Uganda, and Tanzania. Created via `scripts/generate_jubilee_data.py`.

## Project structure

- `scripts/generate_jubilee_data.py`: generates the transaction dataset used by the analysis.
- `scripts/generate_notebook.py`: rewrites the hybrid AML notebook after the dataset exists.
- `data/jubilee_transactions.csv`: fresh dataset referenced by the notebook.
- `notebooks/hybrid_aml_jubilee.ipynb`: notebook demonstrating the rule-based and ML hybrid pipeline.

Additional analysis can feed real Jubilee policy data into `scripts/generate_jubilee_data.py` and rerun the pipeline for updated insights.