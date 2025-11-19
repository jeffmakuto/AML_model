# AML_model

Hybrid AML transaction monitoring tailored for Jubilee Insurance in Kenya combines transparent rule scoring with explainable machine learning (XGBoost plus Isolation Forest). The notebook now reads as a structured research manuscript—complete with abstract, methodology, results, discussion, and references—detailing how the hybrid system improves suspicious-transaction detection while preserving compliance explainability.

## Setup

- Create and activate a Python 3.11 virtual environment (the workspace already configures `.venv`).
- Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

- Regenerate the dataset and rebuild the research notebook:

```powershell
python scripts/generate_jubilee_data.py
python scripts/generate_notebook.py
```
- Open `notebooks/hybrid_aml_jubilee.ipynb` in Jupyter or VS Code to review the research narrative, execute experiments, and export tables or figures for publication.
- Execute the cells in order to reproduce reported metrics (ROC AUC, confusion matrices, anomaly coverage). Cells are organized by paper sections (Introduction, Literature Review, Methodology, Results, etc.) to aid drafting.

## Streamlit dashboard

- Launch the interactive dashboard to explore hybrid scores, filter transactions, and inspect high-risk cases interactively:

```powershell
streamlit run streamlit_app.py
```
- The app loads `data/jubilee_transactions.csv`, provides filter controls, and surfaces the top critical transactions along with risk-tier visualizations for investigator walk-throughs.

## Data

- `data/jubilee_transactions.csv`: 65,000+ transactions for 10,000 Jubilee Insurance customers (minimum five transactions per customer) spanning Kenya, Uganda, and Tanzania. Generated via `scripts/generate_jubilee_data.py` and includes rule scores, ML probabilities, anomaly flags, and hybrid risk tiers after notebook execution.

## Project structure

- `scripts/generate_jubilee_data.py`: generates the transaction dataset used by the analysis.
- `scripts/generate_notebook.py`: rewrites the hybrid AML research notebook after the dataset exists.
- `data/jubilee_transactions.csv`: fresh dataset referenced by the notebook and Streamlit app.
- `notebooks/hybrid_aml_jubilee.ipynb`: research-style notebook covering motivation, literature review, methodology, results, discussion, and references.

Additional analysis can feed real Jubilee policy data into `scripts/generate_jubilee_data.py` and rerun the pipeline for updated insights.