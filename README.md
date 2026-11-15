# Predictive Maintenance System Using Machine Learning

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://machine-failure-predictor-ai4i.streamlit.app)
![Python](https://img.shields.io/badge/python-3.10-blue)
![Tests](https://img.shields.io/badge/tests-34%20passing-brightgreen)

End-to-end ML pipeline that predicts industrial machine failures from sensor
data, with a tested Python package and an interactive Streamlit dashboard.

**🚀 Live demo:** <https://machine-failure-predictor-ai4i.streamlit.app>

![Dashboard demo](docs/dashboard_demo.gif)

> Built on the synthetic [AI4I 2020](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
> benchmark (10,000 rows, 3.39% failures). It demonstrates methodology and
> engineering practice, not guaranteed performance in a real factory.

## Results

Class-weighted Random Forest (200 trees, threshold 0.30), evaluated **once**
on an untouched 1,500-row test set:

| Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| 0.902 | 0.902 | 0.902 | 0.991 | 0.937 |

**46 of 51** failures detected, 5 missed, 5 false alarms.

## How it works

1. **Data prep** — drop IDs, exclude leakage columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`), one-hot encode `Type`, stratified 70/15/15 split.
2. **Features** — `Power`, `Temp_diff` and `Wear_Torque` engineered from the raw sensors.
3. **Model selection** — baseline, Logistic Regression, Random Forest (± SMOTE) and XGBoost compared on the validation set only; threshold tuned to 0.30 for recall.
4. **Frozen evaluation** — one test-set run; `scripts/save_final_model.py` refuses to save a model that doesn't reproduce these numbers.

## Dashboard

Frozen results · single-machine prediction with what-if curve · batch CSV
scoring and download · feature importance and threshold explorer · EDA charts.

## Run locally

```bash
git clone https://github.com/merinjo90/predictive_maintenance.git
cd predictive_maintenance
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m pytest                    # 34 tests (data tests need data/raw/ai4i2020.csv)
streamlit run app/dashboard.py      # dashboard at http://localhost:8501
```

To retrain, place `ai4i2020.csv` in `data/raw/` and run
`PYTHONPATH=. python scripts/save_final_model.py`. Use the pinned versions in
`requirements.txt` (scikit-learn 1.7.2) — newer versions train a different model.

## Project structure

```
app/        Streamlit dashboard (+ lean requirements.txt for the live demo)
src/        data_processing · features · train_model · evaluate · predict
scripts/    save_final_model.py · record_dashboard_gif.py
models/     frozen model + metadata
notebooks/  EDA and model comparison
tests/      pytest suite
```

## License

Educational portfolio project. Dataset: AI4I 2020, UCI Machine Learning
Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
