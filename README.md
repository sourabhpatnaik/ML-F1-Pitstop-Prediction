# 🏎️ F1 Pit Stop Strategy AI

A Machine Learning powered web application that predicts whether a Formula 1 driver will make a pit stop in the next lap — based on real-time race telemetry, tyre performance, and lap data. Built with a **Random Forest classifier** and deployed via an F1-themed **Streamlit** dashboard.

---

## 🚀 Live Demo

> Run locally using the steps in the [Getting Started](#-getting-started) section.

---

## 📌 Problem Statement

Pit stop decisions are one of the most critical aspects of F1 race strategy. This project simulates that decision-making process using historical lap data — predicting from race conditions whether a driver *should* pit in the next lap, helping teams optimize their strategy.

---

## 📸 App Preview

| Telemetry Inputs | Prediction Result | Race Insights |
|---|---|---|
| Sidebar sliders for all race parameters | Pit / Continue verdict with confidence % | Real-time tyre, degradation & position insights |

---

## 🧠 ML Pipeline

```
Raw Race Data → Null Removal → Outlier Filtering → Column Renaming
     → Train/Test Split → OrdinalEncoder (Compound)
     → MinMaxScaler (Numerical) → Random Forest Classifier → Prediction
```

---

## 📊 Dataset

| Attribute | Details |
|---|---|
| File | `f1_strategy_dataset.csv` |
| Target column | `PitNextLap` — `0` = No Pit, `1` = Pit Next Lap |
| Dropped columns | `Driver`, `Race`, `Year`, `PitStop` (identifier / leakage columns) |
| Outlier handling | Threshold-based filtering on `LapTime`, `LapTimeDelta`, `CumulativeDegradation` |

---

## 🔍 Exploratory Data Analysis

- **Class distribution** analyzed via countplot for `PitNextLap` (0 vs 1)
- **TyreLife vs PitNextLap** — higher tyre life increases pit stop probability due to wear
- **RaceProgress vs PitNextLap** — pit stops occur more frequently in mid-to-late race stages
- **Correlation Heatmap** key findings:
  - `LapNumber` & `RaceProgress` — strong positive correlation (~0.96)
  - `TyreLife` & `NormalizedTyreLife` — high positive correlation (~0.77)
  - `TyreLife` (~0.27), `LapNumber` (~0.16) show the strongest influence on `PitNextLap`
  - Low multicollinearity across most features

---

## ⚙️ Feature Engineering

Applied separately on `X_train` and `X_test` (using fitted encoders on train only):

| Step | Technique | Applied To |
|---|---|---|
| Categorical Encoding | `OrdinalEncoder` | `Compound` column |
| Numerical Scaling | `MinMaxScaler` | All numeric features |

**Input Features:**
`LapNumber`, `Stint`, `TyreLife`, `Position`, `LapTime`, `LapTimeDelta`, `CumulativeDegradation`, `RaceProgress`, `NormalizedTyreLife`, `PositionChange`, `Compound`

---

## 🤖 Model Comparison

| Model | Accuracy |
|---|---|
| K-Nearest Neighbours (KNN) | 92.65% |
| **Random Forest** ✅ | **93.64%** |
| Logistic Regression | 76.93% |

**Random Forest** was selected as the final model for its highest accuracy and robustness with tabular race data.

| Detail | Value |
|---|---|
| `n_estimators` | 100 |
| `class_weight` | balanced |
| Train/Test Split | 80% / 20% |
| Prediction Threshold | 0.4 (adjusted for class imbalance) |
| Export format | `joblib` → `RandForest_Model.pkl` |

---

## 🖥️ Streamlit App Features

- **Sidebar telemetry inputs** — sliders and number inputs for all 11 race parameters
- **Live stats bar** — lap number, race progress, tyre life, position, stint, compound displayed as metric cards
- **Prediction result panel** — PIT THIS LAP / CONTINUE RACE verdict with model confidence %
- **Probability gauge** — custom matplotlib semicircular gauge showing pit stop probability
- **Race insights panel** — dynamic insights based on tyre wear, degradation, position, race stage, and compound
- F1-styled dark UI with `Titillium Web` font and Ferrari red (`#e10600`) accents

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)

---

## 📁 Project Structure

```
f1-pitstop-prediction/
│
├── F1_Pitstop_Prediction.ipynb    # EDA, preprocessing & model training notebook
├── app.py                         # Streamlit web application
├── RandForest_Model.pkl           # Exported model + encoders (joblib)
├── f1_strategy_dataset.csv        # Dataset
└── README.md
```

---

## ⚡ Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/f1-pitstop-prediction.git
cd f1-pitstop-prediction
```

### 2. Install dependencies
```bash
pip install streamlit scikit-learn pandas numpy matplotlib joblib
```

### 3. Train the model (if not already exported)
Run all cells in `F1_Pitstop_Prediction.ipynb` to generate `RandForest_Model.pkl`.

### 4. Launch the app
```bash
streamlit run app.py
```

---

## 📈 Results

The **Random Forest classifier achieved 93.64% accuracy**, outperforming KNN (92.65%) and Logistic Regression (76.93%). With a tuned decision threshold of 0.4 to handle class imbalance, the model reliably predicts pit stop windows from race telemetry.

---

## 🙋 Author

**Soro**  
Data Science Student | Innomatics Research Labs  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/your-username)
