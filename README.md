# Wind Power Generation Forecasting

This project focuses on **forecasting wind power generation** using **machine learning models** based on meteorological parameters such as temperature, humidity, dew point, wind speed, and wind direction. The dataset combines data from four different locations and applies data preprocessing, exploratory data analysis (EDA), and model evaluation to identify the most accurate prediction technique.

---

## 📊 Project Workflow
1. **Data Collection**
   - Data is sourced from four locations (`Location1.csv`, `Location2.csv`, `Location3.csv`, `Location4.csv`).
   - Each file contains hourly data including temperature, humidity, wind speed (10m & 100m), wind direction, wind gusts, and power output.

2. **Data Preprocessing**
   - Combined all four location datasets into a single DataFrame.
   - Checked for missing values and duplicates.
   - One-hot encoded the `Location` column.
   - Removed time-based column (`Time`) to focus on numerical features.

3. **Exploratory Data Analysis (EDA)**
   - Visualized data distributions using **Seaborn histplots** and **boxplots**.
   - Explored feature relationships with **scatter plots** between predictors and `Power`.
   - Analyzed feature correlations using a **heatmap**.

4. **Model Training**
   - Split dataset into **80% training** and **20% testing** sets.
   - Standardized features using **StandardScaler**.
   - Trained and compared the following regression models:
     - **Linear Regression**
     - **Random Forest Regressor**
     - **XGBoost Regressor**
   - Performed **hyperparameter tuning** using `GridSearchCV` on the XGBoost model.

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/<your-username>/Wind_Power_Generation_Forecasting.git
cd Wind_Power_Generation_Forecasting
pip install pandas numpy scikit-learn matplotlib seaborn xgboost
```

---

## 🧠 Models and Performance

| Model | Mean Absolute Error (MAE) | Mean Squared Error (MSE) | R² Score |
|--------|----------------------------|---------------------------|-----------|
| Linear Regression | 0.1380 | 0.0326 | 0.5106 |
| Random Forest Regressor | 0.1069 | 0.0216 | 0.6765 |
| XGBoost Regressor | 0.1160 | 0.0249 | 0.6270 |
| **Tuned XGBoost (Best)** | **0.1137** | **0.0239** | **0.6413** |

**✅ Best Model:** Tuned XGBoost Regressor  
**Performance:** R² ≈ 0.64 — showing a good predictive correlation with actual power output.

---

## 📦 Key Libraries Used

- `pandas` – Data manipulation and preprocessing  
- `numpy` – Numerical operations  
- `matplotlib`, `seaborn` – Data visualization  
- `scikit-learn` – Model training and evaluation  
- `xgboost` – Gradient boosting regression  
- `joblib` – Parallel computation utilities

---

## 📈 Results and Insights

- Wind speed at **100m height** has the strongest correlation with power generation.  
- Random Forest and XGBoost outperform Linear Regression significantly.  
- Hyperparameter tuning of XGBoost further improves accuracy and reduces errors.  

---

## 🔮 Future Work
- Incorporate **deep learning models (LSTM/GRU)** for temporal sequence forecasting.  
- Deploy the model using **Flask or FastAPI** for real-time wind power prediction.  
- Integrate live weather API data for dynamic model updates.

---
