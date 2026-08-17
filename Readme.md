# Customer Spending Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting a customer's **Average Monthly Spending** using supervised machine learning techniques.

The dataset contains customer-level information such as the number of visits and the number of items purchased from different product categories. The project follows a complete machine learning pipeline, starting from data loading and preprocessing to model training, evaluation, model saving, and Flask deployment.

The main objective is to build a machine learning model that can estimate a customer's monthly spending based on their purchasing behavior.

---

## 🎯 Objectives

* Load and inspect the customer dataset.
* Clean and preprocess the data.
* Handle missing values and duplicate records.
* Perform Exploratory Data Analysis (EDA).
* Remove irrelevant columns.
* Perform feature engineering.
* Apply categorical encoding where required.
* Split the data into training and testing sets.
* Apply feature scaling.
* Train supervised machine learning models.
* Evaluate and compare different models.
* Perform feature selection.
* Improve model performance using hyperparameter tuning.
* Save the trained model.
* Deploy the model using Flask.

---

## 📂 Dataset

The dataset used in this project is:

`Cust_Spend_Data.csv`

### Dataset Features

| Column            | Description                                     |
| ----------------- | ----------------------------------------------- |
| `Cust_ID`         | Unique customer identification number           |
| `Name`            | Customer name                                   |
| `Avg_Mthly_Spend` | Average monthly spending of the customer        |
| `No_Of_Visits`    | Number of visits made by the customer           |
| `Apparel_Items`   | Number of apparel items purchased               |
| `FnV_Items`       | Number of fruits and vegetables items purchased |
| `Staples_Items`   | Number of staple items purchased                |

### Target Variable

**`Avg_Mthly_Spend`**

The model predicts the average monthly spending of a customer.

---

## 🔄 Machine Learning Workflow

The project follows the following workflow:

### 1. Data Loading

The dataset is loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("Cust_Spend_Data.csv")
```

### 2. Data Inspection

The dataset is inspected using:

* `head()`
* `tail()`
* `shape`
* `info()`
* `describe()`
* `isnull().sum()`
* `duplicated().sum()`

This helps understand the structure, data types, missing values, and duplicate records.

### 3. Data Cleaning

Data cleaning includes:

* Checking incorrect data types.
* Identifying missing values.
* Checking duplicate records.
* Removing irrelevant information.
* Handling invalid values.

### 4. Missing Value Handling

Missing values are checked and handled using appropriate techniques such as:

* Mean/median imputation for numerical variables.
* Mode imputation for categorical variables.

In the current dataset, missing values are checked before model training.

### 5. Removing Unwanted Columns

The following columns are not used as predictive features:

```text
Cust_ID
Name
```

`Cust_ID` is an identifier and `Name` is a customer-specific field, so they do not provide useful numerical information for predicting spending.

---

## 6. Duplicate Handling

Duplicate records are identified using:

```python
df.duplicated().sum()
```

Duplicate rows are removed using:

```python
df = df.drop_duplicates()
```

---

## 📊 7. Exploratory Data Analysis

EDA is performed to understand relationships between customer behavior and monthly spending.

The analysis includes:

* Distribution of monthly spending.
* Number of visits vs monthly spending.
* Product category purchases vs monthly spending.
* Correlation analysis.
* Histograms.
* Box plots.
* Scatter plots.

Example:

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()
```

---

## ⚙️ 8. Feature Engineering

New meaningful features are created from existing variables.

### Total Items

```text
Total_Items =
Apparel_Items + FnV_Items + Staples_Items
```

This represents the total number of products purchased by a customer.

### Items Per Visit

```text
Items_Per_Visit =
Total_Items / No_Of_Visits
```

This represents the average number of items purchased during each visit.

### Category Diversity

The number of different product categories purchased by a customer can also be used as a feature.

Feature engineering helps the model understand customer purchasing behavior more effectively.

---

## 🔤 9. Categorical Encoding

Categorical variables are converted into numerical form before training the machine learning model.

For categorical features, **One-Hot Encoding** can be applied using:

```python
from sklearn.preprocessing import OneHotEncoder
```

One-hot encoding converts categories into binary columns.

For example:

```text
Category
--------
Regular
Premium
New
```

can become:

```text
Category_New
Category_Premium
Category_Regular
```

The current dataset mainly contains numerical predictive variables, so one-hot encoding is only required if categorical features are added or present in an expanded version of the dataset.

---

## 🔍 10. Data Verification

After preprocessing, the dataset is verified to ensure:

* No unwanted columns remain.
* Missing values are handled.
* Duplicate records are removed.
* Numerical values have appropriate data types.
* Encoded features are correctly generated.
* No target leakage is present.

---

## ✂️ 11. Train/Test Split

The dataset is divided into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
```

The training set is used to train the model, while the testing set is used to evaluate its performance.

---

## 📏 12. Feature Scaling

Feature scaling is applied to numerical features when required.

`StandardScaler` can be used:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
```

Scaling transforms features so that they have a comparable range.

---

# 🤖 13. Supervised Machine Learning

Since `Avg_Mthly_Spend` is a continuous numerical variable, this is a **regression problem**.

The following models can be trained and compared:

### Linear Regression

```python
from sklearn.linear_model import LinearRegression
```

### Random Forest Regression

```python
from sklearn.ensemble import RandomForestRegressor
```

### Gradient Boosting Regression

```python
from sklearn.ensemble import GradientBoostingRegressor
```

The best-performing model is selected based on evaluation metrics.

---

# 📈 14. Model Evaluation

The models are evaluated using regression metrics.

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Measures the average squared difference between actual and predicted values.

### Root Mean Squared Error (RMSE)

The square root of MSE.

### R² Score

Measures how well the model explains the variation in the target variable.

Example:

```python
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
```

A good regression model generally has:

* Lower MAE
* Lower RMSE
* Higher R²

---

# 🎯 15. Feature Selection

Feature selection is performed to identify the most important variables for predicting customer spending.

Possible techniques include:

* Correlation analysis
* Feature importance
* SelectKBest
* Recursive Feature Elimination (RFE)

For tree-based models, feature importance can be obtained using:

```python
model.feature_importances_
```

---

# 🚀 16. Model Improvement

Model performance can be improved using:

* Hyperparameter tuning.
* Feature engineering.
* Feature selection.
* Cross-validation.
* Removing irrelevant features.
* Comparing multiple algorithms.

Grid Search or Randomized Search can be used for hyperparameter tuning.

Example:

```python
from sklearn.model_selection import GridSearchCV
```

---

# 💾 17. Model Saving

After selecting the best model, it is saved using `joblib`.

```python
import joblib

joblib.dump(model, "customer_spending_model.pkl")
```

The scaler and preprocessing pipeline can also be saved if required.

```python
joblib.dump(scaler, "scaler.pkl")
```

---

# 🌐 18. Flask Deployment

The trained machine learning model is deployed using Flask.

The Flask application accepts customer information from the user and returns the predicted average monthly spending.

### Example Input

```text
Number of Visits: 10
Apparel Items: 5
FnV Items: 8
Staples Items: 12
```

### Example Output

```text
Predicted Average Monthly Spending: ₹XXXX
```

---

# 📁 Project Structure

```text
Customer-Spending-Prediction/
│
├── dataset/
│   └── Cust_Spend_Data.csv
│
├── notebooks/
│   └── customer_spending_prediction.ipynb
│
├── model/
│   ├── customer_spending_model.pkl
│   └── preprocessing.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **Joblib**
* **Flask**
* **HTML/CSS**

---

# 📦 Installation

Clone the repository:

```bash
git clone <repository-url>
cd Customer-Spending-Prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

### Train the Model

Run the Jupyter Notebook:

```text
notebooks/customer_spending_prediction.ipynb
```

This performs data preprocessing, feature engineering, model training, evaluation, and model saving.

### Run Flask Application

```bash
python app.py
```

Open the application in your browser at:

```text
http://127.0.0.1:5000/
```

---

# 📋 Requirements

The `requirements.txt` file should contain:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
flask
```

---

# ⚠️ Dataset Limitation

The current dataset contains a **very small number of observations**. Therefore, model performance metrics may not reliably represent real-world performance.

For a production-quality machine learning model, a substantially larger dataset should be used.

---

# 🔮 Future Scope

The project can be further improved by:

* Using a larger customer dataset.
* Adding demographic information.
* Adding purchase frequency and transaction history.
* Using customer segmentation.
* Implementing advanced ensemble models.
* Deploying the application on a cloud platform.
* Creating a dashboard for customer spending analysis.
* Implementing real-time prediction.

---

# 👩‍💻 Author

**Ashlesha Daulanpure**

B.Tech – Artificial Intelligence Engineering
Maharashtra Institute of Technology, Chhatrapati Sambhajinagar

---

## 📜 License

This project is created for educational and academic purposes.
