# 🏦 Loan Approval Prediction System

**ML Assignment 2 - Binary Classification**
**Author:** Sumit Mondal (2024dc04216)

A comprehensive machine learning system for predicting loan approval status using multiple classification algorithms. This project includes model training, evaluation, and an interactive Streamlit web application for predictions.

Streamlit Link: https://bits-ml-assignment2-wsackzs9qgdz4px7mg8wbx.streamlit.app/
[![Live Demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bits-ml-assignment2-wsackzs9qgdz4px7mg8wbx.streamlit.app/)
---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Project Structure](#project-structure)
- [Results](#results)
- [Contributing](#contributing)

---

## 🎯 Overview

This project implements and compares **6 different machine learning algorithms** for loan approval classification:

1. **Logistic Regression**
2. **Decision Tree**
3. **K-Nearest Neighbors (KNN)**
4. **Naive Bayes**
5. **Random Forest** 🏆
6. **XGBoost**

The system provides:

- ✅ Comprehensive model training and evaluation
- ✅ Interactive web interface for predictions
- ✅ Batch prediction support via CSV upload
- ✅ Detailed performance metrics visualization
- ✅ Model comparison dashboard

---

## 📊 Dataset

**Source:** [Kaggle - Loan Approval Classification Data](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)

**Dataset Details:**

- **Total Records:** 45,000
- **Features:** 13 input features + 1 target variable
- **Target:** `loan_status` (Binary: 0 = Rejected, 1 = Approved)

### Feature Description

| Feature                            | Type        | Description                                           |
| ---------------------------------- | ----------- | ----------------------------------------------------- |
| `person_age`                     | Numeric     | Age of the applicant                                  |
| `person_gender`                  | Categorical | Gender (male/female)                                  |
| `person_education`               | Categorical | Education level (High School, Bachelor, Master, etc.) |
| `person_income`                  | Numeric     | Annual income in USD                                  |
| `person_emp_exp`                 | Numeric     | Years of employment experience                        |
| `person_home_ownership`          | Categorical | RENT/OWN/MORTGAGE/OTHER                               |
| `loan_amnt`                      | Numeric     | Requested loan amount                                 |
| `loan_intent`                    | Categorical | Purpose (EDUCATION, MEDICAL, VENTURE, etc.)           |
| `loan_int_rate`                  | Numeric     | Interest rate (%)                                     |
| `loan_percent_income`            | Numeric     | Loan amount / Annual income                           |
| `cb_person_cred_hist_length`     | Numeric     | Credit history length (years)                         |
| `credit_score`                   | Numeric     | Credit score (300-850)                                |
| `previous_loan_defaults_on_file` | Categorical | Yes/No                                                |

---

## ✨ Features

### 🎯 Prediction Modes

1. **Manual Entry:** Input individual loan application details through an intuitive form
2. **Batch Prediction:** Upload CSV files for processing multiple applications at once

### 📊 Visualizations

- Model performance comparison charts
- Prediction probability distributions
- Interactive radar charts for multi-metric analysis
- Real-time confidence scores

### 🔧 Technical Features

- Automatic feature encoding and scaling
- Model persistence using joblib
- Comprehensive error handling
- Template CSV generation for batch predictions

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/DevOps-MATT1/BITS-ML-Assignment2.git
cd BITS-ML-Assignment2
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset

Download `loan_data.csv` from [Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data) and place it in the project root directory.

---

## 📖 Usage

### Training Models

Run the Jupyter notebook to train all models and generate artifacts:

```bash
jupyter notebook ML_Assignment_2_COMPLETE.ipynb
```

Then execute all cells in the notebook.

**This will create:**

- `all_models.pkl` - All trained models, scaler, and feature names (merged in single file)
- `model_comparison_metrics.csv` - Performance metrics

**Expected Output:**

```
======================================================================
LOAN APPROVAL CLASSIFICATION - MODEL TRAINING
======================================================================

✓ Loading dataset: loan_data.csv
  Dataset shape: (45000, 14)

======================================================================
DATA PREPROCESSING
======================================================================
  Encoding categorical variables...
  Splitting data (75% train, 25% test)...
  Scaling features...

======================================================================
MODEL TRAINING
======================================================================
  Training: Logistic Regression...
    ✓ Accuracy: 0.9000 | F1: 0.7685 | AUC: 0.9563
  
  Training: XGBoost...
    ✓ Accuracy: 0.9361 | F1: 0.8483 | AUC: 0.9792
  
  ... (all 6 models)

🏆 BEST MODEL: XGBoost (F1 Score: 0.8483)
```

### Running the Web Application

Launch the Streamlit app:

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 🤖 Models

### Model Comparison

| Model                         | Accuracy         | Precision        | Recall           | F1 Score         | AUC              | MCC              |
| ----------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
| **XGBoost** 🏆          | **0.9361** | **0.8974** | **0.8044** | **0.8483** | **0.9792** | **0.8100** |
| **Random Forest**       | 0.9277           | 0.8973           | 0.7620           | 0.8241           | 0.9747           | 0.7832           |
| **Logistic Regression** | 0.9000           | 0.7914           | 0.7468           | 0.7685           | 0.9563           | 0.7053           |
| **Decision Tree**       | 0.8979           | 0.7677           | 0.7748           | 0.7713           | 0.8539           | 0.7055           |
| **KNN**                 | 0.8904           | 0.7918           | 0.6876           | 0.7360           | 0.9213           | 0.6700           |
| **Naive Bayes**         | 0.8060           | 0.6663           | 0.2548           | 0.3686           | 0.7805           | 0.3255           |

### Model Selection Recommendations

- **Best Overall Performance:** XGBoost (highest accuracy, AUC, recall, F1, and MCC)
- **Best Alternative:** Random Forest (excellent ensemble performance)
- **Fastest Prediction:** Logistic Regression or Naive Bayes
- **Most Interpretable:** Decision Tree or Logistic Regression
- **Best for Imbalanced Data:** XGBoost or Random Forest

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── ML_Assignment_2_COMPLETE.ipynb  # Jupyter notebook for training
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── loan_data.csv                   # Dataset (download separately)
│
├── all_models.pkl                  # All trained models + scaler + features (generated)
└── model_comparison_metrics.csv    # Performance metrics (generated)
```

---

## 📈 Results

### Key Findings

1. **Ensemble Methods Dominate:** XGBoost and Random Forest significantly outperform other models
2. **High AUC Scores:** Most models achieve AUC > 0.90, indicating excellent discriminative ability
3. **Naive Bayes Underperforms:** Low recall suggests the independence assumption doesn't hold well
4. **Feature Engineering Impact:** One-hot encoding and scaling are crucial for model performance

### Model Insights

**XGBoost (Best Model):**

- Achieves highest performance across all key metrics
- Excellent gradient boosting with sequential error correction
- Built-in regularization prevents overfitting
- Superior recall (0.8044) minimizes false rejections
- **Recommendation:** Use for production deployment

**Random Forest:**

- Strong second-place performance (very competitive with XGBoost)
- Handles non-linear relationships effectively through ensemble averaging
- Provides good feature importance rankings
- **Recommendation:** Excellent alternative to XGBoost

**Logistic Regression:**

- Strong baseline performance (90% accuracy)
- Fast inference time
- Highly interpretable coefficients
- **Recommendation:** Use when interpretability is critical

---

## 🎨 Web Application Features

### Manual Prediction Form

- Interactive form with validation
- Real-time loan-to-income ratio calculation
- Visual prediction confidence display
- Color-coded approval/rejection status

### Batch Prediction

- CSV template download
- Preview uploaded data
- Batch processing with progress indication
- Downloadable results with probabilities

### Model Comparison Dashboard

- Side-by-side metric comparison
- Interactive bar charts
- Multi-dimensional radar plots
- Sortable performance tables

---

## 🔧 Technical Details

### Data Preprocessing Pipeline

1. **Categorical Encoding:** One-hot encoding with `drop_first=True`
2. **Feature Scaling:** StandardScaler for Logistic Regression and KNN
3. **Train-Test Split:** 75-25 split with stratification
4. **Missing Values:** None in original dataset

### Model-Specific Notes

- **Logistic Regression & KNN:** Require scaled features
- **Tree-based models:** Work directly with unscaled features
- **Gradient Boosting:** Uses softmax for probability estimates
- **All models:** Support `predict_proba` for confidence scores

---

## 🐛 Troubleshooting

### Common Issues

**1. File Not Found Error**

```
ERROR: 'loan_data.csv' not found!
```

**Solution:** Download the dataset from Kaggle and place in project root

**2. Module Import Error**

```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:** Install dependencies: `pip install -r requirements.txt`

**3. Model File Not Found**

```
Model file 'all_models.pkl' not found!
```

**Solution:** Run all cells in `ML_Assignment_2_COMPLETE.ipynb` first

**4. Feature Mismatch in Batch Prediction**

```
ValueError: Feature names mismatch
```

**Solution:** Ensure CSV has all required columns matching the template

---

## 📝 Future Enhancements

- [ ] Add SHAP/LIME for model interpretability
- [ ] Implement hyperparameter tuning (GridSearchCV)
- [ ] Add cross-validation results
- [ ] Deploy to cloud (Streamlit Cloud/Heroku)
- [ ] Add user authentication
- [ ] Integrate with SQL database for storing predictions
- [ ] Add A/B testing for model comparison in production
- [ ] Implement real-time monitoring and retraining

---

## 📜 License

This project is created for educational purposes as part of ML Assignment 2.

---

## 👨‍💻 Author

**Sumit Mondal**
Student ID: 2024dc04216

---

## 🙏 Acknowledgments

- Dataset: [Tawei Lo on Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data)
- Libraries: scikit-learn, Streamlit, Pandas, Plotly
- Course: Machine Learning

---

## 📧 Contact

For questions or suggestions, please open an issue in the repository or contact the author.

---

**⭐ If you found this project helpful, please consider giving it a star!**
