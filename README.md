# 📘 Loan Approval Prediction – Classification Models Comparison

---

## a. Problem Statement

The objective of this project is to build and evaluate multiple **classification models** to predict whether a loan application will be **approved or not** based on applicant and loan-related attributes.
The task involves applying different machine learning algorithms on the **same dataset**, evaluating them using standard performance metrics, and comparing their effectiveness.

---

## b. Dataset Description 

The dataset used for this project is a **Loan Approval dataset**, which contains information related to loan applicants such as personal details, financial status, and loan characteristics.

* **Target Variable**:

  * `Loan_Status`

    * `Y` → Loan Approved
    * `N` → Loan Not Approved

* **Features include**:

  * Applicant income and co-applicant income
  * Loan amount and loan term
  * Credit history
  * Gender, marital status, education, self-employment status
  * Property area

The dataset contains both **categorical and numerical variables**, requiring preprocessing such as encoding and scaling before model training.

---

## c. Models Used 

Six classification models were implemented on the **same preprocessed dataset** and evaluated using the following metrics:

* Accuracy
* AUC (Area Under ROC Curve)
* Precision
* Recall
* F1 Score
* Matthews Correlation Coefficient (MCC)

---

### 📊 Model Comparison Table (Evaluation Metrics)

| **ML Model Name**        | **Accuracy** | **AUC**    | **Precision** | **Recall** | **F1 Score** | **MCC**    |
| ------------------------ | ------------ | ---------- | ------------- | ---------- | ------------ | ---------- |
| Logistic Regression      | 0.9000       | 0.9563     | 0.7914        | 0.7468     | 0.7685       | 0.7053     |
| Decision Tree            | 0.8979       | 0.8539     | 0.7677        | 0.7748     | 0.7713       | 0.7055     |
| kNN                      | 0.8904       | 0.9213     | 0.7918        | 0.6876     | 0.7360       | 0.6700     |
| Naive Bayes              | 0.8060       | 0.7805     | 0.6663        | 0.2548     | 0.3686       | 0.3255     |
| Random Forest (Ensemble) | **0.9277**   | **0.9747** | **0.8973**    | **0.7620** | **0.8241**   | **0.7832** |
| XGBoost (Ensemble*)      | 0.9222       | 0.9719     | 0.8788        | 0.7540     | 0.8116       | 0.7665     |

> **Note**: Gradient Boosting Classifier from Scikit-learn was used as an alternative to XGBoost due to Python 3.13 compatibility constraints.

---

### 📝 Observations on Model Performance 

| **ML Model Name**        | **Observation about model performance**                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Logistic Regression      | Demonstrated strong overall performance with high AUC and balanced precision–recall, making it a reliable baseline linear model for loan approval prediction.            |
| Decision Tree            | Achieved good accuracy and recall but showed lower AUC, indicating potential overfitting and weaker generalization compared to ensemble methods.                         |
| kNN                      | Provided good precision but lower recall, meaning some eligible applicants were misclassified. Performance is sensitive to feature scaling and choice of *k*.            |
| Naive Bayes              | Performed the weakest among all models, with very low recall and F1-score, due to violation of the feature independence assumption.                                      |
| Random Forest (Ensemble) | Achieved the **best overall performance** across most metrics, including highest accuracy, AUC, F1-score, and MCC, indicating strong robustness and generalization.      |
| XGBoost (Ensemble)       | Delivered excellent predictive performance with very high AUC and F1-score, slightly lower than Random Forest but still highly effective due to boosting-based learning. |

---

### ✅ Conclusion

Among all the models evaluated, **ensemble methods outperformed individual classifiers**, with **Random Forest** emerging as the best-performing model. Naive Bayes showed the least effectiveness due to its simplifying assumptions, while Logistic Regression provided a strong and interpretable baseline.
