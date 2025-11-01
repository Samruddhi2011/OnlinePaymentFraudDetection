# Online Payment Fraud Detection System  

## Project Overview  

The **Online Payment Fraud Detection System** is a machine learning–based model designed to detect fraudulent transactions in real-time payment environments. It addresses the growing risk of digital financial fraud by analyzing transaction patterns, identifying anomalies, and predicting fraud likelihood with high accuracy.  

Unlike rule-based methods, this project uses **supervised and unsupervised learning** to automatically adapt to evolving fraud behaviors.  

---

## Features  

-  **Machine Learning Algorithms:** Logistic Regression, Random Forest, Gradient Boosting  
-  **Handles Data Imbalance:** Uses **SMOTE** (Synthetic Minority Oversampling Technique)  
-  **Dimensionality Reduction:** PCA (Principal Component Analysis)  
-  **Clustering for Anomaly Detection:** K-Means and Agglomerative Clustering  
-  **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, ROC-AUC  
-  **Visualization:** Histograms, scatter plots, and PCA component plots for pattern analysis  

---

##  Tech Stack  

| Component | Technology |
|------------|-------------|
| Language | Python |
| Frameworks | scikit-learn, pandas, numpy, matplotlib, seaborn |
| Environment | Jupyter Notebook |
| Models Used | Logistic Regression, Random Forest, Gradient Boosting |
| Techniques | PCA, Clustering, SMOTE |

---

##  Methodology  

### 1 Data Preprocessing
- Handled missing values and scaled numerical features  
- Encoded categorical variables  
- Created new indicators such as 24-hour transaction frequency  

### 2 Exploratory Data Analysis  
- Visualized transaction patterns using histograms and scatter plots  
- Detected outliers and correlations  

### 3 PCA for Dimensionality Reduction  
- Reduced dataset features to retain **90% variance**  

### 4 Clustering  
- Used **K-Means** and **Agglomerative Clustering** to detect anomalous patterns  

### 5 Model Training & Evaluation  
- Compared Logistic Regression, Random Forest, and Gradient Boosting models  
- Evaluated with accuracy, recall, precision, and ROC-AUC  

---

## Results  

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|-----------|-----------|---------|-----------|----------|
| Logistic Regression | 71.1% | 0.0 | 0.0 | 0.0 | 0.51 |
| Gradient Boosting | 87.5% | 0.85 | 0.99 | 0.84 | High |
| Random Forest | 99.6% | 0.99 | 0.99 | 0.99 | 0.99 |

**Random Forest** achieved the best performance with **99.6% accuracy**, proving its robustness in detecting complex fraud patterns.

---

## Advantages  

- High accuracy and reliability through ensemble models  
- Scalable for large transaction datasets  
- Real-time fraud detection potential  
- Adaptable for banking, e-commerce, and insurance applications  

---

## Limitations  

- Data imbalance remains a key challenge  
- High computational requirement for real-time deployment  
- Requires periodic retraining to adapt to evolving fraud tactics  

---

## Future Scope  

- Integration of **deep learning** (LSTM/RNN) for sequential fraud pattern detection  
- Implementation of **GAN-based synthetic data generation** for better balancing  
- Real-time fraud alert dashboard and live transaction scoring system  

---

## Project Files  
- OnlinePaymentFraudDetection/
- ┣ src/
- ┃ ┗ ONLINE_PAYMENT_FRAUD_DETECTION_MINI_PROJECT.py
- ┣ data/
- ┃ ┗ fraud_dataset.csv
- ┗ README.md

---

## Author  

**Samruddhi Patodi**  
B.Tech – Computer Science & Business Systems (CSBS)  
NMIMS MPSTME, Mumbai  

---

## Tags  

`#MachineLearning` `#Python` `#FraudDetection` `#RandomForest` `#GradientBoosting`  
`#PCA` `#Clustering` `#SMOTE` `#MiniProject` `#NMIMS` `#DataScience`
