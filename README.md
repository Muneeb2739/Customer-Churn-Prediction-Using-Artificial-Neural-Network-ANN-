# Customer Churn Prediction using ANN

A Customer Churn Prediction system built using an Artificial Neural Network (ANN) to predict whether a customer is likely to leave a service.

The project includes model training, evaluation, preprocessing, and an interactive Streamlit web application.

## 🚀 Features

- Customer churn prediction using ANN
- Churn probability prediction
- Feature scaling and preprocessing
- Class weighting for imbalanced data
- Dropout regularization
- Early stopping
- ROC-AUC evaluation
- Interactive Streamlit application

## 🧠 Machine Learning Model

The project uses an Artificial Neural Network (ANN) built with TensorFlow/Keras.

### Model Architecture

- Input Layer
- Dense Layer — 32 neurons
- Dropout
- Dense Layer — 16 neurons
- Dropout
- Output Layer — 1 neuron
- ReLU activation for hidden layers
- Sigmoid activation for output

Class weights and early stopping were used to improve the model's ability to identify churn customers.

## 📊 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 75% |
| Churn Precision | 0.53 |
| Churn Recall | 0.71 |
| Churn F1-Score | 0.60 |
| ROC-AUC | 0.82 |

The ROC-AUC score of **0.82** indicates that the model has good ability to distinguish between customers who are likely to churn and customers who are likely to stay.

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Streamlit

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── app.py
├── requirements.txt
│
└── notebook/
    ├── customer_churn_ann.pkl
    ├── scaler.pkl
    └── feature_columns.pkl
````

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Customer-Churn-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit application

```bash
streamlit run app.py
```

The application will open in your web browser.

## 🔄 How the Application Works

```text
Customer Information
        ↓
Feature Preparation
        ↓
Feature Scaling
        ↓
ANN Model
        ↓
Churn Probability
        ↓
Churn / No Churn
```

## 🎯 Objective

The main objective of this project is to identify customers who are at risk of leaving a service.

By predicting potential churn customers, businesses can take early retention actions and improve customer retention.

## 📌 Key Learning

Through this project, I worked with:

* Artificial Neural Networks
* Binary Classification
* Feature Scaling
* Class Imbalance
* Dropout Regularization
* Early Stopping
* Model Evaluation
* ROC-AUC
* Streamlit Deployment

## 👨‍💻 Author

**Mohammed Muneeb**

Data Science | Data Analyst | Data Scientist | Machine Learning | Deep Learning | NLP | Gen Ai


