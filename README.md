# Sms_Spam_Detection-Project
# 📱 SMS Spam Detection 📩🔍

A Machine Learning project to classify SMS messages as **Spam** or **Ham (Not Spam)** using natural language processing (NLP) and classification models.

---

## 🚀 Project Overview

In the era of mobile communication, spam messages are not only annoying but also pose a security risk. This project helps detect and filter **spam SMS** messages using machine learning and text classification techniques.

**Techniques used**: Text Preprocessing, TF-IDF, Naive Bayes Classifier, and Evaluation Metrics.

---

## 📊 Dataset

- **Name**: [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Source**: UCI Machine Learning Repository
- **Format**: CSV
- **Columns**:
  - `label`: `ham` (not spam) or `spam`
  - `message`: text content of the SMS

---

## 🧠 Features & Workflow

1. **Text Cleaning & Preprocessing**
   - Lowercasing
   - Removing punctuation and special characters
   - Stopword removal
   - Lemmatization (optional)

2. **Vectorization**
   - Using **TF-IDF** (Term Frequency - Inverse Document Frequency)

3. **Model Training**
   - Model: **Multinomial Naive Bayes**
   - Optional: SVM, Logistic Regression, Random Forest

4. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion Matrix

5. **Prediction**
   - Predict if a new message is spam or not

---

## 🧰 Tech Stack

| Component        | Tool/Library             |
|------------------|--------------------------|
| Programming Lang | Python                   |
| Data Handling    | Pandas, NumPy            |
| NLP              | Scikit-learn, NLTK       |
| Modeling         | Scikit-learn             |
| Visualization    | Matplotlib, Seaborn      |
| Deployment (opt) | Streamlit / Flask        |


## ⚙️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sms-spam-detection.git

# Navigate into the folder
cd sms-spam-detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
jupyter notebook


Message: "Congratulations! You've won a $1000 Walmart gift card!"
Prediction: 🚫 SPAM

Message: "Hey, are we still meeting at 6 PM?"
Prediction: ✅ HAM

