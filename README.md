# Amazon Sentiment Analysis Project 🚀

This project is a web-based application built with **Streamlit** that predicts the sentiment of Amazon product reviews using Machine Learning.

## 🌟 Key Features
- **Balanced Dataset:** Handled data imbalance (1227 Positive vs 8400 Negative) using Under-sampling to ensure fair predictions.
- **Advanced NLP:** Used `TfidfVectorizer` with `N-grams (1, 3)` to understand context like "not bad".
- **Strong Model:** Implemented `RandomForestClassifier` for higher accuracy and better handling of complex phrases.
- **Interactive UI:** A clean and simple web interface for real-time sentiment prediction.

## 🛠️ Technologies Used
- **Python**
- **Scikit-learn** (RandomForest, TF-IDF)
- **Pandas** (Data Manipulation)
- **Streamlit** (Web Framework)
- **Pickle** (Model Serialization)

## 📊 Model Performance
After balancing the dataset and optimizing the N-grams, the model achieved:
- **Accuracy:** ~82.1% (Balanced Accuracy)
- **Context Awareness:** Successfully identifies negations like "not bad" as Positive.

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
