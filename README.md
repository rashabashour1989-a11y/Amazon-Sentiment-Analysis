# Amazon Prpduct Reviews Sentiment Analysis
## Overview
   This project is a **Sentiment Analysis Engine** traind on Amazon product reviews. It uses Machine Learning to classify user feedback into **Positive** or **Negative** category with high precision.
## Features
   + **Text Preprocessing :** Custom cleaning pipeline (Lowercasing, Special characters removal, Stopwords handling).
   + **Vectorization :** Implemented **TF-IDF** (Term Frequency-Inverse Document Frequency) to convert text into meaningful numerical data.
   + **Model :** Trained using **Logistic Regression** , achieving ab accuracy of approximately **93.99%**.
   + **Interactive Inference :** Includes a function to test custom reviews in real-time.
## Data Insights
   + **Initial Dataset :** Thousand of real Amazon reviews.
   + **Class Distribution :** The dataset showed a significant skew towards negative reviews (8419 Negative vs 1227 Positive), which helped the model become highly sensitive to critical feedback.
   + **The Neutral Choice :** During preprocessing, 3-star reviews were excluded to create a clear binary classification (Highly Positive vs Highlt Negative).
## Installation & Usage
   1. Clone the repository
      git clone https://github.com/rashabashour1989-a11y/amazon-sentiment-analysis.git
   2. Install dependencies:
      pip install pandas scikit-learn
      matplotlib seaborn wordcloud
   3. Run the Jupyter Notebook Amazon_Sentiment_Analysis.ipynb
## Example Test
   **Input :** 'The product is amazing !'
   
   **Output :** Positive
   
   **Input :** 'The color is different from the picture'
   
   **Output :** Negative
    
