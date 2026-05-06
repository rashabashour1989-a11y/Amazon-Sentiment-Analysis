import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords

# Download stopwords for the Cloud environment
nltk.download('stopwords')

# 1. Setup cleaning function (Must match your Jupyter training logic)
stop_words = set(stopwords.words('english'))
negation_words = {'not', 'no', 'never', 'dont', 'didnt', 'doesnt', 'wasnt', 'werent'}
final_stop_words = stop_words - negation_words 

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    # Keep negation words to help the model understand negative sentiment
    text = ' '.join([word for word in text.split() if word not in final_stop_words])
    return text

# 2. Load the model and vectorizer
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, 'Sentiment_model.pkl')
vector_path = os.path.join(base_path, 'vectorizer.pkl')

# Load files using pickle
model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vector_path, 'rb'))

# 3. Streamlit UI Layout
st.set_page_config(page_title="Amazon Review Analyzer")
st.title("🛍️ Amazon Sentiment Analyzer")
st.markdown("Write a product review below, and the AI will tell you if it's **Positive** or **Negative**.")

user_input = st.text_area("Enter your review here:", placeholder="e.g., The product is not good...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        # Pre-process the input text
        cleaned_input = clean_text(user_input)
        
        # Transform input using the loaded vectorizer
        transformed_input = vectorizer.transform([cleaned_input])
        
        # Make prediction
        prediction = model.predict(transformed_input)
        
        # Display results (Assuming 1 for Positive, 0 for Negative)
        if prediction[0] == 1:
            st.success("Result: Positive 😄")
        else:
            st.error("Result: Negative ☹️")
