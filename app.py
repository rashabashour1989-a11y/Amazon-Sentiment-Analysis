import streamlit as st
import pickle

import os

# الحصول على المسار الحالي للملف
base_path = os.path.dirname(__file__)

# تحميل الموديل باستخدام المسار الكامل
model_path = os.path.join(base_path, 'Sentiment_model.pkl')
vector_path = os.path.join(base_path, 'vectorizer.pkl')

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vector_path, 'rb'))

# 2. إعداد واجهة المستخدم (UI)
st.set_page_config(page_title="Amazon Review Analyzer", page_icon="🛍️")

st.title("🛍️ Amazon Sentiment Analyzer")
st.markdown("Write a product review below, and the AI will tell you if it's **Positive** or **Negative**!")

# خانة إدخال النص
user_input = st.text_area("Enter your review here:", placeholder="e.g., This product is amazing and fast!")

# زر التحليل
if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # 3. معالجة النص وتوقعه
        # تحويل النص باستخدام الـ vectorizer
        transformed_input = vectorizer.transform([user_input])
        
        # التوقع
        prediction = model.predict(transformed_input)[0]
        
        # 4. عرض النتيجة بشكل جميل
        if prediction == 1:
            st.success("### Result: Positive 😍")
            st.balloons() # تأثير بالونات احتفالية
        else:
            st.error("### Result: Negative 😡")

# إضافة ملاحظة في الأسفل
st.sidebar.info("This app uses a Logistic Regression model trained on Amazon Reviews.")
