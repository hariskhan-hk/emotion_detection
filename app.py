import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import altair as alt
import pickle

# Load Logistic Regression model and TfidfVectorizer
model = pickle.load(open(r"E:\Marketlytics\Emotion-Detection-using-text\model (1).pkl", "rb"))
vectorizer = pickle.load(open(r"E:\Marketlytics\Emotion-Detection-using-text\vectorizer (1).pkl", "rb"))

emotions_emoji_dict = {"anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂", "neutral": "😐", "sad": "😔",
                       "sadness": "😔", "shame": "😳", "surprise": "😮"}

def preprocess_text(text):
    # Add any text preprocessing steps if needed
    return text

def predict_emotions(docx):
    # Preprocess the input text
    processed_text = preprocess_text(docx)

    # Vectorize the processed text
    vector_input = vectorizer.transform([processed_text])

    # Predict emotions
    result = model.predict(vector_input)[0]
    return result

def get_prediction_proba(docx):
    # Preprocess the input text
    processed_text = preprocess_text(docx)

    # Vectorize the processed text
    vector_input = vectorizer.transform([processed_text])

    # Get prediction probabilities
    results = model.predict_proba(vector_input)
    return results

def main():
    st.title("Sentiment Analysis using Text")
    raw_text = st.text_area("Type Here")

    if st.button('Submit'):
        col1, col2 = st.columns(2)

        prediction = predict_emotions(raw_text)
        probability = get_prediction_proba(raw_text)

        with col1:
            st.success("Prediction")
            if prediction in emotions_emoji_dict:
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"<p style='font-size: 24px;'>{prediction.upper()}</p><p style='font-size: 36px;'>{emoji_icon}</p>", unsafe_allow_html=True)
            else:
                st.warning("Unknown emotion prediction")

            st.success("Prediction")
            emoji_icon = emotions_emoji_dict[prediction]
            st.write(f"<p style='font-size: 24px;'>{prediction.upper()}</p><p style='font-size: 36px;'>{emoji_icon}</p>", unsafe_allow_html=True)

        with col2:
            st.success("Prediction Probability")
            proba_df = pd.DataFrame(probability, columns=model.classes_)
            proba_df_clean = proba_df.T.reset_index()
            proba_df_clean.columns = ["emotions", "probability"]

            fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions', y='probability', color='emotions')
            st.altair_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()
