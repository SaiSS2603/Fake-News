import streamlit as st
import pandas as pd
import pickle
import sklearn
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from joblib import dump, load


# Load pre-trained models and vectorizer (assuming they are in the same directory)
with open("logistic_regression_model.pkl", "rb") as model_file:
    LR = pickle.load(model_file)
with open("decision_tree_model.pkl", "rb") as model_file2:
    DT = pickle.load(model_file2)
with open("gradient_boosting_model.pkl", "rb") as model_file:
    GB = pickle.load(model_file)
with open("random_forest_model.pkl", "rb") as model_file:
    RF = pickle.load(model_file)
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Define function to preprocess user input
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)  # Remove square brackets content
    text = re.sub(r"[^\w\s]", " ", text)  # Replace non-alphanumeric and whitespace with space
    text = re.sub(r"https?://\S+|www\.\S+", "", text)  # Remove URLs and links
    text = re.sub(r"<.*?>+", "", text)  # Remove HTML tags
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)  # Remove punctuation
    text = re.sub(r"\w*\d\w*", "", text)  # Remove words containing numbers
    return text

# App title and header
st.title("Fake News Detector")
st.header("Enter news to be verified:")

# User input field for news text
user_news = st.text_area("News Text:")

# Submit button
if st.button("Predict"):
    if user_news:
        # Preprocess the input text
        preprocessed_text = preprocess_text(user_news)

        # Convert text to vector using the trained vectorizer
        new_x_test = vectorizer.transform([preprocessed_text])

        # Make predictions using all four models
        lr_pred = LR.predict(new_x_test)[0]
        dt_pred = DT.predict(new_x_test)[0]
        gb_pred = GB.predict(new_x_test)[0]
        rf_pred = RF.predict(new_x_test)[0]

        # Define dictionary for prediction labels
        prediction_labels = {0: "Fake News", 1: "Not Fake News"}

        # Display predictions from each model
        st.subheader("Prediction Results:")
        st.write("**Logistic Regression:**", prediction_labels[lr_pred])
        st.write("**Decision Tree:**", prediction_labels[dt_pred])
        st.write("**Gradient Boosting:**", prediction_labels[gb_pred])
        st.write("**Random Forest:**", prediction_labels[rf_pred])

        # Display overall conclusion based on majority vote
        if lr_pred == dt_pred == gb_pred == rf_pred:
            st.write("\n**Overall Conclusion:**", prediction_labels[lr_pred])
        else:
            st.write(
                "\n**Overall Conclusion:** Models disagree. Consider the results from each model and conduct further research for a more informed decision."
            )
    else:
        st.error("Please enter news text to be analyzed.")

# Additional information (optional)
st.markdown(
    """**Note:** This is a basic demonstration and may not be completely accurate. It's important to consult multiple sources and exercise critical thinking when evaluating news."""
)
