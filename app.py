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


import cloudpickle
# Load the models
model_file_path1 = 'logistic_regression_model.pkl'
with open(model_file_path1, 'rb') as model_file1:
    LR = cloudpickle.load(model_file1)

model_file_path2 = 'decision_tree_model.pkl'
with open(model_file_path2, 'rb') as model_file2:
    DT = cloudpickle.load(model_file2)

model_file_path3 = 'gradient_boosting_model.pkl'
with open(model_file_path3, 'rb') as model_file3:
    GB = cloudpickle.load(model_file3)

model_file_path4 = 'random_forest_model.pkl'
with open(model_file_path4, 'rb') as model_file4:
    RF = cloudpickle.load(model_file4)

model_file_path5 = 'tfidf_vectorizer.pkl'
with open(model_file_path5, 'rb') as vectorizer_file:
    vectorization = cloudpickle.load(vectorizer_file)

# Function to get model predictions


def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www\.\S+','',text)
    text = re.sub('<.*?>+',b'',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\w*\d\w*','',text)
    return text
def predict_news(news, model):
    new_x_test = [news]
    new_x_test = pd.DataFrame({'text': new_x_test})
    new_x_test['text'] = new_x_test['text'].apply(wordopt)
    new_xv_test = vectorization.transform(new_x_test['text'])
    pred = model.predict(new_xv_test)
    return output_label(pred[0])

# Streamlit UI
st.title("Fake News Detection App")

news_input = st.text_area("Enter news text:", "")
model_choice = st.selectbox("Select Model:", ["Logistic Regression", "Decision Tree", "Gradient Boosting", "Random Forest"])

if st.button("Predict"):
    if model_choice == "Logistic Regression":
        prediction = predict_news(news_input, LR)
    elif model_choice == "Decision Tree":
        prediction = predict_news(news_input, DT)
    elif model_choice == "Gradient Boosting":
        prediction = predict_news(news_input, GB)
    elif model_choice == "Random Forest":
        prediction = predict_news(news_input, RF)

    st.success(f"The news is classified as: {prediction}")
