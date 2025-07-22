import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import os

app = Flask(__name__)

# Load and preprocess data
def load_data():
    # Download or use a local dataset (assume 'spam.csv' in root)
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def train_model(df):
    X = df['text']
    y = df['label_num']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'confusion': confusion_matrix(y_test, y_pred).tolist()
    }
    return model, vectorizer, metrics

df = load_data()
model, vectorizer, metrics = train_model(df)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    proba = None
    if request.method == 'POST':
        message = request.form['message']
        msg_vec = vectorizer.transform([message])
        pred = model.predict(msg_vec)[0]
        proba = model.predict_proba(msg_vec)[0][1]
        prediction = 'Spam' if pred == 1 else 'Ham'
    return render_template('index.html', prediction=prediction, proba=proba, metrics=metrics)

if __name__ == '__main__':
    app.run(debug=True)
