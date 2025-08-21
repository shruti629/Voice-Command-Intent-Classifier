import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# Step 1: Load and Prepare Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("all_intents.csv")   # <- use your merged dataset
    return df

df = load_data()

# =========================
# Step 2: Train Model
# =========================
@st.cache_resource
def train_model():
    X = df['Command Text']
    y = df['Intent']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Logistic Regression classifier
    model = LogisticRegression(max_iter=300, solver="lbfgs")
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, acc, classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred, labels=model.classes_), model.classes_

model, vectorizer, accuracy, report, cm, labels = train_model()

# =========================
# Step 3: Streamlit UI
# =========================
st.title("🎤 Voice Command Intent Classifier")
st.write("This app classifies user commands (text) into predefined **intents**.")
st.write("Example intents: `PlayMusic`, `GetWeather`, `BookRestaurant`, etc.")

# User Input
user_input = st.text_input("Enter your voice command:")

if user_input:
    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0]
    st.success(f"✅ Predicted Intent: **{prediction}**")

# Performance section
show_performance = st.checkbox("Show model performance")

if show_performance:
    st.subheader("📊 Model Performance")
    st.write(f"Accuracy on Test Data: **{accuracy:.2f}**")

    with st.expander("See detailed classification report"):
        st.json(report)

    with st.expander("See confusion matrix"):
        fig, ax = plt.subplots(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        st.pyplot(fig)


