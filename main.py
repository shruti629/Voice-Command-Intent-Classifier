import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle


# Step 1: Load and Prepare Data

@st.cache_data
def load_data():
    df = pd.read_csv("voice_intents_dataset.csv")
    return df

df = load_data()


# Step 2: Train Model
@st.cache_resource
def train_model():
    X = df['Command Text']
    y = df['Intent']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train SVM Classifier
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train_tfidf, y_train)

    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)

    return model, vectorizer, acc, classification_report(y_test, y_pred, output_dict=True), confusion_matrix(y_test, y_pred)

model, vectorizer, accuracy, report, cm = train_model()


# Step 3: Streamlit
st.title(" Voice Command Intent Classifier")
st.write("Enter a voice command and get the predicted **intent**.")
st.write(" Example: (PlayMusic, SetAlarm, SendMessage, etc.)")

# Input box
user_input = st.text_input("Enter your voice command:")

if user_input:
    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0]
    st.success(f"Predicted Intent: **{prediction}**")

show_performance = st.checkbox("Show model performance")

if show_performance:
    # Show model accuracy
    st.subheader("📊 Model Performance")
    st.write(f"Accuracy on Test Data: **{accuracy:.2f}**")

    # Expandable section for more metrics
    with st.expander("See detailed classification report"):
        st.json(report)

    with st.expander("See confusion matrix"):
        st.write(cm)
