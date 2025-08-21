#  Voice Command Intent Classifier
This project builds an **Intent Classifier** that uses Natural Language Processing (NLP) to categorize user voice/text commands into predefined classes such as `PlayMusic`, `GetWeather`, `BookRestaurant`, etc.  


---

## Features
-  Uses **SNIPS dataset** of labeled voice command transcripts  
-  Text vectorization with **TF-IDF (bigrams)**  
-  Trains a **Logistic Regression classifier** for intent recognition  
-  Model evaluation with accuracy, classification report, and confusion matrix  
-  Streamlit web app for live testing: enter a command → get predicted intent instantly  
-  Extendable: can integrate Speech-to-Text for direct voice input in future  

---


##  Dataset
- Source: **SNIPS Voice Command Dataset**  
- Example samples:  

| Command Text                 | Intent            |
|-------------------------------|------------------|
| "Play some jazz music"       | PlayMusic        |
| "What’s the weather in Delhi?"| GetWeather       |
| "Book a table at Domino’s"   | BookRestaurant   |
| "Rate the book Harry Potter" | RateBook         |

---

##  Setup Instructions
### 1. Clone Repository

    https://github.com/shruti629/Voice-Command-Intent-Classifier.git
    cd Voice-Command-Intent-Classifier
    
### 2. Create Virtual Environment

    python -m venv venv
    source venv/bin/activate   # (Linux/Mac)
    venv\Scripts\activate      # (Windows)
### 3. Install Requirements

    pip install -r requirements.txt
### 4. Run Streamlit App

    streamlit run app.py

## Dashboard 
https://voice-command-intent-classifier-ev5upjjrueqkffa4zetcpm.streamlit.app/




