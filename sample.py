import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd
import streamlit as st

# Download NLTK data (run once)
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Sample dataset
data = {
    "Review": [
        "This product is amazing!",
        "Worst purchase ever.",
        "I love it, highly recommend.",
        "Terrible quality, do not buy.",
        "It’s okay, could be better.",
    ],
    "Sentiment": ["positive", "negative", "positive", "negative", "neutral"],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Preprocessing
def preprocess_text(text):
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return " ".join(words)

df["Processed_Review"] = df["Review"].apply(preprocess_text)

# Split the data
X = df["Processed_Review"]
y = df["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Test the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Streamlit Web App
st.title("Sentiment Analyzer")
st.write("Enter a product review to analyze its sentiment.")

user_input = st.text_area("Enter your review:", "")
if st.button("Analyze"):
    processed_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    sentiment = model.predict(vectorized_input)[0]
    st.write(f"The sentiment of the review is: **{sentiment}**")
