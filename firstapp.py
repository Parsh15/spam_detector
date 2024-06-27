
import nltk
import pickle
import streamlit as st
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
cv=pickle.load(open('Countvectorizer.pkl', 'rb'))
model=pickle.load(open('model.pkl','rb'))

st.title('Email Classification')
sms = st.text_area("Enter the message")


def transform_text(text):
    # Convert to lowercase
    text = text.lower()

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    alphanumeric_tokens = [t for t in tokens if t.isalnum()]

    # Remove stopwords
    filtered_tokens = [t for t in alphanumeric_tokens if t not in stopwords.words('english')]

    # Apply stemming
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(t) for t in filtered_tokens]

    # Join tokens into a single string
    processed_text = " ".join(stemmed_tokens)

    return processed_text


if st.button("PREDICT"):
    transform1 = transform_text(sms)
    textinput = cv.transform([transform1])
    k = model.predict(textinput)
    if k == 1:
        st.header("SPAM")

    else:
        st.header("NOT SPAM")