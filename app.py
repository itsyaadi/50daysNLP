from flask import Flask, request, jsonify,render_template
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Download NLTK assets (only once)
nltk.download('stopwords')

# Parameters
voc_size = 5000
sent_length = 20

# Load model
model = load_model("fake_news_model.keras")

# Preprocess
ps = PorterStemmer()
def preprocess(text):
    review = re.sub('[^a-zA-Z0-9]', ' ', text)
    review = review.lower().split()
    review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
    return ' '.join(review)

def encode_text(text):
    encoded = one_hot(text, voc_size)
    padded = pad_sequences([encoded], maxlen=sent_length, padding='pre')
    return padded

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data.get('title', '')
    cleaned = preprocess(input_text)
    padded = encode_text(cleaned)
    pred = model.predict(padded)[0][0]
    result = 'Real' if pred > 0.5 else 'Fake'
    return jsonify({'prediction': result, 'confidence': float(pred)})

if __name__ == '__main__':
    app.run(debug=True)