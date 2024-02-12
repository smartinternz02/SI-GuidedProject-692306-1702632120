import pandas as pd
import numpy as np
import tensorflow as tf
import re
import string
import pickle
from flask import Flask, request, render_template

# Define preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Convert text to lowercase
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove whitespace
    text = text.strip()
    return text

# Load author_to_id dictionary from file
try:
    with open('author_to_id.pickle', 'rb') as f:
        author_to_id = pickle.load(f)
    if not isinstance(author_to_id, dict):
        print("Warning: 'author_to_id.pickle' does not contain a dictionary. Initializing an empty dictionary.")
        author_to_id = {}
except FileNotFoundError:
    print("Error: 'author_to_id.pickle' file not found.")
    author_to_id = {}

print("Loaded 'author_to_id' dictionary:", author_to_id)

with open('tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

# Load the saved model from file
model = tf.keras.models.load_model('my_model.h5')

app = Flask(__name__)

# Define route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Define route for predict page
@app.route('/predict', methods=['POST'])
def predict():
    # Get input text from user
    input_text = request.form['text']

    # Preprocess input text
    input_text = preprocess_text(input_text)

    # Convert input text to sequence of integer IDs
    input_text_id = tokenizer.texts_to_sequences([input_text])

    # Pad input sequence with zeros
    input_text_id_padded = tf.keras.preprocessing.sequence.pad_sequences(input_text_id, maxlen=300)

    # Make prediction using the model
    y_pred = model.predict(np.asarray(input_text_id_padded))

    # Map predicted probabilities to author names
  #  id_to_author = {v: k for k, v in author_to_id.items()}
   # y_pred_author = id_to_author[y_pred.argmax()]
    
    # Map predicted probabilities to author names
    id_to_author = {v: k for k, v in author_to_id.items()}
    predicted_author_id = y_pred.argmax()
    y_pred_author = id_to_author.get(predicted_author_id, "Unknown Author")
    

    # Return result to author page
    return render_template('author.html', input_text=input_text, author=y_pred_author)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

