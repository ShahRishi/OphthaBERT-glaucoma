from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
import re
import json
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to the config file')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = json.load(f)

# Initialize the Flask app
app = Flask(__name__)

# Load the tokenizer and models
tokenizer = DistilBertTokenizer.from_pretrained(config["tokenizer"])

# Binary classification model
binary_model = DistilBertForSequenceClassification.from_pretrained(config["binary_model"], num_labels=2)
binary_model.load_adapter(config["binary_adapter"])
binary_model.eval()

# Subtypes classification model
subtypes_model = DistilBertForSequenceClassification.from_pretrained(config["subtypes_model"], num_labels=5)
subtypes_model.load_adapter(config["subtypes_adapter"])
subtypes_model.eval()

# Helper function to preprocess text
def remove_stopwords(text):
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'\"-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text.lower()

@app.route('/predict/binary', methods=['POST'])
def predict_binary():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400

        text = data['text']
        text = remove_stopwords(text)
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = binary_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        label_map = {0: "control", 1: "case"}
        result = {
            'text': text,
            'predicted_label': label_map[predicted_class]
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



@app.route('/predict/subtypes', methods=['POST'])
def predict_subtypes():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400

        text = data['text']
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = subtypes_model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        # Map prediction to label
        label_map = {0: "POAG", 1: "Other", 2: "PDS", 3:"PACG", 4:"PXF"}
        result = {
            'text': text,
            'predicted_label': label_map[predicted_class]
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=config["port"])
