from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch

# Initialize the Flask app
app = Flask(__name__)

# Load the tokenizer and models
tokenizer = DistilBertTokenizer.from_pretrained("ShahRishi/OphthaBERT")

# Binary classification model
binary_model = DistilBertForSequenceClassification.from_pretrained("ShahRishi/OphthoBERT", num_labels=2)
binary_model.load_adapter("ShahRishi/ophthabert-glaucoma-binary")
binary_model.eval()

# Subtypes classification model
subtypes_model = DistilBertForSequenceClassification.from_pretrained("ShahRishi/OphthoBERT", num_labels=5)
subtypes_model.load_adapter("ShahRishi/ophthabert-glaucoma-subtypes")
subtypes_model.eval()


@app.route('/predict/binary', methods=['POST'])
def predict_binary():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400

        text = data['text']
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
    app.run(host='0.0.0.0', port=8080)