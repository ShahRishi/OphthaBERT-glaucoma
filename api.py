from flask import Flask, request, jsonify
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from adapters import AutoAdapterModel

app = Flask(__name__)

model = DistilBertForSequenceClassification.from_pretrained("medicalai/ClinicalBERT")
tokenizer = DistilBertTokenizer.from_pretrained('medicalai/ClinicalBERT')
model.load_adapter("./models/ophtabert-binary")

model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if 'text' not in data:
            return jsonify({'error': 'Text field is required'}), 400

        text = data['text']
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        with torch.no_grad():
            outputs = model(**inputs)
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
