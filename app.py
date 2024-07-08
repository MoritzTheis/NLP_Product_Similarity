from flask import Flask, request, jsonify, render_template
import torch
from sentence_transformers import SentenceTransformer, util


# Wichtig: Bitte das folgende Modell aus dem Drive herunterladen und 
# in das Wurzelverzeichnis einfügen: 
# https://drive.google.com/file/d/1-911LWjBkAAUHwfYBj1kZXcCKt5gvob-/view?usp=drive_link

app = Flask(__name__)

# Load the pre-trained PyTorch model onto CPU
model_path = "TestSimCSE2Epochs.pt"
device = torch.device('cpu')
model = torch.load(model_path, map_location=device)
model = model.to(device)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode_texts():
    data = request.json
    text1 = data.get('text1', '')
    text2 = data.get('text2', '')
    
    # Create embeddings using SentenceTransformer
    embeddings = model.encode([text1, text2], convert_to_tensor=True, device=str(device))
    
    # Calculate cosine similarity
    cosine_similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    
    return jsonify({
        "cosine_similarity": cosine_similarity
    })

if __name__ == '__main__':
    app.run(debug=True)
