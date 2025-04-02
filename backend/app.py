from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from React

# Load your pretrained model (ensure you have saved it as shown in your training code)
model = load_model('saved_models/audio_classification.keras')

# Define a mapping for class labels (update as needed)
label_map = {
    0: "air_conditioner", 1: "car_horn", 2: "children_playing",
    3: "dog_bark", 4: "drilling", 5: "engine_idling",
    6: "gun_shot", 7: "jackhammer", 8: "siren", 9: "street_music"
}

def preprocess_audio(file_path):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Extract MFCC features (use 40 coefﬁcients as in your training script)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

@app.route('/classify', methods=['POST'])
def classify_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    temp_filename = "temp_audio.wav"
    file.save(temp_filename)

    # Preprocess and predict
    features = preprocess_audio(temp_filename)
    predictions = model.predict(features)
    predicted_idx = int(np.argmax(predictions, axis=1)[0])

    # Remove the temporary file
    os.remove(temp_filename)

    result = {
        "predicted_label": predicted_idx,
        "class": label_map.get(predicted_idx, "unknown")
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
