import logging
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from LLM import infer_from_data

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    # Serve only the page itself, not the whole project folder
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/infer', methods=['POST'])
def infer():
    data = request.json
    
    # Bundle the 11 questionnaire inputs into a text string for the LLM
    prompt_text = f"""
    Daily Digital Immersion: {data.get('q1')} hours
    Late-night usage: {data.get('q2')}
    Task Context: {data.get('q3')}
    Interaction Cadence: {data.get('q4')}
    Device Telemetry Tracked: {data.get('q5')}
    Keystroke Biometrics Logged: {data.get('q6')}
    Attention/Scroll Tracking: {data.get('q7')}
    Facial Scans Active: {data.get('q8')}
    Privacy Tooling: {data.get('q9')}
    Dark Patterns Present: {data.get('q10')}
    Cross-Device Geolocation Linked: {data.get('q11')}
    """
    
    # Calls your existing LLM.py function
    result = infer_from_data(prompt_text)
    
    if result:
        return jsonify(result)
    else:
        return jsonify({"error": "Failed to generate inference from local model."}), 500

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    app.run(debug=True, port=5000)
