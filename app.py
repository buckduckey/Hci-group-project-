from flask import Flask, request, jsonify
from flask_cors import CORS
from LLM import infer_from_data

app = Flask(__name__)
CORS(app)

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
    app.run(debug=True, port=5000)
